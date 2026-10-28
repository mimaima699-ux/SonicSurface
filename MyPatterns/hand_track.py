#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RealSense 指尖三维跟踪: MediaPipe 定位 + 对齐深度 + 阵列坐标标定。"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass

import numpy as np

import my_patterns
from my_patterns import Board, Modulator, pattern_for_kind

# MSMF 的硬件解码变换(HW MFT)是很多 USB 相机 OnReadSample 报错(-1072873822,
# 打开成功但永远读不到帧)的元凶;关闭之。必须在创建任何 VideoCapture 之前设置。
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

CALIB_PATH = Path(__file__).resolve().parent / "results/hand_tracking/hand_track_calib_3d.json"
DIAGNOSTIC_PATH = CALIB_PATH.with_name(CALIB_PATH.stem + "_diagnostic.json")
CAM_SETTINGS_PATH = Path(__file__).resolve().parent / "results/hand_tracking/hand_track_cam.json"
N = my_patterns.N_EMITTERS
OFF_PATTERN = np.full(N, np.nan)
DEFAULT_BOUNDS = ((-0.08, 0.08), (0.07, 0.20), (-0.08, 0.08))
EXPERIMENTAL_BOUNDS = ((-0.03, 0.03), (0.09, 0.16), (-0.03, 0.03))


# ======================================================================
# 滤波与死区
# ======================================================================

class OneEuro:
    """One Euro 滤波(Casiez 2012),支持二维或三维位置。"""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = None
        self._dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            return x.copy()
        dt = max(dt, 1e-3)
        dx = (x - self._x) / dt
        a = self._alpha(self.d_cutoff, dt)
        self._dx = a * dx + (1.0 - a) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()

    def reset(self):
        self._x = None
        self._dx = None


class Deadband:
    """位置变化小于半径时不重发(省串口带宽,防图案微跳)。"""

    def __init__(self, radius):
        self.radius = float(radius)
        self.last = None

    def update(self, pos):
        pos = np.asarray(pos, float)
        if self.last is None or np.linalg.norm(pos - self.last) >= self.radius:
            self.last = pos.copy()
            return pos.copy()
        return None

    def reset(self):
        self.last = None


def in_bounds(pos, bounds):
    pos = np.asarray(pos, float)
    return (len(pos) == len(bounds) and np.isfinite(pos).all()
            and all(lo <= v <= hi for v, (lo, hi) in zip(pos, bounds)))


def tracking_gate(uv, camera_point, pos, bounds):
    """Explain why a detected fingertip may not produce an active focus."""
    if uv is None:
        return "no_hand"
    if camera_point is None:
        return "no_depth"
    if pos is None or not np.isfinite(pos).all():
        return "invalid_position"
    axes = outside_axes(pos, bounds)
    if axes:
        return f"outside_{axes[0]}"
    return "inside"


def outside_axes(pos, bounds):
    return [axis for value, (lo, hi), axis in zip(pos, bounds, "xyz")
            if value < lo or value > hi]


def experimental_bounds(span, height_min=0.09, height_max=0.16):
    """Limit provisional focus to the existing array operating envelope."""
    span = float(span)
    if not np.isfinite(span) or not 0.02 <= span <= 0.08:
        raise SystemExit("--experimental-span 须在 0.02-0.08 m 之间")
    if (not np.isfinite(height_min) or not np.isfinite(height_max)
            or not 0.07 <= height_min < height_max <= 0.20):
        raise SystemExit("实验高度范围须在 0.07-0.20 m 内且下限小于上限")
    return ((-span, span), (height_min, height_max), (-span, span))


# ======================================================================
# 控制核心 —— 与数据来源解耦(摄像头/合成路径共用)
# ======================================================================

class PalmFollower:
    """喂入 (时刻 t 秒, 指尖坐标 (x,y,z) 或 None=丢手),按需 emit(256 相位)。
    stats 记录 sent/off_times/positions 供离线验证。"""

    def __init__(self, emit, kind="focus", dist=0.12, m=1,
                 deadband_m=0.005, lost_s=0.5, bounds=DEFAULT_BOUNDS):
        self.emit = emit
        self.kind = kind
        self.dist = dist
        self.m = m
        self.lost_s = lost_s
        self.bounds = bounds
        # beta 量级适配米制速度:静止 2Hz 压抖动,~80mm/s 手速时 cutoff≈4.4Hz、滞后≈3mm
        self.filter = OneEuro(min_cutoff=2.0, beta=30.0)
        self.band = Deadband(deadband_m)
        self._prev_t = None
        self._last_seen = None
        self._off = True
        self._active_since = None
        self._prev_phases = None
        self._trap_cache = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {"sent": 0, "off_times": [], "positions": [], "active_s": 0.0}

    def _trap_phases(self, pos):
        cell = tuple(int(np.rint(float(v) / 0.005)) for v in pos)
        if cell in self._trap_cache:
            phases = self._trap_cache.pop(cell)
        else:
            phases = my_patterns.phases_trap(
                tuple(v * 0.005 for v in cell),
                x0=self._prev_phases, maxiter=30)
        self._trap_cache[cell] = phases.copy()
        if len(self._trap_cache) > 256:
            self._trap_cache.popitem(last=False)
        if self._prev_phases is not None:
            offset = np.angle(np.vdot(np.exp(1j * self._prev_phases),
                                      np.exp(1j * phases)))
            phases = my_patterns._wrap(phases - offset)
        return phases.copy()

    def _go_off(self, t):
        if self._active_since is not None:
            self.stats["active_s"] += max(0.0, t - self._active_since)
            self._active_since = None
        if not self._off:
            self.emit(OFF_PATTERN.copy())
            self._off = True
            self.stats["off_times"].append(t)
        self.filter.reset()
        self.band.reset()
        self._prev_phases = None
        self._prev_t = None

    def feed(self, t, pos):
        with self._lock:
            self._feed(t, pos)

    def _feed(self, t, pos):
        if pos is not None and not in_bounds(pos, self.bounds):
            self._go_off(t)                      # 越界:立即静默
            return
        if pos is not None:
            self._last_seen = t
            dt = 1.0 / 30.0 if self._prev_t is None else t - self._prev_t
            self._prev_t = t
            p = self.filter(pos, dt)
            new = self.band.update(p)
            if new is not None:
                if self.kind == "trap":
                    ph = self._trap_phases(new)
                else:
                    ph = pattern_for_kind(self.kind, float(new[0]), float(new[2]),
                                          float(new[1]), m=self.m)
                self.emit(ph)
                if self._active_since is None:
                    self._active_since = t
                self._off = False
                self._prev_phases = ph
                self.stats["sent"] += 1
                self.stats["positions"].append((t, *map(float, new)))
        else:
            if self._last_seen is not None and t - self._last_seen > self.lost_s:
                self._go_off(t)                  # 丢手超时:静默

    def close(self, t=None):
        with self._lock:
            if self._active_since is not None:
                end = self._last_seen if t is None else t
                self.stats["active_s"] += max(0.0, end - self._active_since)
                self._active_since = None
            if not self._off:
                self.emit(OFF_PATTERN.copy())
                self._off = True


class LossSupervisor:
    """独立于相机 read() 的丢手监督,避免阻塞读取延迟关断。"""

    def __init__(self, follower, on_error=None):
        self.follower = follower
        self.on_error = on_error
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(0.05):
            now = time.perf_counter()
            try:
                with self.follower._lock:
                    last = self.follower._last_seen
                    if last is not None and now - last > self.follower.lost_s:
                        self.follower._go_off(now)
            except Exception as exc:
                self.error = exc
                if self.on_error is not None:
                    try:
                        self.on_error()
                    except Exception:
                        pass
                return

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


# ======================================================================
# 摄像头:MediaPipe 掌心检测 + 单应标定
# ======================================================================

MODEL_PATH = Path(__file__).resolve().parent / "assets/hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
TARGET_LANDMARKS = {"thumb_tip": 4, "index_tip": 8, "middle_tip": 12,
                    "ring_tip": 16, "pinky_tip": 20}
TARGETS = ("palm", *TARGET_LANDMARKS)


@dataclass
class HandObservation:
    landmarks: np.ndarray
    mask: np.ndarray

    def point(self, target):
        if target == "palm":
            point = 0.5 * (self.landmarks[0] + self.landmarks[9])
        else:
            point = self.landmarks[TARGET_LANDMARKS[target]]
        h, w = self.mask.shape
        if not np.isfinite(point).all() or not (0 <= point[0] < w and 0 <= point[1] < h):
            return None
        return tuple(point)


def hand_mask_from_landmarks(points, shape):
    """Approximate binary hand silhouette from palm polygon and finger bones."""
    import cv2
    h, w = shape[:2]
    pts = np.asarray(points, float)
    if pts.shape != (21, 2):
        raise ValueError("Expected 21 hand landmarks")
    pixels = np.rint(pts).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    palm_width = float(np.linalg.norm(pts[5] - pts[17]))
    base = max(2, int(round(palm_width * 0.17)))
    palm = cv2.convexHull(pixels[[0, 1, 2, 5, 9, 13, 17]])
    cv2.fillConvexPoly(mask, palm, 255)
    for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                  (13, 14, 15, 16), (17, 18, 19, 20)):
        for i, (a, b) in enumerate(zip(chain, chain[1:])):
            radius = max(2, int(round(base * (1.0 - 0.18 * i))))
            cv2.line(mask, tuple(pixels[a]), tuple(pixels[b]), 255, 2 * radius)
            cv2.circle(mask, tuple(pixels[b]), radius, 255, -1)
    return mask


def draw_hand_observation(frame, observation, target):
    """Overlay the approximate mask and selected target on a preview frame."""
    import cv2
    display = frame.copy()
    if observation is None:
        return display
    tint = np.zeros_like(display)
    tint[:, :] = (40, 170, 40)
    display[observation.mask > 0] = cv2.addWeighted(
        display, 0.7, tint, 0.3, 0)[observation.mask > 0]
    point = observation.point(target)
    if point is not None:
        x, y = point
        cv2.circle(display, (int(round(x)), int(round(y))), 9, (0, 0, 255), 2)
    return display


def _download_model(dest, url=MODEL_URL):
    import urllib.request
    print(f"首次使用新版 MediaPipe:下载手势模型 {dest.name}(约 8MB)...")
    try:
        urllib.request.urlretrieve(url, dest)
        print("模型下载完成")
    except Exception as exc:
        raise SystemExit(f"模型下载失败({exc})。请手动下载:\n  {url}\n存为 {dest} 后重试")


class PalmDetector:
    """MediaPipe 掌心检测封装,兼容两代 API:
    旧版 mp.solutions.hands(0.10.x,免模型文件)或新版 Tasks API(1.x,首次下载模型)。
    返回掌心像素坐标 (u, v):wrist(0) 与 middle_mcp(9) 中点;未检测到手返回 None。"""

    def __init__(self, num_hands=1):
        import mediapipe as mp
        self._legacy = None
        self._landmarker = None
        if hasattr(mp, "solutions"):
            self._legacy = mp.solutions.hands.Hands(static_image_mode=False,
                                                    max_num_hands=num_hands,
                                                    model_complexity=0)
        else:
            if not MODEL_PATH.exists():
                _download_model(MODEL_PATH)
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                num_hands=num_hands)
            self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._legacy is not None:
            res = self._legacy.process(rgb)
            if not res.multi_hand_landmarks:
                return None
            lm = res.multi_hand_landmarks[0].landmark
        else:
            res = self._landmarker.detect(mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                return None
            lm = res.hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        points = np.array([(point.x * w, point.y * h) for point in lm], float)
        if points.shape != (21, 2) or not np.isfinite(points).all():
            return None
        return HandObservation(points, hand_mask_from_landmarks(points, frame_bgr.shape))

    def palm(self, frame_bgr):
        observation = self.detect(frame_bgr)
        return observation.point("palm") if observation is not None else None

    def close(self):
        for obj in (self._legacy, self._landmarker):
            if obj is not None:
                obj.close()


# ----------------------------------------------------------------------
# 摄像头设置 —— 亮度/变焦/分辨率整定(camtest 调好存 JSON,其余模式自动套用)
# ----------------------------------------------------------------------

def load_cam_settings(path=CAM_SETTINGS_PATH):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _backend_id(name):
    import cv2
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[name]


def _quiet_cv():
    """屏蔽 OpenCV 探测时的 WARN/ERROR 刷屏(如 obsensor 'index out of range')。"""
    try:
        import cv2
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass


def _open_attempt(cam, backend_name, settings):
    """单次打开尝试:指定后端 + 应用设置 + 暖机验证能出帧。返回 cap 或 None。"""
    import cv2
    cap = cv2.VideoCapture(cam, _backend_id(backend_name))
    if not cap.isOpened():
        cap.release()
        return None
    _apply_cam_settings(cap, settings)
    if _warmup(cap):
        return cap
    cap.release()
    return None


def _apply_cam_settings(cap, settings):
    import cv2
    if settings.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings["fourcc"]))
    for key, prop in (("width", cv2.CAP_PROP_FRAME_WIDTH),
                      ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                      ("fps", cv2.CAP_PROP_FPS),
                      ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
                      ("exposure", cv2.CAP_PROP_EXPOSURE),
                      ("brightness", cv2.CAP_PROP_BRIGHTNESS),
                      ("gain", cv2.CAP_PROP_GAIN),
                      ("zoom", cv2.CAP_PROP_ZOOM)):
        if settings.get(key) is not None:
            cap.set(prop, settings[key])


def _warmup(cap, seconds=3.0):
    """等驱动吐出首帧;UVC 相机开头几帧 read() 返回空属正常,不重试会误判"打不开"。"""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if cap.read()[0]:
            return True
        time.sleep(0.05)
    return False


def read_frame(cap, timeout_s=2.0):
    """读一帧,短暂失败自动重试(偶发丢帧不致命),超时才返回 False。"""
    end = time.perf_counter() + timeout_s
    while True:
        ok, frame = cap.read()
        if ok:
            return ok, frame
        if time.perf_counter() >= end:
            return False, None
        time.sleep(0.05)


def open_camera(cam, settings=None):
    """按设置打开摄像头:后端链 × 补救设置 逐组合尝试,暖机等首帧。返回 (cap, 后端名)。
    打不开/出不了帧的常见来源与对策:
      被其它程序独占占用 → 关程序;后端协商失败(DSHOW 拒开) → 自动换 MSMF;
      MSMF 硬件变换不兼容 → 模块级环境变量已关;USB2 带宽不足 → 自动降级 MJPG/640x480;
      驱动首帧慢 → 暖机重试。
    另两个画面坑:默认 640x480 从 16:9 传感器裁切 → 看起来"变焦过高",换 1280x720;
    默认手动低曝光 → 过暗,开自动曝光。"""
    settings = dict(settings or {})
    first = settings.pop("backend", "dshow")
    order = [first] + [b for b in ("dshow", "msmf", "any") if b != first]
    rescues = ({},                                       # 用户设置原样
               {"fourcc": "MJPG"},                       # USB2 带宽不足 → 换编码
               {"fourcc": "MJPG", "width": 640, "height": 480})   # 再降分辨率
    _quiet_cv()
    tried = []
    for backend_name in order:
        for rescue in rescues:
            merged = {**settings, **rescue}
            cap = _open_attempt(cam, backend_name, merged)
            if cap is not None:
                return cap, backend_name
            tried.append(backend_name + (f"+{sorted(rescue)}" if rescue else ""))
            time.sleep(0.3)
    raise SystemExit(
        f"摄像头 {cam} 打不开或读不到画面(已尝试 {' / '.join(tried)})。\n"
        "排查:①关闭占用摄像头的程序(Windows 相机/微信/QQ/浏览器/会议软件/"
        "任务管理器里未退出的 python.exe);②重插 USB、换接口(优先主板后置口,别接集线器);\n"
        "③Windows 设置→隐私和安全→相机→允许桌面应用访问;④加 --backend msmf 再试")


class RealSenseSource:
    """RealSense 帧源;depth=True 时将深度对齐到彩色帧。"""

    def __init__(self, width=1280, height=720, fps=30, depth=False):
        import pyrealsense2 as rs
        self._rs = rs
        devices = rs.context().query_devices()
        if len(devices) == 0:
            raise SystemExit("RealSense SDK 未发现设备，请检查连接与设备管理器")
        self.serial = devices[0].get_info(rs.camera_info.serial_number)
        profiles = [p for sensor in devices[0].query_sensors()
                    for p in sensor.get_stream_profiles()]
        self.stream_name = "color" if any(
            p.stream_type() == rs.stream.color for p in profiles) else "infrared"
        self.depth_enabled = depth
        self.depth_m = None
        self.intrinsics = None
        self._depth_frame = None
        self._color_bgr = None
        if depth and self.stream_name != "color":
            raise SystemExit("三维跟踪需要 RealSense 彩色流和深度流;当前设备没有彩色流")
        if self.stream_name == "color":
            stream, fmt, index = rs.stream.color, rs.format.bgr8, 0
        else:
            stream, fmt, index = rs.stream.infrared, rs.format.y8, 1
        candidates = [(width, height), (640, 480)]
        supported = [size for size in candidates if any(
            p.stream_type() == stream and p.format() == fmt and p.fps() == fps
            and p.as_video_stream_profile().width() == size[0]
            and p.as_video_stream_profile().height() == size[1]
            and (index == 0 or p.stream_index() == index)
            for p in profiles)]
        if not supported:
            raise SystemExit(f"RealSense 未提供 {self.stream_name} BGR8/Y8 {fps}fps 视频模式")
        self.width, self.height = supported[0]
        self.pipeline = rs.pipeline()
        config = rs.config()
        if index:
            config.enable_stream(stream, index, self.width, self.height, fmt, fps)
        else:
            config.enable_stream(stream, self.width, self.height, fmt, fps)
        if depth:
            depth_modes = [p for p in profiles if p.stream_type() == rs.stream.depth
                           and p.format() == rs.format.z16 and p.fps() == fps]
            if not depth_modes:
                raise SystemExit("RealSense 未提供 30fps Z16 深度流")
            mode = next((p for p in depth_modes if
                         (p.as_video_stream_profile().width(),
                          p.as_video_stream_profile().height()) == (self.width, self.height)), None)
            if mode is None:
                raise SystemExit("RealSense 彩色和深度流没有共同分辨率")
            video = mode.as_video_stream_profile()
            config.enable_stream(rs.stream.depth, video.width(), video.height(), rs.format.z16, fps)
        try:
            self.profile = self.pipeline.start(config)
            if depth:
                self._align = rs.align(rs.stream.color)
                self._depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
                self._pointcloud = rs.pointcloud()
        except Exception as exc:
            hardware_hint = ("设备序列号为 ffffffffffff；若同时识别成通用 D400，"
                             "优先检查相机模组与处理板之间的排线连接。\n"
                             if self.serial == "ffffffffffff" else "")
            raise SystemExit(
                f"RealSense {self.stream_name} {self.width}x{self.height} 流启动失败({exc})。\n"
                f"{hardware_hint}"
                "在 RealSense Viewer 中尝试相同流；若也报 hwmon 错误，请检查设备占用、"
                "固件/驱动和 USB 连接") from exc
        first = None
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and first is None:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                first = self._frame(frames) if frames else None
            except Exception:
                pass
        if first is None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            raise SystemExit(f"RealSense 5 秒内等不到 {self.stream_name} 帧")

    def _frame(self, frames):
        if self.stream_name == "color":
            return frames.get_color_frame()
        return frames.get_infrared_frame(1)

    def read(self):
        deadline = time.perf_counter() + 0.2
        while time.perf_counter() < deadline:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
            except Exception:
                return False, None
            if self.depth_enabled and frames:
                frames = self._align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    self.depth_m = None
                    self._depth_frame = None
                    continue
                depth_data = np.asanyarray(depth_frame.get_data())
                if depth_data.shape != (color_frame.get_height(), color_frame.get_width()):
                    self.depth_m = None
                    self._depth_frame = None
                    continue
                self.depth_m = depth_data.astype(np.float32) * self._depth_scale
                self._depth_frame = depth_frame
                self.intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            frame = self._frame(frames) if frames else None
            if frame is not None:
                image = np.asanyarray(frame.get_data())
                if self.stream_name == "infrared":
                    import cv2
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if self.depth_enabled:
                    self._color_bgr = image
                return True, image
        return False, None

    def isOpened(self):
        return True

    def release(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def point_3d(self, uv, mask=None):
        """从指尖局部深度反投影;孔洞或背景边缘深度不可靠时返回 None。"""
        if not self.depth_enabled or self.depth_m is None or self.intrinsics is None:
            return None
        depth = fingertip_depth(self.depth_m, uv, mask)
        if depth is None:
            return None
        return np.asarray(self._rs.rs2_deproject_pixel_to_point(
            self.intrinsics, list(map(float, uv)), depth), float)

    def point_cloud(self, uv=None):
        """Downsample the aligned SDK point cloud, retaining detail near the target."""
        if self._depth_frame is None or self._color_bgr is None:
            return np.empty((0, 3)), np.empty((0, 3))
        h, w = self.depth_m.shape
        points = self._pointcloud.calculate(self._depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3)
        sample = np.zeros((h, w), bool)
        sample[::12, ::12] = True
        if uv is not None:
            u, v = map(int, np.rint(uv))
            x0, x1 = max(0, u - 100), min(w, u + 101)
            y0, y1 = max(0, v - 100), min(h, v + 101)
            sample[y0:y1:3, x0:x1:3] = True
        valid = sample & np.isfinite(vertices).all(axis=2)
        valid &= (self.depth_m >= 0.07) & (self.depth_m <= 0.50)
        return vertices[valid], self._color_bgr[valid, ::-1] / 255.0

    def _sensor(self):
        device = self.profile.get_device()
        return (device.first_color_sensor() if self.stream_name == "color"
                else device.first_depth_sensor())

    def get_exposure(self):
        try:
            return float(self._sensor().get_option(self._rs.option.exposure))
        except Exception:
            return None

    def get_auto_exposure(self):
        try:
            return bool(self._sensor().get_option(self._rs.option.enable_auto_exposure))
        except Exception:
            return None

    def set_exposure(self, value_us):
        try:
            self._sensor().set_option(self._rs.option.enable_auto_exposure, 0.0)
            self._sensor().set_option(self._rs.option.exposure, float(value_us))
        except Exception as exc:
            print(f"  曝光设置失败: {exc}")

    def set_auto_exposure(self, enabled):
        try:
            self._sensor().set_option(self._rs.option.enable_auto_exposure,
                                      1.0 if enabled else 0.0)
        except Exception as exc:
            print(f"  自动曝光设置失败: {exc}")


def open_source(cam, settings=None, realsense=False, depth=False):
    """统一帧源入口:显式 --realsense 或 hand_track_cam.json 记录了 realsense
    则走 SDK 彩色/红外流,其余走 cv2 后端链。返回 (帧源, 名字)。"""
    settings = dict(settings or {})
    if realsense or settings.get("backend") == "realsense":
        source = RealSenseSource(depth=True) if depth else RealSenseSource()
        if settings.get("rs_auto_exposure") is not None:
            source.set_auto_exposure(bool(settings["rs_auto_exposure"]))
        if settings.get("rs_auto_exposure") is False and settings.get("rs_exposure_us") is not None:
            source.set_exposure(settings["rs_exposure_us"])
        return source, "realsense"
    return open_camera(cam, settings)


def list_cams(max_index=7, snap_dir=None):
    """逐 (后端, 索引) 组合探测并各存快照 cam_<后端>_<索引>.png。
    DSHOW 与 MSMF 的设备枚举顺序可能不同——同一台 USB 相机可能在 DSHOW 下
    打不开、在 MSMF 下是任意索引,所以两个后端分开扫。"""
    import cv2
    snap_dir = Path(snap_dir) if snap_dir else Path(__file__).resolve().parent / "results/hand_tracking"
    print("探测摄像头(每组合存快照 cam_<后端>_<索引>.png):")
    _quiet_cv()
    found = 0
    for backend in ("dshow", "msmf"):
        for i in range(max_index + 1):
            cap = _open_attempt(i, backend, {})
            if cap is None and backend == "msmf":
                cap = _open_attempt(i, backend, {"fourcc": "MJPG"})
            if cap is None:
                continue
            ok, frame = read_frame(cap)
            if ok:
                path = snap_dir / f"cam_{backend}_{i}.png"
                cv2.imwrite(str(path), frame)
                h, w = frame.shape[:2]
                print(f"  --cam {i} --backend {backend}: {w}x{h}  快照 {path.name}")
                found += 1
            cap.release()
    if not found:
        print("  一个摄像头都没找到(检查 USB/驱动)")
    else:
        print("捂住镜头重跑一次,变黑的那张快照就是对应相机;后续命令带上同款 --cam/--backend")
    return found


def depth_colormap(depth_m):
    """Colorize aligned depth from 7 to 50 cm; invalid pixels remain black."""
    import cv2
    scaled = np.clip((np.asarray(depth_m) - 0.07) / 0.43, 0, 1)
    colored = cv2.applyColorMap(np.uint8(np.rint((1 - scaled) * 255)), cv2.COLORMAP_TURBO)
    valid = np.isfinite(depth_m) & (depth_m >= 0.07) & (depth_m <= 0.50)
    colored[~valid] = 0
    return colored


def open_preview_window(name, image):
    """Open a resizable preview without reducing the image pixel data."""
    import cv2
    height, width = image.shape[:2]
    scale = min(1.0, 1400 / width, 800 / height)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, int(round(width * scale)), int(round(height * scale)))


class PointCloudPreview:
    """Interactive camera-coordinate point cloud for depth diagnosis."""

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = plt.figure("RealSense 3D", figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set(xlim=(-0.35, 0.35), ylim=(0.05, 0.50), zlim=(-0.25, 0.25),
                    xlabel="Camera X (m)", ylabel="Depth (m)", zlabel="Up (m)")
        self.ax.set_box_aspect((0.7, 0.45, 0.5))
        self.scatter = self.ax.scatter([], [], [], s=2, depthshade=False)
        self._last_update = 0.0
        plt.show(block=False)

    def update(self, source, uv):
        if not self.plt.fignum_exists(self.fig.number):
            return
        now = time.perf_counter()
        if now - self._last_update >= 0.2:
            vertices, colors = source.point_cloud(uv)
            if len(vertices):
                self.scatter._offsets3d = (vertices[:, 0], vertices[:, 2], -vertices[:, 1])
                self.scatter.set_color(colors)
                self.fig.canvas.draw_idle()
            self._last_update = now
        self.plt.pause(0.001)

    def close(self):
        self.plt.close(self.fig)


def camtest(cam=0, settings_path=CAM_SETTINGS_PATH, backend=None, realsense=False,
            target="index_tip", depth=False, pointcloud=False):
    """摄像头诊断/整定:实时窗口用键盘调参,ESC 退出并保存 hand_track_cam.json。
    W/S 曝光±  A/D 亮度±  Q/E 增益±  -/= 变焦±  X 自动曝光开关
    R 循环分辨率(640x480→1280x720→1920x1080)  P 打印当前参数
    画面上实时显示分辨率/读回参数/掌心检测状态(MediaPipe 直接验证亮度够不够检测)。
    RealSense 模式(--realsense)仅支持 W/S 曝光与 X 自动曝光,
    优先 1280x720,设备不支持时使用 640x480。"""
    import cv2

    settings = load_cam_settings(settings_path)
    if backend:
        settings["backend"] = backend
    if (depth or pointcloud) and not (realsense or settings.get("backend") == "realsense"):
        raise SystemExit("深度图/三维点云需要 --realsense 和 RealSense 深度相机")
    overrides = {k: v for k, v in settings.items() if k != "backend" and v is not None}
    cap, used_backend = open_source(cam, settings, realsense=realsense,
                                    depth=depth or pointcloud)
    print(f"相机已打开(后端 {used_backend})")
    detector = PalmDetector()
    cloud_view = PointCloudPreview() if pointcloud else None
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    res_idx = 0
    print("camtest: W/S曝光 A/D亮度 Q/E增益 -/=变焦 X自动曝光 R分辨率 P打印 ESC保存退出")
    fail = 0
    window_name = "camtest (ESC = save & quit)"
    window_open = False
    fullscreen = False
    view_mode = 0
    while True:
        ok, frame = read_frame(cap)
        if not ok:
            fail += 1
            print("  等待画面...(读不到帧)")
            if fail >= 5:                    # 连续 ~10s 无帧
                raise SystemExit("持续读不到画面:摄像头被占用或已断开,重插后重试")
            continue
        fail = 0
        observation = detector.detect(frame)
        uv = observation.point(target) if observation is not None else None
        disp = draw_hand_observation(frame, observation, target)
        is_rs = isinstance(cap, RealSenseSource)
        if is_rs:
            h_px, w_px = frame.shape[:2]
            exp = cap.get_exposure()
            auto_exp = cap.get_auto_exposure()
            info = (f"RealSense {w_px}x{h_px} @30fps",
                    f"exp={exp:.0f}us auto={'ON' if auto_exp else 'OFF' if auto_exp is False else '?'}"
                    if exp is not None else "exp=?",
                    f"{target}={'OK' if uv is not None else '--'}")
        else:
            info = (f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                    f"@{cap.get(cv2.CAP_PROP_FPS):.0f}fps",
                    f"exp={cap.get(cv2.CAP_PROP_EXPOSURE):g} "
                    f"bri={cap.get(cv2.CAP_PROP_BRIGHTNESS):g} "
                    f"gain={cap.get(cv2.CAP_PROP_GAIN):g} "
                    f"zoom={cap.get(cv2.CAP_PROP_ZOOM):g} "
                    f"autoexp={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g}",
                    f"{target}={'OK' if uv is not None else '--'}")
        for i, line in enumerate(info):
            cv2.putText(disp, line, (10, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1, cv2.LINE_AA)
        if depth or pointcloud:
            depth_display = depth_colormap(cap.depth_m)
            if uv is not None:
                cv2.circle(depth_display, tuple(map(int, np.rint(uv))), 9, (255, 255, 255), 2)
                point = cap.point_3d(uv, observation.mask)
                tip_depth = f"tip depth={point[2]:.3f}m" if point is not None else "tip depth=--"
            else:
                tip_depth = "tip depth=--"
            cv2.putText(depth_display, f"Depth 0.07-0.50m  {tip_depth}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if view_mode == 0:
                disp = np.hstack((disp, depth_display))
            elif view_mode == 2:
                disp = depth_display
            if cloud_view is not None:
                cloud_view.update(cap, uv)
        if not window_open:
            open_preview_window(window_name, disp)
            window_open = True
        cv2.imshow(window_name, disp)

        def nudge(prop, step, name):
            cur = cap.get(prop)
            new = cur + step
            cap.set(prop, new)
            overrides[name] = new
            print(f"  {name}: {cur:g} -> {new:g} (读回 {cap.get(prop):g})")

        key = cv2.waitKey(30) & 0xFF
        if key == 27:                                  # ESC
            break
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif (depth or pointcloud) and key == ord("v"):
            view_mode = (view_mode + 1) % 3
            print(f"  预览: {('并排', '彩色', '深度')[view_mode]}")
            if not fullscreen:
                width = frame.shape[1] * (2 if view_mode == 0 else 1)
                height = frame.shape[0]
                scale = min(1.0, 1400 / width, 800 / height)
                cv2.resizeWindow(window_name, int(round(width * scale)),
                                 int(round(height * scale)))
        elif is_rs and key in (ord("w"), ord("s")):
            cur = cap.get_exposure()
            if cur is not None:
                new = max(1.0, cur + (1000.0 if key == ord("w") else -1000.0))
                cap.set_exposure(new)
                print(f"  auto_exposure: off; exposure: {cur:.0f} -> {new:.0f} us")
        elif is_rs and key == ord("x"):
            on = cap.get_auto_exposure()
            if on is not None:
                cap.set_auto_exposure(not on)
                print(f"  auto_exposure: {'on' if on else 'off'} -> "
                      f"{'off' if on else 'on'}")
        elif is_rs and key in (ord("a"), ord("d"), ord("q"), ord("e"), ord("-"),
                               ord("_"), ord("="), ord("+"), ord("r")):
            print("  RealSense 模式仅支持 W/S 曝光与 X 自动曝光")
        elif key == ord("w"):
            nudge(cv2.CAP_PROP_EXPOSURE, +1, "exposure")
        elif key == ord("s"):
            nudge(cv2.CAP_PROP_EXPOSURE, -1, "exposure")
        elif key == ord("d"):
            nudge(cv2.CAP_PROP_BRIGHTNESS, +8, "brightness")
        elif key == ord("a"):
            nudge(cv2.CAP_PROP_BRIGHTNESS, -8, "brightness")
        elif key == ord("e"):
            nudge(cv2.CAP_PROP_GAIN, +8, "gain")
        elif key == ord("q"):
            nudge(cv2.CAP_PROP_GAIN, -8, "gain")
        elif key in (ord("-"), ord("_")):
            nudge(cv2.CAP_PROP_ZOOM, -20, "zoom")
        elif key in (ord("="), ord("+")):
            nudge(cv2.CAP_PROP_ZOOM, +20, "zoom")
        elif key == ord("x"):
            cur = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            new = 3 if cur != 3 else 1                 # UVC/DShow 约定: 1=手动 3=自动
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, new)
            overrides["auto_exposure"] = new
            print(f"  auto_exposure: {cur:g} -> {new:g} "
                  f"(读回 {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g})")
        elif key == ord("r"):
            res_idx = (res_idx + 1) % len(resolutions)
            w, h = resolutions[res_idx]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            overrides.update(width=w, height=h)
            print(f"  请求 {w}x{h},读回 {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        elif key == ord("p"):
            print(f"  hand_track_cam.json 内容: {json.dumps(overrides)}")
    save = {"backend": used_backend}
    if used_backend == "realsense":
        save["rs_auto_exposure"] = cap.get_auto_exposure()
        if save["rs_auto_exposure"] is False:
            save["rs_exposure_us"] = cap.get_exposure()
    else:
        save["fourcc"] = "MJPG"
        save.update(overrides)
    cap.release()
    if cloud_view is not None:
        cloud_view.close()
    cv2.destroyAllWindows()
    detector.close()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(save, indent=1), encoding="utf-8")
    print(f"摄像头设置已保存: {settings_path}")
    print("注意:分辨率/变焦改变会使既有单应标定失效,改完设置请重跑 calibrate")
    return save


def pixel_to_world(H, uv):
    """findHomography 约定: [x', y', 1]^T ~ H @ [u, v, 1]^T(列向量左乘)。"""
    v = H @ np.array([uv[0], uv[1], 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def fingertip_depth(depth_m, uv, mask=None, radius=4):
    """取目标点周围前景深度中位数;拒绝孔洞和深度不连续边界。"""
    h, w = depth_m.shape
    u, v = np.rint(uv).astype(int)
    if not (radius <= u < w - radius and radius <= v < h - radius):
        return None
    patch = depth_m[v-radius:v+radius+1, u-radius:u+radius+1]
    good = np.isfinite(patch) & (patch >= 0.07) & (patch <= 0.50)
    if mask is not None:
        good &= mask[v-radius:v+radius+1, u-radius:u+radius+1] > 0
    values = patch[good]
    if len(values) < 12:
        return None
    median = float(np.median(values))
    near = values[np.abs(values - median) <= 0.012]
    if len(near) < max(12, len(values) * 0.65):
        return None
    return float(np.median(near))


def fit_rigid_transform(camera_points, world_points):
    """Fit an orthogonal camera-to-array transform, including mirrored axis labels."""
    camera = np.asarray(camera_points, float)
    world = np.asarray(world_points, float)
    if camera.shape != world.shape or camera.ndim != 2 or camera.shape[1] != 3 or len(camera) < 6:
        raise ValueError("三维标定至少需要 6 对坐标")
    if not np.isfinite(camera).all() or not np.isfinite(world).all():
        raise ValueError("标定坐标包含无效数值")
    cc, wc = camera.mean(axis=0), world.mean(axis=0)
    if np.linalg.svd(world - wc, compute_uv=False)[-1] < 0.01:
        raise ValueError("标定点缺少足够的高度/三维跨度")
    if np.linalg.svd(camera - cc, compute_uv=False)[-1] < 0.01:
        raise ValueError("相机采样点近乎共面;请检查指尖深度是否取到了背景")
    U, _, Vt = np.linalg.svd((camera - cc).T @ (world - wc))
    # Array x/z directions may be labeled with either handedness; distances
    # used by the acoustic solver are invariant under an axis reflection.
    R = Vt.T @ U.T
    t = wc - R @ cc
    errors = np.linalg.norm(camera @ R.T + t - world, axis=1)
    return R, t, errors


def calibration_from_diagnostic(diagnostic):
    """Recheck a rejected sample set for a tightly limited hardware experiment."""
    try:
        camera = np.asarray(diagnostic["points"]["camera"], float)
        world = np.asarray(diagnostic["points"]["world"], float)
        R, t, errors = fit_rigid_transform(camera, world)
        rms_mm = float(np.sqrt(np.mean(errors ** 2)) * 1000)
        max_mm = float(np.max(errors) * 1000)
        serial = str(diagnostic["serial"])
        frame_size = list(diagnostic["frame_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"实验诊断文件不完整: {exc}") from exc
    if (camera.shape != (18, 3) or len(serial) < 3 or len(frame_size) != 2
            or rms_mm > 15 or max_mm > 30):
        raise SystemExit(f"诊断数据不符合实验限制(RMS {rms_mm:.1f}mm,最大 {max_mm:.1f}mm)")
    return {"version": 2, "mode": "realsense_3d_experimental",
            "R": R.tolist(), "t": t.tolist(), "rms_mm": rms_mm, "max_mm": max_mm,
            "target": "index_tip", "frame_size": frame_size, "stream": "color",
            "serial": serial}


def validate_calibration(calib, target, experimental=False):
    expected_mode = "realsense_3d_experimental" if experimental else "realsense_3d"
    if calib.get("version") != 2 or calib.get("mode") != expected_mode:
        raise SystemExit("旧二维标定不能用于高度跟踪;请运行 calibrate --realsense 重新标定")
    if calib.get("target") != target:
        raise SystemExit(f"标定目标是 {calib.get('target')};请用 --target {target} 重新标定")
    R = np.asarray(calib.get("R"), float)
    t = np.asarray(calib.get("t"), float)
    if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all()
            or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-3)
            or abs(abs(np.linalg.det(R)) - 1) > 1e-3
            or float(calib.get("rms_mm", np.inf)) > (15 if experimental else 10)
            or float(calib.get("max_mm", np.inf)) > (30 if experimental else 20)):
        raise SystemExit("三维标定无效或残差过大;请重新标定")
    return R, t


def calibration_targets(grid, span, dist, height_step):
    axes = np.linspace(-span, span, grid)
    return [(float(x), float(y), float(z))
            for z in axes for x in axes for y in (dist, dist + height_step)]


def calibrate(cam=0, save=CALIB_PATH, grid=3, span=0.06, backend=None,
              realsense=False, target="index_tip", dist=0.10, height_step=0.05):
    """每个 x/z 点连续采两个高度,拟合相机到阵列的刚体变换。"""
    import cv2

    if grid < 2 or span <= 0 or dist < 0.07 or height_step < 0.02:
        raise SystemExit("标定需要 grid >= 2、span > 0、dist >= 0.07、height-step >= 0.02")
    if not realsense and load_cam_settings().get("backend") != "realsense":
        raise SystemExit("三维标定需要 --realsense 和彩色+深度流")

    targets = calibration_targets(grid, span, dist, height_step)
    samples = [None] * len(targets)

    cam_settings = load_cam_settings()
    if backend:
        cam_settings["backend"] = backend
    cap, _ = open_source(cam, cam_settings, realsense=True, depth=True)
    detector = None
    idx, stable, frame_size = 0, [], None
    window_name = "calibrate 3D (space=sample v=view r=redo q=quit)"
    window_open = False
    fullscreen = False
    view_mode = 0
    print("三维标定:以阵列中心为 (0,0,0), y 垂直离阵;每个 x/z 点先低后高采样")
    print("指尖准确落在提示的尺量位置后按空格;按 V 查看对齐深度图")
    try:
        detector = PalmDetector()
        while idx < len(targets):
            ok, frame = read_frame(cap)
            if not ok:
                break
            frame_size = (frame.shape[1], frame.shape[0])
            observation = detector.detect(frame)
            uv = observation.point(target) if observation is not None else None
            point = cap.point_3d(uv, observation.mask) if uv is not None else None
            if point is not None:
                stable.append(point)
                stable = stable[-15:]
            else:
                stable.clear()
            disp = draw_hand_observation(frame, observation, target)
            tx, ty, tz = targets[idx]
            lines = (f"[{idx + 1}/{len(targets)}] x={tx:+.3f} y={ty:.3f} z={tz:+.3f} m",
                     f"depth={point[2]:.3f}m" if point is not None else "depth=--",
                     f"samples={len(stable)} SPACE=save R=redo Q=quit")
            for row, line in enumerate(lines):
                cv2.putText(disp, line, (10, 30 + row * 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 1, cv2.LINE_AA)
            if view_mode:
                depth_display = depth_colormap(cap.depth_m)
                if uv is not None:
                    cv2.circle(depth_display, tuple(map(int, np.rint(uv))), 9,
                               (255, 255, 255), 2)
                for row, line in enumerate(lines):
                    cv2.putText(depth_display, line, (10, 30 + row * 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                                1, cv2.LINE_AA)
                disp = depth_display if view_mode == 1 else np.hstack((disp, depth_display))
            if not window_open:
                open_preview_window(window_name, disp)
                window_open = True
            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise SystemExit("标定被放弃")
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if key == ord("v"):
                view_mode = (view_mode + 1) % 3
                print(f"  标定预览: {('彩色', '深度', '并排')[view_mode]}")
                if not fullscreen:
                    width = frame.shape[1] * (2 if view_mode == 2 else 1)
                    height = frame.shape[0]
                    scale = min(1.0, 1400 / width, 800 / height)
                    cv2.resizeWindow(window_name, int(round(width * scale)),
                                     int(round(height * scale)))
            if key == ord("r") and idx > 0:
                idx -= 1
                samples[idx] = None
                stable.clear()
            elif key == ord(" ") and len(stable) >= 8:
                cloud = np.asarray(stable[-10:])
                center = np.median(cloud, axis=0)
                if np.max(np.linalg.norm(cloud - center, axis=1)) > 0.008:
                    print("  指尖或深度跳动超过 8mm,请停稳后重采")
                    stable.clear()
                    continue
                samples[idx] = center
                print(f"  {idx + 1}/{len(targets)} "
                      f"(x={tx:+.3f}, y={ty:.3f}, z={tz:+.3f}) <- {center.round(4)}")
                if idx % 2 == 1:
                    measured = float(np.linalg.norm(samples[idx] - samples[idx - 1]))
                    print(f"  相机三维两点直线距离 {measured * 1000:.1f}mm "
                          f"(标称高度步长 {height_step * 1000:.1f}mm;仅供参考)")
                idx += 1
                stable.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()
    if any(p is None for p in samples):
        raise SystemExit("有标定点未采样,标定未完成")

    src = np.array(samples, float)
    dst = np.array(targets, float)
    try:
        R, t, errors = fit_rigid_transform(src, dst)
    except ValueError as exc:
        raise SystemExit(f"三维标定失败: {exc}") from exc
    rms_mm = float(np.sqrt(np.mean(errors ** 2)) * 1000)
    max_mm = float(np.max(errors) * 1000)
    print(f"拟合轴约定: determinant={np.linalg.det(R):+.0f};逐点残差(mm):")
    for i, (expected, error) in enumerate(zip(dst, errors), 1):
        print(f"  {i:2d} x={expected[0]:+.3f} y={expected[1]:.3f} "
              f"z={expected[2]:+.3f}: {error * 1000:.1f}")
    if rms_mm > 10 or max_mm > 20:
        diagnostic = save.with_name(save.stem + "_diagnostic.json")
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_text(json.dumps({
            "rms_mm": rms_mm, "max_mm": max_mm, "determinant": float(np.linalg.det(R)),
            "R": R.tolist(), "t": t.tolist(), "errors_mm": (errors * 1000).tolist(),
            "points": {"world": dst.tolist(), "camera": src.tolist()},
            "serial": cap.serial, "frame_size": frame_size,
        }, indent=1), encoding="utf-8")
        raise SystemExit(f"标定残差过大(RMS {rms_mm:.1f}mm,最大 {max_mm:.1f}mm),未保存正式标定。"
                         f"逐点诊断已存 {diagnostic}")
    data = {"version": 2, "mode": "realsense_3d", "R": R.tolist(), "t": t.tolist(),
            "rms_mm": rms_mm, "max_mm": max_mm, "grid": grid, "span_m": span,
            "heights_m": [dist, dist + height_step], "target": target,
            "frame_size": frame_size, "stream": "color", "serial": cap.serial,
            "determinant": float(np.linalg.det(R)),
            "points": {"world": dst.tolist(), "camera": src.tolist()}}
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"标定完成: RMS {rms_mm:.1f} mm, 最大 {max_mm:.1f} mm → {save}")
    return data


# ======================================================================
# 主循环
# ======================================================================

def run_tracking(calib=None, port=None, kind="focus", dist=0.12, m=1,
                 freq=200.0, duty=0.7, no_mod=False, cam=0, dry=False,
                 sim=False, sim_seconds=8.0, deadband_m=None, lost_s=0.5,
                 bounds=DEFAULT_BOUNDS, backend=None, realsense=False,
                 target="index_tip", experimental=False, max_seconds=None,
                 experimental_span=0.03, experimental_height=(0.09, 0.16)):
    """跟踪主循环。返回 PalmFollower.stats。
    dry=True 只打印不连板;sim=True 用合成圆路径(无需摄像头/标定)。"""
    if not sim and calib is None:
        raise SystemExit("摄像头模式需要标定文件:先运行 calibrate,或用 --sim")
    if dry and port:
        raise SystemExit("--dry 与 --port 互斥")
    if experimental:
        if sim or kind != "focus" or target != "index_tip" or no_mod:
            raise SystemExit("实验模式仅允许 RealSense 食指尖 focus + 调制")
        if not (100 <= freq <= 200 and 0 < duty <= 0.2):
            raise SystemExit("实验模式要求 100-200Hz 且 --duty 不超过 0.2")
        if max_seconds is None:
            max_seconds = 20.0
        if not (0 < max_seconds <= 20):
            raise SystemExit("实验模式最多运行 20 秒")
        if not dry and port is None:
            raise SystemExit("实验模式必须明确指定板卡 --port")
        bounds = experimental_bounds(experimental_span, *experimental_height)
        lost_s = min(lost_s, 0.2)
        if deadband_m is None:
            deadband_m = 0.002
    elif max_seconds is not None and max_seconds <= 0:
        raise SystemExit("--max-seconds 必须大于零")
    if deadband_m is None:
        deadband_m = 0.005
    if not 0 < deadband_m <= 0.02:
        raise SystemExit("--deadband 须在 0-0.02 m 之间")
    if not sim:
        R, translation = validate_calibration(calib, target, experimental=experimental)
        if not realsense and load_cam_settings().get("backend") != "realsense":
            raise SystemExit("三维跟踪需要 --realsense 彩色+深度流")

    board = None
    mod = None
    supervisor = None
    cap = None
    detector = None
    if not dry:
        board = Board(port)
        if no_mod:
            board.off()
            emit = board.send
        else:
            # 预装 off:跟踪开始前(未检测到手)阵列静默,调制线程待命
            try:
                mod = Modulator(board, OFF_PATTERN, freq=freq, duty=duty).start()
            except Exception:
                board.close()
                raise
            emit = mod.set_pattern
    else:
        def emit(ph):
            if np.isnan(ph).all():
                print("  [off]")
            else:
                print(f"  emit focus/kind pattern ({time.perf_counter() % 100:.3f}s)")

    follower = PalmFollower(emit, kind=kind, dist=dist, m=m,
                            deadband_m=deadband_m, lost_s=lost_s, bounds=bounds)
    gate_counts = {name: 0 for name in ("no_frame", "no_hand", "no_depth",
                    "invalid_position", "outside_x", "outside_y", "outside_z",
                    "inside", "steady")}
    outside_axis_counts = {axis: 0 for axis in "xyz"}
    outside_y_values = []
    try:
        if sim:
            rng = np.random.default_rng(0)
            fps, radius = 30.0, 0.05
            n = int(sim_seconds * fps)
            for i in range(n):
                t = i / fps
                th = 2.0 * np.pi * t / 4.0            # 4 秒一圈
                pos = (np.array([radius * np.cos(th), dist,
                                 radius * np.sin(th)]) + rng.normal(0.0, 0.002, 3))
                if 3.0 <= t < 4.0:                    # 模拟 1 秒丢手
                    pos = None
                follower.feed(t, pos)
                time.sleep(1.0 / fps)
        else:
            import cv2
            cam_settings = load_cam_settings()
            if backend:
                cam_settings["backend"] = backend
            cap, _ = open_source(cam, cam_settings, realsense=True, depth=True)
            if calib.get("serial") and calib["serial"] != cap.serial:
                raise RuntimeError("RealSense 序列号与标定时不同;请重新标定")
            detector = PalmDetector()
            if not dry:
                supervisor = LossSupervisor(
                    follower, on_error=mod.stop if mod is not None else board.off).start()
            print(f"{'限时实验' if experimental else '三维跟踪'}中: target={target} "
                  f"kind={kind} {'dry' if dry else port}(Ctrl+C 或窗口 q 退出)")
            if experimental:
                print(f"  中心 x/z ±{experimental_span * 100:.0f}cm, "
                      f"y {bounds[1][0] * 100:.0f}-{bounds[1][1] * 100:.0f}cm, "
                      f"deadband={deadband_m * 1000:.0f}mm, duty={duty:.2f}, "
                      f"丢手 {lost_s:.1f}s 关断, {max_seconds:.0f}s 自动停止")
            missing_since = None
            started = time.perf_counter()
            window_name = "hand track (q=quit)"
            window_open = False
            fullscreen = False
            while True:
                if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                    break
                if supervisor is not None and supervisor.error is not None:
                    raise RuntimeError("丢手监督线程关断失败") from supervisor.error
                if mod is not None and mod.error is not None:
                    raise RuntimeError("调制线程串口写入失败") from mod.error
                ok, frame = read_frame(cap, timeout_s=0.2)
                if not ok:
                    gate_counts["no_frame"] += 1
                    now = time.perf_counter()
                    follower.feed(now, None)
                    if missing_since is None:
                        missing_since = now
                    if now - missing_since >= 2.0:
                        raise RuntimeError("摄像头连续 2 秒没有画面，已关闭发射")
                    continue
                missing_since = None
                t = time.perf_counter()
                if calib.get("frame_size") and tuple(calib["frame_size"]) != (frame.shape[1], frame.shape[0]):
                    raise RuntimeError("相机分辨率与标定时不同;请重新标定")
                if calib.get("stream") and calib["stream"] != (cap.stream_name if isinstance(cap, RealSenseSource) else "uvc"):
                    raise RuntimeError("相机彩色/红外流与标定时不同;请重新标定")
                observation = detector.detect(frame)
                uv = observation.point(target) if observation is not None else None
                camera_point = cap.point_3d(uv, observation.mask) if uv is not None else None
                pos = R @ camera_point + translation if camera_point is not None else None
                gate = tracking_gate(uv, camera_point, pos, bounds)
                gate_counts[gate] += 1
                axes = outside_axes(pos, bounds) if pos is not None and np.isfinite(pos).all() else []
                for axis in axes:
                    outside_axis_counts[axis] += 1
                if "y" in axes:
                    outside_y_values.append(float(pos[1]))
                sent_before = follower.stats["sent"]
                follower.feed(t, pos)
                if gate == "inside" and follower.stats["sent"] == sent_before:
                    gate_counts["steady"] += 1
                disp = draw_hand_observation(frame, observation, target)
                status = ("ACTIVE (unchanged phase)" if gate == "inside"
                          and follower.stats["sent"] == sent_before else
                          "ACTIVE (phase updated)" if gate == "inside" else
                          "outside " + ",".join(axes) if axes else gate)
                cv2.putText(disp, status, (10, 104), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 210, 0) if gate == "inside" else (0, 0, 255),
                            2, cv2.LINE_AA)
                if pos is not None:
                    cv2.putText(disp, f"x={pos[0]*100:+.1f}cm  "
                                f"y={pos[1]*100:.1f}cm  z={pos[2]*100:+.1f}cm",
                                (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(disp, f"allowed y={bounds[1][0]*100:.0f}-"
                                f"{bounds[1][1]*100:.0f}cm", (10, 158),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
                                1, cv2.LINE_AA)
                if uv:
                    if pos is not None:
                        cv2.putText(disp, f"x={pos[0]:+.3f} y={pos[1]:.3f} z={pos[2]:+.3f}",
                                    (int(uv[0]) + 12, int(uv[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                                    cv2.LINE_AA)
                    else:
                        cv2.putText(disp, "depth unavailable", (10, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                if not window_open:
                    open_preview_window(window_name, disp)
                    window_open = True
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if supervisor is not None:
                supervisor.stop()
            follower.close(t=sim_seconds if sim else time.perf_counter())
            if mod is not None:
                mod.stop()
            elif board is not None:
                board.off()
        finally:
            if board is not None:
                board.close()
            if detector is not None:
                detector.close()
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()
        print(f"跟踪结束: 相位更新 {follower.stats['sent']} 次, "
              f"焦点激活 {follower.stats['active_s']:.1f}s, "
              f"{len(follower.stats['off_times'])} 次静默, 已关发射")
        if not sim:
            print("逐帧诊断: " + ", ".join(f"{key}={value}" for key, value in
                  gate_counts.items() if value))
            if any(outside_axis_counts.values()):
                print("越界轴次数: " + ", ".join(
                    f"{axis}={count}" for axis, count in outside_axis_counts.items()))
            if outside_y_values:
                low, mid, high = np.percentile(outside_y_values, [5, 50, 95])
                print(f"outside_y 高度(cm): 5%={low * 100:.1f}, "
                      f"中位数={mid * 100:.1f}, 95%={high * 100:.1f}; "
                      f"允许 {bounds[1][0] * 100:.0f}-{bounds[1][1] * 100:.0f}")
    follower.stats["gate_counts"] = gate_counts
    follower.stats["outside_axis_counts"] = outside_axis_counts
    return follower.stats


def main(argv=None):
    p = argparse.ArgumentParser(prog="hand_track.py",
                                description="RealSense 食指尖三维跟踪 → 声场随手移动")
    sub = p.add_subparsers(dest="cmd", required=True)

    ls = sub.add_parser("cams", help="列出摄像头索引并存快照(内置相机常占 0,USB 相机找别的号)")
    ls.add_argument("--max", type=int, default=7)

    ct = sub.add_parser("camtest", help="摄像头诊断/整定(亮度/变焦/分辨率),保存 hand_track_cam.json")
    ct.add_argument("--cam", type=int, default=0)
    ct.add_argument("--backend", choices=("dshow", "msmf", "any"), default=None,
                    help="强制视频后端(打不开画面时试 msmf)")
    ct.add_argument("--realsense", action="store_true",
                    help="Intel RealSense D4xx:用官方 SDK 采彩色或红外流")
    ct.add_argument("--target", choices=TARGETS, default="index_tip",
                    help="预览目标点(默认食指尖)")
    ct.add_argument("--depth", action="store_true",
                    help="RealSense:并排显示彩色画面与对齐深度热图")
    ct.add_argument("--pointcloud", action="store_true",
                    help="RealSense:显示深度热图及可旋转的实时三维点云")

    c = sub.add_parser("calibrate", help="RealSense 三维标定,两层高度网格采样")
    c.add_argument("--cam", type=int, default=0)
    c.add_argument("--backend", choices=("dshow", "msmf", "any"), default=None)
    c.add_argument("--realsense", action="store_true")
    c.add_argument("--save", default=str(CALIB_PATH))
    c.add_argument("--grid", type=int, default=3, help="标定网格边长(3 → 3x3 点)")
    c.add_argument("--span", type=float, default=0.06, help="网格半宽 [m]")
    c.add_argument("--target", choices=TARGETS, default="index_tip",
                   help="标定目标点(默认食指尖)")
    c.add_argument("--dist", type=float, default=0.10,
                   help="第一层指尖高度 [m],默认 0.10")
    c.add_argument("--height-step", type=float, default=0.05,
                   help="第二层比第一层高多少 [m],默认 0.05")

    t = sub.add_parser("track", help="跟踪发射")
    t.add_argument("--cam", type=int, default=0)
    t.add_argument("--backend", choices=("dshow", "msmf", "any"), default=None)
    t.add_argument("--realsense", action="store_true")
    t.add_argument("--calib", default=str(CALIB_PATH))
    t.add_argument("--experimental", action="store_true",
                   help="用最近一次失败的三维诊断做限时低占空比中心区域实验")
    t.add_argument("--experimental-span", type=float, default=0.03,
                   help="实验模式 x/z 半宽 [m],默认 0.03,最大 0.08")
    t.add_argument("--experimental-height", type=float, nargs=2, metavar=("MIN", "MAX"),
                   default=(0.09, 0.16), help="实验模式高度范围 [m],默认 0.09 0.16")
    t.add_argument("--port", default=None, help="串口(COM 名或序号);缺省须配 --dry")
    t.add_argument("--dry", action="store_true", help="只打印坐标,不连板")
    t.add_argument("--sim", action="store_true", help="合成圆路径离线跑(无需摄像头)")
    t.add_argument("--sim-seconds", type=float, default=8.0)
    t.add_argument("--kind", choices=("focus", "trap", "vortex", "twin", "bottle"),
                   default="focus")
    t.add_argument("--m", type=int, default=1)
    t.add_argument("--dist", type=float, default=0.12, help="仅 --sim 使用的合成高度 [m]")
    t.add_argument("--freq", type=float, default=200.0, help="调制频率 Hz")
    t.add_argument("--duty", type=float, default=0.7)
    t.add_argument("--no-mod", action="store_true", help="不调制,直接逐帧发送")
    t.add_argument("--deadband", type=float, default=None,
                   help="重发死区 [m],普通模式默认 0.005,实验模式默认 0.002")
    t.add_argument("--lost", type=float, default=0.5, help="丢手静默阈值 [s]")
    t.add_argument("--max-seconds", type=float, default=None,
                   help="自动结束时间 [s];实验模式默认并最多 20 秒")
    t.add_argument("--target", choices=TARGETS, default="index_tip",
                   help="跟踪目标点(必须与标定一致;默认食指尖)")

    args = p.parse_args(argv)
    if args.cmd == "cams":
        list_cams(max_index=args.max)
    elif args.cmd == "camtest":
        camtest(cam=args.cam, backend=args.backend, realsense=args.realsense,
                target=args.target, depth=args.depth, pointcloud=args.pointcloud)
    elif args.cmd == "calibrate":
        calibrate(cam=args.cam, save=Path(args.save), grid=args.grid,
                  span=args.span, backend=args.backend, realsense=args.realsense,
                  target=args.target, dist=args.dist, height_step=args.height_step)
    else:
        calib = None
        if not args.sim:
            path = DIAGNOSTIC_PATH if args.experimental and args.calib == str(CALIB_PATH) else Path(args.calib)
            if not path.exists():
                raise SystemExit(f"找不到标定文件 {path}:先运行 calibrate")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            calib = calibration_from_diagnostic(loaded) if args.experimental else loaded
            print(f"标定载入: RMS {calib.get('rms_mm', '?')} mm")
        run_tracking(calib=calib, port=args.port, kind=args.kind, dist=args.dist,
                     m=args.m, freq=args.freq, duty=args.duty, no_mod=args.no_mod,
                     cam=args.cam, dry=args.dry, sim=args.sim,
                     sim_seconds=args.sim_seconds, deadband_m=args.deadband,
                     lost_s=args.lost, backend=args.backend,
                     realsense=args.realsense, target=args.target,
                     experimental=args.experimental, max_seconds=args.max_seconds,
                     experimental_span=args.experimental_span,
                     experimental_height=args.experimental_height)


if __name__ == "__main__":
    main()
