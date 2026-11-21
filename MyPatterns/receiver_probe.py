#!/usr/bin/env python3
"""Track a visible receiver can and aim at its hidden receiving face."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

import hand_track
from my_patterns import Board
from receiver_side_geometry import estimate_side_receiver


FOCUS_BOUNDS = hand_track.DEFAULT_BOUNDS


class FaceTracker:
    """Track a manually boxed receiver body using a local template search."""

    def __init__(self, min_confidence=0.72):
        self.min_confidence = min_confidence
        self.template = None
        self.box = None
        self.confidence = 0.0

    def select(self, frame, box):
        import cv2

        x, y, w, h = map(int, box)
        height, width = frame.shape[:2]
        if w < 20 or h < 20 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError("请完整框选银色圆柱本体，至少 20x20 像素")
        self.template = cv2.GaussianBlur(cv2.cvtColor(frame[y:y+h, x:x+w],
                                                 cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if float(self.template.std()) < 8:
            self.template = None
            raise ValueError("选区对比度太低；请给银色圆柱增加背景对比度")
        self.box = (x, y, w, h)
        self.confidence = 1.0

    def clear(self):
        self.template = None
        self.box = None
        self.confidence = 0.0

    def update(self, frame):
        import cv2

        if self.template is None:
            return None
        x, y, w, h = self.box
        height, width = frame.shape[:2]
        margin = max(24, int(0.65 * max(w, h)))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
        search = cv2.GaussianBlur(cv2.cvtColor(frame[y0:y1, x0:x1],
                                               cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if search.shape[0] < h or search.shape[1] < w:
            self.confidence = 0.0
            return None
        score = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, top_left = cv2.minMaxLoc(score)
        self.confidence = float(confidence)
        if not np.isfinite(confidence) or confidence < self.min_confidence:
            return None
        self.box = (x0 + top_left[0], y0 + top_left[1], w, h)
        return self.box

    def center(self):
        if self.box is None:
            return None
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)


def live_board_pose(cap, calibration):
    """Recheck board pose at startup when the calibration uses board detection."""
    R, translation = hand_track.validate_calibration(calibration, "index_tip")
    if calibration["mode"] != "realsense_board_3d":
        return R, translation
    manual = calibration.get("board_pose_method") == "depth_corners"
    grid = None
    previous = None
    deadline = time.perf_counter() + 8
    while time.perf_counter() < deadline:
        ok, frame = hand_track.read_frame(cap, timeout_s=0.3)
        if not ok:
            continue
        if tuple(calibration["frame_size"]) != (frame.shape[1], frame.shape[0]):
            raise RuntimeError("相机分辨率与板面标定不同")
        try:
            if manual and grid is None:
                grid = hand_track.grid_from_camera_corners(cap,
                                                            calibration["board_corners_px"])
            fresh, _ = hand_track.board_pose_from_frame(
                cap, frame, calibration["orientation"], grid=grid,
                manual_corners=manual)
        except ValueError:
            previous = None
            continue
        if manual and not hand_track.board_poses_agree(
                calibration, fresh, position_m=0.02, angle_deg=6):
            raise RuntimeError("相机与板卡相对位置已变化；请重新运行 boardcalibrate")
        if hand_track.board_poses_agree(previous, fresh):
            return np.asarray(fresh["R"], float), np.asarray(fresh["t"], float)
        previous = fresh
    raise RuntimeError("8 秒内未稳定识别板卡；请露出完整阵列并固定相机")


def _window_closed(cv2, name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _select_image_box(frame):
    """Select a rectangle without OpenCV's blocking selectROI dialog."""
    import cv2

    name = "receiver body: drag box, Enter=select, q/Esc=quit"
    start = None
    end = None
    box = None

    def mouse(event, x, y, _flags, _param):
        nonlocal start, end, box
        if event == cv2.EVENT_LBUTTONDOWN:
            start = end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and start is not None:
            end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and start is not None:
            x0, y0 = start
            box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            start = end = None

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, mouse)
    try:
        while True:
            if _window_closed(cv2, name):
                return None
            display = frame.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 0), 2)
            if start is not None and end is not None:
                cv2.rectangle(display, start, end, (255, 255, 0), 2)
            cv2.imshow(name, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key in (13, 10, 32) and box is not None:
                return box
    finally:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass


def inspect_image(path, box=None):
    """Preview receiver-body selection on an ordinary RGB photo, without depth."""
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"无法读取图片: {path}")
    if box is None:
        print("拖框选银色圆柱本体（不含引脚）；Enter/空格确认，q/Esc/关闭窗口退出")
        box = _select_image_box(frame)
        if box is None:
            print("已退出图片选框")
            return None
    tracker = FaceTracker()
    tracker.select(frame, box)
    tracker.update(frame)
    x, y, w, h = tracker.box
    center = tracker.center()
    print(f"圆柱本体选框: x={x}, y={y}, w={w}, h={h}; "
          f"图像中心=({center[0]:.1f}, {center[1]:.1f}) px; "
          "RGB 图片没有深度，不能计算隐藏接收面位置")
    return center


class ReceiverFollower:
    """Require repeated 3D fixes before allowing a receiver-driven focus."""

    def __init__(self, emit, confirmation_frames=3):
        self.follower = hand_track.PalmFollower(
            emit, kind="focus", deadband_m=0.005, lost_s=0.2,
            bounds=FOCUS_BOUNDS)
        self.recent = deque(maxlen=confirmation_frames)
        self.last_position = None
        self.active = False

    def pause(self, now):
        self.follower.pause(now)
        self.recent.clear()
        self.last_position = None
        self.active = False

    def feed(self, now, position):
        if position is None or not hand_track.in_bounds(position, FOCUS_BOUNDS):
            self.pause(now)
            return False
        if self.active and self.follower.focus_position() is None:
            self.active = False
            self.recent.clear()
            self.last_position = None
        position = np.asarray(position, float)
        if self.last_position is not None and np.linalg.norm(
                position - self.last_position) > 0.03:
            self.pause(now)
            return False
        self.last_position = position.copy()
        self.recent.append(position.copy())
        if not self.active:
            if len(self.recent) < self.recent.maxlen:
                return False
            center = np.median(np.asarray(self.recent), axis=0)
            if np.max(np.linalg.norm(np.asarray(self.recent) - center, axis=1)) > 0.005:
                return False
            self.active = True
        self.follower.feed(now, position)
        return True

    def focus_position(self):
        return self.follower.focus_position()


def run(args):
    """Track the can and focus on its estimated hidden receiving face."""
    import cv2

    if args.port and args.dry:
        raise SystemExit("--dry 与 --port 不能同时使用")
    length_mm = getattr(args, "length_mm", 12.0)
    diameter_mm = getattr(args, "diameter_mm", 15.0)
    if length_mm is None or diameter_mm is None:
        raise SystemExit("需要银色圆柱本体的长度和直径")
    try:
        calibration = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(
            f"找不到板面标定: {args.calib}\n"
            "先运行 .\\ultrasound_solver\\.venv\\Scripts\\python.exe -X utf8 "
            "MyPatterns\\hand_track.py boardcalibrate --realsense"
        ) from exc
    settings = hand_track.load_cam_settings()
    cap, _ = hand_track.open_source(0, settings, realsense=True, depth=True)
    board = None
    supervisor = None
    window = "receiver: drag silver can; r=reset; q/Esc=quit"
    tracker = FaceTracker()
    drag_start = None
    drag_end = None
    pending_box = None
    window_open = False
    controller = None
    try:
        if calibration.get("serial") and calibration["serial"] != cap.serial:
            raise RuntimeError("RealSense 序列号与标定不同")
        if calibration.get("rotation", "none") != cap.rotation:
            raise RuntimeError("相机画面方向与标定不同")
        ok, first_frame = hand_track.read_frame(cap, timeout_s=0.3)
        if not ok:
            raise RuntimeError("RealSense 没有彩色/深度帧")
        if tuple(calibration.get("frame_size", ())) != (
                first_frame.shape[1], first_frame.shape[0]):
            raise RuntimeError("相机分辨率与标定不同")
        if calibration.get("mode") == "realsense_board_3d":
            print("先移开接收器，露出完整阵列；板面定位完成后再移回接收器")
        R, translation = live_board_pose(cap, calibration)
        if args.port:
            board = Board(args.port)
            board.off()
        controller = ReceiverFollower(board.send if board is not None
                                      else lambda _phases: None)
        if board is not None:
            supervisor = hand_track.LossSupervisor(
                controller.follower, on_error=board.off).start()
        print("框选整个银色圆柱本体，不含引脚；请确认引脚端背向阵列、接收面朝向阵列")
        print(f"圆柱本体长 {length_mm:g} mm、直径 {diameter_mm:g} mm；"
              "假定接收面朝向阵列且轴线近似垂直阵列")
        print("r: 清除并关断；q/Esc/关闭窗口: 退出并关断")
        if board is None:
            print("当前为 dry 预览；不会连接板卡。发射须显式加 --port COM8")

        def mouse(event, x, y, _flags, _param):
            nonlocal drag_start, drag_end, pending_box
            if event == cv2.EVENT_LBUTTONDOWN:
                controller.pause(time.perf_counter())
                tracker.clear()
                pending_box = None
                drag_start = drag_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
                drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
                x0, y0 = drag_start
                pending_box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
                drag_start = drag_end = None

        hand_track.open_preview_window(window, first_frame)
        cv2.setMouseCallback(window, mouse)
        window_open = True
        cv2.imshow(window, first_frame)
        while True:
            if supervisor is not None and supervisor.error is not None:
                raise RuntimeError("相机失联监督关断失败") from supervisor.error
            if window_open and _window_closed(cv2, window):
                break
            ok, frame = hand_track.read_frame(cap, timeout_s=0.3)
            now = time.perf_counter()
            if not ok:
                controller.pause(now)
                if window_open and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue
            if pending_box is not None:
                controller.pause(now)
                try:
                    tracker.select(frame, pending_box)
                    print("圆柱本体已选中；等待隐藏接收面三维坐标")
                except ValueError as exc:
                    tracker.clear()
                    print(exc)
                pending_box = None
            box = tracker.update(frame)
            if box is not None:
                position, depth_source = estimate_side_receiver(
                    cap, box, R, translation, length_mm / 1000.0,
                    diameter_mm / 1000.0)
            else:
                position, depth_source = None, "select can"
            if box is None or position is None:
                controller.pause(now)
                if box is None and tracker.template is not None:
                    tracker.clear()
            elif not hand_track.in_bounds(position, FOCUS_BOUNDS):
                controller.pause(now)
            elif not controller.feed(now, position):
                if box is not None and controller.last_position is None:
                    tracker.clear()
                    print("接收器位置跳变，已关断；请重新框选")
            active = controller.focus_position() is not None
            display = frame.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 0), 2)
            if position is not None:
                target_uv = hand_track.project_board_point(
                    position, R, translation, cap)
                if target_uv is not None:
                    cv2.drawMarker(display, tuple(map(int, target_uv)), (0, 0, 255),
                                   cv2.MARKER_CROSS, 18, 2)
            if drag_start is not None and drag_end is not None:
                cv2.rectangle(display, drag_start, drag_end, (255, 255, 0), 2)
            if position is not None:
                cv2.putText(display,
                            "receiver x={:+.1f} y={:.1f} z={:+.1f} cm".format(
                                *(position * 100)), (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2)
            if active:
                state = "FOCUS ON (static 40kHz)" if board is not None else "DRY focus preview"
            elif box is None:
                state = "OFF: select silver receiver body"
            elif position is None:
                state = "OFF: hidden face not localized"
            elif not hand_track.in_bounds(position, FOCUS_BOUNDS):
                state = "OFF: outside focus bounds"
            else:
                state = "OFF: waiting for stable position"
            cv2.putText(display, state, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 220, 0) if active else (0, 0, 255), 2)
            cv2.putText(display, f"track={tracker.confidence:.2f} depth={depth_source}",
                        (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if not window_open:
                hand_track.open_preview_window(window, display)
                cv2.setMouseCallback(window, mouse)
                window_open = True
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                controller.pause(time.perf_counter())
                tracker.clear()
        return controller.follower.stats
    except KeyboardInterrupt:
        return controller.follower.stats if controller is not None else None
    finally:
        original_error = sys.exc_info()[1]
        cleanup_errors = []
        for cleanup in (
                supervisor.stop if supervisor is not None else None,
                (lambda: controller.pause(time.perf_counter()))
                if controller is not None else None,
                board.off if board is not None else None,
                board.close if board is not None else None,
                cap.release,
                cv2.destroyAllWindows):
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors and original_error is None:
            raise cleanup_errors[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="定位接收器并让静态焦点实时跟随")
    parser.add_argument("--image", type=Path,
                        help="仅在 RGB 照片上检查圆柱本体选框，不连接相机或阵列")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="图片模式的圆柱本体选框像素；省略则鼠标框选")
    parser.add_argument("--port", help="显式指定后打开板卡并让静态焦点跟随接收器")
    parser.add_argument("--dry", action="store_true", help="只预览；默认不连接板卡")
    parser.add_argument("--calib", type=Path, default=hand_track.BOARD_CALIB_PATH)
    parser.add_argument("--length-mm", type=float, default=12.0,
                        help="银色圆柱本体轴向长度，默认 12 mm")
    parser.add_argument("--diameter-mm", type=float, default=15.0,
                        help="银色圆柱本体外径，默认 15 mm")
    args = parser.parse_args(argv)
    if not 5.0 <= args.length_mm <= 100.0 or not 5.0 <= args.diameter_mm <= 100.0:
        parser.error("圆柱长度和外径必须在 5-100 mm 之间")
    if args.image is not None:
        if args.port is not None:
            parser.error("--image 不能与 --port 同用")
        inspect_image(args.image, args.box)
        return
    if args.box is not None:
        parser.error("--box 仅用于 --image")
    run(args)


if __name__ == "__main__":
    main()
