#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
my_patterns.py — SonicSurface 自制图案生成 / 声场预览 / 串口发送脚本
================================================================

不依赖 Ultraino AcousticFieldSim(Java GUI)的独立控制脚本。

工作链路:
    图案函数(算 256 个相位) → 可选 matplotlib 声场预览 → 串口发送
    254 + 256 个相位字节(0..31, 32=关闭) + 253   (230400 baud, 8N1)

重要约定:
    * 相位校正已在你的 FPGA 固件里(Distribute.vhd 的 PHASE_CORRECTION),
      本脚本发送【未校正】的原始相位 —— 校正只做一处,切勿重复。
    * 坐标系: 阵列位于 x-z 平面(法线 +y),EMITTERS 为实测换能器坐标。
      焦点等目标用 (x, y, z),y 是离阵距离,单位米。
    * 幅度命令(251)/200Hz 调制(252)是幅度版固件的功能,primary 固件不支持。

命令行示例:
    python my_patterns.py ports
    python my_patterns.py focus 0 0 --dist 0.16 --port COM7
    python my_patterns.py vortex 0 0 --m 2 --preview
    python my_patterns.py twin 0 0 --angle 90 --save pv.png --dry
    python my_patterns.py bottle 0 0
    python my_patterns.py points 0.025 0.025 -0.025 -0.025
    python my_patterns.py image ../Reproduction/smiley.png --preview
    python my_patterns.py anim orbit --kind focus --radius 0.03 --port COM7
    python my_patterns.py anim images a.png b.png --port COM7
    python my_patterns.py off
"""

from __future__ import annotations

import argparse
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

PREVIEW_DIR = Path(__file__).resolve().parent / "results/previews"

# ----------------------------------------------------------------------
# 阵列常数(与官方 ControlSoftware/Python/SonicSurface.py 保持一致)
# ----------------------------------------------------------------------
PHASE_DIVS = 32          # 相位级数(0..31,32 = 换能器关闭)
N_EMITTERS = 256         # 16x16
WAVELENGTH = 0.00865     # 25 degC 空气中 40kHz 声波波长 [m]
APERTURE = 0.009         # 活塞模型发射口径 [m]
K = 2.0 * np.pi / WAVELENGTH
PITCH = 0.0105           # 阵元间距 [m](注意 > λ/2,大角度偏转会有栅瓣)

EMITTERS_POS = [-0.07874993,0,0.08074993,-0.057749946,0,0.08074993,-0.047249947,0,0.08074993,-0.06824995,0,0.070249945,-0.07874993,0,0.070249915,-0.047249954,0,0.07024993,-0.057749953,0,0.07024993,-0.06824992,0,0.08074993,-0.07874993,0,0.05974994,-0.057749946,0,0.059749942,-0.047249947,0,0.059749942,-0.06824993,0,0.04924994,-0.07874993,0,0.04924994,-0.047249947,0,0.049249943,-0.057749946,0,0.049249943,-0.06824993,0,0.05974994,-0.07874993,0,0.03874995,-0.057749946,0,0.038749944,-0.047249947,0,0.038749944,-0.06824993,0,0.028249947,-0.07874993,0,0.028249947,-0.047249954,0,0.028249951,-0.057749953,0,0.028249951,-0.06824993,0,0.03874994,-0.07874993,0,0.017749948,-0.057749953,0,0.017749952,-0.047249954,0,0.017749952,-0.068249926,0,0.0072499607,-0.07874993,0,0.007249957,-0.04724995,0,0.007249959,-0.05774995,0,0.007249959,-0.06824993,0,0.017749948,-0.07874993,0,-0.0032500343,-0.05774995,0,-0.0032500399,-0.04724995,0,-0.0032500362,-0.06824993,0,-0.013750033,-0.07874993,0,-0.013750033,-0.04724995,0,-0.013750035,-0.05774995,0,-0.013750035,-0.06824993,0,-0.0032500343,-0.07874993,0,-0.024250032,-0.05774995,0,-0.024250034,-0.04724995,0,-0.024250034,-0.06824993,0,-0.03475002,-0.07874994,0,-0.034750026,-0.047249954,0,-0.03475003,-0.057749953,0,-0.03475003,-0.06824993,0,-0.024250032,-0.07874993,0,-0.045250025,-0.057749953,0,-0.04525003,-0.047249947,0,-0.04525002,-0.06824993,0,-0.055750016,-0.07874993,0,-0.055750024,-0.047249947,0,-0.05575002,-0.057749953,0,-0.055750027,-0.06824993,0,-0.045250017,-0.078749925,0,-0.06624998,-0.05774995,0,-0.06624999,-0.04724995,0,-0.06624999,-0.06824994,0,-0.076749995,-0.07874994,0,-0.07674998,-0.04724995,0,-0.07674999,-0.05774999,0,-0.07674999,-0.068249926,0,-0.06625,-0.03674995,0,0.08074994,-0.015749965,0,0.08074993,-0.005249969,0,0.08074993,-0.026249964,0,0.07024993,-0.03674996,0,0.07024994,-0.005249969,0,0.07024993,-0.015749965,0,0.07024993,-0.026249964,0,0.08074993,-0.03674995,0,0.05974994,-0.015749961,0,0.059749935,-0.0052499655,0,0.05974994,-0.02624996,0,0.04924994,-0.036749955,0,0.04924994,-0.0052499655,0,0.04924994,-0.015749961,0,0.04924994,-0.02624996,0,0.059749935,-0.036749955,0,0.038749944,-0.01574996,0,0.03874995,-0.0052499655,0,0.038749944,-0.026249964,0,0.028249947,-0.036749955,0,0.028249947,-0.0052499664,0,0.02824995,-0.015749965,0,0.028249947,-0.026249958,0,0.03874995,-0.036749955,0,0.017749948,-0.015749965,0,0.017749948,-0.0052499664,0,0.01774995,-0.026249962,0,0.007249958,-0.036749955,0,0.007249959,-0.0052499673,0,0.007249957,-0.015749963,0,0.007249958,-0.026249964,0,0.017749948,-0.036749955,0,-0.003250037,-0.015749961,0,-0.003250038,-0.005249968,0,-0.003250038,-0.026249958,0,-0.013750037,-0.036749955,0,-0.013750033,-0.005249967,0,-0.013750033,-0.01574996,0,-0.013750037,-0.02624996,0,-0.003250038,-0.036749955,0,-0.024250032,-0.01574996,0,-0.024250036,-0.0052499673,0,-0.024250032,-0.02624996,0,-0.034750026,-0.036749955,0,-0.034750026,-0.0052499673,0,-0.034750026,-0.015749961,0,-0.034750026,-0.026249958,0,-0.024250036,-0.036749955,0,-0.04525003,-0.015749963,0,-0.04525002,-0.0052499655,0,-0.04525002,-0.026249964,0,-0.05575002,-0.036749955,0,-0.055750027,-0.0052499655,0,-0.05575002,-0.015749965,0,-0.05575002,-0.026249962,0,-0.04525002,-0.03674995,0,-0.06624999,-0.015749963,0,-0.06624999,-0.005249969,0,-0.06624999,-0.026249962,0,-0.07674999,-0.03674996,0,-0.07674999,-0.005249969,0,-0.07674999,-0.015749963,0,-0.07674999,-0.026249962,0,-0.06624999,0.0052500297,0,0.08074993,0.026250018,0,0.08074993,0.036750015,0,0.08074993,0.015750019,0,0.07024993,0.0052500297,0,0.07024993,0.036750015,0,0.07024993,0.026250018,0,0.07024993,0.015750019,0,0.08074993,0.0052500297,0,0.05974994,0.026250016,0,0.059749935,0.036750015,0,0.05974994,0.015750017,0,0.049249936,0.0052500297,0,0.04924994,0.036750015,0,0.04924994,0.026250016,0,0.049249936,0.015750017,0,0.059749935,0.005250028,0,0.038749944,0.02625002,0,0.038749944,0.036750015,0,0.038749944,0.01575002,0,0.028249951,0.005250028,0,0.028249947,0.036750015,0,0.028249951,0.02625002,0,0.028249951,0.01575002,0,0.038749944,0.0052500293,0,0.017749952,0.02625002,0,0.017749952,0.036750015,0,0.017749952,0.01575002,0,0.007249958,0.0052500293,0,0.007249957,0.036750015,0,0.007249959,0.02625002,0,0.007249958,0.01575002,0,0.017749952,0.005250029,0,-0.003250037,0.02625002,0,-0.003250038,0.036750015,0,-0.003250038,0.01575002,0,-0.013750033,0.005250028,0,-0.013750033,0.036750015,0,-0.013750037,0.02625002,0,-0.013750033,0.01575002,0,-0.003250038,0.005250028,0,-0.024250032,0.02625002,0,-0.024250034,0.036750015,0,-0.024250036,0.015750019,0,-0.034750022,0.0052500297,0,-0.034750026,0.036750015,0,-0.034750026,0.026250018,0,-0.034750022,0.01575002,0,-0.024250034,0.0052500297,0,-0.04525002,0.026250018,0,-0.04525002,0.036750015,0,-0.045250017,0.015750019,0,-0.05575002,0.0052500297,0,-0.05575002,0.03675001,0,-0.05575002,0.026250018,0,-0.05575002,0.015750019,0,-0.04525002,0.0052500297,0,-0.06624999,0.02625002,0,-0.06624999,0.03675002,0,-0.06624998,0.01575002,0,-0.07674999,0.0052500297,0,-0.07674999,0.03675001,0,-0.07674998,0.02625002,0,-0.07674999,0.01575002,0,-0.06624999,0.047250018,0,0.08074993,0.068249986,0,0.08074993,0.078749985,0,0.08074993,0.05775,0,0.07024994,0.04725001,0,0.07024992,0.078749985,0,0.07024993,0.068249986,0,0.07024993,0.05775001,0,0.08074993,0.047250006,0,0.059749946,0.068249986,0,0.05974994,0.078749985,0,0.059749946,0.057750013,0,0.049249936,0.04725001,0,0.04924994,0.078749985,0,0.049249947,0.068249986,0,0.04924994,0.05775001,0,0.05974994,0.047250006,0,0.03874995,0.068249986,0,0.03874994,0.078749985,0,0.03874995,0.057750005,0,0.028249947,0.047250006,0,0.028249947,0.078749985,0,0.028249947,0.068249986,0,0.028249947,0.057750005,0,0.038749937,0.047250006,0,0.017749948,0.068249986,0,0.017749948,0.078749985,0,0.017749948,0.05775001,0,0.007249957,0.04725001,0,0.007249959,0.078749985,0,0.0072499607,0.068249986,0,0.007249955,0.057750005,0,0.017749948,0.04725001,0,-0.0032500362,0.068249986,0,-0.003250038,0.078749985,0,-0.003250038,0.057750005,0,-0.013750033,0.047250006,0,-0.013750033,0.078749985,0,-0.013750037,0.068249986,0,-0.013750037,0.057750005,0,-0.0032500362,0.047250006,0,-0.024250032,0.068249986,0,-0.024250036,0.078749985,0,-0.024250036,0.057750005,0,-0.034750022,0.047250006,0,-0.03475003,0.078749985,0,-0.034750026,0.068249986,0,-0.034750026,0.057750005,0,-0.024250032,0.047250014,0,-0.045250017,0.068249986,0,-0.045250017,0.078749985,0,-0.045250025,0.057750013,0,-0.055750016,0.047250014,0,-0.055750016,0.078749985,0,-0.055750024,0.068249986,0,-0.055750016,0.057750013,0,-0.045250017,0.047250018,0,-0.06625,0.06824999,0,-0.06625,0.07874999,0,-0.06624998,0.05775001,0,-0.07674999,0.04725001,0,-0.07674999,0.07874998,0,-0.07674998,0.06824978,0,-0.076749995,0.057750016,0,-0.06624998]

EMITTERS = np.array(EMITTERS_POS, dtype=float).reshape(-1, 3)
CENTER = EMITTERS.mean(axis=0)
# 每个阵元相对阵列中心的方位角(约定与官方 vortexAt 一致: atan2(dx, dz))
_AZ = np.arctan2(EMITTERS[:, 0] - CENTER[0], EMITTERS[:, 2] - CENTER[2])


def _wrap(p):
    return np.mod(p, 2.0 * np.pi)


# ======================================================================
# 一、图案库 —— 每个函数返回 256 个相位(弧度,0..2π;NaN 表示关闭)
# ======================================================================

def phases_focus(pos):
    """单焦点: 各阵元相位补足到目标点路程差的整数波长 → 同相叠加。"""
    d = np.linalg.norm(EMITTERS - np.asarray(pos, float), axis=1)
    frag = np.ceil(d / WAVELENGTH) - d / WAVELENGTH
    return 2.0 * np.pi * frag


def phases_vortex(pos, m=1, rotation=0.0):
    """涡旋: focus + 方位角×m。中心必为零点(空心漩涡),m 为拓扑荷。"""
    return _wrap(phases_focus(pos) + _AZ * m + rotation)


def phases_twin(pos, angle_deg=90.0):
    """双阱: focus + 半个阵列加 π(angle 控制劈裂方向)→ 两个俘获点夹一个暗区。"""
    flip = np.mod(_AZ + np.deg2rad(angle_deg), 2.0 * np.pi) > np.pi
    return _wrap(phases_focus(pos) + np.pi * flip)


def phases_bottle(pos):
    """瓶形阱: 涡旋(m=1) + 双阱(转 90°)的组合签名。"""
    flip = np.mod(_AZ + np.pi / 2.0, 2.0 * np.pi) > np.pi
    return _wrap(phases_focus(pos) + _AZ + np.pi * flip)


def phases_steer(angle_deg):
    """波束偏转(平面波): 向 +x 方向倾斜 angle_deg。
    注意: 阵元间距 10.5mm > λ/2(4.3mm),偏转角大时会出现栅瓣,仅适合小角度。"""
    a = np.deg2rad(angle_deg)
    n = np.array([np.sin(a), np.cos(a), 0.0])
    return _wrap(-K * (EMITTERS @ n))


def phases_trap(focus=(0.0, 0.12, 0.0), dy=0.035, ring_r=0.03, n_ring=10, w=60.0,
                maxiter=500, x0=None):
    """高精度触觉焦点:焦点 + 上下 ±dy 高度各一圈环形零点(BFGS 直接优化)。
    作用:轻微收窄轴向感知边缘(实测 12cm 焦点保留 ~94%,14cm 处 43%→39%,16cm→5%)。
    注意:轴向感知范围本质上是衍射受限的(主瓣沿轴 ~3cm 厚),零点只能挪能量不能压薄主瓣;
    真正的精度旋钮是配合低 duty(如 --duty 0.3)把边缘压到触觉阈值以下。
    x0 传入上一帧相位(如跟踪/移动场景)可温启动,小幅位移时收敛快得多。"""
    from scipy.optimize import minimize

    focus = tuple(focus)
    nulls = []
    for h in (focus[1] - dy, focus[1] + dy):
        for a in np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False):
            nulls.append([focus[0] + ring_r * np.cos(a), h, focus[2] + ring_r * np.sin(a)])
    pts = np.vstack([np.atleast_2d(focus), np.array(nulls)])
    M = len(pts)
    G = _propagators(pts)

    def obj(phi):
        x = np.exp(1j * phi)
        f = G.T @ x
        J = -np.abs(f[0]) ** 2 + w * np.sum(np.abs(f[1:]) ** 2)
        grad = -2.0 * np.real(1j * np.conj(f[0]) * x * G[:, 0])
        for p in range(1, M):
            grad += w * 2.0 * np.real(1j * np.conj(f[p]) * x * G[:, p])
        return J, grad

    start = phases_focus(focus) if x0 is None else np.asarray(x0, float)
    res = minimize(obj, start, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return _wrap(res.x)


def pattern_for_kind(kind, x, z, dist, m=1, angle_deg=90.0, x0=None):
    """按类型生成图案相位(burst/hold/tap/跟踪共用,让按钮模式也支持任意位置/图案)。
    x0 传入上一帧相位时 trap 温启动(小位移迭代次数大幅减少)。"""
    pos = (x, dist, z)
    if kind == "focus":
        return phases_focus(pos)
    if kind == "trap":
        return phases_trap(pos, x0=x0)
    if kind == "vortex":
        return phases_vortex(pos, m=m)
    if kind == "twin":
        return phases_twin(pos, angle_deg=angle_deg)
    if kind == "bottle":
        return phases_bottle(pos)
    raise ValueError(kind)


def phases_multipoint(points, amps=None, iters=20):
    """多点 IBP(迭代反传播): 让 P 个点同时成为焦点。
    points: (P,3) 或 (P,4)(第 4 列为该点相对幅度 0..1)。
    与官方 SonicSurface.multiFocusIBP 同算法。"""
    pts = np.atleast_2d(np.asarray(points, float))
    if pts.shape[1] == 4:
        amps = pts[:, 3]
        pts = pts[:, :3]
    if amps is None:
        amps = np.ones(pts.shape[0])
    amps = np.asarray(amps, float)

    G = _propagators(pts)              # (256, P) 发射器→目标点传播子
    Gb = G.conj().T                    # (P, 256) 反传播
    x = np.ones(N_EMITTERS, dtype=complex)
    for _ in range(iters):
        f = x @ G                      # 正传到目标点
        f /= np.abs(f)                 # 目标面约束: 只保留相位
        f *= amps
        x = f @ Gb                     # 反传回发射面
        x /= np.abs(x)                 # 发射面约束: 等幅(纯相位阵列)
    return _wrap(np.angle(x))


# ----------------------------------------------------------------------
# 图像图案 —— 论文第 4 节算法(GS/角谱迭代)
# 移植自 Reproduction/sonicsurface_algorithm.py(即 MATLAB 原版移植)
# ----------------------------------------------------------------------
def _fftasa(p0, z, N, delta):
    """角谱法传播(FOCUS 工具箱 fftasa 的移植,无介质衰减)。"""
    from scipy import fft

    wavelen = WAVELENGTH
    fftpressz0 = fft.fft2(p0, s=(N, N))
    wavenum = 2.0 * np.pi / wavelen
    kx = (np.arange(-N / 2, N / 2)) * wavelen / (N * delta)
    kxspace, kyspace = np.meshgrid(kx, kx)
    kxsq_ysq = fft.fftshift(kxspace ** 2 + kyspace ** 2)
    kzspace = wavenum * np.sqrt(1.0 - kxsq_ysq + 0j)

    if z > 0:
        H = np.conj(np.exp(1j * z * kzspace))
    else:
        H = np.exp(-1j * z * kzspace) * (kxsq_ysq <= 1.0)

    D = (N - 1) * delta
    thres = np.sqrt(0.5 * D ** 2 / (0.5 * D ** 2 + z ** 2))
    H = H * (np.sqrt(kxsq_ysq) <= thres).astype(float)
    return fft.ifft2(fftpressz0 * H, s=(N, N))


def _imresize(arr, new_shape, order):
    from scipy.ndimage import zoom
    factors = (new_shape[0] / arr.shape[0], new_shape[1] / arr.shape[1])
    return zoom(arr, factors, order=order, mode="reflect")


def _gs_image_phases(target_amp, dist, iters, phase_res, amp_res, n_grid):
    """在 n_grid×n_grid 的离散阵列上迭代求发射相位(论文第 4 节)。
    返回 (phases[n_grid,n_grid], 得到的幅度切片)。"""
    w = target_amp.shape[0]
    slice_size = n_grid * PITCH
    emitter_px = w / n_grid

    # 发射面掩模: n_grid×n_grid 个相切的圆形发射器
    ix = np.arange(w) + 1  # 与 MATLAB 版一致(1-based)
    iy = np.arange(w) + 1
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    diffX = IX - (np.floor((IX - 1) / emitter_px) * emitter_px + emitter_px / 2)
    diffY = IY - (np.floor((IY - 1) / emitter_px) * emitter_px + emitter_px / 2)
    mask = (diffX ** 2 + diffY ** 2 < emitter_px ** 2 / 4.0).astype(float)

    target = np.zeros((w, w), dtype=complex)
    emission = np.zeros((w, w), dtype=complex)
    for _ in range(iters):
        target = target_amp * np.exp(1j * np.angle(target))       # 目标幅度约束
        emission = _fftasa(target, -dist, w, slice_size / w)      # 反传到发射面
        amp = np.abs(emission)
        phase = np.angle(emission)
        downAmp = _imresize(amp, (n_grid, n_grid), order=1)
        downPhase = _imresize(phase, (n_grid, n_grid), order=1)
        if amp_res == 0:                       # 相位调制: 幅度全 1
            downAmp = np.ones_like(downAmp)
        else:
            downAmp = np.trunc(downAmp * amp_res) / amp_res
        downPhase = np.trunc(downPhase / np.pi * phase_res) * np.pi / phase_res
        amp = _imresize(downAmp, (w, w), order=0)
        phase = _imresize(downPhase, (w, w), order=0)
        emission = amp * mask * np.exp(1j * phase)                # 发射面约束
        target = _fftasa(emission, dist, w, slice_size / w)       # 正传到目标面

    # 从每个发射器中心提取相位(行主序 [a,b])
    centers = np.round((np.arange(1, n_grid + 1)) * emitter_px - emitter_px / 2).astype(int) - 1
    return np.angle(emission[centers[:, None], centers[None, :]]), np.abs(target)


def _gs_from_model(G, amps, iters, seed, x0=None):
    """GS 迭代。默认以目标场共轭反传初始化;其它 seed 用于多起点搜索。"""
    if x0 is None:
        if seed == 0:
            x = np.exp(1j * np.angle(np.conj(G @ amps)))
        else:
            rng = np.random.default_rng(seed)
            x = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, N_EMITTERS))
    else:
        x = np.exp(1j * np.asarray(x0, float))
    Gb = G.conj().T
    for _ in range(iters):
        f = x @ G
        f = amps * np.exp(1j * np.angle(f))
        x = f @ Gb
        x /= np.maximum(np.abs(x), 1e-12)
    return _wrap(np.angle(x))


def _array_image_model(target2d, dist, res_t=64, half=0.084):
    """Shared array model and target samples for all image phase algorithms."""
    from PIL import Image

    tgt = np.asarray(target2d, float)
    u = np.linspace(-half, half, res_t)
    U, V = np.meshgrid(u, u[::-1])           # 目标行 0 ↔ z=+half(图片"上"=+z,与预览约定一致)
    pts = np.stack([U.ravel(), np.full(U.size, dist), V.ravel()], axis=1)
    amps = np.asarray(tgt, float).ravel() if tgt.size == res_t * res_t else \
        np.asarray(Image.fromarray((tgt * 255).astype(np.uint8)).resize((res_t, res_t), Image.LANCZOS), float).ravel() / 255.0
    amps = amps / max(amps.max(), 1e-9)

    G = _propagators(pts)                    # (256, M) —— 与 field_at 同一物理模型
    return G, amps


def _gs_array_phases(target2d, dist, iters=50, res_t=64, half=0.084, seed=0,
                     return_model=False, x0=None):
    """阵列感知 GS:在实测阵元模型与目标切面之间交替投影。"""
    G, amps = _array_image_model(target2d, dist, res_t, half)
    phases = _gs_from_model(G, amps, iters, seed, x0=x0)
    obtained = np.abs(np.exp(1j * phases) @ G).reshape(res_t, res_t)
    if return_model:
        return phases, obtained, G, amps
    return phases, obtained


@lru_cache(maxsize=8)
def _grid_channel_map(n_grid=16):
    """理想网格(行主序)→ 实测通道 的最近邻匹配。"""
    idx = np.arange(n_grid)
    xs = (idx - (n_grid - 1) / 2.0) * PITCH          # 图像列 b → x(左→右)
    zs = ((n_grid - 1) / 2.0 - idx) * PITCH          # 图像行 a → z(上→+z)
    X, Z = np.meshgrid(xs, zs)                        # X[a,b]=xs[b], Z[a,b]=zs[a]
    ideal = np.stack([X, np.zeros_like(X), Z], axis=-1).reshape(-1, 3)
    d = np.linalg.norm(ideal[:, None, :] - EMITTERS[None, :, :], axis=2)
    ch = d.argmin(axis=1)
    if np.unique(ch).size != n_grid * n_grid:
        raise RuntimeError("网格→通道匹配出现重复,阵列坐标异常")
    return ch


def phases_from_image(path, dist=0.16, iters=50, phase_res=32, amp_res=0,
                      flip_x=False, flip_z=False, transpose=False, n_grid=16,
                      method="array", x0=None, quant_refine=False):
    """任意图片 → 256 通道发射相位。
    method='array'(默认): 阵列感知 GS——用与物理仿真/真实声场一致的活塞传播子
        在实测 256 阵元上直接迭代,重建质量最好(推荐上板用)。
    method='paper': 论文 §4 原版算法(连续口径 + 角谱传播 + 网格→通道映射),
        用于复现对拍;因其设计模型偏理想,物理重建会叠较多斑点。
    返回 (phases, info);info 含 target/obtained 幅度切片与 MSE。
    方向约定: 图片"上"对应 +z,"右"对应 +x(从 +y 往下看)。
    图案镜像/颠倒时用 flip_x / flip_z / transpose 修正。"""
    from PIL import Image

    a = np.asarray(Image.open(path).convert("L").resize((256, 256), Image.LANCZOS), float)
    a = a / max(a.max(), 1e-9)
    if transpose:
        a = a.T
    if flip_x:
        a = a[:, ::-1]
    if flip_z:
        a = a[::-1, :]

    if method == "array":
        phases, obtained = _gs_array_phases(a, dist, iters=iters, x0=x0)
    elif method in ("backprop", "direct"):
        from image.image_direct import backprop_phases, optimize_phases
        G, amps = _array_image_model(a, dist)
        initial = backprop_phases(G, amps)
        if method == "direct":
            phases, direct_info = optimize_phases(G, amps, initial)
        else:
            phases = initial
        obtained = np.abs(np.exp(1j * phases) @ G).reshape(64, 64)
    elif method in ("gradient", "gradient_smooth"):
        from image.image_gradient import (image_metrics, quantize_phases,
                                    reference_scale, refine_image_phases)
        initial, _, G, amps = _gs_array_phases(a, dist, iters=iters,
                                               return_model=True)
        # Select using a denser grid than the one used by GS and L-BFGS.
        eval_res = 96
        u = np.linspace(-0.084, 0.084, eval_res)
        X, Z = np.meshgrid(u, u[::-1])
        eval_points = np.stack([X.ravel(), np.full(X.size, dist), Z.ravel()], axis=1)
        G_eval = _propagators(eval_points)
        eval_target = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                                 .resize((eval_res, eval_res), Image.Resampling.LANCZOS),
                                 float).ravel() / 255.0
        scale = reference_scale(G_eval, quantize_phases(initial), eval_target)

        def evaluate(candidate):
            amplitude = np.abs(np.exp(1j * quantize_phases(candidate)) @ G_eval)
            return image_metrics(amplitude, eval_target, scale)

        baseline_metrics = evaluate(initial)
        phases = initial
        selected_metrics = baseline_metrics
        selected_seed = 0
        selected_kind = "GS"
        smooth = method == "gradient_smooth"
        for seed in range(3):
            start = initial if seed == 0 else _gs_from_model(G, amps, iters, seed)
            _, detail = refine_image_phases(G, amps, start)
            options = [("GS", start)] + [("gradient", value) for value in detail["candidates"]]
            for kind, option in options:
                metrics = evaluate(option)
                if (metrics["stroke_p10"] >= 0.98 * baseline_metrics["stroke_p10"]
                        and metrics["stroke_cv"] <= 1.05 * baseline_metrics["stroke_cv"]
                        and metrics["background_mean"] <= 1.05 * baseline_metrics["background_mean"]
                        and metrics["background_top_1pct"] < selected_metrics["background_top_1pct"]):
                    phases, selected_metrics = option, metrics
                    selected_seed, selected_kind = seed, kind
        if smooth:
            reference_metrics = selected_metrics
            _, detail = refine_image_phases(G, amps, phases,
                                             texture_weight=4.0)
            for option in detail["candidates"]:
                metrics = evaluate(option)
                if (metrics["stroke_p10"] >= 0.98 * reference_metrics["stroke_p10"]
                        and metrics["stroke_cv"] <= 1.05 * reference_metrics["stroke_cv"]
                        and metrics["background_mean"] <= 1.05 * reference_metrics["background_mean"]
                        and metrics["background_top_1pct"] <= 1.05 * reference_metrics["background_top_1pct"]
                        and metrics["stroke_texture"] < selected_metrics["stroke_texture"]):
                    phases, selected_metrics = option, metrics
                    selected_kind = "gradient_smooth"
        refinement = {"selected_seed": selected_seed, "selected_kind": selected_kind,
                      "baseline_metrics": baseline_metrics,
                      "selected_metrics": selected_metrics}
        obtained = np.abs(np.exp(1j * phases) @ G).reshape(64, 64)
    elif method == "paper":
        ph_grid, obtained = _gs_image_phases(a, dist, iters, phase_res, amp_res, n_grid)
        ch = _grid_channel_map(n_grid)   # ch[网格单元] = 实测通道号
        ph_flat = ph_grid.reshape(-1)
        phases = np.empty(N_EMITTERS)    # 正方向放置: 通道 ch[g] 接收网格单元 g 的相位
        phases[ch] = ph_flat
        phases = _wrap(phases)
    else:
        raise ValueError(method)

    if quant_refine:
        from image.image_gradient import refine_quantized_bins
        G_quant, target_quant = _array_image_model(a, dist)
        phases, quant_info = refine_quantized_bins(G_quant, target_quant, phases)
        obtained = np.abs(np.exp(1j * phases) @ G_quant).reshape(64, 64)

    t_at_res = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                          .resize((obtained.shape[1], obtained.shape[0]), Image.LANCZOS), float) / 255.0
    mse = float(np.mean((t_at_res - obtained / max(obtained.max(), 1e-12)) ** 2))
    info = {"target": a, "obtained": obtained, "mse": mse}
    if method in ("gradient", "gradient_smooth"):
        info["refinement"] = refinement
    if method == "direct":
        info["direct"] = direct_info
    if quant_refine:
        info["quant_refinement"] = quant_info
    return phases, info


# ======================================================================
# 二、声场预览 —— 活塞模型直接积分(发送前先看一眼)
# ======================================================================

def _propagators(points):
    """(256, M) 活塞传播子: directivity·e^{ikd}/d(与官方 SonicSurface.py 同模型)。"""
    diff = np.asarray(points, float)[None, :, :] - EMITTERS[:, None, :]
    d = np.linalg.norm(diff, axis=2) + 1e-9
    cosang = diff[:, :, 1] / d                      # 法线 = (0,1,0)
    sin = np.sqrt(np.clip(1.0 - cosang ** 2, 0.0, 1.0))
    dire = np.sinc(0.5 * APERTURE * K * sin / np.pi)
    return dire / d * np.exp(1j * K * d)


def field_at(points, phases):
    """points (M,3) 处的复声压(相对单位)。"""
    src = np.exp(1j * np.asarray(phases, float))
    return _propagators(points).T @ src


def preview_slice(phases, axis="y", at=0.16, half=0.09, res=110,
                  title="field", targets=None, save=None, show=True):
    """预览当前相位下的声场幅度切片。
    axis='y': 距阵 at 米的俯视切面(x-z); axis='z'/'x': 过阵列中心侧视/正视剖面。"""
    import matplotlib.pyplot as plt

    if axis == "y":
        u = np.linspace(-half, half, res)
        v = np.linspace(-half, half, res)
        U, V = np.meshgrid(u, v)                    # U:x  V:z
        pts = np.stack([U.ravel(), np.full(U.size, at), V.ravel()], axis=1)
        extent = [u[0], u[-1], v[0], v[-1]]
        labels = ("x [cm]", "z [cm]")
    elif axis in ("z", "x"):
        u = np.linspace(-half, half, res)
        v = np.linspace(0.01, at + half, res)
        U, V = np.meshgrid(u, v)                    # U:横向  V:y(距离)
        if axis == "z":
            pts = np.stack([U.ravel(), V.ravel(), np.full(U.size, 0.0)], axis=1)
            labels = ("x [cm]", "y [cm]")
        else:
            pts = np.stack([np.full(U.size, 0.0), V.ravel(), U.ravel()], axis=1)
            labels = ("z [cm]", "y [cm]")
        extent = [u[0], u[-1], v[0], v[-1]]
    else:
        raise ValueError("axis must be 'y', 'z' or 'x'")

    amp = np.abs(field_at(pts, phases)).reshape(U.shape)
    vmax = np.percentile(amp, 99.2)                 # 截掉近场热点,保住切片细节

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(amp, origin="lower", extent=np.array(extent) * 100,
                   cmap="viridis", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"{title}  |p| slice ({axis}={at * 100:.0f} cm, p99-clip)")
    fig.colorbar(im, ax=ax, label="|p| (a.u.)")

    if targets:
        for t in np.atleast_2d(np.asarray(targets, float)):
            if axis == "y" and abs(t[1] - at) < 0.015:
                ax.plot(t[0] * 100, t[2] * 100, "o", ms=14, mfc="none", mec="w", mew=1.6)
                ax.plot(t[0] * 100, t[2] * 100, "+", color="w", ms=9)
            elif axis == "z" and abs(t[2]) < 0.015:
                ax.plot(t[0] * 100, t[1] * 100, "o", ms=14, mfc="none", mec="w", mew=1.6)
            elif axis == "x" and abs(t[0]) < 0.015:
                ax.plot(t[2] * 100, t[1] * 100, "o", ms=14, mfc="none", mec="w", mew=1.6)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130)
        print(f"preview saved: {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return amp


# ======================================================================
# 三、传输层 —— 串口发送(协议见论文表 1 / Distribute.vhd)
# ======================================================================

class Board:
    """SonicSurface 板卡连接。port 可以是 'COM7'、序号(1 起)或 None(交互选择)。

    线程安全:所有串口写共用一把可重入锁,允许多线程并发
    send/send_commit/off/close(如调制线程 + 手掌跟踪主循环同时写)。
    写超时(默认 1s)时清空缓冲后整帧重发一次,再失败才抛异常。"""

    def __init__(self, port=None, baud=230400, write_timeout=1.0):
        import serial
        import serial.tools.list_ports
        self._serial_mod = serial
        ports = serial.tools.list_ports.comports()
        if port is None:
            self.list_ports()
            port = input("enter port index: ")
        if str(port).isdigit():
            port = ports[int(port) - 1].device
        self.lock = threading.RLock()
        self.conn = serial.Serial(str(port), baud, write_timeout=write_timeout)

    @staticmethod
    def list_ports():
        import serial.tools.list_ports
        print("serial ports:")
        for i, p in enumerate(serial.tools.list_ports.comports(), start=1):
            print(f"  {i}: {p.device}  {p.description}")

    def _write(self, data):
        """带锁串口写;超时 → 重置缓冲后整帧重发一次。"""
        with self.lock:
            # A commit may already have reached the FPGA when flush times out.
            # Retrying it would swap the two buffers a second time.
            for attempt in range(2 if len(data) > 1 else 1):
                try:
                    written = self.conn.write(data)
                    if written != len(data):
                        raise self._serial_mod.SerialTimeoutException(
                            f"short serial write: {written}/{len(data)} bytes")
                    self.conn.flush()
                    return
                except self._serial_mod.SerialTimeoutException:
                    if attempt == 1 or len(data) == 1:
                        raise
                    self.conn.reset_output_buffer()
                    self.conn.reset_input_buffer()

    def send(self, phases):
        """发送一副相位并 commit(253)。phases: 256 个弧度;NaN → 关闭该通道。"""
        ph = np.asarray(phases, float)
        assert ph.shape == (N_EMITTERS,), f"need {N_EMITTERS} phases, got {ph.shape}"
        off = np.isnan(ph)
        with np.errstate(invalid="ignore"):
            q = np.rint(_wrap(ph) * PHASE_DIVS / (2.0 * np.pi)) % PHASE_DIVS
        q[off] = PHASE_DIVS
        self._write(bytes([254]) + q.astype(np.uint8).tobytes() + bytes([253]))

    def off(self):
        self.send(np.full(N_EMITTERS, np.nan))

    def send_commit(self):
        """只发单字节 253:在双缓冲的两副图案之间瞬间互换。
        预装 A、B 两副图案后,高频发 253 即可实现超过 200Hz 的图案开关调制
        (逐帧发送只有 ~44Hz,此法一个字节 43us)。"""
        self._write(bytes([253]))

    def close(self):
        with self.lock:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Modulator:
    """后台调制线程:图案/关闭两副相位预装 FPGA 双缓冲,循环单字节 253
    互换产生 freq Hz 幅度调制。burst/hold/live/手掌跟踪共用。

    用法:
        mod = Modulator(board, phases, freq=100, duty=0.7).start()
        mod.set_pattern(new_phases)     # 调制运行中换图案
        mod.freq = 200.0                # 属性直接改,下个周期生效
        mod.stop()                      # 停止并关闭发射(幂等)

    时序:>5ms 的等待用 time.sleep(Python 3.11+ 在 Windows 有高精度计时器,
    旧版本会退化到 ~15ms),尾部忙等保亚毫秒精度;高占空比高频时仍占部分 CPU,
    低频时不再像纯忙等那样占满一个核。"""

    def __init__(self, board, pattern, freq=100.0, duty=0.7):
        self.board = board
        self.pattern = np.asarray(pattern, float).copy()
        self.freq = float(freq)
        self.duty = float(duty)
        self.cycles = 0
        self._stop = threading.Event()
        self._thread = None
        self._off_sent = False
        self.error = None
        self._is_on = False

    def start(self):
        # 预装:图案入前台,关闭图入前台(此刻静默,图案转后台)
        self.board.send(self.pattern)
        self.board.send(np.full(N_EMITTERS, np.nan))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _wait(self, seconds):
        """等待 seconds 秒;>5ms 部分 sleep,尾部忙等;stop 置位立即返回。"""
        target = time.perf_counter() + seconds
        while not self._stop.is_set():
            remaining = target - time.perf_counter()
            if remaining <= 0.0:
                return
            if remaining > 0.005:
                time.sleep(remaining - 0.001)

    def _run(self):
        try:
            while not self._stop.is_set():
                with self.board.lock:
                    self.board.send_commit()       # → 开
                    self._is_on = True
                self.cycles += 1
                self._wait(self.duty / self.freq)
                if self._stop.is_set():
                    break
                with self.board.lock:
                    self.board.send_commit()       # → 关
                    self._is_on = False
                self._wait((1.0 - self.duty) / self.freq)
        except Exception as exc:
            self.error = exc
            self._stop.set()
            try:
                self.board.off()
            except Exception:
                pass

    def set_pattern(self, phases, silent_swap=False):
        """原子更新双缓冲:新图案和关闭图各写一帧,保持后续 commit 交替。"""
        if self.error is not None:
            raise RuntimeError("modulation thread failed") from self.error
        if self._stop.is_set():
            raise RuntimeError("modulator is stopped")
        self.pattern = np.asarray(phases, float).copy()
        with self.board.lock:
            self.board.send(self.pattern)
            self.board.send(np.full(N_EMITTERS, np.nan))
            if self._is_on:
                self.board.send_commit()

    def stop(self, turn_off=True):
        """停止调制;turn_off=True 时再发关闭图保证静默。
        幂等:重复调用不会重复发送关断帧。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise TimeoutError("modulation thread did not stop")
            self._thread = None
        if turn_off and not self._off_sent:
            self.board.off()
            self._off_sent = True

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ======================================================================
# 四、动画 —— 相位序列循环发送
# ======================================================================

def frames_orbit(dist=0.16, radius=0.03, n=72, kind="focus", m=1, angle_deg=90.0):
    """焦点/涡旋/双阱沿圆轨道运动。"""
    for t in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False):
        pos = (radius * np.cos(t), dist, radius * np.sin(t))
        if kind == "focus":
            yield phases_focus(pos)
        elif kind == "vortex":
            yield phases_vortex(pos, m=m)
        elif kind == "twin":
            yield phases_twin(pos, angle_deg=angle_deg)
        else:
            raise ValueError(kind)


def frames_scan(dist=0.16, x0=-0.05, x1=0.05, z=0.0, n=50, kind="focus"):
    """焦点沿 x 往返扫描(可听/可触的幅度调制感)。"""
    xs = np.concatenate([np.linspace(x0, x1, n), np.linspace(x1, x0, n)])
    fn = {"focus": phases_focus, "vortex": phases_vortex, "twin": phases_twin}[kind]
    for x in xs:
        yield fn((x, dist, z))


def frames_vortex_spin(dist=0.16, m=1, n=72):
    """定点涡旋自旋(只转相位,焦点不动)。"""
    for rot in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False):
        yield phases_vortex((0.0, dist, 0.0), m=m, rotation=rot)


def frames_alternate(points_a, points_b, dist=0.16):
    """时分复用: 两组焦点交替 → 一颗阵列"同时"悬浮多个点。"""
    a = phases_multipoint(np.column_stack([points_a, np.full(len(points_a), dist)]))
    b = phases_multipoint(np.column_stack([points_b, np.full(len(points_b), dist)]))
    while True:
        yield a
        yield b


def frames_images(paths, dist=0.16, iters=50, hold_s=2.0, fps=20):
    """多张图片轮播(GS 逐张算好后循环发送)。"""
    pats = []
    for path in paths:
        phases = phases_from_image(path, dist=dist, iters=iters,
                                   x0=pats[-1] if pats else None)[0]
        pats.append(phases)
    hold = max(1, int(hold_s * fps))
    while True:
        for ph in pats:
            for _ in range(hold):
                yield ph


def animate(board, frames, fps=20):
    """循环发送帧序列,Ctrl+C 停止并关闭发射。"""
    dt = 1.0 / fps
    try:
        for ph in frames:
            board.send(ph)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        board.off()
        print("emission off")


def play_depth_frames(board, frames, fps=20.0, duration=3.0):
    """Cycle precomputed depth frames for a bounded time, then turn all channels off."""
    if not 0 < fps <= 30 or duration <= 0:
        raise ValueError("image3d 分时模式要求 0 < fps <= 30 且 duration > 0")
    interval = 1.0 / fps
    end = time.perf_counter() + duration
    index = 0
    try:
        while time.perf_counter() < end:
            start = time.perf_counter()
            board.send(frames[index % len(frames)])
            index += 1
            time.sleep(max(0.0, interval - (time.perf_counter() - start)))
    finally:
        board.off()
    return index


def burst(board, dist=0.12, dur=1.5, freq=200.0, duty=0.7, pattern=None):
    """按钮式强超声爆发:图案(默认焦点)与关闭图预装入 FPGA 双缓冲,然后以
    freq Hz 交替 commit(单字节 253)→ 40kHz 载波被调制成 freq Hz 的强触觉脉冲。
    duty 为开通占空比(默认 0.7):平均辐射力 ∝ duty,越高越"顶"。
    pattern 可传入任意 256 相位(如图案)使其带上触觉调制。持续 dur 秒后自动停止。
    调制在后台 Modulator 线程,主线程仅等待。"""
    focus = phases_focus((0.0, dist, 0.0)) if pattern is None else pattern
    mod = Modulator(board, focus, freq=freq, duty=duty).start()
    try:
        time.sleep(dur)
    finally:
        mod.stop()
    print(f"burst 完成: {dur:.1f}s @ {freq:.0f}Hz/{duty:.0%} 占空比, {mod.cycles} 个脉冲, 已停止发射")


def tap(board, dist=0.12, pulse=0.05, count=1, gap=0.3, pattern=None):
    """按钮式单击:全功率短脉冲(默认 50ms),pattern 可为任意图案(默认焦点)。
    无周期调制 → 无持续低鸣,只有一声轻"嗒";皮肤靠冲击瞬态感到"被按了一下"。"""
    ph = phases_focus((0.0, dist, 0.0)) if pattern is None else pattern
    for i in range(count):
        board.send(ph)
        time.sleep(pulse)
        board.off()
        if i < count - 1:
            time.sleep(gap)
    print(f"tap 完成: {count} 次 x {pulse*1000:.0f}ms 脉冲")


def hold(board, dist=0.12, freq=100.0, duty=0.7, pattern=None):
    """持续调制发射:burst 的不限时版本。
    图案与关闭图预装双缓冲,后台 Modulator 线程循环 commit 互换产生 freq Hz
    幅度调制,直到用户按回车或 Ctrl+C 才停止。期间可最小化窗口,进程须保持运行。
    硬件应力低于连续悬浮模式(平均功率仅 duty),长时间运行安全;
    调制线程等待以 sleep 为主,不再占满一个 CPU 核。"""
    ph = phases_focus((0.0, dist, 0.0)) if pattern is None else pattern
    mod = Modulator(board, ph, freq=freq, duty=duty).start()
    print(f"hold: {freq:.0f}Hz / {duty:.0%} 持续调制发射中(焦点 {dist*100:.0f}cm)...")
    print("按回车停止并关闭发射")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        mod.stop()
        print(f"已停止(共 {mod.cycles} 个调制周期)。硬件应力提示:本模式比悬浮更轻,可长时间运行。")


def live(board, dist=0.12, freq=100.0, duty=0.7, kind="focus", x=0.0, z=0.0, m=1):
    """实时调节台:持续调制发射中用键盘即时调参数,无需停止重发。
    ↑/↓   调制频率 ±10 Hz(10~1000)
    ←/→   占空比 ±0.05(0.1~0.9)
    W/S   x 位置 ±5 mm     A/D   z 位置 ±5 mm     R/F   高度 ±5 mm
    1~5   切换图案 focus/trap/vortex/twin/bottle
    回车或 q 退出(自动关发射)
    调制在后台 Modulator 线程;trap 从上一帧结果温启动,移动位置比冷算快得多。"""
    import msvcrt

    kinds = ("focus", "trap", "vortex", "twin", "bottle")
    S = {"kind": kind, "x": x, "z": z, "dist": dist, "freq": freq,
         "duty": duty, "m": m, "ph": pattern_for_kind(kind, x, z, dist, m=m),
         "quit": False}

    def reload():
        if S["kind"] == "trap":
            print(" (计算 trap...)", end="", flush=True)
        S["ph"] = pattern_for_kind(S["kind"], S["x"], S["z"], S["dist"], m=S["m"],
                                   x0=S["ph"] if S["kind"] == "trap" else None)
        mod.set_pattern(S["ph"], silent_swap=True)

    def show():
        print(f"\r[{S['kind']:6s}] ({S['x']:+.3f},{S['dist']:.2f},{S['z']:+.3f})m  "
              f"{S['freq']:5.0f}Hz  {S['duty']:.0%}   ", end="", flush=True)

    mod = Modulator(board, S["ph"], freq=freq, duty=duty).start()
    print("live 调节台: ↑↓频率 ←→占空比 W/S/A/D位置 R/F高度 1-5图案 回车退出")
    while not S["quit"]:
        while msvcrt.kbhit():
            k = msvcrt.getch()
            if k in (b"\x00", b"\xe0"):          # 方向键两字节前缀
                a = msvcrt.getch()
                if a == b"H":
                    S["freq"] = min(1000.0, S["freq"] + 10.0); mod.freq = S["freq"]
                elif a == b"P":
                    S["freq"] = max(10.0, S["freq"] - 10.0); mod.freq = S["freq"]
                elif a == b"M":
                    S["duty"] = min(0.9, S["duty"] + 0.05); mod.duty = S["duty"]
                elif a == b"K":
                    S["duty"] = max(0.1, S["duty"] - 0.05); mod.duty = S["duty"]
            elif k in (b"\r", b"q"):
                S["quit"] = True
            elif k == b"w": S["x"] += 0.005; reload()
            elif k == b"s": S["x"] -= 0.005; reload()
            elif k == b"a": S["z"] += 0.005; reload()
            elif k == b"d": S["z"] -= 0.005; reload()
            elif k == b"r": S["dist"] += 0.005; reload()
            elif k == b"f": S["dist"] = max(0.05, S["dist"] - 0.005); reload()
            elif k in (b"1", b"2", b"3", b"4", b"5"):
                S["kind"] = kinds[int(k) - 1]; reload()
            show()
        time.sleep(0.002)     # 键盘轮询让出 CPU;调制在 Modulator 线程
    mod.stop()
    print("\n已停止发射")


# ======================================================================
# 五、命令行
# ======================================================================

def _xz(s):
    x, z = s.split(",")
    return float(x), float(z)


def _build_parser():
    p = argparse.ArgumentParser(
        prog="my_patterns.py",
        description="SonicSurface 自制图案脚本(算相位 → 可选预览 → 串口发送)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="坐标: 阵列在 x-z 平面,法线 +y;焦点参数为 X Z,距离用 --dist(默认 0.16 m)。")

    # 公共参数既挂在主解析器上也挂到每个子命令,方便写在命令行任意位置
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", default=None, help="串口(COM 名或序号;缺省交互选择)")
    common.add_argument("--dist", type=float, default=0.16, help="目标面离阵距离 [m]")
    common.add_argument("--preview", action="store_true", help="发送前弹出声场预览")
    common.add_argument("--save", metavar="PNG", help="预览存图(不弹窗)")
    common.add_argument("--dry", action="store_true", help="只计算/预览,不连板子")
    common.add_argument("--axis", choices=("y", "z", "x"), default="y", help="预览切面(默认 y 俯视)")
    common.add_argument("--fps", type=float, default=20.0, help="动画帧率")

    sub = p.add_subparsers(dest="cmd", required=True)

    def sp(name, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    sp("ports", help="列出串口")
    sp("off", help="关闭发射")

    s = sp("focus", help="单焦点")
    s.add_argument("x", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.0)

    s = sp("vortex", help="涡旋")
    s.add_argument("x", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.0)
    s.add_argument("--m", type=int, default=1, help="拓扑荷")

    s = sp("twin", help="双阱")
    s.add_argument("x", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.0)
    s.add_argument("--angle", type=float, default=90.0, help="劈裂方向 [deg]")

    s = sp("bottle", help="瓶形阱")
    s.add_argument("x", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.0)

    s = sp("points", help="多点 IBP: X1 Z1 X2 Z2 ...")
    s.add_argument("xz", type=float, nargs="+", help="成对给出 x z(至少 2 点)")

    s = sp("image", help="图片图案(GS 迭代)")
    s.add_argument("path")
    s.add_argument("--method", choices=("array", "gradient", "gradient_smooth", "backprop", "direct", "paper"), default="array",
                   help="array=GS;gradient=GS精修;gradient_smooth=亮区平滑;backprop=单次反传;direct=无GS直接优化;paper=论文原版")
    s.add_argument("--mod", type=float, default=None,
                   help="给定调制频率(如 200)则图案以 200Hz 开关调制发射,手掌可感")
    s.add_argument("--dur", type=float, default=3.0, help="调制发射持续秒数(配 --mod)")
    s.add_argument("--iters", type=int, default=50)
    s.add_argument("--quant-refine", action="store_true",
                   help="量化到32档后逐通道试±1档,只接受目标函数改善")
    s.add_argument("--flipx", action="store_true")
    s.add_argument("--flipz", action="store_true")
    s.add_argument("--transpose", action="store_true")

    s = sp("image3d", help="多深度图片共用一组相位(实验模式)")
    s.add_argument("--layer", nargs=2, action="append", required=True,
                   metavar=("PATH", "DEPTH_M"),
                   help="重复提供图片及离阵深度,例如 --layer F.png 0.12 --layer A.png 0.16")
    s.add_argument("--steps", type=int, default=160, help="直接相位优化的最大迭代数")
    s.add_argument("--mode", choices=("sequential", "static"), default="sequential",
                   help="sequential=逐层GS分时; static=单帧多层直接优化(串扰较大)")
    s.add_argument("--mod", type=float, default=None, help="限时触觉调制频率 Hz")
    s.add_argument("--dur", type=float, default=3.0, help="调制持续秒数")

    s = sp("cup3d", help="稀疏 3D 杯子轮廓: 焦点沿杯口/侧壁/杯底扫描")
    s.add_argument("--bottom", type=float, default=0.10, help="杯底离阵深度 [m]")
    s.add_argument("--top", type=float, default=0.16, help="杯口离阵深度 [m]")
    s.add_argument("--radius", type=float, default=0.03, help="杯口半径 [m]")
    s.add_argument("--rim-points", type=int, default=8)
    s.add_argument("--base-points", type=int, default=8)
    s.add_argument("--wall-levels", type=int, default=2)
    s.add_argument("--dur", type=float, default=3.0, help="显式连接串口时的播放时长 [s]")

    s = sp("mesh3d", help="STL/OBJ/PLY 模型转稀疏 3D 焦点扫描路径")
    s.add_argument("path", help="3D 模型文件；模型 Z 轴视为杯子竖直方向")
    s.add_argument("--near", type=float, default=0.10, help="模型底部离阵距离 [m]")
    s.add_argument("--width", type=float, default=0.06, help="横向最大宽度 [m]")
    s.add_argument("--height", type=float, default=0.06, help="深度方向最大高度 [m]")
    s.add_argument("--points", type=int, default=24, help="扫描焦点数量(8..120)")
    s.add_argument("--feature-angle", type=float, default=40.0,
                   help="提取网格锐边的角度阈值 [deg]")
    s.add_argument("--dur", type=float, default=3.0, help="显式连接串口时的播放时长 [s]")

    s = sp("steer", help="波束偏转(小角度)")
    s.add_argument("angle", type=float, help="偏向 +x 的角度 [deg]")

    s = sp("burst", help="按钮式强超声爆发(200Hz 调制,默认 70%% 占空比)")
    s.add_argument("x", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.0)
    s.add_argument("--kind", choices=("focus", "trap", "vortex", "twin", "bottle"),
                   default="focus", help="图案类型(trap=高精度触觉焦点)")
    s.add_argument("--m", type=int, default=1, help="vortex 拓扑荷")
    s.add_argument("--dur", type=float, default=1.5, help="持续秒数")
    s.add_argument("--freq", type=float, default=200.0, help="调制频率 Hz")
    s.add_argument("--duty", type=float, default=0.7, help="开通占空比 0.1~0.9,越高冲击力越强")

    s = sp("tap", help="单击脉冲:50ms 全功率焦点,几乎无声,像被按了一下")
    s.add_argument("x", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.0)
    s.add_argument("--kind", choices=("focus", "trap", "vortex", "twin", "bottle"),
                   default="focus", help="图案类型")
    s.add_argument("--m", type=int, default=1, help="vortex 拓扑荷")
    s.add_argument("--pulse", type=float, default=0.05, help="单次脉冲时长 [s]")
    s.add_argument("--count", type=int, default=1, help="连续点几下")
    s.add_argument("--gap", type=float, default=0.3, help="多次点击间隔 [s]")

    s = sp("hold", help="持续调制发射(burst 不限时版):按回车才停")
    s.add_argument("x", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.0)
    s.add_argument("--kind", choices=("focus", "trap", "vortex", "twin", "bottle"),
                   default="focus", help="图案类型(trap=高精度触觉焦点)")
    s.add_argument("--m", type=int, default=1, help="vortex 拓扑荷")
    s.add_argument("--freq", type=float, default=100.0, help="调制频率 Hz(实测 100~200 体感最强)")
    s.add_argument("--duty", type=float, default=0.7, help="开通占空比 0.1~0.9")

    s = sp("live", help="实时调节台:持续发射中用键盘即时调频率/占空比/位置/图案")
    s.add_argument("x", type=float, nargs="?", default=0.0)
    s.add_argument("z", type=float, nargs="?", default=0.0)
    s.add_argument("--kind", choices=("focus", "trap", "vortex", "twin", "bottle"),
                   default="focus", help="初始图案类型")
    s.add_argument("--m", type=int, default=1, help="vortex 拓扑荷")
    s.add_argument("--freq", type=float, default=100.0, help="初始调制频率 Hz")
    s.add_argument("--duty", type=float, default=0.7, help="初始占空比")

    s = sp("anim", help="动画")
    s.add_argument("type", choices=("orbit", "scan", "spin", "mpx", "images"))
    s.add_argument("--kind", choices=("focus", "vortex", "twin"), default="focus")
    s.add_argument("--radius", type=float, default=0.03)
    s.add_argument("--x0", type=float, default=-0.05)
    s.add_argument("--x1", type=float, default=0.05)
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--m", type=int, default=1)
    s.add_argument("--pa", type=_xz, default="0.03,0", help="mpx 组A 中心 x,z")
    s.add_argument("--pb", type=_xz, default="-0.03,0", help="mpx 组B 中心 x,z")
    s.add_argument("--paths", nargs="+", help="images 轮播图片")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.cmd == "ports":
        Board.list_ports()
        return

    # ---- 计算相位 ----
    pos = (getattr(args, "x", 0.0), args.dist, getattr(args, "z", 0.0))
    targets = None
    if args.cmd == "off":
        pass
    elif args.cmd == "focus":
        phases, title = phases_focus(pos), f"focus {pos}"
        targets = [pos]
    elif args.cmd == "vortex":
        phases, title = phases_vortex(pos, m=args.m), f"vortex m={args.m} {pos}"
        targets = [pos]
    elif args.cmd == "twin":
        phases, title = phases_twin(pos, angle_deg=args.angle), f"twin {args.angle:.0f}deg {pos}"
        targets = [pos]
    elif args.cmd == "bottle":
        phases, title = phases_bottle(pos), f"bottle {pos}"
        targets = [pos]
    elif args.cmd == "points":
        if len(args.xz) < 4 or len(args.xz) % 2:
            raise SystemExit("points 需要成对的 x z,例: points 0.03 0 -0.03 0")
        pts = np.array(args.xz, float).reshape(-1, 2)
        pts3 = np.column_stack([pts, np.full(len(pts), args.dist)])
        phases, title = phases_multipoint(pts3), f"IBP {len(pts)} points"
        targets = pts3.tolist()
    elif args.cmd == "image":
        phases, info = phases_from_image(
            args.path, dist=args.dist, iters=args.iters, method=args.method,
            flip_x=args.flipx, flip_z=args.flipz, transpose=args.transpose,
            quant_refine=args.quant_refine)
        title = f"image {args.path} (MSE {info['mse']:.3f})"
        print(f"image MSE = {info['mse']:.4f}")
        if "refinement" in info:
            print(f"selected: {info['refinement']['selected_kind']} "
                  f"(seed {info['refinement']['selected_seed']})")
        if "quant_refinement" in info:
            q = info["quant_refinement"]
            print(f"quantized score {q['baseline_score']:.5f} -> "
                  f"{q['selected_score']:.5f} ({q['accepted_moves']} bin moves)")
    elif args.cmd == "image3d":
        from geometry.image_3d import focus_depth_contrast, load_layers, optimize_layers

        try:
            layers = load_layers(args.layer)
            if args.mode == "static":
                phases, details = optimize_layers(layers, _propagators, steps=args.steps)
            else:
                if args.mod is not None:
                    raise ValueError("分时模式不支持 --mod；完整相位帧无法按此方式做 200 Hz 调制")
                if not 0 < args.fps <= 30 or args.dur <= 0:
                    raise ValueError("分时模式要求 0 < fps <= 30 且 dur > 0")
                image_start = time.perf_counter()
                frames = [phases_from_image(layer["path"],
                          dist=layer["depth_m"], method="array")[0]
                          for layer in layers]
                details = {"total_seconds": time.perf_counter() - image_start,
                           "method": "2D array GS per depth"}
            details["single_focus_reference"] = focus_depth_contrast(
                [layer["depth_m"] for layer in layers], phases_focus, _propagators)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        title = "image3d " + ", ".join(
            f"{layer['depth_m'] * 100:.0f}cm" for layer in layers)
        if args.mode == "static":
            print(f"image3d static optimized in {details['optimization_seconds']:.2f}s "
                  f"({details['iterations']} iterations)")
        else:
            print(f"image3d sequential: {len(frames)} GS frames computed in "
                  f"{details['total_seconds']:.2f}s")
        for focus in details["single_focus_reference"]:
            ratios = ", ".join(f"{value:.2f}" for value in
                               focus["relative_amplitude_at_depths"])
            print(f"  focus {focus['focus_depth_m'] * 100:.0f}cm "
                  f"relative axial amplitudes: [{ratios}]")
    elif args.cmd == "cup3d":
        from geometry.cup_3d import cup_outline, focus_frames

        try:
            if not 0 < args.fps <= 30 or args.dur <= 0:
                raise ValueError("cup3d 要求 0 < fps <= 30 且 dur > 0")
            cup_points, cup_labels = cup_outline(
                bottom=args.bottom, top=args.top, radius=args.radius,
                rim_points=args.rim_points, base_points=args.base_points,
                wall_levels=args.wall_levels)
            frames, generation_seconds = focus_frames(cup_points, phases_focus)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"cup3d: {len(frames)} focus frames in {generation_seconds:.3f}s; "
              f"one cycle >= {len(frames) / args.fps:.2f}s at {args.fps:g} fps")
    elif args.cmd == "mesh3d":
        from geometry.cup_3d import focus_frames
        from geometry.mesh_3d import mesh_outline

        try:
            if not 0 < args.fps <= 30 or args.dur <= 0:
                raise ValueError("mesh3d 要求 0 < fps <= 30 且 dur > 0")
            mesh_points, mesh_labels, mesh_info = mesh_outline(
                args.path, near=args.near, width=args.width,
                height=args.height, count=args.points,
                feature_angle=args.feature_angle)
            frames, generation_seconds = focus_frames(mesh_points, phases_focus)
        except (OSError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"mesh3d: {len(frames)} {mesh_info['source']} focus frames in "
              f"{generation_seconds:.3f}s; one cycle >= "
              f"{len(frames) / args.fps:.2f}s at {args.fps:g} fps")
    elif args.cmd == "steer":
        phases, title = phases_steer(args.angle), f"steer {args.angle:.0f}deg"
    elif args.cmd in ("burst", "tap", "hold", "live"):
        phases = pattern_for_kind(args.kind, args.x, args.z, args.dist, m=args.m)
        title = f"{args.cmd} {args.kind} ({args.x:+.2f},{args.dist:.2f},{args.z:+.2f})"
        targets = [(args.x, args.dist, args.z)]
    elif args.cmd == "anim":
        title = f"anim {args.type}"
        phases = None
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")

    # ---- 预览 ----
    if args.cmd == "mesh3d" and (args.preview or args.save):
        from geometry.cup_3d import save_scan_preview

        if args.save and not args.preview:
            import matplotlib
            matplotlib.use("Agg")
        save_path = args.save or str(PREVIEW_DIR / "mesh3d_preview.png")
        bottom, top = mesh_points[:, 1].min(), mesh_points[:, 1].max()
        metrics_path, report = save_scan_preview(
            save_path, mesh_points, mesh_labels, frames, _propagators,
            bottom, top, args.width / 2, args.fps, generation_seconds,
            title="Mesh scan path", metadata=mesh_info)
        print(f"preview saved: {save_path}, {metrics_path}")
        print(f"  maximum time-average / one-focus peak intensity: "
              f"{report['maximum_time_average_to_single_focus_peak']:.3f}")
        if args.preview:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(mpimg.imread(save_path))
            ax.axis("off")
            plt.show()
    elif args.cmd == "cup3d" and (args.preview or args.save):
        from geometry.cup_3d import save_cup_preview

        if args.save and not args.preview:
            import matplotlib
            matplotlib.use("Agg")
        save_path = args.save or str(PREVIEW_DIR / "cup3d_preview.png")
        metrics_path, report = save_cup_preview(
            save_path, cup_points, cup_labels, frames, _propagators,
            args.bottom, args.top, args.radius, args.fps, generation_seconds)
        print(f"preview saved: {save_path}, {metrics_path}")
        print(f"  maximum time-average / one-focus peak intensity: "
              f"{report['maximum_time_average_to_single_focus_peak']:.3f}")
        if args.preview:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(mpimg.imread(save_path))
            ax.axis("off")
            plt.show()
    elif args.cmd == "image3d" and (args.preview or args.save):
        from geometry.image_3d import save_preview, save_sequential_preview

        if args.save and not args.preview:
            import matplotlib
            matplotlib.use("Agg")
        save_path = args.save or str(PREVIEW_DIR / "image3d_preview.png")
        if args.mode == "static":
            side_path, metrics_path, report = save_preview(
                save_path, phases, layers, _propagators, details)
            print(f"preview saved: {save_path}, {side_path}, {metrics_path}")
            metrics = report["layers"]
        else:
            metrics_path, metrics = save_sequential_preview(
                save_path, frames, layers, _propagators, details)
            print(f"preview saved: {save_path}, {metrics_path}")
        for layer in metrics:
            print(f"  {layer['depth_m'] * 100:.0f}cm: MSE={layer['shared_scale_mse']:.3f}, "
                  f"stroke={layer['stroke_mean']:.3f}, "
                  f"background={layer['background_mean']:.3f}")
        if args.preview:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(mpimg.imread(save_path))
            ax.axis("off")
            plt.show()
    elif args.cmd not in ("off", "anim") and (args.preview or args.save):
        if args.save and not args.preview:
            import matplotlib
            matplotlib.use("Agg")
        half = 0.085 if args.cmd == "image" else 0.09
        preview_slice(phases, axis=args.axis, at=args.dist, half=half,
                      title=title, targets=targets,
                      save=args.save, show=bool(args.preview))

    if args.dry:
        print("dry run, nothing sent")
        return

    if args.cmd in ("cup3d", "mesh3d") and not args.port:
        raise SystemExit("3D 焦点扫描实物播放须显式指定 --port；先用 --dry 看仿真")

    # ---- 发送 ----
    with Board(args.port) as b:
        if args.cmd == "off":
            b.off()
            print("emission off")
        elif args.cmd == "burst":
            print(f"burst: {args.kind} @({args.x:+.2f},{args.dist:.2f},{args.z:+.2f}), "
                  f"{args.dur}s, {args.freq:.0f}Hz/{args.duty:.0%}")
            burst(b, dist=args.dist, dur=args.dur, freq=args.freq, duty=args.duty, pattern=phases)
        elif args.cmd == "tap":
            tap(b, dist=args.dist, pulse=args.pulse, count=args.count, gap=args.gap, pattern=phases)
        elif args.cmd == "hold":
            hold(b, dist=args.dist, freq=args.freq, duty=args.duty, pattern=phases)
        elif args.cmd == "live":
            live(b, dist=args.dist, freq=args.freq, duty=args.duty,
                 kind=args.kind, x=args.x, z=args.z, m=args.m)
        elif args.cmd == "anim":
            if args.type == "orbit":
                frames = frames_orbit(args.dist, args.radius, args.n, args.kind, m=args.m)
            elif args.type == "scan":
                frames = frames_scan(args.dist, args.x0, args.x1, n=args.n, kind=args.kind)
            elif args.type == "spin":
                frames = frames_vortex_spin(args.dist, m=args.m, n=args.n)
            elif args.type == "mpx":
                axz, bxz = args.pa, args.pb
                frames = frames_alternate([(axz[0] - 0.015, axz[1]), axz, (axz[0] + 0.015, axz[1])],
                                          [(bxz[0], bxz[1] - 0.015), bxz, (bxz[0], bxz[1] + 0.015)],
                                          dist=args.dist)
            elif args.type == "images":
                if not args.paths:
                    raise SystemExit("anim images 需要 --paths a.png b.png ...")
                frames = frames_images(args.paths, dist=args.dist, fps=args.fps)
            animate(b, frames, fps=args.fps)
        elif args.cmd == "image3d" and args.mode == "sequential":
            count = play_depth_frames(b, frames, fps=args.fps, duration=args.dur)
            print(f"image3d: sent {count} frames; emission off")
        elif args.cmd == "cup3d":
            count = play_depth_frames(b, frames, fps=args.fps, duration=args.dur)
            print(f"cup3d: sent {count} focus frames; emission off")
        elif args.cmd == "mesh3d":
            count = play_depth_frames(b, frames, fps=args.fps, duration=args.dur)
            print(f"mesh3d: sent {count} focus frames; emission off")
        elif args.cmd in ("image", "image3d") and args.mod:
            print(f"调制发射: {args.mod:.0f}Hz, {args.dur}s ...")
            burst(b, dur=args.dur, freq=args.mod, pattern=phases)
        else:
            b.send(phases)
            print(f"sent: {title} — 板子持续发射中(用 off 命令停止)")


if __name__ == "__main__":
    main()
