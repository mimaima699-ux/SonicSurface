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
import time
from functools import lru_cache

import numpy as np

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


def _gs_array_phases(target2d, dist, iters=50, res_t=64, half=0.084, seed=0):
    """阵列感知 GS:在【实测 256 阵元(活塞传播子)】与目标切面之间做 Gerchberg-Saxton。
    设计所用物理模型与 preview/真实声场一致,物理重建质量远好于论文的连续口径版
    (实测:字母 F 物理相关 0.19 → 0.55)。
    注意:暗区必须强制幅度为零(标准 GS)——放任暗区自由会保留背景能量,结果退化为亮环。"""
    from PIL import Image

    tgt = np.asarray(target2d, float)
    u = np.linspace(-half, half, res_t)
    U, V = np.meshgrid(u, u[::-1])           # 目标行 0 ↔ z=+half(图片"上"=+z,与预览约定一致)
    pts = np.stack([U.ravel(), np.full(U.size, dist), V.ravel()], axis=1)
    amps = np.asarray(tgt, float).ravel() if tgt.size == res_t * res_t else \
        np.asarray(Image.fromarray((tgt * 255).astype(np.uint8)).resize((res_t, res_t), Image.LANCZOS), float).ravel() / 255.0
    amps = amps / max(amps.max(), 1e-9)

    G = _propagators(pts)                    # (256, M) —— 与 field_at 同一物理模型
    Gb = G.conj().T

    rng = np.random.default_rng(seed)
    x = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, N_EMITTERS))   # 随机初相破对称
    for _ in range(iters):
        f = x @ G                            # 正传到切面
        f = amps * np.exp(1j * np.angle(f))  # 目标约束:盖幅度、保相位(暗区→零)
        x = f @ Gb                           # 反传回阵面
        x /= np.abs(x)                       # 相位-only 约束
    obtained = np.abs(x @ G).reshape(res_t, res_t)
    return _wrap(np.angle(x)), obtained


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
                      method="array"):
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
        phases, obtained = _gs_array_phases(a, dist, iters=iters)
    elif method == "paper":
        ph_grid, obtained = _gs_image_phases(a, dist, iters, phase_res, amp_res, n_grid)
        ch = _grid_channel_map(n_grid)   # ch[网格单元] = 实测通道号
        ph_flat = ph_grid.reshape(-1)
        phases = np.empty(N_EMITTERS)    # 正方向放置: 通道 ch[g] 接收网格单元 g 的相位
        phases[ch] = ph_flat
        phases = _wrap(phases)
    else:
        raise ValueError(method)

    t_at_res = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                          .resize((obtained.shape[1], obtained.shape[0]), Image.LANCZOS), float) / 255.0
    mse = float(np.mean((t_at_res - obtained / max(obtained.max(), 1e-12)) ** 2))
    return phases, {"target": a, "obtained": obtained, "mse": mse}


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
    """SonicSurface 板卡连接。port 可以是 'COM7'、序号(1 起)或 None(交互选择)。"""

    def __init__(self, port=None, baud=230400):
        import serial
        import serial.tools.list_ports
        self._serial_mod = serial
        ports = serial.tools.list_ports.comports()
        if port is None:
            self.list_ports()
            port = input("enter port index: ")
        if str(port).isdigit():
            port = ports[int(port) - 1].device
        self.conn = serial.Serial(str(port), baud)

    @staticmethod
    def list_ports():
        import serial.tools.list_ports
        print("serial ports:")
        for i, p in enumerate(serial.tools.list_ports.comports(), start=1):
            print(f"  {i}: {p.device}  {p.description}")

    def send(self, phases):
        """发送一副相位并 commit(253)。phases: 256 个弧度;NaN → 关闭该通道。"""
        ph = np.asarray(phases, float)
        assert ph.shape == (N_EMITTERS,), f"need {N_EMITTERS} phases, got {ph.shape}"
        off = np.isnan(ph)
        with np.errstate(invalid="ignore"):
            q = np.rint(_wrap(ph) * PHASE_DIVS / (2.0 * np.pi)) % PHASE_DIVS
        q[off] = PHASE_DIVS
        self.conn.write(bytes([254]) + q.astype(np.uint8).tobytes() + bytes([253]))
        self.conn.flush()

    def off(self):
        self.send(np.full(N_EMITTERS, np.nan))

    def send_commit(self):
        """只发单字节 253:在双缓冲的两副图案之间瞬间互换。
        预装 A、B 两副图案后,高频发 253 即可实现超过 200Hz 的图案开关调制
        (逐帧发送只有 ~44Hz,此法一个字节 43us)。"""
        self.conn.write(bytes([253]))
        self.conn.flush()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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
    pats = [phases_from_image(p, dist=dist, iters=iters)[0] for p in paths]
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


def burst(board, dist=0.12, dur=1.5, freq=200.0, duty=0.7, pattern=None):
    """按钮式强超声爆发:图案(默认焦点)与关闭图预装入 FPGA 双缓冲,然后以
    freq Hz 交替 commit(单字节 253)→ 40kHz 载波被调制成 freq Hz 的强触觉脉冲。
    duty 为开通占空比(默认 0.7):平均辐射力 ∝ duty,越高越"顶"。
    pattern 可传入任意 256 相位(如图案)使其带上触觉调制。持续 dur 秒后自动停止。"""
    focus = phases_focus((0.0, dist, 0.0)) if pattern is None else pattern
    offp = np.full(N_EMITTERS, np.nan)
    board.send(focus)     # 装入 A(焦点)并 commit → A 在发射
    board.send(offp)      # 装入 B(关闭)并 commit → B 在发射,后台缓冲=A
    t_on = duty / freq
    t_off = (1.0 - duty) / freq
    t_end = time.perf_counter() + dur
    n = 0
    while time.perf_counter() < t_end:
        board.send_commit()                    # → 开
        n += 1
        target = time.perf_counter() + t_on
        while time.perf_counter() < target:    # 周期几 ms,忙等保时序精度
            pass
        board.send_commit()                    # → 关
        target = time.perf_counter() + t_off
        while time.perf_counter() < target:
            pass
    board.off()
    print(f"burst 完成: {dur:.1f}s @ {freq:.0f}Hz/{duty:.0%} 占空比, {n} 个脉冲, 已停止发射")


def tap(board, dist=0.12, pulse=0.05, count=1, gap=0.3):
    """按钮式单击:全功率焦点短脉冲(默认 50ms)。
    无周期调制 → 无持续低鸣,只有一声轻"嗒";皮肤靠冲击瞬态感到"被按了一下"。"""
    ph = phases_focus((0.0, dist, 0.0))
    for i in range(count):
        board.send(ph)
        time.sleep(pulse)
        board.off()
        if i < count - 1:
            time.sleep(gap)
    print(f"tap 完成: {count} 次 x {pulse*1000:.0f}ms 脉冲")


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
    s.add_argument("--method", choices=("array", "paper"), default="array",
                   help="array=阵列感知GS(推荐);paper=论文原版算法")
    s.add_argument("--mod", type=float, default=None,
                   help="给定调制频率(如 200)则图案以 200Hz 开关调制发射,手掌可感")
    s.add_argument("--dur", type=float, default=3.0, help="调制发射持续秒数(配 --mod)")
    s.add_argument("--iters", type=int, default=50)
    s.add_argument("--flipx", action="store_true")
    s.add_argument("--flipz", action="store_true")
    s.add_argument("--transpose", action="store_true")

    s = sp("steer", help="波束偏转(小角度)")
    s.add_argument("angle", type=float, help="偏向 +x 的角度 [deg]")

    s = sp("burst", help="按钮式强超声爆发(200Hz 调制,默认 70% 占空比)")
    s.add_argument("--dur", type=float, default=1.5, help="持续秒数")
    s.add_argument("--freq", type=float, default=200.0, help="调制频率 Hz")
    s.add_argument("--duty", type=float, default=0.7, help="开通占空比 0.1~0.9,越高冲击力越强")

    s = sp("tap", help="单击脉冲:50ms 全功率焦点,几乎无声,像被按了一下")
    s.add_argument("--pulse", type=float, default=0.05, help="单次脉冲时长 [s]")
    s.add_argument("--count", type=int, default=1, help="连续点几下")
    s.add_argument("--gap", type=float, default=0.3, help="多次点击间隔 [s]")

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
            flip_x=args.flipx, flip_z=args.flipz, transpose=args.transpose)
        title = f"image {args.path} (MSE {info['mse']:.3f})"
        print(f"image MSE = {info['mse']:.4f}")
    elif args.cmd == "steer":
        phases, title = phases_steer(args.angle), f"steer {args.angle:.0f}deg"
    elif args.cmd == "burst":
        phases, title = phases_focus((0.0, args.dist, 0.0)), f"burst @{args.dist:.2f}m"
        targets = [(0.0, args.dist, 0.0)]
    elif args.cmd == "tap":
        phases, title = phases_focus((0.0, args.dist, 0.0)), f"tap @{args.dist:.2f}m"
        targets = [(0.0, args.dist, 0.0)]
    elif args.cmd == "anim":
        title = f"anim {args.type}"
        phases = None
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")

    # ---- 预览 ----
    if args.cmd not in ("off", "anim") and (args.preview or args.save):
        half = 0.085 if args.cmd == "image" else 0.09
        preview_slice(phases, axis=args.axis, at=args.dist, half=half,
                      title=title, targets=targets,
                      save=args.save, show=bool(args.preview))

    if args.dry:
        print("dry run, nothing sent")
        return

    # ---- 发送 ----
    with Board(args.port) as b:
        if args.cmd == "off":
            b.off()
            print("emission off")
        elif args.cmd == "burst":
            print(f"burst: 焦点 {args.dist*100:.0f}cm, {args.dur}s, {args.freq:.0f}Hz 调制...")
            burst(b, dist=args.dist, dur=args.dur, freq=args.freq, duty=args.duty)
        elif args.cmd == "tap":
            tap(b, dist=args.dist, pulse=args.pulse, count=args.count, gap=args.gap)
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
        elif args.cmd == "image" and args.mod:
            print(f"调制发射: {args.mod:.0f}Hz, {args.dur}s ...")
            burst(b, dur=args.dur, freq=args.mod, pattern=phases)
        else:
            b.send(phases)
            print(f"sent: {title} — 板子持续发射中(用 off 命令停止)")


if __name__ == "__main__":
    main()
