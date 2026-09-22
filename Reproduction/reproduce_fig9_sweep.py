"""Reproduce Figure 9 of the SonicSurface paper: the resolution sweet-spot analysis.

One parameter is varied at a time (emitter size, phase resolution, amplitude
resolution) around the SMD-board defaults (emitterSize=10 mm, phaseRes=32,
no amplitude modulation), MSE is computed for each target pattern, exactly as
described in Section 5.2 of the paper.
"""

import os
import json
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

from sonicsurface_algorithm import calcEmissionForTargetAmpSlice

HERE = os.path.dirname(os.path.abspath(__file__))
ALGO_DIR = os.path.join(os.path.dirname(HERE), "Algorithms", "EmissionPhasesAmpsForTargetSlice")

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Segoe UI", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False

# dataviz reference palette (validated: adjacent pairs pass in light mode)
C = {"smiley": "#2a78d6", "A": "#eb6834", "circle": "#1baf7a", "curve": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

DIST, SLICE, FREQ, SPEED, ITERS = 0.16, 0.16, 40000, 340, 50


def load_pattern(name):
    img = np.array(Image.open(os.path.join(ALGO_DIR, f"{name}.png")).convert("L"), dtype=float)
    return img / img.max()


def mse(target, obtained):
    t = target / target.max()
    o = obtained / obtained.max()
    return float(np.sum((t - o) ** 2) / o.size)


def run(name, emitter_size, phase_res, amp_res):
    target = load_pattern(name)
    _, _, amp_slice = calcEmissionForTargetAmpSlice(
        target, DIST, ITERS, SLICE, FREQ, SPEED, emitter_size, amp_res, phase_res)
    return mse(target, amp_slice)


def main():
    patterns = ["smiley", "A", "circle", "curve"]
    sizes = [0.010, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
    phase_res_list = [2, 4, 8, 16, 32]
    amp_res_list = [0, 2, 4, 8, 16]  # 0 = no amplitude control

    results = {"sizes_mm": [s * 1000 for s in sizes],
               "phase_res": phase_res_list, "amp_res": amp_res_list,
               "mse_size": {}, "mse_phase": {}, "mse_amp": {}}

    t0 = time.time()
    for p in patterns:
        results["mse_size"][p] = [run(p, s, 32, 0) for s in sizes]
        results["mse_phase"][p] = [run(p, 0.010, pr, 0) for pr in phase_res_list]
        results["mse_amp"][p] = [run(p, 0.010, 32, ar) for ar in amp_res_list]
        print(f"[{time.time()-t0:6.1f}s] {p} done", flush=True)

    with open(os.path.join(HERE, "fig9_sweep_results.json"), "w") as f:
        json.dump(results, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.patch.set_facecolor("#fcfcfb")

    def style(ax, title, xlabel):
        ax.set_facecolor("#fcfcfb")
        ax.set_title(title, color=INK, fontsize=12, pad=10)
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
        ax.set_ylabel("MSE（均方误差）", color=INK2, fontsize=10)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color("#c3c2b7"); s.set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=9)

    wl_mm = 8.65  # wavelength at 40 kHz in air
    for p in patterns:
        axes[0].plot(results["sizes_mm"], results["mse_size"][p], "-o",
                     color=C[p], linewidth=2, markersize=5, label=p)
        axes[1].plot(results["phase_res"], results["mse_phase"][p], "-o",
                     color=C[p], linewidth=2, markersize=5)
        axes[2].plot(results["amp_res"], results["mse_amp"][p], "-o",
                     color=C[p], linewidth=2, markersize=5)

    axes[0].axvline(wl_mm / 4, color=MUTED, linestyle="--", linewidth=1)
    axes[0].annotate("λ/4 ≈ 2.2 mm", xy=(wl_mm / 4, axes[0].get_ylim()[1]),
                     xytext=(wl_mm / 4 + 0.4, axes[0].get_ylim()[1] * 0.92),
                     color=MUTED, fontsize=9)
    axes[0].invert_xaxis()
    style(axes[0], "发射器尺寸（空间分辨率）", "发射器直径 (mm)，左大右小")
    axes[1].set_xticks(phase_res_list)
    style(axes[1], "相位分辨率", "每周期相位级数")
    axes[2].set_xticks(amp_res_list)
    axes[2].set_xticklabels(["无幅度\n调制" if a == 0 else str(a) for a in amp_res_list])
    style(axes[2], "幅度分辨率", "幅度级数")

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("SonicSurface 图9复现：分辨率对成像质量的影响（阵列上方 16 cm，50 次迭代）",
                 color=INK, fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(HERE, "fig9_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print("saved:", out)

    print("\nMSE 表（甜点验证）")
    for p in patterns:
        print(f"  {p:7s} size: {[round(v,4) for v in results['mse_size'][p]]}")
        print(f"  {'':7s} phase: {[round(v,4) for v in results['mse_phase'][p]]}")
        print(f"  {'':7s} amp:   {[round(v,4) for v in results['mse_amp'][p]]}")


if __name__ == "__main__":
    main()
