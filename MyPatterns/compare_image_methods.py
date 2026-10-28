"""Offline comparison of image phase methods; never opens a serial port."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from image.image_gradient import image_metrics, quantize_phases, reference_scale
from my_patterns import _propagators, phases_from_image


METHODS = (
    ("array", "GS"),
    ("gradient", "GS + gradient"),
    ("gradient_smooth", "GS + smooth gradient"),
    ("backprop", "Backprop"),
    ("direct", "Direct L-BFGS-B"),
)


def compare(image_path, output_dir, dist=0.16, resolution=128):
    image_path = Path(image_path)
    phases = {}
    details = {}
    seconds = {}
    for method, _ in METHODS:
        started = time.perf_counter()
        phases[method], details[method] = phases_from_image(
            image_path, dist=dist, method=method)
        seconds[method] = time.perf_counter() - started

    started = time.perf_counter()
    u = np.linspace(-0.084, 0.084, resolution)
    x, z = np.meshgrid(u, u[::-1])
    points = np.stack((x.ravel(), np.full(x.size, dist), z.ravel()), axis=1)
    G = _propagators(points)
    target = np.asarray(Image.fromarray((details["array"]["target"] * 255).astype(np.uint8))
                        .resize((resolution, resolution), Image.Resampling.LANCZOS),
                        dtype=float) / 255.0
    target = np.clip(target, 0, 1)
    scale = reference_scale(G, quantize_phases(phases["array"]), target)
    amplitudes = {}
    metrics = {}
    for method, _ in METHODS:
        amplitude = np.abs(np.exp(1j * quantize_phases(phases[method])) @ G)
        amplitudes[method] = amplitude.reshape(target.shape)
        metrics[method] = image_metrics(amplitudes[method], target, scale)
        metrics[method]["shared_scale_mse"] = float(np.mean(
            (amplitudes[method] / scale - target) ** 2))

    output_dir.mkdir(parents=True, exist_ok=True)
    limit = max(*(float(np.percentile(amp / scale, 99.2))
                  for amp in amplitudes.values()), 1e-9)
    fig, axes = plt.subplots(1, 1 + len(METHODS),
                             figsize=(3.5 * (1 + len(METHODS)), 3.5),
                             constrained_layout=True)
    axes[0].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Target")
    for ax, (method, label) in zip(axes[1:], METHODS):
        ax.imshow(amplitudes[method] / scale, cmap="magma", vmin=0, vmax=limit)
        ax.set_title(label)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    figure_path = output_dir / f"{image_path.stem}_comparison.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    seconds["evaluation_and_plot"] = time.perf_counter() - started

    report = {
        "image": str(image_path), "distance_m": dist,
        "evaluation_grid": [resolution, resolution],
        "reference_scale": scale,
        "metrics": metrics,
        "seconds": seconds,
        "direct_optimization": details["direct"]["direct"],
        "note": "Model prediction only; all methods use 32-bin phases and the GS stroke-mean scale.",
    }
    report_path = output_dir / f"{image_path.stem}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return figure_path, report_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path,
                        default=[Path(__file__).resolve().parent / "assets/demo_F.png",
                                 Path(__file__).resolve().parents[1] / "Reproduction/smiley.png"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results/direct_comparison")
    parser.add_argument("--dist", type=float, default=0.16)
    args = parser.parse_args()
    for image in args.images:
        figure, metrics, report = compare(image, args.output_dir, args.dist)
        print(f"{image.name}: {figure} / {metrics}", flush=True)
        for method, label in METHODS:
            m = report["metrics"][method]
            print(f"  {label:21s} {report['seconds'][method]:6.2f}s "
                  f"MSE={m['shared_scale_mse']:.3f} "
                  f"stroke_p10={m['stroke_p10']:.3f} "
                  f"stroke_cv={m['stroke_cv']:.3f} "
                  f"background_top_1pct={m['background_top_1pct']:.3f}", flush=True)


if __name__ == "__main__":
    main()
