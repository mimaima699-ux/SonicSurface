"""Phase-only synthesis and offline evaluation for several image depth planes."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize

from image.image_gradient import image_metrics, loss_and_gradient, quantize_phases


HALF_WIDTH = 0.084


def load_layers(specs, resolution=40):
    """Load (path, depth in meters) pairs into equally sampled x-z planes."""
    if len(specs) < 2:
        raise ValueError("image3d 至少需要两个 --layer 图片 深度")
    depths = [float(depth) for _, depth in specs]
    if any(not np.isfinite(depth) or depth <= 0 for depth in depths):
        raise ValueError("所有深度必须是正数，单位米")
    if len(set(depths)) != len(depths):
        raise ValueError("两个图层不能位于相同深度")
    layers = []
    for path, depth in specs:
        image = Image.open(path).convert("L").resize(
            (resolution, resolution), Image.Resampling.LANCZOS)
        target = np.asarray(image, dtype=float)
        target /= max(target.max(), 1e-12)
        target = np.clip(target, 0, 1)
        if not (target >= 0.5).any() or not (target <= 0.1).any():
            raise ValueError(f"图层需要亮笔画和暗背景: {path}")
        layers.append({"path": str(path), "depth_m": float(depth), "target": target})
    return layers


def plane_points(depth, resolution):
    u = np.linspace(-HALF_WIDTH, HALF_WIDTH, resolution)
    x, z = np.meshgrid(u, u[::-1])
    return np.stack((x.ravel(), np.full(x.size, depth), z.ravel()), axis=1)


def focus_depth_contrast(depths, focus_phases, propagators):
    """Reference axial selectivity for a single focused spot at each depth."""
    centers = np.array([(0.0, depth, 0.0) for depth in depths])
    G = propagators(centers)
    results = []
    for i, depth in enumerate(depths):
        phases = quantize_phases(focus_phases((0.0, depth, 0.0)))
        amplitudes = np.abs(np.exp(1j * phases) @ G)
        results.append({
            "focus_depth_m": depth,
            "relative_amplitude_at_depths": (amplitudes / max(amplitudes[i], 1e-12)).tolist(),
        })
    return results


def optimize_layers(layers, propagators, steps=160):
    """Optimize one phase vector over all planes, with no GS initialization."""
    points = np.concatenate([
        plane_points(layer["depth_m"], layer["target"].shape[0])
        for layer in layers
    ])
    target = np.concatenate([layer["target"].ravel() for layer in layers])
    G = propagators(points)
    initial = np.mod(np.angle(np.conj(G @ target)), 2 * np.pi)
    initial_field = np.abs(np.exp(1j * initial) @ G)
    bright = target >= 0.5
    scale = max(float(initial_field[bright].mean()), 1e-12)
    objective = lambda phases: loss_and_gradient(phases, G, target, scale)
    started = time.perf_counter()
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": steps})
    initial_score = objective(quantize_phases(initial))[0]
    optimized_score = objective(quantize_phases(result.x))[0]
    phases = result.x if optimized_score < initial_score else initial
    return np.mod(phases, 2 * np.pi), {
        "optimization_seconds": time.perf_counter() - started,
        "iterations": int(result.nit),
        "initial_score": float(initial_score),
        "selected_score": float(min(initial_score, optimized_score)),
        "reference_scale": scale,
        "grid": list(layers[0]["target"].shape),
    }


def evaluate_layers(phases, layers, propagators, resolution=80):
    """Measure quantized fields on a denser grid and report depth leakage."""
    q = quantize_phases(phases)
    targets = []
    fields = []
    for layer in layers:
        target = np.asarray(Image.fromarray(
            np.uint8(np.rint(layer["target"] * 255))).resize(
                (resolution, resolution), Image.Resampling.LANCZOS), dtype=float) / 255.0
        target = np.clip(target, 0, 1)
        field = np.abs(np.exp(1j * q) @ propagators(
            plane_points(layer["depth_m"], resolution))).reshape(resolution, resolution)
        targets.append(target)
        fields.append(field)

    bright_values = [field[target >= 0.5].mean()
                     for field, target in zip(fields, targets)]
    scale = max(float(np.mean(bright_values)), 1e-12)
    report = []
    for i, (layer, target, field) in enumerate(zip(layers, targets, fields)):
        metrics = image_metrics(field, target, scale)
        metrics["shared_scale_mse"] = float(np.mean((field / scale - target) ** 2))
        metrics["depth_m"] = layer["depth_m"]
        metrics["image"] = layer["path"]
        # At the other layer's image coordinates, compare intended and leaked brightness.
        metrics["other_layer_stroke_mean"] = [
            float((field[other_target >= 0.5] / scale).mean())
            for j, other_target in enumerate(targets) if j != i
        ]
        report.append(metrics)
    return targets, fields, {"evaluation_grid": [resolution, resolution],
                             "reference_scale": scale, "layers": report}


def save_preview(path, phases, layers, propagators, details, resolution=80):
    """Save matched-scale depth slices and an x-y section, plus JSON metrics."""
    import matplotlib.pyplot as plt

    targets, fields, report = evaluate_layers(phases, layers, propagators, resolution)
    depths = [layer["depth_m"] for layer in layers]
    ys = np.linspace(max(0.01, min(depths) - 0.04), max(depths) + 0.04, resolution)
    xs = np.linspace(-HALF_WIDTH, HALF_WIDTH, resolution)
    x, y = np.meshgrid(xs, ys)
    side_points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    side = np.abs(np.exp(1j * quantize_phases(phases)) @
                  propagators(side_points)).reshape(resolution, resolution)
    scale = report["reference_scale"]
    limit = max(float(np.percentile(np.concatenate([
        field.ravel() / scale for field in fields] + [side.ravel() / scale]), 99.2)),
        1e-12)
    fig, axes = plt.subplots(2, len(layers), figsize=(4 * len(layers), 8),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, (layer, target, field) in enumerate(zip(layers, targets, fields)):
        axes[0, i].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Target: {Path(layer['path']).name}, {layer['depth_m']*100:.0f} cm")
        axes[1, i].imshow(field / scale, cmap="magma", vmin=0, vmax=limit)
        axes[1, i].set_title(f"Field at {layer['depth_m']*100:.0f} cm")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    side_path = path.with_name(path.stem + "_side.png")
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.imshow(side / scale, origin="lower", extent=[xs[0] * 100, xs[-1] * 100,
              ys[0] * 100, ys[-1] * 100], aspect="auto", cmap="magma",
              vmin=0, vmax=limit)
    for depth in depths:
        ax.axhline(depth * 100, color="white", alpha=0.5, linewidth=0.8)
    ax.set(xlabel="x [cm]", ylabel="depth y [cm]", title="Side section (z=0)")
    fig.savefig(side_path, dpi=150)
    plt.close(fig)

    report["optimization"] = details
    report["note"] = "Offline model prediction after 32-bin phase quantization; not tactile measurement."
    metrics_path = path.with_suffix(".json")
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return side_path, metrics_path, report


def save_sequential_preview(path, frames, layers, propagators, details, resolution=80):
    """Show each GS frame at every depth to expose out-of-plane leakage."""
    import matplotlib.pyplot as plt

    targets = []
    fields = []
    for layer in layers:
        target = np.asarray(Image.fromarray(np.uint8(np.rint(
            layer["target"] * 255))).resize((resolution, resolution),
            Image.Resampling.LANCZOS), dtype=float) / 255.0
        targets.append(np.clip(target, 0, 1))
    for phases in frames:
        q = quantize_phases(phases)
        fields.append([
            np.abs(np.exp(1j * q) @ propagators(
                plane_points(layer["depth_m"], resolution))).reshape(
                    resolution, resolution)
            for layer in layers
        ])
    scale = max(float(np.mean([
        fields[i][i][targets[i] >= 0.5].mean() for i in range(len(layers))])),
        1e-12)
    limit = max(float(np.percentile(np.concatenate([
        field.ravel() / scale for row in fields for field in row]), 99.2)), 1e-12)
    fig, axes = plt.subplots(len(layers), len(layers) + 1,
                             figsize=(4 * (len(layers) + 1), 4 * len(layers)),
                             constrained_layout=True, squeeze=False)
    report = []
    for i, layer in enumerate(layers):
        axes[i, 0].imshow(targets[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Target frame {i+1}")
        own = fields[i][i]
        metrics = image_metrics(own, targets[i], scale)
        metrics["shared_scale_mse"] = float(np.mean((own / scale - targets[i]) ** 2))
        metrics["other_depth_stroke_mean"] = [
            float(fields[i][j][targets[i] >= 0.5].mean() / scale)
            for j in range(len(layers)) if j != i
        ]
        metrics["depth_contrast_ratio"] = [
            float(metrics["stroke_mean"] / max(leak, 1e-12))
            for leak in metrics["other_depth_stroke_mean"]
        ]
        metrics["image"] = layer["path"]
        metrics["depth_m"] = layer["depth_m"]
        report.append(metrics)
        for j, field in enumerate(fields[i]):
            axes[i, j + 1].imshow(field / scale, cmap="magma", vmin=0, vmax=limit)
            axes[i, j + 1].set_title(
                f"Frame {i+1} at {layers[j]['depth_m'] * 100:.0f} cm")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    metrics_path = path.with_suffix(".json")
    metrics_path.write_text(json.dumps({
        "mode": "sequential", "layers": report,
        "reference_scale": scale, "evaluation_grid": [resolution, resolution],
        "generation_seconds": details,
        "note": "Each GS frame is evaluated separately after 32-bin quantization; not simultaneous 3D or measured touch."
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics_path, report
