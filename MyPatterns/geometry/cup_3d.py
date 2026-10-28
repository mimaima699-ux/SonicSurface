"""Sparse three-dimensional cup outline and offline focal-scan preview."""

import json
import time
from pathlib import Path

import numpy as np

from image.image_gradient import quantize_phases


def cup_outline(bottom=0.10, top=0.16, radius=0.03,
                rim_points=8, base_points=8, wall_levels=2):
    """Return a cup-axis-along-y scan path in (x, y, z) meters."""
    if not (np.isfinite(bottom) and np.isfinite(top) and 0 < bottom < top):
        raise ValueError("杯底和杯口深度必须满足 0 < bottom < top")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("杯口半径必须为正数")
    if rim_points < 4 or base_points < 4 or wall_levels < 0:
        raise ValueError("杯口/杯底至少各 4 点，侧壁层数不能为负")
    base_radius = radius * 0.78
    points = []
    labels = []
    for name, count, depth, ring_radius in (
            ("rim", rim_points, top, radius),
            ("base", base_points, bottom, base_radius)):
        for angle in np.linspace(0, 2 * np.pi, count, endpoint=False):
            points.append((ring_radius * np.cos(angle), depth,
                           ring_radius * np.sin(angle)))
            labels.append(name)
    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        for depth in np.linspace(bottom, top, wall_levels + 2)[1:-1]:
            fraction = (depth - bottom) / (top - bottom)
            wall_radius = base_radius + (radius - base_radius) * fraction
            points.append((wall_radius * np.cos(angle), depth,
                           wall_radius * np.sin(angle)))
            labels.append("wall")
    return np.asarray(points), labels


def focus_frames(points, focus_phases):
    started = time.perf_counter()
    frames = np.array([focus_phases(point) for point in points])
    return frames, time.perf_counter() - started


def save_cup_preview(path, points, labels, frames, propagators,
                     bottom, top, radius, fps=20, generation_seconds=0.0,
                     resolution=80):
    return save_scan_preview(path, points, labels, frames, propagators,
                             bottom, top, radius, fps, generation_seconds,
                             resolution=resolution, title="Cup scan path")


def save_scan_preview(path, points, labels, frames, propagators,
                      bottom, top, radius, fps=20, generation_seconds=0.0,
                      resolution=80, title="3D scan path", metadata=None):
    """Render path and time-averaged intensity at three cross sections."""
    import matplotlib.pyplot as plt

    xy = np.linspace(-radius * 1.7, radius * 1.7, resolution)
    x, z = np.meshgrid(xy, xy[::-1])
    y_axis = np.linspace(max(0.01, bottom - 0.04), top + 0.04, resolution)
    xs, ys = np.meshgrid(xy, y_axis)
    upper_name, lower_name = (("Rim depth", "Base depth") if title == "Cup scan path"
                              else ("Far depth", "Near depth"))
    planes = [
        (upper_name, np.stack((x.ravel(), np.full(x.size, top), z.ravel()), axis=1)),
        (lower_name, np.stack((x.ravel(), np.full(x.size, bottom), z.ravel()), axis=1)),
        ("Side section z=0", np.stack((xs.ravel(), ys.ravel(),
                                      np.zeros(xs.size)), axis=1)),
    ]
    weights = np.exp(1j * quantize_phases(frames))
    fields = []
    for _, sample_points in planes:
        G = propagators(sample_points)
        fields.append(np.mean(np.abs(weights @ G) ** 2, axis=0).reshape(
            resolution, resolution))
    peak_at_foci = np.abs(np.sum(weights * propagators(points).T, axis=1)) ** 2
    peak_reference = max(float(peak_at_foci.mean()), 1e-12)
    peak_average_fraction = max(float(field.max()) for field in fields) / peak_reference

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    colors = {"rim": "#df5a37", "base": "#267e85", "wall": "#744ba8",
              "feature": "#df5a37", "surface": "#267e85"}
    for name in dict.fromkeys(labels):
        mask = np.array([label == name for label in labels])
        ax3d.scatter(points[mask, 0] * 100, points[mask, 2] * 100,
                     points[mask, 1] * 100, label=name,
                     color=colors.get(name, "#744ba8"), s=36)
    ax3d.set(xlabel="x [cm]", ylabel="z [cm]", zlabel="depth y [cm]",
             title=title)
    ax3d.set_zlim(bottom * 100 - 2, top * 100 + 2)
    ax3d.legend(loc="upper left")
    for index, (name, _) in enumerate(planes):
        ax = fig.add_subplot(2, 2, index + 2)
        field = fields[index] / peak_reference
        if index < 2:
            extent = [xy[0] * 100, xy[-1] * 100, xy[-1] * 100, xy[0] * 100]
            ax.set(xlabel="x [cm]", ylabel="z [cm]")
        else:
            extent = [xy[0] * 100, xy[-1] * 100,
                      y_axis[0] * 100, y_axis[-1] * 100]
            ax.set(xlabel="x [cm]", ylabel="depth y [cm]")
            ax.axhline(bottom * 100, color="white", linewidth=0.7, alpha=0.5)
            ax.axhline(top * 100, color="white", linewidth=0.7, alpha=0.5)
        im = ax.imshow(field, origin="lower", extent=extent, cmap="magma",
                       vmin=0, vmax=peak_average_fraction, aspect="auto")
        ax.set_title(f"{name}: time-averaged intensity")
        fig.colorbar(im, ax=ax, label="fraction of mean single-focus peak")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    report = {
        "points": points.tolist(), "labels": labels,
        "frame_count": len(frames), "fps": fps,
        "cycle_seconds_at_requested_fps": len(frames) / fps,
        "serial_minimum_seconds_230400_8n1": len(frames) * 258 * 10 / 230400,
        "phase_generation_seconds": generation_seconds,
        "mean_single_focus_peak_intensity": peak_reference,
        "maximum_time_average_to_single_focus_peak": peak_average_fraction,
        "note": "Offline time average of 32-bin single-focus frames. Not a simultaneous field or measured tactile 3D shape."
    }
    if metadata:
        report["model"] = metadata
    metrics_path = path.with_suffix(".json")
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics_path, report
