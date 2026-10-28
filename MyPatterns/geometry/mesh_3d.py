"""Convert a 3D mesh into a sparse ultrasonic focal-scan path."""

from pathlib import Path

import numpy as np


def _line_samples(polydata, spacing):
    """Sample PyVista line cells, including the interior of long edges."""
    packed = polydata.lines
    samples = []
    offset = 0
    while offset < len(packed):
        length = int(packed[offset])
        ids = packed[offset + 1:offset + 1 + length]
        for a, b in zip(ids[:-1], ids[1:]):
            start, end = polydata.points[a], polydata.points[b]
            count = max(2, int(np.ceil(np.linalg.norm(end - start) / spacing)) + 1)
            samples.extend(np.linspace(start, end, count))
        offset += length + 1
    return np.asarray(samples).reshape(-1, 3)


def _farthest_points(candidates, count):
    """Cover the candidate geometry with a deterministic, sparse point set."""
    count = min(count, len(candidates))
    selected = [int(np.argmax(candidates[:, 1]))]
    distances = np.sum((candidates - candidates[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(distances, np.sum(
            (candidates - candidates[index]) ** 2, axis=1))
    return candidates[selected]


def _nearest_order(points):
    """Limit travel between successive focus frames."""
    remaining = list(range(1, len(points)))
    order = [0]
    while remaining:
        last = points[order[-1]]
        next_index = min(remaining, key=lambda index: np.sum((points[index] - last) ** 2))
        order.append(next_index)
        remaining.remove(next_index)
    return points[order]


def mesh_outline(path, near=0.10, width=0.06, height=0.06,
                 count=24, feature_angle=40.0):
    """Map mesh Z-up to array y-depth, then sample sharp/boundary edges."""
    import pyvista as pv

    if Path(path).suffix.lower() not in (".stl", ".obj", ".ply"):
        raise ValueError("目前支持 STL、OBJ、PLY 网格")
    if any(not np.isfinite(value) or value <= 0 for value in (near, width, height)):
        raise ValueError("near、width、height 必须为正数，单位米")
    if not 8 <= count <= 120:
        raise ValueError("焦点数须在 8 到 120 之间")
    mesh = pv.read(path)
    mesh = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
    original = np.asarray(mesh.points)
    if len(original) < 3 or not np.all(np.isfinite(original)):
        raise ValueError("模型缺少有效的三维表面")
    spans = np.ptp(original, axis=0)
    lateral = max(spans[0], spans[1])
    if lateral <= 0 or spans[2] <= 0:
        raise ValueError("模型需要横向尺寸和 Z 方向高度；导入前请调整朝向")
    scale = min(width / lateral, height / spans[2])
    center = (original.min(axis=0) + original.max(axis=0)) / 2
    min_z = original[:, 2].min()

    def transform(points):
        result = np.empty_like(points, dtype=float)
        result[:, 0] = (points[:, 0] - center[0]) * scale
        result[:, 1] = near + (points[:, 2] - min_z) * scale
        result[:, 2] = (points[:, 1] - center[1]) * scale
        return result

    edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=True,
                                       manifold_edges=False, non_manifold_edges=True,
                                       feature_angle=feature_angle)
    edge_points = transform(_line_samples(edges, max(lateral, spans[2]) / 30))
    if len(edge_points) >= count:
        candidates, source = edge_points, "feature"
    else:
        candidates, source = transform(original), "surface"
    candidates = np.unique(np.round(candidates, 5), axis=0)
    if len(candidates) < 3:
        raise ValueError("模型可用的表面采样点不足")
    points = _nearest_order(_farthest_points(candidates, count))
    report = {
        "path": str(path), "source": source,
        "requested_points": count, "selected_points": len(points),
        "input_bounds": list(mesh.bounds), "meters_per_model_unit": float(scale),
        "near_m": near, "physical_bounds_m": {
            "x": [float(points[:, 0].min()), float(points[:, 0].max())],
            "depth_y": [float(points[:, 1].min()), float(points[:, 1].max())],
            "z": [float(points[:, 2].min()), float(points[:, 2].max())],
        },
        "maximum_jump_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).max()),
    }
    return points, [source] * len(points), report
