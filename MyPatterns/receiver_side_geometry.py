"""Estimate a hidden receiver face from the visible side of a cylindrical can."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import hand_track


def _circle_candidates(points, radius):
    """Yield fixed-radius circle centers through pairs of side-surface points."""
    rng = np.random.default_rng(2718)
    count = len(points)
    for _ in range(350):
        i, j = rng.integers(count, size=2)
        delta = points[j] - points[i]
        separation = float(np.linalg.norm(delta))
        if not 0.55 * radius <= separation <= 1.95 * radius:
            continue
        midpoint = (points[i] + points[j]) / 2
        normal = np.array([-delta[1], delta[0]]) / separation
        offset = np.sqrt(max(0.0, radius * radius - separation * separation / 4))
        yield midpoint + normal * offset
        yield midpoint - normal * offset


def _angle_span(points, center):
    angles = np.sort(np.mod(np.arctan2(points[:, 1] - center[1],
                                      points[:, 0] - center[0]), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return 2 * np.pi - float(np.max(gaps))


def _fit_circle(points, radius):
    tolerance = max(0.0015, min(0.0025, 0.16 * radius))
    best_center = None
    best_mask = None
    best_count = 0
    for center in _circle_candidates(points, radius):
        residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        mask = residuals < tolerance
        count = int(mask.sum())
        if count > best_count:
            best_center, best_mask, best_count = center, mask, count
    if best_center is None or best_count < max(40, 0.55 * len(points)):
        return None, None
    fit = least_squares(
        lambda center: np.linalg.norm(points[best_mask] - center, axis=1) - radius,
        best_center, loss="soft_l1", f_scale=0.001)
    center = fit.x
    residuals = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    mask = residuals < tolerance
    if (mask.sum() < max(40, 0.55 * len(points))
            or np.median(residuals[mask]) > 0.0018
            or _angle_span(points[mask], center) < 1.1):
        return None, None
    return center, mask


def _has_cylindrical_curvature(points, center, radius):
    """Reject a plane that only resembles a small arc of the known circle."""
    shifted = points - center
    angles = np.arctan2(shifted[:, 1], shifted[:, 0])
    mean_angle = np.angle(np.mean(np.exp(1j * angles)))
    unwrapped = (angles - mean_angle + np.pi) % (2 * np.pi) - np.pi
    low, high = np.percentile(unwrapped, (10, 90))
    if high - low < 0.7:
        return False
    # A visible cylinder bows away from the chord joining its outer sectors.
    left = shifted[unwrapped <= low + 0.2 * (high - low)]
    right = shifted[unwrapped >= high - 0.2 * (high - low)]
    middle = shifted[(unwrapped >= low + 0.4 * (high - low))
                     & (unwrapped <= low + 0.6 * (high - low))]
    if min(len(left), len(right), len(middle)) < 8:
        return False
    chord = np.median(right, axis=0) - np.median(left, axis=0)
    chord_length = np.linalg.norm(chord)
    if chord_length < 0.45 * radius:
        return False
    normal = np.array([-chord[1], chord[0]]) / chord_length
    a, b, mid = np.median(left, axis=0), np.median(right, axis=0), np.median(middle, axis=0)
    sagitta = abs(float(np.dot(mid - (a + b) / 2, normal)))
    expected = radius - np.sqrt(max(0.0, radius ** 2 - (chord_length / 2) ** 2))
    return sagitta >= max(0.0009, 0.35 * expected)


def estimate_side_receiver(cap, box, R, t, length_m, diameter_m):
    """Return (receiving-face XYZ in board frame, diagnostic reason).

    The selected ROI must contain the complete metal can, without the leads.
    The can axis is assumed close to the board normal, with its receiving face
    at the lower-Y end. This requires a visible, depth-supported cylindrical
    side and a measured can length/diameter; uncertain geometry returns None.
    """
    if not (np.isfinite(length_m) and 0.005 <= length_m <= 0.10
            and np.isfinite(diameter_m) and 0.005 <= diameter_m <= 0.08):
        return None, "invalid measured dimensions"
    depth = getattr(cap, "depth_m", None)
    intrinsics = getattr(cap, "intrinsics", None)
    if depth is None or intrinsics is None:
        return None, "no aligned RealSense depth"
    depth = np.asarray(depth)
    if depth.ndim != 2:
        return None, "invalid depth image"
    R, t = np.asarray(R, float), np.asarray(t, float)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        return None, "invalid board pose"
    x, y, w, h = map(int, box)
    height, width = depth.shape
    if (w < 20 or h < 20 or x < 0 or y < 0 or x + w > width
            or y + h > height):
        return None, "select the complete metal can"

    pad = max(8, int(0.25 * max(w, h)))
    x0, x1 = max(0, x - pad), min(width, x + w + pad)
    y0, y1 = max(0, y - pad), min(height, y + h + pad)
    surround = depth[y0:y1, x0:x1]
    outside = np.ones(surround.shape, bool)
    outside[y-y0:y-y0+h, x-x0:x-x0+w] = False
    background = surround[outside & np.isfinite(surround)
                          & (surround >= 0.07) & (surround <= 0.50)]
    if len(background) < max(40, 0.05 * w * h):
        return None, "insufficient background depth"
    foreground_limit = float(np.median(background)) - 0.010
    step = max(1, int(np.ceil(np.sqrt(w * h / 1800))))
    rows, cols = np.mgrid[y:y+h:step, x:x+w:step]
    values = depth[rows, cols]
    valid = np.isfinite(values) & (values >= 0.07) & (values < foreground_limit)
    if valid.sum() < 70 or valid.sum() < 0.16 * values.size:
        return None, "insufficient can depth"
    pixels = np.column_stack((cols[valid], rows[valid]))
    depths = values[valid]
    deproject = cap._rs.rs2_deproject_pixel_to_point
    camera_points = np.asarray([
        deproject(intrinsics,
                  list(hand_track.display_to_sensor_uv(uv, cap.width, cap.height,
                                                       cap.rotation)), float(distance))
        for uv, distance in zip(pixels, depths)], float)
    if camera_points.shape != (len(pixels), 3):
        return None, "invalid camera deprojection"
    world = camera_points @ R.T + t
    world = world[np.isfinite(world).all(axis=1)]
    if len(world) < 70:
        return None, "insufficient valid 3D points"

    radius = diameter_m / 2
    horizontal = world[:, (0, 2)]
    axis = np.linalg.svd(horizontal - np.mean(horizontal, axis=0),
                         full_matrices=False)[2][0]
    width = np.ptp(horizontal @ axis)
    if not 0.75 * diameter_m <= width <= 1.25 * diameter_m:
        return None, "can width disagrees with measured diameter"
    center, inliers = _fit_circle(world[:, (0, 2)], radius)
    if center is None:
        return None, "cylindrical side not resolved"
    side = world[inliers]
    centered_side = side - np.mean(side, axis=0)
    plane_rms = np.linalg.svd(centered_side, compute_uv=False)[-1] / np.sqrt(len(side))
    if (plane_rms < 0.0008
            or not _has_cylindrical_curvature(side[:, (0, 2)], center, radius)):
        return None, "visible surface is too flat to confirm a cylinder"
    lower, upper = np.percentile(side[:, 1], (3, 97))
    observed_length = upper - lower
    if not 0.72 * length_m <= observed_length <= 1.12 * length_m:
        return None, "can length disagrees with measured length"
    midpoint = (lower + upper) / 2
    low_points = side[side[:, 1] < midpoint, :][:, (0, 2)]
    high_points = side[side[:, 1] >= midpoint, :][:, (0, 2)]
    if len(low_points) < 20 or len(high_points) < 20:
        return None, "one end lacks depth support"
    low_center, _ = _fit_circle(low_points, radius)
    high_center, _ = _fit_circle(high_points, radius)
    if (low_center is None or high_center is None
            or np.linalg.norm(low_center - high_center) > 0.004):
        return None, "can axis is tilted or depth is inconsistent"
    receiving_y = midpoint - length_m / 2
    target = np.array([center[0], receiving_y, center[1]])
    if not np.isfinite(target).all() or receiving_y < 0.0:
        return None, "receiving face is outside board space"
    return target, "side-cylinder depth"


