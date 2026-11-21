"""Locate the 16x16 array and recover its pose from the known emitter pitch."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

import my_patterns

GRID_SIZE = 16
PITCH_M = 0.0105
GRID = np.array([(column, row) for row in range(GRID_SIZE)
                 for column in range(GRID_SIZE)], np.float32)
ORIENTATIONS = ("right-up", "right-down", "left-up", "left-down",
                "up-right", "up-left", "down-right", "down-left")


@dataclass
class BoardGrid:
    pixels: np.ndarray
    observed: np.ndarray
    matched: np.ndarray
    homography: np.ndarray
    rms_px: float
    corners: np.ndarray
    model: str = "pixel_homography"


def _largest_cluster(points):
    if len(points) < 100:
        raise ValueError("可见圆心过少;请移开手并让整块阵列进入画面")
    tree = cKDTree(points)
    distances = tree.query(points, k=2)[0][:, 1]
    spacing = float(np.median(distances))
    parents = np.arange(len(points))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for a, b in tree.query_pairs(max(10.0, spacing * 1.8)):
        ra, rb = root(a), root(b)
        parents[ra] = rb
    labels = np.array([root(i) for i in range(len(points))])
    cluster = points[labels == np.bincount(labels).argmax()]
    if len(cluster) < 100:
        raise ValueError("找不到完整的阵列圆点区域")
    return cluster


def _corner_seeds(points):
    x, y = points.T
    return np.float32([points[np.argmin(x + y)], points[np.argmax(x - y)],
                       points[np.argmax(x + y)], points[np.argmin(x - y)]])


def _fit_seed(points, corners):
    ideal_corners = np.float32([[0, 0], [15, 0], [15, 15], [0, 15]])
    H = cv2.getPerspectiveTransform(ideal_corners, corners)
    tree = cKDTree(points)
    for iteration in range(6):
        predicted = cv2.perspectiveTransform(GRID.reshape(-1, 1, 2), H).reshape(-1, 2)
        spacing = min(np.median(np.linalg.norm(predicted[1:] - predicted[:-1], axis=1)),
                      np.median(np.linalg.norm(predicted[16:] - predicted[:-16], axis=1)))
        radius = max(3.0, float(spacing) * (0.8 if iteration == 0 else 0.5))
        distances, nearest = tree.query(predicted)
        ranked = np.argsort(distances)
        paired = {}
        claimed = set()
        for index in ranked:
            candidate = int(nearest[index])
            if distances[index] < radius and candidate not in claimed:
                paired[int(index)] = candidate
                claimed.add(candidate)
        if len(paired) < 75:
            return None
        ids = np.array(sorted(paired))
        H_new, inliers = cv2.findHomography(GRID[ids], points[[paired[i] for i in ids]],
                                            cv2.RANSAC, max(3.0, radius * 0.65))
        if H_new is None or inliers is None or inliers.sum() < 75:
            return None
        H = H_new
    predicted = cv2.perspectiveTransform(GRID.reshape(-1, 1, 2), H).reshape(-1, 2)
    distances, nearest = tree.query(predicted)
    spacing = np.median(np.linalg.norm(predicted[16:] - predicted[:-16], axis=1))
    valid = distances < max(3.0, spacing * 0.4)
    # One detected circle must never stand in for two adjacent emitters.
    for index in np.argsort(distances):
        if not valid[index]:
            continue
        duplicates = np.flatnonzero(valid & (nearest == nearest[index]))
        valid[duplicates[duplicates != index]] = False
    error = float(np.sqrt(np.mean(distances[valid] ** 2))) if valid.any() else np.inf
    return BoardGrid(predicted, points[nearest], valid, H, error,
                     cv2.perspectiveTransform(ideal_corners.reshape(-1, 1, 2), H).reshape(4, 2))


def fit_grid_to_centers(centers, min_matches=145):
    """Fit a perspective lattice with missing and extra circle candidates."""
    points = _largest_cluster(np.asarray(centers, np.float32))
    corners = _corner_seeds(points)
    candidates = []
    # Board supports can look like an extra row of circles. Try shorter
    # quadrilaterals, then require actual observations on every outer edge.
    for trim in np.arange(0, 3.01, 0.25):
        seed = corners.copy()
        seed[2] += (corners[1] - corners[2]) * (trim / 15.0)
        seed[3] += (corners[0] - corners[3]) * (trim / 15.0)
        fitted = _fit_seed(points, seed)
        if fitted is not None:
            candidates.append(fitted)
    if not candidates:
        raise ValueError("圆心无法组成 16x16 阵列网格")
    covered = [item for item in candidates if item.matched.sum() >= min_matches
               and min(item.matched.reshape(16, 16)[0].sum(),
                       item.matched.reshape(16, 16)[-1].sum(),
                       item.matched.reshape(16, 16)[:, 0].sum(),
                       item.matched.reshape(16, 16)[:, -1].sum()) >= 5]
    if not covered:
        raise ValueError("阵列边缘圆孔匹配不足;请露出完整四边并避免支架进入阵列")
    grid = max(covered, key=lambda item: (item.matched.sum(), -item.rms_px))
    valid = grid.matched
    if valid.sum() < min_matches:
        raise ValueError(f"仅匹配 {valid.sum()}/256 个阵元;请移开遮挡并正对整块阵列")
    spacing = np.median(np.linalg.norm(grid.pixels[16:] - grid.pixels[:-16], axis=1))
    if grid.rms_px > max(3.5, spacing * 0.28):
        raise ValueError(f"阵列圆心重投影误差 {grid.rms_px:.1f}px 过大")
    quadrants = [valid.reshape(16, 16)[rows, cols].sum()
                 for rows in (slice(0, 8), slice(8, 16))
                 for cols in (slice(0, 8), slice(8, 16))]
    if min(quadrants) < 25:
        raise ValueError("阵列有一个角被遮挡;请显示全部四角")
    return grid


def detect_board_grid(frame):
    """Find ring centers over the full frame, then reject non-lattice clutter."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    size = min(gray.shape)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=max(7, size // 55), param1=80, param2=12,
                               minRadius=max(3, size // 125),
                               maxRadius=max(9, size // 45))
    if circles is None:
        raise ValueError("画面中没有检测到换能器圆孔")
    return fit_grid_to_centers(circles[0, :, :2])


def grid_from_corners(corners, pixel_to_ray=None, ray_to_pixel=None):
    """Predict array centers in undistorted ray space when optics are known."""
    corners = np.asarray(corners, np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("请依次点击阵列左上、右上、右下、左下的最外侧阵元中心")
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        raise ValueError("四角顺序不正确;请右键撤销并按左上、右上、右下、左下重选")
    ideal = np.float32([[0, 0], [15, 0], [15, 15], [0, 15]])
    H = cv2.getPerspectiveTransform(ideal, corners)
    if abs(np.linalg.det(H)) < 1e-9:
        raise ValueError("四角位置无法组成有效阵列")
    if (pixel_to_ray is None) != (ray_to_pixel is None):
        raise ValueError("相机去畸变转换必须成对提供")
    if pixel_to_ray is None:
        pixels = cv2.perspectiveTransform(GRID.reshape(-1, 1, 2), H).reshape(-1, 2)
        model = "pixel_homography"
    else:
        ray_corners = np.asarray([pixel_to_ray(uv) for uv in corners], np.float32)
        if ray_corners.shape != (4, 2) or not np.isfinite(ray_corners).all():
            raise ValueError("四角去畸变失败;请检查相机内参")
        ray_H = cv2.getPerspectiveTransform(ideal, ray_corners)
        rays = cv2.perspectiveTransform(GRID.reshape(-1, 1, 2), ray_H).reshape(-1, 2)
        pixels = np.asarray([ray_to_pixel(ray) for ray in rays], np.float32)
        model = "undistorted_rays"
    if pixels.shape != (len(GRID), 2) or not np.isfinite(pixels).all():
        raise ValueError("阵列网格投影失败;请检查镜头参数")
    spacing = min(np.median(np.linalg.norm(pixels[1:16] - pixels[:15], axis=1)),
                  np.median(np.linalg.norm(pixels[16:] - pixels[:-16], axis=1)))
    if spacing < 5:
        raise ValueError("阵列在画面中过小;请让板子占据更多像素")
    return BoardGrid(pixels, pixels.copy(), np.ones(len(GRID), bool), H, 0.0,
                     corners, model)


def board_object_points(orientation="right-up"):
    """The image grid orientation names the physical +X and +Z edges."""
    directions = {"right-up": ((1, 0), (0, -1)),
                  "left-up": ((-1, 0), (0, -1)),
                  "right-down": ((1, 0), (0, 1)),
                  "left-down": ((-1, 0), (0, 1)),
                  "up-right": ((0, -1), (1, 0)),
                  "down-right": ((0, 1), (1, 0)),
                  "up-left": ((0, -1), (-1, 0)),
                  "down-left": ((0, 1), (-1, 0))}
    if orientation not in directions:
        raise ValueError(f"未知阵列方向: {orientation}")
    xdir, zdir = map(np.asarray, directions[orientation])
    centered = (GRID - 7.5) * PITCH_M
    x = centered @ xdir
    z = centered @ zdir
    return np.column_stack((x + my_patterns.CENTER[0], np.zeros(len(x)),
                            z + my_patterns.CENTER[2])).astype(np.float32)


def estimate_board_pose(grid, intrinsics, display_to_sensor, orientation):
    """PnP gives metric pose; positive Y is always the camera-facing side."""
    objects = board_object_points(orientation)
    pixels = np.asarray([display_to_sensor(p) for p in grid.observed], np.float32)
    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                  [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]], float)
    distortion = np.asarray(intrinsics.coeffs, float)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objects[grid.matched], pixels[grid.matched], K, distortion,
        iterationsCount=300, reprojectionError=8.0, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or inliers is None or len(inliers) < 100:
        raise ValueError("阵列三维姿态无法稳定求解")
    projected, _ = cv2.projectPoints(objects[grid.matched], rvec, tvec, K, distortion)
    residual = np.linalg.norm(projected.reshape(-1, 2) - pixels[grid.matched], axis=1)
    residual = residual[inliers.ravel()]
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if rms > 7.0:
        raise ValueError(f"阵列三维重投影误差 {rms:.1f}px 过大")
    camera_from_board = cv2.Rodrigues(rvec)[0]
    board_from_camera = camera_from_board.T
    translation = -board_from_camera @ tvec.reshape(3)
    if translation[1] < 0:
        board_from_camera[1] *= -1
        translation[1] *= -1
    if not 0.07 <= translation[1] <= 0.7:
        raise ValueError(f"相机离阵列面 {translation[1]:.3f}m,超出可信范围")
    return board_from_camera, translation, rms, len(inliers)


def _rigid_board_fit(camera, board):
    camera_center = camera.mean(axis=0)
    board_center = board.mean(axis=0)
    U, _, Vt = np.linalg.svd((camera - camera_center).T @ (board - board_center))
    correction = np.diag((1.0, 1.0, np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ correction @ U.T
    t = board_center - R @ camera_center
    return R, t


def estimate_board_pose_from_depth(grid, camera_points, orientation):
    """Robustly align measured array surface points with emitter coordinates."""
    measured = np.asarray(camera_points, float)
    if measured.shape != (len(GRID), 3):
        raise ValueError("阵列深度点数量不正确")
    usable = grid.matched & np.isfinite(measured).all(axis=1)
    camera = measured[usable]
    board = board_object_points(orientation)[usable].astype(float)
    if len(camera) < 50:
        raise ValueError(f"阵列有效深度点只有 {len(camera)} 个,至少需要 50 个")
    rng = np.random.default_rng(0)
    best = np.zeros(len(camera), bool)
    best_rms = np.inf
    for _ in range(300):
        ids = rng.choice(len(camera), 3, replace=False)
        if np.linalg.norm(np.cross(board[ids[1]] - board[ids[0]],
                                   board[ids[2]] - board[ids[0]])) < 1e-4:
            continue
        R, t = _rigid_board_fit(camera[ids], board[ids])
        errors = np.linalg.norm(camera @ R.T + t - board, axis=1)
        inliers = errors < 0.012
        if inliers.sum() > best.sum() or (inliers.sum() == best.sum()
                                        and inliers.any()
                                        and np.mean(errors[inliers] ** 2) < best_rms):
            best = inliers
            best_rms = np.mean(errors[inliers] ** 2)
    if best.sum() < max(50, int(np.ceil(len(camera) * 0.45))):
        raise ValueError(f"阵列深度平面内点只有 {best.sum()}/{len(camera)} 个")
    R, t = _rigid_board_fit(camera[best], board[best])
    errors = np.linalg.norm(camera @ R.T + t - board, axis=1)
    best = errors < 0.012
    if best.sum() < 50:
        raise ValueError("阵列深度平面拟合不稳定")
    spread = np.zeros(len(GRID), bool)
    spread[np.flatnonzero(usable)[best]] = True
    quarters = [spread.reshape(16, 16)[rows, cols].sum()
                for rows in (slice(0, 8), slice(8, 16))
                for cols in (slice(0, 8), slice(8, 16))]
    if min(quarters) < 5:
        raise ValueError("阵列有效深度点集中在局部;请检查四角深度画面")
    R, t = _rigid_board_fit(camera[best], board[best])
    rms_m = float(np.sqrt(np.mean(
        np.linalg.norm(camera[best] @ R.T + t - board[best], axis=1) ** 2)))
    if rms_m > 0.0085:
        raise ValueError(f"阵列深度平面误差 {rms_m * 1000:.1f}mm 过大")
    if t[1] < 0:
        R[1] *= -1
        t[1] *= -1
    if not 0.07 <= t[1] <= 0.7:
        raise ValueError(f"相机离阵列面 {t[1]:.3f}m,超出可信范围")
    return R, t, rms_m * 1000, int(best.sum()), len(camera)
