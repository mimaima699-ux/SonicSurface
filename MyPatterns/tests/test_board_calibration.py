import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import my_patterns

from board_calibration import (BoardGrid, GRID, board_object_points,
                               estimate_board_pose, estimate_board_pose_from_depth,
                               fit_grid_to_centers, grid_from_corners)
from hand_track import (PalmFollower, RealSenseSource, board_poses_agree,
                        board_pose_from_frame, boardcalibrate, run_tracking)
from hand_track import validate_calibration


class BoardCalibrationTests(unittest.TestCase):
    def test_perspective_grid_survives_missing_and_extra_circles(self):
        rng = np.random.default_rng(12)
        corners = np.float32([[170, 650], [420, 665], [465, 905], [110, 900]])
        H = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [15, 0], [15, 15], [0, 15]]), corners)
        expected = cv2.perspectiveTransform(GRID.reshape(-1, 1, 2), H).reshape(-1, 2)
        visible = expected[rng.random(len(expected)) > 0.13]
        noisy = visible + rng.normal(0, 0.55, visible.shape)
        clutter = rng.uniform([550, 50], [700, 1100], (35, 2))
        grid = fit_grid_to_centers(np.vstack((noisy, clutter)))
        self.assertGreaterEqual(grid.matched.sum(), 200)
        self.assertLess(grid.rms_px, 2)
        np.testing.assert_allclose(grid.corners, corners, atol=3)

    def test_support_circles_do_not_replace_last_board_row(self):
        corners = np.float32([[175, 650], [415, 660], [460, 870], [115, 865]])
        H = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [15, 0], [15, 15], [0, 15]]), corners)
        emitters = cv2.perspectiveTransform(GRID.reshape(-1, 1, 2), H).reshape(-1, 2)
        supports = np.float32([[112, 900], [170, 900], [260, 900],
                               [340, 900], [460, 900]])
        grid = fit_grid_to_centers(np.vstack((emitters, supports)))
        self.assertEqual(grid.matched.reshape(16, 16)[-1].sum(), 16)
        np.testing.assert_allclose(grid.corners, corners, atol=2)

    def test_known_board_pose_recovers_metric_camera_to_array_transform(self):
        objects = board_object_points("right-up")
        np.testing.assert_allclose(objects[:, (0, 2)].mean(axis=0),
                                   my_patterns.CENTER[[0, 2]], atol=1e-6)
        K = np.array([[650, 0, 640], [0, 650, 360], [0, 0, 1]], float)
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        rvec = cv2.Rodrigues(rotation)[0]
        tvec = np.array([0.01, -0.02, 0.35])
        pixels, _ = cv2.projectPoints(objects, rvec, tvec, K, np.zeros(5))
        pixels = pixels.reshape(-1, 2)
        grid = BoardGrid(pixels, pixels, np.ones(256, bool), np.eye(3), 0,
                         np.zeros((4, 2)))
        intrinsics = SimpleNamespace(fx=650, fy=650, ppx=640, ppy=360,
                                     coeffs=[0, 0, 0, 0, 0])
        R, t, rms_px, inliers = estimate_board_pose(
            grid, intrinsics, lambda point: point, "right-up")
        camera_center = tvec
        np.testing.assert_allclose(R @ camera_center + t, np.zeros(3), atol=1e-4)
        self.assertGreater(t[1], 0)
        self.assertLess(rms_px, 1e-3)
        self.assertEqual(inliers, 256)

    def test_depth_pose_rejects_holes_and_background(self):
        rng = np.random.default_rng(9)
        board = board_object_points("right-up").astype(float)
        true_R = cv2.Rodrigues(np.array([0.18, -0.12, 0.22]))[0]
        true_t = np.array([0.015, 0.31, -0.025])
        camera = (board - true_t) @ true_R
        camera += rng.normal(0, 0.001, camera.shape)
        camera[::9] = np.nan
        camera[4::11] += [0, 0, 0.05]
        grid = BoardGrid(np.zeros((256, 2)), np.zeros((256, 2)),
                         np.ones(256, bool), np.eye(3), 0,
                         np.zeros((4, 2)))
        R, t, rms_mm, inliers, valid = estimate_board_pose_from_depth(
            grid, camera, "right-up")
        self.assertGreater(inliers, 170)
        self.assertLess(valid, 256)
        self.assertLess(rms_mm, 3)
        np.testing.assert_allclose(R, true_R, atol=0.02)
        np.testing.assert_allclose(t, true_t, atol=0.005)

    def test_board_frame_falls_back_to_color_when_depth_unavailable(self):
        grid = BoardGrid(np.zeros((256, 2)), np.zeros((256, 2)),
                         np.ones(256, bool), np.eye(3), 3.0,
                         np.zeros((4, 2)))
        cap = SimpleNamespace(depth_m=np.ones((40, 40)),
                              point_3d=lambda uv: None, width=40, height=40,
                              rotation="none", serial="camera", intrinsics=object())
        with patch("board_calibration.detect_board_grid", return_value=grid), \
                patch("board_calibration.estimate_board_pose",
                      return_value=(np.eye(3), np.array([0, 0.3, 0]), 4.0, 120)):
            data, _ = board_pose_from_frame(cap, np.zeros((40, 40, 3)), "right-up")
        self.assertEqual(data["board_pose_method"], "color")

    def test_bad_generated_board_quality_is_recoverable(self):
        grid = BoardGrid(np.zeros((256, 2)), np.zeros((256, 2)),
                         np.ones(256, bool), np.eye(3), 6.0,
                         np.zeros((4, 2)))
        cap = SimpleNamespace(depth_m=None, width=40, height=40,
                              rotation="none", serial="camera", intrinsics=object())
        with patch("board_calibration.detect_board_grid", return_value=grid), \
                patch("board_calibration.estimate_board_pose",
                      return_value=(np.eye(3), np.array([0, 0.3, 0]), 4.0, 120)):
            with self.assertRaises(ValueError):
                board_pose_from_frame(cap, np.zeros((40, 40, 3)), "right-up")

    def test_clicked_corners_and_depth_recover_board_pose(self):
        corners = np.float32([[180, 210], [420, 220], [460, 470], [130, 460]])
        grid = grid_from_corners(corners)
        board = board_object_points("right-up").astype(float)
        R = cv2.Rodrigues(np.array([0.12, 0.08, -0.14]))[0]
        t = np.array([0.01, 0.31, -0.02])
        camera = (board - t) @ R

        def point_at(uv):
            index = np.argmin(np.linalg.norm(grid.pixels - uv, axis=1))
            return camera[index]

        cap = SimpleNamespace(depth_m=np.ones((600, 600)),
                              point_3d=point_at, width=600, height=600,
                              rotation="none", serial="camera")
        data, _ = board_pose_from_frame(
            cap, np.zeros((600, 600, 3)), "right-up",
            grid=grid, manual_corners=True)
        self.assertEqual(data["board_pose_method"], "depth_corners")
        self.assertEqual(data["board_inliers"], 256)
        np.testing.assert_allclose(data["t"], t, atol=1e-4)
        np.testing.assert_allclose(data["board_corners_px"], corners)

    def test_corner_grid_interpolates_in_undistorted_ray_space(self):
        ray_corners = np.float32([[-0.38, -0.27], [0.34, -0.25],
                                  [0.40, 0.32], [-0.35, 0.30]])

        def to_pixel(ray):
            ray = np.asarray(ray, float)
            scale = 1 + 0.32 * np.dot(ray, ray)
            return (ray * scale * 700 + [640, 360]).tolist()

        def to_ray(pixel):
            ray = (np.asarray(pixel, float) - [640, 360]) / 700
            for _ in range(12):
                ray = (np.asarray(pixel, float) - [640, 360]) / (
                    700 * (1 + 0.32 * np.dot(ray, ray)))
            return ray.tolist()

        corners = np.float32([to_pixel(ray) for ray in ray_corners])
        corrected = grid_from_corners(corners, to_ray, to_pixel)
        naive = grid_from_corners(corners)
        ray_H = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [15, 0], [15, 15], [0, 15]]), ray_corners)
        rays = cv2.perspectiveTransform(GRID.reshape(-1, 1, 2), ray_H).reshape(-1, 2)
        expected = np.float32([to_pixel(ray) for ray in rays])
        np.testing.assert_allclose(corrected.pixels, expected, atol=0.01)
        self.assertEqual(corrected.model, "undistorted_rays")
        self.assertGreater(np.max(np.linalg.norm(naive.pixels - expected, axis=1)), 2)

    def test_clicked_corners_require_depth(self):
        grid = grid_from_corners(np.float32(
            [[180, 210], [420, 220], [460, 470], [130, 460]]))
        cap = SimpleNamespace(depth_m=None, width=600, height=600,
                              rotation="none", serial="camera")
        with self.assertRaisesRegex(ValueError, "四角深度定位失败"):
            board_pose_from_frame(cap, np.zeros((600, 600, 3)),
                                  "right-up", grid=grid, manual_corners=True)

    def test_clicked_corners_reject_crossed_order(self):
        with self.assertRaisesRegex(ValueError, "四角顺序"):
            grid_from_corners([[180, 210], [460, 470],
                               [420, 220], [130, 460]])

    def test_saved_board_orientation_and_fit_quality_are_required(self):
        valid = {"version": 2, "mode": "realsense_board_3d",
                 "target": "index_tip", "R": np.eye(3).tolist(), "t": [0, 0.3, 0],
                 "orientation": "right-up", "board_matched": 175,
                 "board_inliers": 130, "board_grid_rms_px": 4.0,
                 "board_pose_rms_px": 5.0}
        validate_calibration(valid, "index_tip")
        with self.assertRaises(SystemExit):
            validate_calibration({**valid, "orientation": None}, "index_tip")
        with self.assertRaises(SystemExit):
            validate_calibration({**valid, "board_matched": 80}, "index_tip")
        with self.assertRaises(SystemExit):
            validate_calibration({**valid, "board_pose_rms_px": 12}, "index_tip")

    def test_adjacent_board_poses_must_be_stable(self):
        first = {"orientation": "right-up", "R": np.eye(3).tolist(),
                 "t": [0, 0.30, 0]}
        self.assertTrue(board_poses_agree(first, {**first, "t": [0.005, 0.30, 0]}))
        self.assertFalse(board_poses_agree(first, {**first, "t": [0.025, 0.30, 0]}))
        self.assertFalse(board_poses_agree(first, {**first, "orientation": "left-up"}))

    def test_boardcalibrate_saves_only_after_axis_confirmation(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        source = SimpleNamespace(read=lambda: (True, frame), release=lambda: None)
        seen_orientations = []

        def pose(_source, _frame, orientation):
            seen_orientations.append(orientation)
            return {"orientation": orientation, "board_matched": 180,
                    "board_inliers": 130, "board_pose_rms_px": 4.0,
                    "R": np.eye(3).tolist(), "t": [0, 0.3, 0]}, object()

        with tempfile.TemporaryDirectory() as folder:
            save = Path(folder) / "board.json"
            with patch("hand_track.open_source", return_value=(source, "realsense")), \
                    patch("hand_track.board_pose_from_frame", side_effect=pose), \
                    patch("hand_track.draw_board_pose", return_value=frame), \
                    patch("hand_track.open_preview_window"), patch("cv2.imshow"), \
                    patch("cv2.waitKey", side_effect=[ord("d"), 255, ord("c")]), \
                    patch("cv2.destroyAllWindows"):
                boardcalibrate(save=save)
            self.assertEqual(seen_orientations,
                             ["right-up", "right-down", "right-down"])
            self.assertEqual(json.loads(save.read_text())["orientation"], "right-down")

    def test_auto_board_tracking_uses_fresh_pose(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        cap = object.__new__(RealSenseSource)
        cap.serial = "test-camera"
        cap.rotation = "ccw"
        cap.stream_name = "color"
        cap.read = lambda: (True, frame)
        cap.point_3d = lambda uv, mask: np.array([0, 0.12, 0])
        cap.release = lambda: None
        calib = {"version": 2, "mode": "realsense_board_3d",
                 "target": "index_tip", "R": np.eye(3).tolist(),
                 "t": [0, 0, 0], "orientation": "right-up",
                 "board_matched": 180, "board_inliers": 130,
                 "board_grid_rms_px": 3.0, "board_pose_rms_px": 4.0,
                 "serial": "test-camera", "rotation": "ccw",
                 "frame_size": [640, 480], "stream": "color"}
        fresh = {**calib, "t": [0.02, 0, 0]}
        positions = []
        feed = PalmFollower.feed

        def record(follower, timestamp, position):
            positions.append(position)
            return feed(follower, timestamp, position)

        observation = SimpleNamespace(point=lambda target: (320, 240),
                                      mask=np.ones((480, 640), np.uint8))
        with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                patch("hand_track.board_pose_from_frame", return_value=(fresh, None)), \
                patch("hand_track.PalmDetector") as detector, \
                patch("hand_track.PalmFollower.feed", record), \
                patch("hand_track.draw_hand_observation", return_value=frame), \
                patch("hand_track.open_preview_window"), patch("cv2.imshow"), \
                patch("cv2.setMouseCallback"), \
                patch("cv2.waitKey", return_value=ord("q")), \
                patch("cv2.destroyAllWindows"):
            detector.return_value.detect.return_value = observation
            result = run_tracking(calib=calib, realsense=True, dry=True)
        np.testing.assert_allclose(positions[0], [0.02, 0.12, 0])
        self.assertEqual(result["gate_counts"]["inside"], 1)

    def test_manual_depth_tracking_rechecks_saved_corners(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        cap = object.__new__(RealSenseSource)
        cap.serial = "test-camera"
        cap.rotation = "none"
        cap.stream_name = "color"
        cap.read = lambda: (True, frame)
        cap.release = lambda: None
        calib = {"version": 2, "mode": "realsense_board_3d",
                 "target": "index_tip", "R": np.eye(3).tolist(),
                 "t": [0, 0.3, 0], "orientation": "right-up",
                 "board_pose_method": "depth_corners", "board_matched": 256,
                 "board_inliers": 180, "board_depth_rms_mm": 2.0,
                 "board_grid_rms_px": 0.0,
                 "board_corners_px": [[170, 120], [400, 125],
                                      [420, 360], [150, 355]],
                 "serial": "test-camera", "rotation": "none",
                 "frame_size": [640, 480], "stream": "color"}
        calls = []

        def pose(_cap, _frame, orientation, grid=None, manual_corners=False):
            calls.append((orientation, grid, manual_corners))
            return calib, grid

        with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                patch("hand_track.grid_from_camera_corners", return_value=object()), \
                patch("hand_track.board_pose_from_frame", side_effect=pose), \
                patch("hand_track.PalmDetector") as detector, \
                patch("hand_track.draw_hand_observation", return_value=frame), \
                patch("hand_track.open_preview_window"), patch("cv2.imshow"), \
                patch("cv2.setMouseCallback"), \
                patch("cv2.waitKey", return_value=ord("q")), \
                patch("cv2.destroyAllWindows"):
            detector.return_value.detect.return_value = None
            run_tracking(calib=calib, realsense=True, dry=True)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(manual and grid is not None
                            for _, grid, manual in calls))
