"""Offline checks for RealSense fingertip tracking and safety."""

import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hand_track import (CALIB_PATH, Deadband, HandObservation, LossSupervisor,
                        OneEuro, PalmFollower, hand_mask_from_landmarks,
                        in_bounds, run_tracking)


class RecorderBoard:
    """鸭子类型板卡:只记录发送的相位,供断言。"""

    def __init__(self):
        self.frames = []

    def send(self, phases):
        self.frames.append(np.asarray(phases, float).copy())

    def off(self):
        self.send(np.full(256, np.nan))

    def close(self):
        pass


class ViewerSnapshotTests(unittest.TestCase):
    def test_phase_diagnostic_reports_commanded_payload(self):
        import my_patterns
        from hand_track import phase_diagnostic

        center = (0.0, 0.12, 0.0)
        side = (0.04, 0.12, 0.0)
        _, first = phase_diagnostic(center, my_patterns.phases_focus(center))
        line, second = phase_diagnostic(
            side, my_patterns.phases_focus(side), first)
        changed = sum(a != b for a, b in zip(first, second))
        self.assertIn("x=+0.040", line)
        self.assertIn(f"changed={changed}/256", line)
        self.assertIn(f"vs_center={changed}/256", line)
        self.assertEqual(second, my_patterns.phase_payload(
            my_patterns.phases_focus(side)))

    @staticmethod
    def tracking_scene():
        from hand_track import RealSenseSource

        calib = {"version": 2, "mode": "realsense_3d", "R": np.eye(3).tolist(),
                 "t": [0, 0, 0], "target": "index_tip", "rms_mm": 1,
                 "max_mm": 2, "frame_size": [640, 480], "stream": "color"}
        points = np.full((21, 2), (320.0, 240.0))
        obs = HandObservation(points, hand_mask_from_landmarks(points, (480, 640, 3)))
        frame = np.zeros((480, 640, 3), np.uint8)
        cap = object.__new__(RealSenseSource)
        cap.stream_name = "color"
        cap.serial = "test-camera"
        cap.read = lambda: (True, frame)
        cap.release = lambda: None
        cap.point_3d = lambda uv, mask: np.array([0.01, 0.12, -0.02])
        return calib, obs, cap

    def test_phase_snapshot_tracks_the_last_emitted_pattern_and_off(self):
        emitted = []
        follower = PalmFollower(lambda phases: emitted.append(np.asarray(phases).copy()))
        self.assertIsNone(follower.phase_snapshot())
        follower.feed(1.0, (0.0, 0.12, 0.0))
        np.testing.assert_allclose(follower.phase_snapshot(), emitted[-1])
        follower.pause(1.1)
        self.assertIsNone(follower.phase_snapshot())

    def test_3d_view_rejects_sim_without_camera(self):
        with self.assertRaisesRegex(SystemExit, "--view-3d"):
            run_tracking(sim=True, dry=True, view_3d=True)

    def test_tracking_sends_measured_and_commanded_points_to_3d_view(self):
        calib, obs, cap = self.tracking_scene()
        with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                patch("hand_track.PalmDetector") as detector, \
                patch("live_diagnostic_3d.LiveDiagnostic3D") as viewer, \
                patch("builtins.print") as printed, \
                patch("cv2.destroyAllWindows"):
            detector.return_value.detect.return_value = obs
            viewer.return_value.poll.return_value = True
            viewer.return_value.consume_pause_toggle.return_value = False
            viewer.return_value.update.return_value = False
            run_tracking(calib=calib, dry=True, realsense=True,
                         view_3d=True, phase_trace=True)
        args = viewer.return_value.update.call_args.args
        np.testing.assert_allclose(args[4], [0.01, 0.12, -0.02])
        np.testing.assert_allclose(args[5], [0.01, 0.12, -0.02])
        self.assertEqual(np.asarray(args[6]).shape, (256,))
        self.assertEqual(args[7], "inside")
        self.assertTrue(any("[phase dry]" in call.args[0]
                            for call in printed.call_args_list))
        viewer.return_value.close.assert_called_once()

    def test_space_freezes_without_new_camera_frames_and_silences_board(self):
        calib, obs, cap = self.tracking_scene()
        board = RecorderBoard()
        reads = []
        frame = np.zeros((480, 640, 3), np.uint8)

        def read():
            reads.append(len(reads) + 1)
            return True, frame

        cap.read = read
        keys = iter((ord(" "), -1, ord(" "), ord("q")))
        with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                patch("hand_track.PalmDetector") as detector, \
                patch("hand_track.Board", return_value=board), \
                patch("hand_track.LossSupervisor") as supervisor, \
                patch("live_diagnostic_3d.LiveDiagnostic3D") as viewer, \
                patch("hand_track.open_preview_window"), \
                patch("cv2.setMouseCallback"), patch("cv2.imshow"), \
                patch("cv2.waitKey", side_effect=lambda _delay: next(keys)), \
                patch("cv2.destroyAllWindows"):
            detector.return_value.detect.return_value = obs
            supervisor.return_value.start.return_value = supervisor.return_value
            supervisor.return_value.error = None
            viewer.return_value.poll.return_value = True
            viewer.return_value.consume_pause_toggle.return_value = False
            viewer.return_value.update.return_value = True
            run_tracking(calib=calib, port="COM_FAKE", no_mod=True,
                         realsense=True, view_3d=True)
        self.assertEqual(len(reads), 2)
        self.assertEqual(detector.return_value.detect.call_count, 2)
        self.assertGreaterEqual(sum(np.isnan(p).all() for p in board.frames), 1)
        viewer.return_value.set_paused.assert_any_call(True)
        viewer.return_value.set_paused.assert_any_call(False)
        viewer.return_value.close.assert_called_once()


def circle_samples(seconds=8.0, fps=30.0, radius=0.05, noise=0.002,
                   drop=(3.0, 4.0), seed=0):
    """合成掌心样本:4 秒一圈的圆 + 2mm 噪声,3~4 秒丢手。"""
    rng = np.random.default_rng(seed)
    for i in range(int(seconds * fps)):
        t = i / fps
        th = 2.0 * np.pi * t / 4.0
        pos = np.array([radius * np.cos(th), 0.12, radius * np.sin(th)])
        pos = pos + rng.normal(0.0, noise, 3)
        yield t, None if drop[0] <= t < drop[1] else pos


class OneEuroTests(unittest.TestCase):
    def test_smooths_noise_on_constant_signal(self):
        rng = np.random.default_rng(1)
        filt = OneEuro(min_cutoff=1.0, beta=0.02)
        outs = [filt(np.array([0.05, -0.05]) + rng.normal(0, 0.003, 2), 1 / 30)
                for _ in range(200)]
        tail = np.array(outs[100:])
        self.assertLess(float(tail.std(axis=0).mean()), 0.003)   # 比输入噪声平滑
        np.testing.assert_allclose(tail.mean(axis=0), [0.05, -0.05], atol=0.002)

    def test_tracks_fast_step_with_low_lag(self):
        filt = OneEuro(min_cutoff=1.0, beta=0.5)                 # 大 beta:快动低延迟
        out = None
        for i in range(30):                                      # 0.5s @30fps
            out = filt(np.array([0.05, 0.0]) if i >= 15 else np.zeros(2), 1 / 30)
        self.assertLess(abs(out[0] - 0.05), 0.005)               # 0.25s 内基本跟上

    def test_reset_forgets_history(self):
        filt = OneEuro()
        for _ in range(50):
            filt(np.array([0.05, 0.05]), 1 / 30)
        filt.reset()
        out = filt(np.array([-0.05, -0.05]), 1 / 30)
        np.testing.assert_allclose(out, [-0.05, -0.05])          # 重置后不再拉回旧值


class DeadbandTests(unittest.TestCase):
    def test_small_moves_swallowed_big_moves_pass(self):
        band = Deadband(0.005)
        self.assertIsNotNone(band.update((0.0, 0.0)))
        self.assertIsNone(band.update((0.003, 0.0)))             # 3mm < 5mm
        self.assertIsNotNone(band.update((0.006, 0.0)))          # 6mm ≥ 5mm
        band.reset()
        self.assertIsNotNone(band.update((0.02, 0.02)))          # 重置后首点必过

    def test_experimental_two_millimeter_motion_updates_phase(self):
        band = Deadband(0.002)
        self.assertIsNotNone(band.update((0.0, 0.12, 0.0)))
        self.assertIsNone(band.update((0.001, 0.12, 0.0)))
        self.assertIsNotNone(band.update((0.003, 0.12, 0.0)))


class BoundsTests(unittest.TestCase):
    def test_bounds(self):
        from hand_track import DEFAULT_BOUNDS
        self.assertTrue(in_bounds((0.0, 0.12, 0.0), DEFAULT_BOUNDS))
        self.assertFalse(in_bounds((0.09, 0.12, 0.0), DEFAULT_BOUNDS))
        self.assertTrue(in_bounds((0.0, 0.21, 0.0), DEFAULT_BOUNDS))
        self.assertFalse(in_bounds((0.0, 0.26, 0.0), DEFAULT_BOUNDS))
        self.assertFalse(in_bounds((0.0, 0.12, -0.1), DEFAULT_BOUNDS))

    def test_tracking_gate_reports_why_focus_is_silent(self):
        from hand_track import DEFAULT_BOUNDS, tracking_gate
        uv = (100.0, 150.0)
        point = np.array([0.0, 0.12, 0.0])
        self.assertEqual(tracking_gate(None, None, None, DEFAULT_BOUNDS), "no_hand")
        self.assertEqual(tracking_gate(uv, None, None, DEFAULT_BOUNDS), "no_depth")
        self.assertEqual(tracking_gate(uv, point, np.array([np.nan, 0.12, 0]),
                                       DEFAULT_BOUNDS), "invalid_position")
        self.assertEqual(tracking_gate(uv, point, np.array([0.09, 0.12, 0]),
                                       DEFAULT_BOUNDS), "outside_x")
        self.assertEqual(tracking_gate(uv, point, np.array([0, 0.26, 0]),
                                       DEFAULT_BOUNDS), "outside_y")
        self.assertEqual(tracking_gate(uv, point, np.array([0, 0.12, 0.09]),
                                       DEFAULT_BOUNDS), "outside_z")
        self.assertEqual(tracking_gate(uv, point, point, DEFAULT_BOUNDS), "inside")


class HandTargetTests(unittest.TestCase):
    def test_projected_focus_matches_rotated_camera_coordinates(self):
        from hand_track import (display_to_sensor_uv, draw_tracking_focus,
                                project_board_point)

        cap = SimpleNamespace(width=1280, height=720, rotation="ccw",
                              intrinsics=SimpleNamespace(fx=650, fy=650,
                                                         ppx=640, ppy=360))
        R = np.eye(3)
        translation = np.array([0, 0, -0.3])
        focus = np.array([0.03, 0.12, 0])
        uv = project_board_point(focus, R, translation, cap)
        np.testing.assert_allclose(uv, (620, 574), atol=1)
        np.testing.assert_allclose(
            display_to_sensor_uv(uv, cap.width, cap.height, cap.rotation),
            (705, 620), atol=1)
        display = np.zeros((1280, 720, 3), np.uint8)
        draw_tracking_focus(display, focus, R, translation, cap)
        self.assertGreater(display[574, 620, 0], 200)
        self.assertGreater(display[574, 620, 2], 200)

    def test_allowed_outline_uses_observed_fingertip_height(self):
        from hand_track import draw_tracking_focus

        cap = SimpleNamespace(width=640, height=480, rotation="none",
                              intrinsics=SimpleNamespace(fx=500, fy=500,
                                                         ppx=320, ppy=240))
        observed = np.array([0.01, 0.14, -0.02])
        display = np.zeros((480, 640, 3), np.uint8)
        with patch("hand_track.project_board_point", return_value=(320, 240)) as project:
            draw_tracking_focus(display, None, np.eye(3), np.zeros(3),
                                cap, observed=observed, gate="inside")
        allowed_points = [call.args[0] for call in project.call_args_list[4:8]]
        self.assertEqual([point[1] for point in allowed_points], [0.14] * 4)

    def test_mask_and_selected_fingertip(self):
        points = np.full((21, 2), (50.0, 60.0))
        points[0] = (50, 85)
        points[5] = (35, 55)
        points[9] = (50, 52)
        points[13] = (65, 55)
        points[17] = (72, 69)
        points[6:9] = ((34, 43), (32, 30), (30, 20))
        mask = hand_mask_from_landmarks(points, (100, 100, 3))
        obs = HandObservation(points, mask)
        self.assertEqual(obs.point("index_tip"), (30.0, 20.0))
        self.assertEqual(obs.point("palm"), (50.0, 68.5))
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask[20, 30], 255)
        self.assertEqual(mask[0, 0], 0)
        points[8] = (-5, 20)
        self.assertIsNone(HandObservation(points, mask).point("index_tip"))

    def test_click_selects_visible_landmark_only(self):
        from hand_track import choose_hand_target

        points = np.full((21, 2), (-20.0, -20.0))
        points[4] = (45, 40)
        points[8] = (85, 40)
        observation = HandObservation(points, np.zeros((100, 100), np.uint8))
        self.assertEqual(choose_hand_target(observation, (47, 42)), "thumb_tip")
        self.assertEqual(choose_hand_target(observation, (84, 41)), "index_tip")
        self.assertIsNone(choose_hand_target(observation, (8, 88)))
        self.assertIsNone(choose_hand_target(None, (45, 40)))

    def test_preview_click_silences_old_focus_before_changing_target(self):
        import cv2
        from hand_track import RealSenseSource

        frame = np.zeros((480, 640, 3), np.uint8)
        cap = object.__new__(RealSenseSource)
        cap.stream_name = "color"
        cap.serial = "camera"
        cap.read = lambda: (True, frame)
        cap.release = lambda: None
        cap.point_3d = lambda uv, mask: np.array([(uv[0] - 320) / 1000,
                                                  0.12, 0.0])
        points = np.full((21, 2), (320.0, 240.0))
        points[8] = (340, 245)
        points[12] = (360, 250)
        obs = HandObservation(points, hand_mask_from_landmarks(points, frame.shape))
        calib = {"version": 2, "mode": "realsense_3d", "R": np.eye(3).tolist(),
                 "t": [0, 0, 0], "target": "index_tip", "rms_mm": 1,
                 "max_mm": 2, "frame_size": [640, 480], "stream": "color"}
        callback = None
        frames = 0

        def set_callback(_window, fn):
            nonlocal callback
            callback = fn

        def key(_delay):
            nonlocal frames
            frames += 1
            if frames == 1:
                callback(cv2.EVENT_LBUTTONDOWN, 360, 250, 0, None)
                return -1
            return ord("q")

        with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                patch("hand_track.PalmDetector") as detector, \
                patch("hand_track.open_preview_window"), \
                patch("cv2.setMouseCallback", side_effect=set_callback), \
                patch("cv2.imshow"), patch("cv2.waitKey", side_effect=key), \
                patch("cv2.destroyAllWindows"):
            detector.return_value.detect.return_value = obs
            stats = run_tracking(calib=calib, dry=True, realsense=True)
        self.assertEqual(len(stats["positions"]), 2)
        self.assertAlmostEqual(stats["positions"][0][1], 0.02)
        self.assertAlmostEqual(stats["positions"][1][1], 0.04)
        self.assertEqual(len(stats["off_times"]), 1)

    def test_tracking_uses_tip_and_rejects_invalid_calibration(self):
        from unittest.mock import patch
        from hand_track import RealSenseSource

        calib = {"version": 2, "mode": "realsense_3d", "R": np.eye(3).tolist(),
                 "t": [0, 0, 0], "target": "index_tip", "rms_mm": 1, "max_mm": 2,
                 "frame_size": [640, 480], "stream": "color"}
        with self.assertRaises(SystemExit):
            run_tracking(calib={"H": np.eye(3).tolist()}, dry=True,
                         realsense=True)

        points = np.full((21, 2), (320.0, 240.0))
        points[8] = (340, 245)
        obs = HandObservation(points, hand_mask_from_landmarks(points, (480, 640, 3)))
        frame = np.zeros((480, 640, 3), np.uint8)
        fake_cap = object.__new__(RealSenseSource)
        fake_cap.stream_name = "color"
        fake_cap.serial = "test-camera"
        fake_cap.read = lambda: (True, frame)
        fake_cap.release = lambda: None
        fake_cap.point_3d = lambda uv, mask: np.array([0.02, 0.13, 0.005])
        seen = []
        original_feed = PalmFollower.feed

        def capture_feed(follower, t, pos):
            seen.append(pos)
            return original_feed(follower, t, pos)

        with patch("hand_track.open_source", return_value=(fake_cap, "any")), \
                patch("hand_track.PalmDetector") as detector, \
                patch("hand_track.PalmFollower.feed", capture_feed), \
                patch("cv2.namedWindow"), patch("cv2.resizeWindow"), \
                patch("cv2.imshow"), patch("cv2.setMouseCallback"), \
                patch("cv2.waitKey", return_value=ord("q")), \
                patch("cv2.destroyAllWindows"):
            detector.return_value.detect.return_value = obs
            run_tracking(calib=calib, dry=True, target="index_tip", realsense=True)
        np.testing.assert_allclose(seen[0], (0.02, 0.13, 0.005), atol=1e-10)


class PartialTipTests(unittest.TestCase):
    @staticmethod
    def scene(x=72, y=68):
        import cv2

        rng = np.random.default_rng(123)
        frame = np.zeros((140, 160, 3), np.uint8)
        patch = rng.integers(60, 220, (25, 25), dtype=np.uint8)
        frame[y-12:y+13, x-12:x+13] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        depth = np.full((140, 160), 0.27, np.float32)
        depth[y-13:y+14, x-13:x+14] = 0.14
        return frame, depth

    def test_clicked_partial_tip_tracks_without_hand_landmarks(self):
        from hand_track import PartialTipTracker

        tracker = PartialTipTracker()
        first, first_depth = self.scene()
        moved, moved_depth = self.scene(78, 71)
        self.assertTrue(tracker.seed(first, first_depth, (72, 68), now=1.0))
        self.assertEqual(tracker.update(moved, moved_depth, now=1.03), (78.0, 71.0))

    def test_partial_tip_rejects_background_and_depth_jump(self):
        from hand_track import PartialTipTracker

        tracker = PartialTipTracker()
        frame, depth = self.scene()
        self.assertFalse(tracker.seed(frame, np.full_like(depth, 0.14),
                                      (72, 68), now=1.0))
        self.assertTrue(tracker.seed(frame, depth, (72, 68), now=1.0))
        jumped = depth.copy()
        jumped[55:82, 59:86] = 0.20
        self.assertIsNone(tracker.update(frame, jumped, now=1.03))
        self.assertIsNone(tracker.update(frame, depth, now=1.06))

    def test_partial_tip_times_out_without_recent_visual_confirmation(self):
        from hand_track import PartialTipTracker

        frame, depth = self.scene()
        tracker = PartialTipTracker(max_gap=0.15)
        self.assertTrue(tracker.seed(frame, depth, (72, 68), now=1.0))
        self.assertIsNone(tracker.update(frame, depth, now=1.17))

    def test_partial_tip_near_frame_edge(self):
        from hand_track import PartialTipTracker

        tracker = PartialTipTracker()
        first, depth = self.scene(13, 68)
        moved, moved_depth = self.scene(16, 69)
        self.assertTrue(tracker.seed(first, depth, (13, 68), now=1.0))
        self.assertEqual(tracker.update(moved, moved_depth, now=1.03),
                         (16.0, 69.0))

    def test_unknown_tip_cannot_be_enabled_in_experimental_mode(self):
        from hand_track import run_tracking

        with self.assertRaisesRegex(SystemExit, "--manual-tip"):
            run_tracking(calib={}, dry=True, realsense=True,
                         experimental=True, manual_tip_enabled=True)

    def test_mediapipe_retries_with_padding_and_maps_back_to_sensor_frame(self):
        import sys
        from hand_track import PalmDetector

        image = np.zeros((80, 100, 3), np.uint8)
        calls = []
        landmarks = [SimpleNamespace(x=(72 + 18) / 136,
                                     y=(40 + 14) / 108) for _ in range(21)]

        def process(frame):
            calls.append(frame.shape)
            return SimpleNamespace(multi_hand_landmarks=(
                [SimpleNamespace(landmark=landmarks)] if len(calls) == 2 else None))

        detector = object.__new__(PalmDetector)
        detector._legacy = SimpleNamespace(process=process)
        detector._landmarker = None
        with patch.dict(sys.modules, {"mediapipe": SimpleNamespace()}):
            observation = detector.detect(image)
        self.assertEqual(calls, [(80, 100, 3), (108, 136, 3)])
        np.testing.assert_allclose(observation.point("index_tip"), (72, 40))

    def test_manual_click_requires_three_matching_frames_before_focus(self):
        import cv2
        from hand_track import RealSenseSource

        frame, depth = self.scene()
        cap = object.__new__(RealSenseSource)
        cap.stream_name = "color"
        cap.serial = "camera"
        cap.depth_m = depth
        cap.read = lambda: (True, frame)
        cap.release = lambda: None
        cap.point_3d = lambda uv, mask: np.array([0.0, 0.12, 0.0])
        calib = {"version": 2, "mode": "realsense_3d",
                 "R": np.eye(3).tolist(), "t": [0, 0, 0],
                 "target": "index_tip", "rms_mm": 1, "max_mm": 2,
                 "frame_size": [160, 140], "stream": "color"}
        state = {"callback": None, "frames": 0}

        def key(_delay):
            state["frames"] += 1
            if state["frames"] == 1:
                state["callback"](cv2.EVENT_LBUTTONDOWN, 72, 68, 0, None)
            return ord("q") if state["frames"] == 4 else -1

        with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                patch("hand_track.PalmDetector") as detector, \
                patch("hand_track.open_preview_window"), \
                patch("cv2.setMouseCallback", side_effect=lambda _win, fn: state.update(callback=fn)), \
                patch("cv2.imshow"), patch("cv2.waitKey", side_effect=key), \
                patch("cv2.destroyAllWindows"):
            detector.return_value.detect.return_value = None
            stats = run_tracking(calib=calib, dry=True, realsense=True,
                                 manual_tip_enabled=True)
        self.assertEqual(stats["sent"], 1)
        self.assertEqual(stats["gate_counts"]["inside"], 1)


class PalmFollowerTests(unittest.TestCase):
    def make_follower(self):
        board = RecorderBoard()
        follower = PalmFollower(board.send, kind="focus", dist=0.12,
                                deadband_m=0.005, lost_s=0.5)
        return follower, board

    def test_sim_loop_follows_filters_and_silences(self):
        follower, board = self.make_follower()
        samples = list(circle_samples())
        in_bounds_count = sum(1 for _, p in samples if p is not None)
        for t, pos in samples:
            follower.feed(t, pos)
        follower.close()
        # 全部帧合法
        self.assertTrue(all(f.shape == (256,) for f in board.frames))
        self.assertTrue(all(np.all(np.isfinite(f)) or np.isnan(f).all()
                            for f in board.frames))
        # 死区生效:发送次数明显少于在界样本数
        self.assertLess(follower.stats["sent"], in_bounds_count * 0.7)
        # 跟踪精度:稳态(t>1s,跳过启动瞬态与丢手段)偏差 < 12mm
        # (2mm 噪声 + One-Euro 在 78mm/s 圆周速度下的滞后)
        for t, x, y, z in follower.stats["positions"]:
            if t < 1.0 or 3.0 <= t < 4.0:
                continue
            th = 2.0 * np.pi * t / 4.0
            d = np.hypot(x - 0.05 * np.cos(th), z - 0.05 * np.sin(th))
            self.assertLess(d, 0.012, f"t={t:.2f} 偏差 {d * 1000:.1f}mm")
        # 丢手 0.5s 后静默一次,复得后恢复,结束时关闭
        self.assertEqual(len(follower.stats["off_times"]), 1)
        self.assertGreater(follower.stats["off_times"][0], 3.4)
        self.assertLess(follower.stats["off_times"][0], 4.0)
        self.assertTrue(np.isnan(board.frames[-1]).all())

    def test_out_of_bounds_silences_immediately(self):
        follower, board = self.make_follower()
        follower.feed(0.0, (0.0, 0.12, 0.0))
        follower.feed(1 / 30, (0.0, 0.27, 0.0))                  # 高度越界
        self.assertEqual(follower.stats["off_times"], [1 / 30]) # 立即静默,不等超时
        follower.feed(2 / 30, (0.01, 0.12, 0.0))                 # 回到界内即恢复
        self.assertGreater(follower.stats["sent"], 1)

    def test_loss_supervisor_silences_without_new_camera_frames(self):
        board = RecorderBoard()
        follower = PalmFollower(board.send, lost_s=0.08)
        follower.feed(time.perf_counter(), (0.0, 0.12, 0.0))
        supervisor = LossSupervisor(follower).start()
        try:
            deadline = time.perf_counter() + 0.5
            while not follower.stats["off_times"] and time.perf_counter() < deadline:
                time.sleep(0.01)
        finally:
            supervisor.stop()
        self.assertIsNone(supervisor.error)
        self.assertEqual(len(follower.stats["off_times"]), 1)
        self.assertTrue(np.isnan(board.frames[-1]).all())

    def test_active_duration_counts_continuous_pattern_not_phase_updates(self):
        board = RecorderBoard()
        follower = PalmFollower(board.send, deadband_m=0.005, lost_s=0.1)
        follower.feed(0.0, (0.0, 0.12, 0.0))
        np.testing.assert_allclose(follower.focus_position(), (0.0, 0.12, 0.0))
        follower.feed(0.05, (0.001, 0.12, 0.0))
        follower.feed(0.20, None)
        follower.close(t=0.25)
        self.assertEqual(follower.stats["sent"], 1)
        self.assertAlmostEqual(follower.stats["active_s"], 0.20)
        self.assertEqual(len(follower.stats["off_times"]), 1)
        self.assertIsNone(follower.focus_position())

    def test_out_of_bounds_before_first_focus_is_not_a_dropout(self):
        board = RecorderBoard()
        follower = PalmFollower(board.send)
        follower.feed(0.0, (0.20, 0.12, 0.0))
        follower.close(t=0.1)
        self.assertEqual(follower.stats["sent"], 0)
        self.assertEqual(follower.stats["off_times"], [])

    def test_trap_warm_start_chain(self):
        """trap 连续跟踪:温启动链不崩、产出合法相位、死区外持续发送。
        (6mm 阶跃经 One-Euro 强平滑后进死区是正常行为,故用 20mm 大步长。)"""
        board = RecorderBoard()
        follower = PalmFollower(board.send, kind="trap", dist=0.12,
                                deadband_m=0.005)
        steps = [(0.020 * i, 0.12, 0.0) for i in range(6)]       # 每步 20mm 滑动
        for i, pos in enumerate(steps):
            follower.feed(i / 30.0, pos)
        follower.close()
        self.assertGreaterEqual(follower.stats["sent"], 2)
        emitted = [f for f in board.frames if np.isfinite(f).all()]
        self.assertGreaterEqual(len(emitted), 2)
        self.assertTrue(np.isnan(board.frames[-1]).all())        # close 静默

    def test_trap_reuses_quantized_cell(self):
        from unittest.mock import patch
        import my_patterns

        follower = PalmFollower(lambda _: None, kind="trap")
        with patch.object(my_patterns, "phases_trap", wraps=my_patterns.phases_trap) as solve:
            a = follower._trap_phases((0.001, 0.12, 0.001))
            follower._prev_phases = a
            b = follower._trap_phases((0.002, 0.12, 0.001))
        self.assertEqual(solve.call_count, 1)
        np.testing.assert_allclose(a, b, atol=1e-12)


class HomographyTests(unittest.TestCase):
    def test_round_trip_recovers_mapping(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv 未安装(摄像头标定路径)")
        # 真值 H: 平移 + 缩放(像素/米 ≈ 1000,中心 320,240,x 镜像)
        H_true = np.array([[-1000.0, 0.0, 320.0],
                           [0.0, 1000.0, 240.0],
                           [0.0, 0.0, 1.0]])
        world = np.array([[x, z] for z in np.linspace(-0.06, 0.06, 3)
                          for x in np.linspace(-0.06, 0.06, 3)], float)
        pixels = []
        for x, z in world:
            v = H_true @ np.array([x, z, 1.0])
            pixels.append([v[0] / v[2], v[1] / v[2]])
        H, _ = cv2.findHomography(np.array(pixels, float), world)
        from hand_track import pixel_to_world
        err = [np.hypot(*(np.array(pixel_to_world(H, p)) - w))
               for p, w in zip(pixels, world)]
        self.assertLess(max(err), 1e-6)


class ThreeDTests(unittest.TestCase):
    def test_calibration_checks_adjacent_distance_and_direction(self):
        from hand_track import calibration_neighbor_checks, calibration_targets

        targets = calibration_targets(3, 0.06, 0.10, 0.05)
        samples = [np.asarray(point) for point in targets]
        samples[17] = None
        checks = calibration_neighbor_checks(samples, targets, 17,
                                             np.asarray(targets[17]))
        self.assertEqual({check["axis"] for check in checks}, {"x", "y", "z"})
        self.assertTrue(all(not check["warn"] for check in checks))
        self.assertEqual({round(check["expected_mm"]) for check in checks}, {50, 60})

        samples[13] = None
        moved = samples[12] + np.array([0.042, 0.016, 0.012])
        checks = calibration_neighbor_checks(samples, targets, 13, moved)
        height = next(check for check in checks if check["axis"] == "y")
        self.assertLess(abs(height["measured_mm"] - 50), 5)
        self.assertGreater(height["angle_deg"], 20)
        self.assertTrue(height["warn"])

    def test_review_allows_redoing_final_point_before_saving(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from hand_track import calibrate, calibration_targets

        targets = calibration_targets(3, 0.06, 0.10, 0.05)
        frame = np.zeros((480, 640, 3), np.uint8)

        class FakeCap:
            serial = "test-camera"
            rotation = "ccw"
            depth_m = np.full((480, 640), 0.20, np.float32)
            index = 0
            frames_at_point = 0
            redone = False
            enter_tried = False

            def read(self):
                self.frames_at_point += 1
                return True, frame

            def point_3d(self, uv, mask):
                point = np.asarray(targets[self.index], float)
                return point + ([0.08, 0, 0] if self.index == 17 and not self.redone
                                else [0, 0, 0])

            def release(self):
                pass

        cap = FakeCap()
        keys = []

        def key(_delay):
            if cap.index == len(targets):
                if not cap.redone:
                    if not cap.enter_tried:
                        cap.enter_tried = True
                        keys.append("enter_invalid")
                        return 13
                    cap.redone = True
                    cap.index = 17
                    cap.frames_at_point = 0
                    keys.append("r")
                    return ord("r")
                keys.append("q_valid")
                return ord("q")
            if cap.frames_at_point >= 8:
                cap.index += 1
                cap.frames_at_point = 0
                return ord(" ")
            return -1

        observation = SimpleNamespace(point=lambda target: (320.0, 240.0),
                                      mask=np.ones((480, 640), np.uint8))
        with tempfile.TemporaryDirectory() as folder, \
                patch("hand_track.open_source", return_value=(cap, "realsense")), \
                patch("hand_track.PalmDetector") as detector, \
                patch("hand_track.draw_hand_observation", return_value=frame.copy()), \
                patch("hand_track.open_preview_window"), \
                patch("cv2.imshow"), patch("cv2.waitKey", side_effect=key), \
                patch("cv2.destroyAllWindows"), redirect_stdout(io.StringIO()):
            detector.return_value.detect.return_value = observation
            saved = calibrate(save=Path(folder) / "calib.json", realsense=True)
        self.assertEqual(keys, ["enter_invalid", "r", "q_valid"])
        self.assertLess(saved["max_mm"], 1e-6)
        self.assertEqual(saved["rotation"], "ccw")

    def test_rotated_fingertip_maps_back_to_sensor_depth(self):
        from types import SimpleNamespace
        from hand_track import (RealSenseSource, display_to_sensor_uv,
                                rotate_image)

        width, height = 20, 16
        raw = np.full((height, width), 0.2, np.float32)
        raw[5:14, 3:12] = 0.13
        sensor_uv = (7, 9)
        for rotation, display_uv in (("none", (7, 9)), ("ccw", (9, 12)),
                                     ("cw", (6, 7)), ("180", (12, 6))):
            with self.subTest(rotation=rotation):
                source = object.__new__(RealSenseSource)
                source.depth_enabled = True
                source.depth_m = rotate_image(raw, rotation)
                source.intrinsics = object()
                source.width, source.height = width, height
                source.rotation = rotation
                source._rs = SimpleNamespace(rs2_deproject_pixel_to_point=
                    lambda intrinsics, uv, depth: [*uv, depth])
                self.assertEqual(
                    display_to_sensor_uv(display_uv, width, height, rotation),
                    sensor_uv)
                point = source.point_3d(display_uv)
                np.testing.assert_allclose(point, [7, 9, 0.13], atol=1e-6)

    def test_changed_camera_rotation_rejects_old_calibration(self):
        from unittest.mock import patch
        from hand_track import RealSenseSource

        calib = {"version": 2, "mode": "realsense_3d", "target": "index_tip",
                 "R": np.eye(3).tolist(), "t": [0, 0, 0],
                 "rms_mm": 11, "max_mm": 25, "serial": "test-camera"}
        fake_cap = object.__new__(RealSenseSource)
        fake_cap.rotation = "ccw"
        fake_cap.serial = "test-camera"
        fake_cap.release = lambda: None
        with patch("hand_track.open_source", return_value=(fake_cap, "realsense")), \
                patch("cv2.destroyAllWindows"):
            with self.assertRaisesRegex(RuntimeError, "重新标定"):
                run_tracking(calib=calib, dry=True, realsense=True)

    def test_formal_calibration_accepts_12mm_rms_and_30mm_max(self):
        from hand_track import validate_calibration
        calib = {"version": 2, "mode": "realsense_3d", "target": "index_tip",
                 "R": np.eye(3).tolist(), "t": [0, 0, 0],
                 "rms_mm": 12.0, "max_mm": 30.0}
        validate_calibration(calib, "index_tip")
        with self.assertRaises(SystemExit):
            validate_calibration({**calib, "rms_mm": 12.01}, "index_tip")
        with self.assertRaises(SystemExit):
            validate_calibration({**calib, "max_mm": 30.01}, "index_tip")

    def test_diagnostic_requires_explicit_experimental_mode(self):
        from hand_track import (calibration_from_diagnostic, calibration_targets,
                                validate_calibration)
        world = np.asarray(calibration_targets(3, 0.06, 0.10, 0.05))
        diagnostic = {"points": {"world": world.tolist(), "camera": world.tolist()},
                      "serial": "test-camera", "frame_size": [1280, 720]}
        calib = calibration_from_diagnostic(diagnostic)
        validate_calibration(calib, "index_tip", experimental=True)
        with self.assertRaises(SystemExit):
            validate_calibration(calib, "index_tip")
        diagnostic["points"]["camera"][0][0] += 0.10
        with self.assertRaises(SystemExit):
            calibration_from_diagnostic(diagnostic)

    def test_experimental_tracking_restricts_output_before_opening_board(self):
        from hand_track import EXPERIMENTAL_BOUNDS, experimental_bounds
        calib = {"version": 2, "mode": "realsense_3d_experimental",
                 "R": np.eye(3).tolist(), "t": [0, 0, 0],
                 "rms_mm": 13.1, "max_mm": 28.6, "target": "index_tip"}
        with self.assertRaises(SystemExit):
            run_tracking(calib=calib, experimental=True, realsense=True,
                         port="COM_TEST", duty=0.7)
        with self.assertRaises(SystemExit):
            run_tracking(calib=calib, experimental=True, realsense=True,
                         port="COM_TEST", duty=0.2, kind="trap")
        with self.assertRaises(SystemExit):
            run_tracking(calib=calib, experimental=True, realsense=True,
                         duty=0.2, max_seconds=21)
        self.assertFalse(in_bounds((0.05, 0.12, 0.0), EXPERIMENTAL_BOUNDS))
        self.assertTrue(in_bounds((0.0, 0.12, 0.0), EXPERIMENTAL_BOUNDS))
        self.assertTrue(in_bounds((0.045, 0.12, 0), experimental_bounds(0.05)))
        self.assertFalse(in_bounds((0.055, 0.12, 0), experimental_bounds(0.05)))
        wide = experimental_bounds(0.08, 0.08, 0.18)
        self.assertTrue(in_bounds((0.075, 0.17, -0.075), wide))
        self.assertFalse(in_bounds((0.075, 0.19, -0.075), wide))
        with self.assertRaises(SystemExit):
            experimental_bounds(0.09)
        with self.assertRaises(SystemExit):
            experimental_bounds(0.08, 0.06, 0.18)
        with self.assertRaises(SystemExit):
            run_tracking(calib=calib, experimental=True, realsense=True,
                         dry=True, duty=0.2, experimental_span=0.09)

    def test_experimental_hardware_path_starts_off_and_closes(self):
        from unittest.mock import patch
        from hand_track import RealSenseSource
        from my_patterns import N_EMITTERS
        from types import SimpleNamespace

        calib = {"version": 2, "mode": "realsense_3d_experimental",
                 "R": np.eye(3).tolist(), "t": [0, 0, 0],
                 "rms_mm": 13.1, "max_mm": 28.6, "target": "index_tip",
                 "serial": "test-camera", "frame_size": [640, 480], "stream": "color"}
        fake_cap = object.__new__(RealSenseSource)
        fake_cap.serial = "test-camera"
        fake_cap.stream_name = "color"
        fake_cap.read = lambda: (True, np.zeros((480, 640, 3), np.uint8))
        fake_cap.release = lambda: None
        fake_cap.point_3d = lambda uv, mask: np.array([0.0, 0.12, 0.0])
        off = np.full(N_EMITTERS, np.nan)
        frames = []

        class FakeBoard:
            def __init__(self, port):
                self.port = port
                self.closed = False

            def off(self):
                frames.append(off.copy())

            def close(self):
                self.closed = True

        class FakeModulator:
            error = None

            def __init__(self, board, pattern, freq, duty):
                self.board = board
                self.duty = duty
                frames.append(np.asarray(pattern).copy())

            def start(self):
                return self

            def set_pattern(self, phases):
                frames.append(np.asarray(phases).copy())

            def stop(self):
                self.board.off()

        with patch("hand_track.Board", FakeBoard), \
                patch("hand_track.Modulator", FakeModulator), \
                patch("hand_track.open_source", return_value=(fake_cap, "realsense")), \
                patch("hand_track.PalmDetector") as detector, \
                patch("hand_track.open_preview_window"), patch("cv2.imshow"), \
                patch("cv2.setMouseCallback"), \
                patch("cv2.waitKey", return_value=ord("q")), \
                patch("cv2.destroyAllWindows"):
            detector.return_value.detect.return_value = SimpleNamespace(
                point=lambda target: (320.0, 240.0),
                mask=np.full((480, 640), 255, np.uint8))
            with patch("hand_track.draw_hand_observation",
                       return_value=np.zeros((480, 640, 3), np.uint8)):
                stats = run_tracking(calib=calib, port="COM_TEST", experimental=True,
                                     realsense=True, duty=0.2, max_seconds=1)
        self.assertEqual(stats["sent"], 1)
        self.assertEqual(stats["gate_counts"]["inside"], 1)
        self.assertTrue(np.isnan(frames[0]).all())
        self.assertTrue(np.isnan(frames[-1]).all())


    def test_calibration_samples_both_heights_before_moving_xy(self):
        from hand_track import calibration_targets
        targets = calibration_targets(3, 0.06, 0.10, 0.05)
        self.assertEqual(len(targets), 18)
        self.assertEqual(targets[0], (-0.06, 0.10, -0.06))
        np.testing.assert_allclose(targets[1], (-0.06, 0.15, -0.06))
        self.assertEqual(targets[2], (0.0, 0.10, -0.06))
        np.testing.assert_allclose(targets[3], (0.0, 0.15, -0.06))
        np.testing.assert_allclose(targets[-2:], [(0.06, 0.10, 0.06),
                                                   (0.06, 0.15, 0.06)])
        for low, high in zip(targets[::2], targets[1::2]):
            self.assertEqual((low[0], low[2]), (high[0], high[2]))
            self.assertAlmostEqual(high[1] - low[1], 0.05)

    def test_rigid_fit_recovers_tilted_camera_and_height(self):
        from hand_track import fit_rigid_transform
        rng = np.random.default_rng(4)
        camera = rng.uniform([-0.05, 0.08, 0.10], [0.05, 0.16, 0.25], (30, 3))
        angle = np.deg2rad(68)
        truth = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])
        offset = np.array([0.02, 0.10, -0.08])
        world = camera @ truth.T + offset
        R, t, errors = fit_rigid_transform(camera, world)
        np.testing.assert_allclose(R, truth, atol=1e-12)
        np.testing.assert_allclose(t, offset, atol=1e-12)
        self.assertLess(max(errors), 1e-12)
        with self.assertRaises(ValueError):
            fit_rigid_transform(camera[:9], np.column_stack((world[:9, 0],
                                                            np.full(9, 0.12), world[:9, 2])))

    def test_mirrored_array_axis_labels_fit_without_false_residual(self):
        from hand_track import fit_rigid_transform, validate_calibration
        rng = np.random.default_rng(13)
        camera = rng.uniform(-0.08, 0.08, (18, 3))
        world = camera @ np.diag([1, -1, 1]) + [0.01, 0.12, -0.02]
        R, t, errors = fit_rigid_transform(camera, world)
        self.assertLess(max(errors), 1e-12)
        self.assertAlmostEqual(np.linalg.det(R), -1)
        calib = {"version": 2, "mode": "realsense_3d", "target": "index_tip",
                 "R": R.tolist(), "t": t.tolist(), "rms_mm": 0, "max_mm": 0}
        validate_calibration(calib, "index_tip")

    def test_depth_holes_and_background_edges(self):
        from hand_track import depth_colormap, fingertip_depth
        depth = np.full((40, 40), 0.15, np.float32)
        mask = np.full((40, 40), 255, np.uint8)
        self.assertAlmostEqual(fingertip_depth(depth, (20, 20), mask), 0.15, places=5)
        depth[16:25, 16:25] = 0
        self.assertIsNone(fingertip_depth(depth, (20, 20), mask))
        depth[16:25, 16:25] = 0.15
        depth[16:25, 16:20] = 0.4
        self.assertAlmostEqual(fingertip_depth(depth, (20, 20), mask),
                               0.15, places=5)
        depth[18, 20] = 0.10
        depth[20, 20] = 0.10
        self.assertAlmostEqual(fingertip_depth(depth, (20, 20), mask),
                               0.15, places=5)
        preview = depth_colormap(np.array([[0, 0.10, 0.30, 0.60]], np.float32))
        self.assertEqual(preview.shape, (1, 4, 3))
        np.testing.assert_array_equal(preview[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(preview[0, 3], [0, 0, 0])
        self.assertFalse(np.array_equal(preview[0, 1], preview[0, 2]))

    def test_height_changes_focus_and_missing_depth_silences(self):
        from unittest.mock import patch
        import hand_track
        emitted = []
        follower = PalmFollower(lambda ph: emitted.append(ph.copy()), deadband_m=0.001,
                                lost_s=0.1)
        with patch.object(hand_track, "pattern_for_kind", return_value=np.zeros(256)) as pattern:
            follower.feed(0.0, (0.0, 0.10, 0.0))
            follower.feed(1 / 30, (0.0, 0.15, 0.0))
        self.assertGreater(pattern.call_count, 1)
        self.assertNotEqual(pattern.call_args_list[0].args[3],
                            pattern.call_args_list[1].args[3])
        follower.feed(0.2, None)
        self.assertTrue(np.isnan(emitted[-1]).all())


class CameraSettingsTests(unittest.TestCase):
    """open_camera / load_cam_settings 的离线检查(FakeCap,不开真摄像头)。"""

    class FakeCap:
        def __init__(self, *args):
            self.args = args                       # (cam) 或 (cam, backend)
            self.props = {}
            self._open = True

        def isOpened(self):
            return self._open

        def set(self, prop, val):
            self.props[prop] = val

        def get(self, prop):
            return self.props.get(prop, 0.0)

        def read(self):
            return True, np.zeros((480, 640, 3), np.uint8)

        def release(self):
            pass

    def test_load_cam_settings_missing_returns_empty(self):
        import tempfile
        from pathlib import Path
        from hand_track import load_cam_settings
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_cam_settings(Path(d) / "nope.json"), {})

    def test_preview_window_is_resizable(self):
        import cv2
        from unittest.mock import patch
        from hand_track import open_preview_window
        with patch("cv2.namedWindow") as named, patch("cv2.resizeWindow") as resize:
            open_preview_window("depth", np.zeros((720, 2560, 3), np.uint8))
        named.assert_called_once_with("depth", cv2.WINDOW_NORMAL)
        resize.assert_called_once_with("depth", 1400, 394)

    @staticmethod
    def _cap_factory(cap_cls):
        """带调用记录的 cv2.VideoCapture 替身工厂。"""
        calls = []

        def factory(*args):
            calls.append(args)
            return cap_cls(*args)

        factory.calls = calls
        return factory

    def test_open_camera_applies_settings(self):
        import cv2
        from unittest.mock import patch
        from hand_track import open_camera
        factory = self._cap_factory(self.FakeCap)
        with patch("cv2.VideoCapture", factory):
            cap, used = open_camera(0, {"width": 1280, "height": 720, "zoom": 100,
                                        "fourcc": "MJPG", "backend": "dshow"})
        self.assertEqual(used, "dshow")
        self.assertEqual(factory.calls, [(0, cv2.CAP_DSHOW)])
        self.assertEqual(cap.props[cv2.CAP_PROP_FOURCC],
                         cv2.VideoWriter_fourcc(*"MJPG"))
        self.assertEqual(cap.props[cv2.CAP_PROP_FRAME_WIDTH], 1280)
        self.assertEqual(cap.props[cv2.CAP_PROP_FRAME_HEIGHT], 720)
        self.assertEqual(cap.props[cv2.CAP_PROP_ZOOM], 100)
        self.assertNotIn(cv2.CAP_PROP_EXPOSURE, cap.props)      # 未给的设置不乱设

    def test_open_camera_falls_back_when_backend_fails(self):
        import cv2
        from unittest.mock import patch
        from hand_track import open_camera

        class FailingDShow(self.FakeCap):
            def __init__(self, *args):
                super().__init__(*args)
                if len(args) > 1 and args[1] == cv2.CAP_DSHOW:
                    self._open = False             # DShow 打不开

        factory = self._cap_factory(FailingDShow)
        with patch("cv2.VideoCapture", factory), patch("time.sleep"):
            cap, used = open_camera(0, {"backend": "dshow", "width": 1280})
        self.assertEqual(used, "msmf")
        self.assertEqual(factory.calls,
                         [(0, cv2.CAP_DSHOW)] * 3 + [(0, cv2.CAP_MSMF)])
        # DShow 三档补救设置全败后换 MSMF,首档即成功
        self.assertTrue(cap.isOpened())
        self.assertEqual(cap.props[cv2.CAP_PROP_FRAME_WIDTH], 1280)

    def test_open_camera_raises_when_never_opens(self):
        from unittest.mock import patch
        from hand_track import open_camera

        class NeverOpens(self.FakeCap):
            def isOpened(self):
                return False

        with patch("cv2.VideoCapture", NeverOpens), patch("time.sleep"):
            with self.assertRaises(SystemExit):
                open_camera(0)


    def test_list_cams_writes_snapshots_per_open_index(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from hand_track import list_cams
        _C = CameraSettingsTests

        # 索引 0、2 能开,其余打不开;两个后端各扫一遍
        class OnlySome(_C.FakeCap):
            def __init__(self, *args):
                super().__init__(*args)
                self._open = args[0] in (0, 2)

        factory = _C._cap_factory(OnlySome)
        with tempfile.TemporaryDirectory() as d, \
                patch("cv2.VideoCapture", factory), patch("time.sleep"):
            found = list_cams(max_index=3, snap_dir=Path(d))
            self.assertEqual(found, 4)              # 2 索引 × 2 后端
            for backend in ("dshow", "msmf"):
                for i in (0, 2):
                    self.assertTrue((Path(d) / f"cam_{backend}_{i}.png").exists())
                for i in (1, 3):
                    self.assertFalse((Path(d) / f"cam_{backend}_{i}.png").exists())


    def test_open_source_routes_realsense(self):
        from unittest.mock import patch
        import hand_track
        fake = object()
        with patch.object(hand_track, "RealSenseSource", return_value=fake) as source:
            src, name = hand_track.open_source(0, {
                "backend": "realsense", "rs_rotation": "ccw"})
            self.assertIs(src, fake)                 # JSON 里记录过则自动走 SDK
            self.assertEqual(name, "realsense")
            source.assert_called_with(depth=False, rotation="ccw")
            src2, name2 = hand_track.open_source(0, {}, realsense=True)
            self.assertIs(src2, fake)
            self.assertEqual(name2, "realsense")

    def test_realsense_manual_exposure_disables_auto_and_is_restored(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import hand_track

        values = {"auto": 1.0, "exposure": 5000.0}
        calls = []

        class Sensor:
            def get_option(self, option):
                return values[option]

            def set_option(self, option, value):
                calls.append((option, value))
                values[option] = value

        sensor = Sensor()
        source = object.__new__(hand_track.RealSenseSource)
        source.stream_name = "infrared"
        source._rs = SimpleNamespace(option=SimpleNamespace(
            enable_auto_exposure="auto", exposure="exposure"))
        source.profile = SimpleNamespace(get_device=lambda: SimpleNamespace(
            first_depth_sensor=lambda: sensor))
        with patch.object(hand_track, "RealSenseSource", return_value=source):
            opened, backend = hand_track.open_source(0, {
                "backend": "realsense", "rs_auto_exposure": False,
                "rs_exposure_us": 6000.0})
        self.assertIs(opened, source)
        self.assertEqual(backend, "realsense")
        self.assertEqual(source.get_auto_exposure(), False)
        self.assertEqual(source.get_exposure(), 6000.0)
        self.assertEqual(calls[-2:], [("auto", 0.0), ("exposure", 6000.0)])

    def test_realsense_uses_bgr8_and_releases_pipeline(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from hand_track import RealSenseSource

        calls = []

        class Pipeline:
            def start(self, config):
                calls.append("start")
                return object()

            def wait_for_frames(self, timeout_ms):
                return SimpleNamespace(get_color_frame=lambda: object())

            def stop(self):
                calls.append("stop")

        class Config:
            def enable_stream(self, *args):
                calls.append(args)

        profile = SimpleNamespace(
            stream_type=lambda: "color", stream_index=lambda: 0,
            format=lambda: "bgr8", fps=lambda: 30,
            as_video_stream_profile=lambda: SimpleNamespace(
                width=lambda: 1280, height=lambda: 720))
        device = SimpleNamespace(query_sensors=lambda: [SimpleNamespace(
            get_stream_profiles=lambda: [profile])], get_info=lambda _: "test-camera")
        fake_rs = SimpleNamespace(
            pipeline=Pipeline, config=Config,
            context=lambda: SimpleNamespace(query_devices=lambda: [device]),
            camera_info=SimpleNamespace(serial_number="serial"),
            stream=SimpleNamespace(color="color", infrared="infrared"),
            format=SimpleNamespace(bgr8="bgr8", y8="y8"))
        with patch.dict(sys.modules, {"pyrealsense2": fake_rs}):
            source = RealSenseSource()
            source.release()
        self.assertEqual(calls[0], ("color", 1280, 720, "bgr8", 30))
        self.assertEqual(calls[-1], "stop")

    def test_realsense_without_rgb_uses_infrared(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from hand_track import RealSenseSource

        calls = []
        ir_frame = SimpleNamespace(get_data=lambda: np.full((480, 640), 42, np.uint8))
        frames = SimpleNamespace(get_infrared_frame=lambda index: ir_frame)

        class Pipeline:
            def start(self, config):
                return object()

            def wait_for_frames(self, timeout_ms):
                return frames

            def stop(self):
                pass

        class Config:
            def enable_stream(self, *args):
                calls.append(args)

        profile = SimpleNamespace(
            stream_type=lambda: "infrared", stream_index=lambda: 1,
            format=lambda: "y8", fps=lambda: 30,
            as_video_stream_profile=lambda: SimpleNamespace(
                width=lambda: 640, height=lambda: 480))
        device = SimpleNamespace(query_sensors=lambda: [SimpleNamespace(
            get_stream_profiles=lambda: [profile])], get_info=lambda _: "test-camera")
        fake_rs = SimpleNamespace(
            pipeline=Pipeline, config=Config,
            context=lambda: SimpleNamespace(query_devices=lambda: [device]),
            camera_info=SimpleNamespace(serial_number="serial"),
            stream=SimpleNamespace(color="color", infrared="infrared"),
            format=SimpleNamespace(bgr8="bgr8", y8="y8"))
        with patch.dict(sys.modules, {"pyrealsense2": fake_rs}):
            source = RealSenseSource()
            ok, image = source.read()
            source.release()
        self.assertEqual(calls[0], ("infrared", 1, 640, 480, "y8", 30))
        self.assertTrue(ok)
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(int(image.mean()), 42)


if __name__ == "__main__":
    unittest.main()
