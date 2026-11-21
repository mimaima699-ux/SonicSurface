"""Offline checks for hidden receiving-face localization and focus following."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from hand_track import BOARD_CALIB_PATH, in_bounds
from receiver_probe import (FOCUS_BOUNDS, FaceTracker, ReceiverFollower,
                            _select_image_box, inspect_image, main, run)


class ReceiverProbeTests(unittest.TestCase):
    def test_image_selection_q_and_window_close_exit(self):
        frame = np.zeros((40, 60, 3), np.uint8)
        for closed, key in ((False, ord("q")), (True, -1)):
            with self.subTest(closed=closed):
                with patch("cv2.namedWindow"), patch("cv2.setMouseCallback"), \
                        patch("cv2.imshow"), patch("cv2.destroyWindow"), \
                        patch("receiver_probe._window_closed", return_value=closed), \
                        patch("cv2.waitKey", return_value=key):
                    self.assertIsNone(_select_image_box(frame))

    def test_cancelled_image_selection_does_not_try_to_track(self):
        with patch("cv2.imread", return_value=np.zeros((40, 60, 3), np.uint8)), \
                patch("receiver_probe._select_image_box", return_value=None), \
                patch("receiver_probe.FaceTracker.select") as select:
            self.assertIsNone(inspect_image("photo.jpg"))
        select.assert_not_called()

    def test_cli_rejects_removed_analysis_and_fixed_target(self):
        for option in (("--analyze", "missing.csv"),
                       ("--target", "0", ".12", "0")):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                main(list(option))

    def test_default_dry_and_port_are_exclusive(self):
        args = SimpleNamespace(port="COM_FAKE", dry=True)
        with self.assertRaisesRegex(SystemExit, "互斥|不能同时"):
            run(args)

    def test_default_calibration_and_measured_can_dimensions(self):
        with patch("receiver_probe.run") as live:
            main([])
        options = live.call_args.args[0]
        self.assertEqual(options.calib, BOARD_CALIB_PATH)
        self.assertEqual(options.length_mm, 12.0)
        self.assertEqual(options.diameter_mm, 15.0)

    def test_cli_rejects_invalid_can_dimensions(self):
        for option in (("--length-mm", "0"), ("--diameter-mm", "101")):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                main(list(option))

    def test_cli_accepts_measured_can_dimensions(self):
        with patch("receiver_probe.run") as live:
            main(["--length-mm", "18", "--diameter-mm", "22"])
        options = live.call_args.args[0]
        self.assertEqual(options.length_mm, 18.0)
        self.assertEqual(options.diameter_mm, 22.0)

    def test_missing_board_calibration_explains_how_to_create_it(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(port=None, dry=True,
                                   calib=Path(root) / "missing.json")
            with self.assertRaisesRegex(SystemExit, "boardcalibrate --realsense"):
                run(args)

    def test_focus_requires_consecutive_valid_frames_and_stops_on_loss(self):
        emitted = []
        controller = ReceiverFollower(lambda ph: emitted.append(np.asarray(ph)))
        for t in (0.0, 0.03):
            self.assertFalse(controller.feed(t, [0, .12, 0]))
        self.assertEqual(emitted, [])
        self.assertTrue(controller.feed(.06, [0, .12, 0]))
        self.assertEqual(len(emitted), 1)
        self.assertFalse(controller.feed(.09, None))
        self.assertTrue(np.isnan(emitted[-1]).all())
        self.assertIsNone(controller.focus_position())

    def test_focus_deadband_and_position_update(self):
        emitted = []
        controller = ReceiverFollower(lambda ph: emitted.append(np.asarray(ph)))
        for i in range(3):
            controller.feed(i * .03, [0, .12, 0])
        controller.feed(.09, [.001, .12, 0])
        self.assertEqual(len(emitted), 1)
        controller.feed(.12, [.02, .12, 0])
        self.assertEqual(len(emitted), 2)
        self.assertFalse(np.allclose(emitted[0], emitted[1]))

    def test_focus_outside_or_jump_silences_immediately(self):
        for invalid in ([.09, .12, 0], [.04, .12, 0]):
            with self.subTest(invalid=invalid):
                emitted = []
                controller = ReceiverFollower(lambda ph: emitted.append(np.asarray(ph)))
                for i in range(3):
                    controller.feed(i * .03, [0, .12, 0])
                self.assertFalse(controller.feed(.09, invalid))
                self.assertTrue(np.isnan(emitted[-1]).all())
                self.assertIsNone(controller.focus_position())

    def test_focus_rearms_after_three_new_fixes(self):
        emitted = []
        controller = ReceiverFollower(lambda ph: emitted.append(np.asarray(ph)))
        for i in range(3):
            controller.feed(i * .03, [0, .12, 0])
        controller.pause(.1)
        self.assertEqual(len(emitted), 2)
        self.assertFalse(controller.feed(.13, [.01, .12, 0]))
        self.assertFalse(controller.feed(.16, [.01, .12, 0]))
        self.assertTrue(controller.feed(.19, [.01, .12, 0]))
        self.assertEqual(len(emitted), 3)

    def test_focus_bounds_match_hand_tracking(self):
        self.assertTrue(in_bounds([.05, .12, -.05], FOCUS_BOUNDS))
        self.assertFalse(in_bounds([.15, .12, 0], FOCUS_BOUNDS))

    def test_template_rejects_unrelated_scene(self):
        rng = np.random.default_rng(4)
        first = rng.integers(0, 255, (160, 200, 3), dtype=np.uint8)
        shifted = np.zeros_like(first)
        shifted[3:, 5:] = first[:-3, :-5]
        tracker = FaceTracker()
        tracker.select(first, (60, 50, 36, 30))
        self.assertEqual(tracker.update(shifted), (65, 53, 36, 30))
        self.assertIsNone(tracker.update(np.zeros_like(first)))

    def test_realtime_no_frame_still_accepts_q(self):
        frame = np.zeros((40, 60, 3), np.uint8)
        cap = SimpleNamespace(serial="test", rotation="none", release=lambda: None)
        with tempfile.TemporaryDirectory() as root:
            calibration = Path(root) / "calibration.json"
            calibration.write_text(
                '{"serial":"test","rotation":"none","frame_size":[60,40]}',
                encoding="utf-8")
            args = SimpleNamespace(calib=calibration, port=None, dry=True,
                                   length_mm=12.0, diameter_mm=15.0)
            with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                    patch("hand_track.read_frame", side_effect=[
                        (True, frame), (True, frame), (False, None)]), \
                    patch("receiver_probe.live_board_pose", return_value=(
                        np.eye(3), np.zeros(3))), \
                    patch("receiver_probe.Board") as board, \
                    patch("receiver_probe._window_closed", return_value=False), \
                    patch("hand_track.open_preview_window"), \
                    patch("cv2.setMouseCallback"), patch("cv2.imshow"), \
                    patch("cv2.waitKey", side_effect=[-1, ord("q")]), \
                    patch("cv2.destroyAllWindows"):
                run(args)
            board.assert_not_called()

    def test_realtime_board_follows_and_silences_on_reselect_depth_and_frame_loss(self):
        frame = np.random.default_rng(8).integers(0, 255, (80, 100, 3),
                                                  dtype=np.uint8)
        cap = SimpleNamespace(serial="test", rotation="none", release=lambda: None)
        events = []

        class FakeBoard:
            def __init__(self, port):
                events.append(("open", port))

            def send(self, phases):
                phases = np.asarray(phases)
                events.append(("off_pattern" if np.isnan(phases).all() else "focus",
                               phases.copy()))

            def off(self):
                events.append(("off", None))

            def close(self):
                events.append(("close", None))

        mouse_callback = None
        frame_count = 0

        def set_mouse_callback(_window, callback):
            nonlocal mouse_callback
            mouse_callback = callback

        def wait_key(_delay):
            nonlocal frame_count
            frame_count += 1
            if frame_count in (1, 4):
                mouse_callback(cv2.EVENT_LBUTTONDOWN, 30, 20, 0, None)
                mouse_callback(cv2.EVENT_LBUTTONUP, 60, 44, 0, None)
            return ord("q") if frame_count == 13 else -1

        with tempfile.TemporaryDirectory() as root:
            calibration = Path(root) / "calibration.json"
            calibration.write_text(
                '{"serial":"test","rotation":"none","frame_size":[100,80]}',
                encoding="utf-8")
            args = SimpleNamespace(calib=calibration, port="COM_FAKE", dry=False,
                                   length_mm=12.0, diameter_mm=15.0)
            valid = (np.array([0.0, .12, 0.0]), "side_fit")
            depth_results = [valid] * 7 + [(None, "no_depth")] + [valid] * 4
            with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                    patch("hand_track.read_frame", side_effect=(
                        [(True, frame)] * 13 + [(False, None)])), \
                    patch("receiver_probe.live_board_pose", return_value=(
                        np.eye(3), np.zeros(3))), \
                    patch("receiver_probe.estimate_side_receiver",
                          side_effect=depth_results) as estimate, \
                    patch("receiver_probe.Board", FakeBoard), \
                    patch("receiver_probe._window_closed", return_value=False), \
                    patch("hand_track.open_preview_window"), \
                    patch("hand_track.LossSupervisor") as supervisor, \
                    patch("cv2.setMouseCallback", side_effect=set_mouse_callback), \
                    patch("cv2.imshow"), \
                    patch("cv2.waitKey", side_effect=wait_key), \
                    patch("cv2.destroyAllWindows"):
                supervisor.return_value.start.return_value = supervisor.return_value
                supervisor.return_value.error = None
                run(args)
            self.assertTrue(estimate.called)
            first_call = estimate.call_args_list[0].args
            self.assertIs(first_call[0], cap)
            self.assertEqual(first_call[1], (30, 20, 30, 24))
            self.assertAlmostEqual(first_call[4], .012)
            self.assertAlmostEqual(first_call[5], .015)

        sent = [kind for kind, _ in events if kind in ("focus", "off_pattern")]
        self.assertEqual(sent, ["focus", "off_pattern", "focus", "off_pattern",
                                "focus", "off_pattern"])
        self.assertEqual(events[0], ("open", "COM_FAKE"))
        self.assertEqual(events[-2:], [("off", None), ("close", None)])
        self.assertEqual(frame_count, 13)

    def test_board_closes_even_when_pause_cleanup_fails(self):
        frame = np.zeros((40, 60, 3), np.uint8)
        cap = SimpleNamespace(serial="test", rotation="none", release=lambda: None)
        board = SimpleNamespace(send=lambda _phases: None,
                                off=unittest.mock.Mock(),
                                close=unittest.mock.Mock())
        with tempfile.TemporaryDirectory() as root:
            calibration = Path(root) / "calibration.json"
            calibration.write_text(
                '{"serial":"test","rotation":"none","frame_size":[60,40]}',
                encoding="utf-8")
            args = SimpleNamespace(calib=calibration, port="COM_FAKE", dry=False,
                                   length_mm=12.0, diameter_mm=15.0)
            with patch("hand_track.open_source", return_value=(cap, "realsense")), \
                    patch("hand_track.read_frame", return_value=(True, frame)), \
                    patch("receiver_probe.live_board_pose", return_value=(
                        np.eye(3), np.zeros(3))), \
                    patch("receiver_probe.Board", return_value=board), \
                    patch("receiver_probe._window_closed", return_value=False), \
                    patch("hand_track.open_preview_window"), \
                    patch("hand_track.LossSupervisor") as supervisor, \
                    patch("receiver_probe.ReceiverFollower.pause", side_effect=[
                        RuntimeError("original pause error"),
                        RuntimeError("cleanup pause error")]), \
                    patch("cv2.setMouseCallback"), patch("cv2.imshow"), \
                    patch("cv2.destroyAllWindows"):
                supervisor.return_value.start.return_value = supervisor.return_value
                supervisor.return_value.error = None
                with self.assertRaisesRegex(RuntimeError, "original pause error"):
                    run(args)
        self.assertGreaterEqual(board.off.call_count, 2)
        board.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
