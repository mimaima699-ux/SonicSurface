"""Model checks for experimental near-field sidelobe suppression."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import my_patterns
from acoustic_optimization import (predict_sidelobes, save_prediction,
                                   suppress_off_axis_peaks)


class AcousticOptimizationTests(unittest.TestCase):
    def test_axis_focus_reduces_held_out_3d_remote_peak_after_quantization(self):
        target = (0.04, 0.12, 0.0)
        phases, report = suppress_off_axis_peaks(target)
        self.assertTrue(report["improved"])
        self.assertEqual(phases.shape, (256,))
        bins = phases * my_patterns.PHASE_DIVS / (2 * np.pi)
        np.testing.assert_allclose(bins, np.rint(bins), atol=1e-10)
        self.assertLess(report["optimized"]["remote_ratio"],
                        0.8 * report["baseline"]["remote_ratio"])
        self.assertGreater(report["optimized"]["target_pressure"],
                           0.75 * report["baseline"]["target_pressure"])
        peak = np.asarray(report["optimized"]["global_peak_position"])
        self.assertLess(np.linalg.norm(peak - target), 0.006)

    def test_corner_focus_improves_without_moving_global_peak(self):
        target = (0.04, 0.12, 0.04)
        _, report = suppress_off_axis_peaks(target)
        self.assertTrue(report["improved"])
        self.assertLess(report["optimized"]["remote_ratio"],
                        0.9 * report["baseline"]["remote_ratio"])
        peak = np.asarray(report["optimized"]["global_peak_position"])
        self.assertLess(np.linalg.norm(peak - target), 0.006)

    def test_center_keeps_normal_focus_and_invalid_height_rejected(self):
        phases, report = suppress_off_axis_peaks((0.0, 0.12, 0.0))
        self.assertFalse(report["improved"])
        expected = (np.rint(my_patterns.phases_focus((0, 0.12, 0))
                            * my_patterns.PHASE_DIVS / (2 * np.pi))
                    % my_patterns.PHASE_DIVS)
        np.testing.assert_allclose(
            phases * my_patterns.PHASE_DIVS / (2 * np.pi), expected)
        with self.assertRaises(ValueError):
            suppress_off_axis_peaks((0, 0.04, 0))

    def test_corner_predicts_multiple_distinct_off_axis_peaks(self):
        target = (0.04, 0.12, 0.04)
        report = predict_sidelobes(target, half=0.15, step=0.003, count=4)
        self.assertFalse(report["measurement"])
        self.assertEqual(len(report["peaks"]), 4)
        self.assertGreater(report["peaks"][0]["relative_pressure"], 0.40)
        self.assertGreater(report["peaks"][1]["relative_pressure"], 0.40)
        self.assertLess(abs(report["peaks"][0]["position_m"][0] + 0.08), 0.012)
        self.assertLess(abs(report["peaks"][1]["position_m"][2] + 0.08), 0.012)
        self.assertAlmostEqual(report["peaks"][0]["position_m"][1], target[1])

    def test_receiver_scan_csv_has_target_and_model_peaks(self):
        report = predict_sidelobes((0.04, 0.12, 0), step=0.005, count=2)
        with TemporaryDirectory() as tmp:
            path = save_prediction(report, Path(tmp) / "scan.csv")
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("model_relative_pressure", lines[0])
        self.assertTrue(lines[1].startswith("target,"))
        self.assertTrue(lines[2].startswith("side_peak_1,"))
