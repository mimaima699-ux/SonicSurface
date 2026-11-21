"""Checks for the model-only three-dimensional field overlay."""

import unittest

import numpy as np

import my_patterns
from diagnostic_field_3d import FieldSlices


class FieldSlicesTests(unittest.TestCase):
    def setUp(self):
        self.model = FieldSlices(lateral_step=0.02, height_step=0.025)
        self.focus = np.array([0.025, 0.12, -0.015])

    def test_three_planes_intersect_at_requested_focus(self):
        frame = self.model.sample(my_patterns.phases_focus(self.focus), self.focus)
        self.assertTrue(frame.active)
        self.assertEqual(set(frame.planes), {"xz", "xy", "zy"})
        self.assertTrue(np.all(frame.planes["xz"].xyz[:, :, 1] == self.focus[1]))
        self.assertTrue(np.all(frame.planes["xy"].xyz[:, :, 2] == self.focus[2]))
        self.assertTrue(np.all(frame.planes["zy"].xyz[:, :, 0] == self.focus[0]))
        for plane in frame.planes.values():
            self.assertEqual(plane.xyz.shape[:2], plane.amplitude.shape)
            self.assertTrue(np.isfinite(plane.amplitude).all())
            self.assertGreaterEqual(plane.amplitude.min(), 0)
        self.assertGreater(frame.target_pressure, 0)
        self.assertGreater(frame.max_pressure, 0)

    def test_board_quantization_matches_direct_model(self):
        phases = my_patterns.phases_focus(self.focus) + 0.012
        frame = self.model.sample(phases, self.focus)
        bins = (np.rint(my_patterns._wrap(phases) * my_patterns.PHASE_DIVS
                       / (2 * np.pi)) % my_patterns.PHASE_DIVS)
        quantized = bins * 2 * np.pi / my_patterns.PHASE_DIVS
        expected = abs(my_patterns.field_at(self.focus[None, :], quantized)[0])
        self.assertAlmostEqual(frame.target_pressure, expected, places=8)

    def test_off_grid_focus_is_considered_for_global_peak(self):
        frame = self.model.sample(my_patterns.phases_focus(self.focus), self.focus)
        sampled_max = max(float(plane.amplitude.max())
                          for plane in frame.planes.values())
        self.assertGreater(frame.target_pressure, sampled_max)
        self.assertAlmostEqual(frame.max_pressure, frame.target_pressure)
        np.testing.assert_array_equal(frame.peak_position, self.focus)

    def test_genuine_stronger_remote_peak_is_preserved(self):
        actual_focus = np.array([.08, .12, 0.0])
        phases = my_patterns.phases_focus(actual_focus)
        frame = self.model.sample(phases, [0.0, .12, 0.0])
        self.assertGreater(frame.max_pressure, frame.target_pressure)
        self.assertGreater(np.linalg.norm(frame.peak_position - frame.focus), .02)
        sampled_max = max(float(plane.amplitude.max())
                          for plane in frame.planes.values())
        self.assertAlmostEqual(frame.max_pressure, sampled_max)

    def test_disabled_emitters_and_off_pattern(self):
        phases = np.full(my_patterns.N_EMITTERS, np.nan)
        off = self.model.sample(phases, None)
        self.assertFalse(off.active)
        self.assertEqual(off.planes, {})
        self.assertIsNone(off.peak_position)
        self.assertEqual(off.max_pressure, 0)

        phases[:12] = 0
        active = self.model.sample(phases, self.focus)
        expected = abs(my_patterns._propagators(self.focus[None, :])[:12, 0].sum())
        self.assertAlmostEqual(active.target_pressure, expected, places=8)
        self.assertTrue(np.isfinite(active.max_pressure))
        self.assertFalse(self.model.sample(np.full_like(phases, np.nan), None).active)

    def test_cache_invalidates_on_phase_focus_and_off_changes(self):
        phases = my_patterns.phases_focus(self.focus)
        first = self.model.sample(phases, self.focus)
        self.assertIs(self.model.sample(phases, self.focus), first)
        moved = self.focus + np.array([.01, 0, 0])
        second = self.model.sample(phases, moved)
        self.assertIsNot(second, first)
        self.assertIsNot(second.planes["zy"].xyz, first.planes["zy"].xyz)
        changed = self.model.sample(phases + .2 * np.arange(len(phases))
                                    / len(phases), moved)
        self.assertIsNot(changed, second)
        self.model.sample(np.full_like(phases, np.nan), None)
        self.assertIsNot(self.model.sample(phases, moved), second)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            self.model.sample(np.zeros(12), self.focus)
        with self.assertRaises(ValueError):
            self.model.sample(np.full(256, np.inf), self.focus)
        with self.assertRaises(ValueError):
            self.model.sample(np.zeros(256), None)


if __name__ == "__main__":
    unittest.main()
