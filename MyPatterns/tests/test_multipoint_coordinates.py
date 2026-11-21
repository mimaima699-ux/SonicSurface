"""Regression checks for the user-facing X/Z multipoint commands."""

import unittest
from unittest.mock import patch

import numpy as np

import my_patterns


class MultipointCoordinateTests(unittest.TestCase):
    def test_points_command_places_height_on_y_axis(self):
        captured = []

        def record(points):
            captured.append(np.asarray(points).copy())
            return np.zeros(my_patterns.N_EMITTERS)

        with patch.object(my_patterns, "phases_multipoint", side_effect=record):
            my_patterns.main(["points", "0.03", "0.02", "-0.03", "-0.02",
                              "--dist", "0.12", "--dry"])
        np.testing.assert_allclose(captured[0],
                                   [[0.03, 0.12, 0.02], [-0.03, 0.12, -0.02]])

    def test_mpx_frames_place_height_on_y_axis(self):
        captured = []

        def record(points):
            captured.append(np.asarray(points).copy())
            return np.zeros(my_patterns.N_EMITTERS)

        with patch.object(my_patterns, "phases_multipoint", side_effect=record):
            frames = my_patterns.frames_alternate([(0.03, 0.02)],
                                                 [(-0.03, -0.02)], dist=0.12)
            next(frames)
        np.testing.assert_allclose(captured[0], [[0.03, 0.12, 0.02]])
        np.testing.assert_allclose(captured[1], [[-0.03, 0.12, -0.02]])


if __name__ == "__main__":
    unittest.main()
