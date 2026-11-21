"""Offline checks for locating the hidden end of a side-view receiver."""

import unittest

import numpy as np

from receiver_side_geometry import estimate_side_receiver


class SyntheticCamera:
    width = 200
    height = 200
    rotation = "none"

    def __init__(self, radius=0.015, length=0.04, visible=True):
        self.intrinsics = object()
        self._rs = self
        self.depth_m = np.full((200, 200), 0.30, float)
        if not visible:
            return
        for v in range(60, 141):
            for u in range(70, 131):
                a = (u - 100) / 400
                radicand = radius ** 2 * (1 + a ** 2) - a ** 2 * 0.20 ** 2
                if radicand <= 0:
                    continue
                distance = (0.20 - np.sqrt(radicand)) / (1 + a ** 2)
                world_y = 0.20 - (v - 100) * distance / 400
                if abs(world_y - 0.20) <= length / 2:
                    self.depth_m[v, u] = distance

    def rs2_deproject_pixel_to_point(self, _intrinsics, uv, distance):
        return [(uv[0] - 100) * distance / 400,
                (uv[1] - 100) * distance / 400, distance]


class ReceiverSideGeometryTests(unittest.TestCase):
    R = np.diag([1.0, -1.0, 1.0])
    t = np.array([0.0, 0.20, -0.20])
    box = (69, 59, 63, 83)

    def test_hidden_face_from_side_depth(self):
        camera = SyntheticCamera()
        target, reason = estimate_side_receiver(camera, self.box,
                                                self.R, self.t, .04, .03)
        self.assertEqual(reason, "side-cylinder depth")
        np.testing.assert_allclose(target, [0, .18, 0], atol=.003)

    def test_measured_12_by_15_mm_can_at_working_distance(self):
        camera = SyntheticCamera(radius=.0075, length=.012)
        rows, cols = np.where(camera.depth_m < .25)
        box = (int(cols.min()) - 2, int(rows.min()) - 2,
               int(np.ptp(cols)) + 5, int(np.ptp(rows)) + 5)
        target, reason = estimate_side_receiver(camera, box,
                                                self.R, self.t, .012, .015)
        self.assertEqual(reason, "side-cylinder depth")
        np.testing.assert_allclose(target, [0, .194, 0], atol=.002)

    def test_background_does_not_look_like_receiver(self):
        camera = SyntheticCamera(visible=False)
        target, reason = estimate_side_receiver(camera, self.box,
                                                self.R, self.t, .04, .03)
        self.assertIsNone(target)
        self.assertIn("depth", reason)

    def test_flat_foreground_cannot_mimic_can_side(self):
        camera = SyntheticCamera(visible=False)
        camera.depth_m[86:115, 83:118] = .185
        target, reason = estimate_side_receiver(camera, (82, 85, 37, 31),
                                                self.R, self.t, .012, .015)
        self.assertIsNone(target)
        self.assertIn("flat", reason)

    def test_tilted_planar_foreground_cannot_mimic_can_side(self):
        camera = SyntheticCamera(visible=False)
        rows, cols = np.mgrid[86:115, 83:118]
        camera.depth_m[86:115, 83:118] = .185 + .00008 * (cols - 100)
        target, _ = estimate_side_receiver(camera, (82, 85, 37, 31),
                                           self.R, self.t, .012, .015)
        self.assertIsNone(target)

    def test_wrong_dimensions_fail_closed(self):
        camera = SyntheticCamera()
        for length, diameter in ((.08, .03), (.04, .05)):
            with self.subTest(length=length, diameter=diameter):
                target, _ = estimate_side_receiver(camera, self.box,
                                                   self.R, self.t,
                                                   length, diameter)
                self.assertIsNone(target)

    def test_depth_holes_fail_closed(self):
        camera = SyntheticCamera()
        camera.depth_m[59:142, 69:132] = 0
        target, _ = estimate_side_receiver(camera, self.box,
                                           self.R, self.t, .04, .03)
        self.assertIsNone(target)


if __name__ == "__main__":
    unittest.main()
