"""Offline checks for layered image synthesis and bounded frame playback."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import my_patterns
from geometry.cup_3d import cup_outline, focus_frames
from geometry.image_3d import focus_depth_contrast, load_layers, optimize_layers, plane_points
from geometry.mesh_3d import mesh_outline


class Image3DTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent
        self.specs = [(root.parent / "assets/demo_F.png", "0.12"),
                      (root.parents[1] / "Algorithms/EmissionPhasesAmpsForTargetSlice/A.png", "0.16")]

    def test_layers_have_distinct_positive_depths_and_bright_pixels(self):
        layers = load_layers(self.specs, resolution=8)
        self.assertEqual(len(layers), 2)
        self.assertEqual(plane_points(0.12, 8).shape, (64, 3))
        self.assertTrue(np.all(plane_points(0.12, 8)[:, 1] == 0.12))
        with self.assertRaises(ValueError):
            load_layers([(self.specs[0][0], "0.12"), (self.specs[1][0], "0.12")])

    def test_static_method_selects_no_worse_quantized_objective(self):
        layers = load_layers(self.specs, resolution=12)
        phases, details = optimize_layers(layers, my_patterns._propagators, steps=12)
        self.assertEqual(phases.shape, (256,))
        self.assertTrue(np.all(np.isfinite(phases)))
        self.assertLessEqual(details["selected_score"], details["initial_score"] + 1e-12)

    def test_single_focus_has_depth_reference(self):
        reference = focus_depth_contrast([0.12, 0.16], my_patterns.phases_focus,
                                         my_patterns._propagators)
        self.assertAlmostEqual(reference[0]["relative_amplitude_at_depths"][0], 1.0)
        self.assertLess(reference[0]["relative_amplitude_at_depths"][1], 0.5)
        self.assertLess(reference[1]["relative_amplitude_at_depths"][0], 0.5)

    def test_sequential_dry_run_uses_2d_gs_without_serial(self):
        args = ["image3d", "--layer", str(self.specs[0][0]), "0.12",
                "--layer", str(self.specs[1][0]), "0.16", "--dry"]
        with patch.object(my_patterns, "phases_from_image",
                          return_value=(np.zeros(256), {})) as synthesis, \
             patch.object(my_patterns, "Board") as board, \
             contextlib.redirect_stdout(io.StringIO()):
            my_patterns.main(args)
        self.assertEqual(synthesis.call_count, 2)
        self.assertTrue(all(call.kwargs["method"] == "array"
                            for call in synthesis.call_args_list))
        board.assert_not_called()

    def test_frame_playback_turns_off_after_write_failure(self):
        class FailingBoard:
            off_called = False

            def send(self, phases):
                raise OSError("write failed")

            def off(self):
                self.off_called = True

        board = FailingBoard()
        with self.assertRaises(OSError):
            my_patterns.play_depth_frames(board, [np.zeros(256)], fps=20, duration=0.1)
        self.assertTrue(board.off_called)

    def test_cup_outline_has_distinct_rim_wall_and_base_depths(self):
        points, labels = cup_outline()
        self.assertEqual(points.shape, (24, 3))
        self.assertEqual({name: labels.count(name) for name in set(labels)},
                         {"rim": 8, "base": 8, "wall": 8})
        self.assertTrue(np.allclose(points[:8, 1], 0.16))
        self.assertTrue(np.allclose(points[8:16, 1], 0.10))
        self.assertTrue(np.all((points[16:, 1] > 0.10) & (points[16:, 1] < 0.16)))
        frames, _ = focus_frames(points, my_patterns.phases_focus)
        self.assertEqual(frames.shape, (24, 256))
        with self.assertRaises(ValueError):
            cup_outline(bottom=0.16, top=0.10)

    def test_cup_dry_run_never_opens_serial(self):
        with patch.object(my_patterns, "Board") as board, \
             contextlib.redirect_stdout(io.StringIO()):
            my_patterns.main(["cup3d", "--dry"])
        board.assert_not_called()

    def test_mesh_import_scales_and_fits_inside_requested_volume(self):
        import pyvista as pv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cube.stl"
            pv.Cube(x_length=2, y_length=2, z_length=2).triangulate().save(path)
            points, labels, info = mesh_outline(path, near=0.10, width=0.06,
                                                height=0.06, count=16)
            self.assertEqual(points.shape, (16, 3))
            self.assertEqual(len(labels), 16)
            self.assertLessEqual(np.abs(points[:, [0, 2]]).max(), 0.03001)
            self.assertGreaterEqual(points[:, 1].min(), 0.10)
            self.assertLessEqual(points[:, 1].max(), 0.16001)
            self.assertEqual(info["selected_points"], 16)
            frames, _ = focus_frames(points, my_patterns.phases_focus)
            self.assertEqual(frames.shape, (16, 256))
            with patch.object(my_patterns, "Board") as board, \
                 contextlib.redirect_stdout(io.StringIO()):
                my_patterns.main(["mesh3d", str(path), "--dry"])
            board.assert_not_called()


if __name__ == "__main__":
    unittest.main()
