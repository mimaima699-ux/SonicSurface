"""Offline checks for the image objective and 32-bin selection."""

import unittest
from unittest.mock import patch

import numpy as np

from image.image_gradient import (_field_loss, loss_and_gradient, quantize_phases,
                            refine_image_phases, refine_quantized_bins)
from image.image_direct import backprop_phases, optimize_phases


class ImageGradientTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.G = rng.normal(size=(6, 12)) + 1j * rng.normal(size=(6, 12))
        self.target = np.array([1] * 5 + [0] * 7, dtype=float)
        self.phases = rng.uniform(0, 2 * np.pi, size=6)

    def test_analytic_gradient_matches_finite_difference(self):
        _, gradient = loss_and_gradient(self.phases, self.G, self.target, 3.0)
        h = 1e-6
        for i in range(self.phases.size):
            delta = np.zeros_like(self.phases)
            delta[i] = h
            plus = loss_and_gradient(self.phases + delta, self.G, self.target, 3.0)[0]
            minus = loss_and_gradient(self.phases - delta, self.G, self.target, 3.0)[0]
            self.assertAlmostEqual(gradient[i], (plus - minus) / (2 * h), places=5)

    def test_texture_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(8)
        G = rng.normal(size=(6, 25)) + 1j * rng.normal(size=(6, 25))
        target = np.zeros((5, 5))
        target[1:4, 1:4] = 1
        phases = rng.uniform(0, 2 * np.pi, size=6)
        _, gradient = loss_and_gradient(phases, G, target, 3.0,
                                         texture_weight=0.8)
        h = 1e-6
        for i in range(phases.size):
            delta = np.zeros_like(phases)
            delta[i] = h
            plus = loss_and_gradient(phases + delta, G, target, 3.0,
                                     texture_weight=0.8)[0]
            minus = loss_and_gradient(phases - delta, G, target, 3.0,
                                      texture_weight=0.8)[0]
            self.assertAlmostEqual(gradient[i], (plus - minus) / (2 * h), places=5)

    def test_refinement_never_selects_a_worse_quantized_score(self):
        phases, info = refine_image_phases(self.G, self.target, self.phases, steps=20)
        self.assertLessEqual(info["selected_score"], info["baseline_score"] + 1e-12)
        self.assertEqual(phases.shape, self.phases.shape)
        self.assertTrue(np.all(np.isfinite(quantize_phases(phases))))

    def test_incremental_loss_matches_full_objective(self):
        field = np.exp(1j * self.phases) @ self.G
        for texture_weight in (0.0, 0.8):
            expected = loss_and_gradient(self.phases, self.G, self.target, 3.0,
                                         texture_weight=texture_weight)[0]
            actual = _field_loss(field, self.target, 3.0,
                                 texture_weight=texture_weight)
            self.assertAlmostEqual(actual, expected, places=12)

    def test_quantized_coordinate_search_improves_board_score(self):
        phases, info = refine_quantized_bins(self.G, self.target, self.phases)
        self.assertLessEqual(info["selected_score"], info["baseline_score"])
        self.assertGreaterEqual(
            info["selected_metrics"]["stroke_p10"],
            0.98 * info["baseline_metrics"]["stroke_p10"])
        np.testing.assert_allclose(phases, quantize_phases(phases), atol=1e-12)
        baseline = quantize_phases(self.phases)
        scale = info["scale"]
        self.assertAlmostEqual(
            loss_and_gradient(phases, self.G, self.target, scale)[0],
            info["selected_score"], places=12)
        self.assertAlmostEqual(
            loss_and_gradient(baseline, self.G, self.target, scale)[0],
            info["baseline_score"], places=12)

    def test_image_quant_refine_exposes_metrics_and_quantized_phases(self):
        from pathlib import Path
        from my_patterns import phases_from_image

        phases, info = phases_from_image(
            Path(__file__).resolve().parents[1] / "assets/demo_F.png", iters=10,
            quant_refine=True)
        self.assertLessEqual(info["quant_refinement"]["selected_score"],
                             info["quant_refinement"]["baseline_score"])
        np.testing.assert_allclose(phases, quantize_phases(phases), atol=1e-12)

    def test_direct_optimization_starts_from_backprop_without_gs(self):
        initial = backprop_phases(self.G, self.target)
        self.assertEqual(initial.shape, self.phases.shape)
        self.assertTrue(np.all(np.isfinite(initial)))
        optimized, info = optimize_phases(self.G, self.target, initial, steps=20)
        self.assertLessEqual(info["selected_score"], info["initial_score"] + 1e-12)
        self.assertTrue(np.all(np.isfinite(quantize_phases(optimized))))

    def test_direct_image_branch_does_not_call_gs(self):
        from pathlib import Path
        from my_patterns import phases_from_image

        image = Path(__file__).resolve().parents[1] / "assets/demo_F.png"
        with patch("my_patterns._gs_array_phases", side_effect=AssertionError("GS called")):
            phases, _ = phases_from_image(image, method="direct")
        self.assertEqual(phases.shape, (256,))


if __name__ == "__main__":
    unittest.main()
