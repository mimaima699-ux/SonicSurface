"""Image phase synthesis without Gerchberg-Saxton iterations."""

import numpy as np
from scipy.optimize import minimize

from .image_gradient import loss_and_gradient, quantize_phases, reference_scale


def backprop_phases(G, target):
    """One adjoint propagation of a real target field, projected to phase only."""
    field = np.conj(G @ np.asarray(target, dtype=float).ravel())
    return np.mod(np.angle(field), 2 * np.pi)


def optimize_phases(G, target, initial, steps=100):
    """Directly minimize image error in the 256 phase variables, starting from backprop."""
    scale = reference_scale(G, initial, target)
    objective = lambda phases: loss_and_gradient(
        phases, G, target, scale, texture_weight=4.0)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": steps})

    def score(phases):
        return objective(quantize_phases(phases))[0]

    initial_score = score(initial)
    optimized_score = score(result.x)
    chosen = result.x if optimized_score < initial_score else initial
    return np.mod(chosen, 2 * np.pi), {
        "iterations": int(result.nit),
        "initial_score": initial_score,
        "optimized_score": optimized_score,
        "selected_score": min(initial_score, optimized_score),
        "used_optimization": bool(optimized_score < initial_score),
    }
