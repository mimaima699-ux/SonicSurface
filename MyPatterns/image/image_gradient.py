"""Phase-only image refinement on a fixed array propagation matrix."""

import numpy as np
from scipy.optimize import minimize


def quantize_phases(phases, divisions=32):
    """Return the continuous angles actually represented by the board bins."""
    bins = np.rint(np.mod(phases, 2 * np.pi) * divisions / (2 * np.pi)) % divisions
    return bins * (2 * np.pi / divisions)


def stroke_neighbors(target):
    """Adjacent samples entirely inside bright strokes, excluding image edges."""
    t = np.asarray(target).ravel()
    side = int(np.sqrt(t.size))
    if side * side != t.size:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    bright = (t.reshape(side, side) >= 0.8)
    indices = np.arange(t.size).reshape(side, side)
    horizontal = bright[:, :-1] & bright[:, 1:]
    vertical = bright[:-1, :] & bright[1:, :]
    first = np.concatenate((indices[:, :-1][horizontal], indices[:-1, :][vertical]))
    second = np.concatenate((indices[:, 1:][horizontal], indices[1:, :][vertical]))
    return first, second


def image_metrics(amplitude, target, scale):
    """Metrics on a shared physical amplitude scale, with no per-image renormalizing."""
    a = np.asarray(amplitude).ravel() / scale
    t = np.asarray(target).ravel()
    bright = t >= 0.5
    dark = t <= 0.1
    if not bright.any() or not dark.any():
        raise ValueError("Image needs both bright strokes and a dark background")
    background = a[dark]
    top_count = max(1, int(np.ceil(background.size * 0.01)))
    first, second = stroke_neighbors(t)
    texture = (float(np.sqrt(np.mean((a[first] - a[second]) ** 2)))
               if first.size else 0.0)
    return {
        "stroke_mean": float(a[bright].mean()),
        "stroke_p10": float(np.percentile(a[bright], 10)),
        "stroke_cv": float(a[bright].std() / max(a[bright].mean(), 1e-12)),
        "stroke_texture": texture,
        "background_mean": float(background.mean()),
        "background_top_1pct": float(np.partition(background, -top_count)[-top_count:].mean()),
        "background_max": float(background.max()),
    }


def reference_scale(G, phases, target):
    """Anchor brightness to the original GS result so darkening everything cannot win."""
    a = np.abs(np.exp(1j * phases) @ G)
    bright = np.asarray(target).ravel() >= 0.5
    if not bright.any() or not (np.asarray(target).ravel() <= 0.1).any():
        raise ValueError("Image needs both bright strokes and a dark background")
    return max(float(a[bright].mean()), 1e-12)


def _field_loss(field, target, scale, texture_weight=0.0, neighbors=None):
    """Value of loss_and_gradient for an already computed complex field."""
    t = np.asarray(target).ravel()
    a = np.maximum(np.abs(field), 1e-12) / scale
    bright = t >= 0.5
    dark = t <= 0.1
    if not bright.any() or not dark.any():
        raise ValueError("Image needs both bright strokes and a dark background")
    weights = np.where(bright, 3.0, np.where(dark, 2.5, 1.0))
    weights /= weights.sum()
    error = a - t
    loss = float(np.dot(weights, error * error))
    top_count = max(1, int(np.ceil(dark.sum() * 0.01)))
    top = np.partition(a[dark], -top_count)[-top_count:]
    loss += float(0.25 * np.mean(top ** 2))
    low = np.minimum(a[bright] - 0.9 * t[bright], 0.0)
    loss += float(4.0 * np.dot(low, low) / bright.sum())
    if texture_weight:
        first, second = stroke_neighbors(t) if neighbors is None else neighbors
        if first.size:
            loss += float(texture_weight * np.mean((a[first] - a[second]) ** 2))
    return loss


def refine_quantized_bins(G, target, initial, divisions=32, passes=1,
                          texture_weight=0.0):
    """Greedy +/- one-bin coordinate search on the phases the board will emit."""
    G = np.asarray(G)
    t = np.asarray(target, float).ravel()
    initial = np.asarray(initial, float).ravel()
    if G.shape != (initial.size, t.size) or divisions < 2 or passes < 1:
        raise ValueError("Invalid propagation model or search parameters")
    bins = (np.rint(np.mod(initial, 2 * np.pi) * divisions / (2 * np.pi))
            .astype(int) % divisions)
    roots = np.exp(2j * np.pi * np.arange(divisions) / divisions)
    field = roots[bins] @ G
    bright = t >= 0.5
    if not bright.any() or not (t <= 0.1).any():
        raise ValueError("Image needs both bright strokes and a dark background")
    scale = max(float(np.abs(field)[bright].mean()), 1e-12)
    neighbors = stroke_neighbors(t) if texture_weight else None
    score = lambda value: _field_loss(value, t, scale, texture_weight, neighbors)
    baseline = score(field)
    baseline_metrics = image_metrics(np.abs(field), t, scale)
    current = baseline
    accepted = 0
    for _ in range(passes):
        changed = False
        for channel in range(initial.size):
            best_bin = bins[channel]
            best_field = None
            best_score = current
            for delta in (-1, 1):
                candidate_bin = (bins[channel] + delta) % divisions
                candidate_field = field + (roots[candidate_bin] - roots[bins[channel]]) * G[channel]
                candidate_score = score(candidate_field)
                if candidate_score < best_score - 1e-12:
                    metrics = image_metrics(np.abs(candidate_field), t, scale)
                    if (metrics["stroke_p10"] >= 0.98 * baseline_metrics["stroke_p10"]
                            and metrics["stroke_cv"] <= 1.05 * baseline_metrics["stroke_cv"]
                            and metrics["background_mean"] <= 1.05 * baseline_metrics["background_mean"]):
                        best_bin, best_field, best_score = candidate_bin, candidate_field, candidate_score
            if best_field is not None:
                bins[channel], field, current = best_bin, best_field, best_score
                accepted += 1
                changed = True
        if not changed:
            break
    return bins * (2 * np.pi / divisions), {
        "baseline_score": baseline, "selected_score": current,
        "accepted_moves": accepted, "scale": scale,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": image_metrics(np.abs(field), t, scale),
    }


def loss_and_gradient(phases, G, target, scale, texture_weight=0.0,
                      neighbors=None):
    """Weighted stroke fidelity and background peak penalty, with analytic VJP."""
    t = np.asarray(target).ravel()
    x = np.exp(1j * phases)
    field = x @ G
    magnitude = np.maximum(np.abs(field), 1e-12)
    a = magnitude / scale
    bright = t >= 0.5
    dark = t <= 0.1
    if not bright.any() or not dark.any():
        raise ValueError("Image needs both bright strokes and a dark background")

    error = a - t
    weights = np.where(bright, 3.0, np.where(dark, 2.5, 1.0))
    weights /= weights.sum()
    loss = float(np.dot(weights, error * error))
    d_a = 2.0 * weights * error

    # Concentrate the sidelobe penalty on the brightest 1% of dark pixels.
    dark_indices = np.flatnonzero(dark)
    top_count = max(1, int(np.ceil(dark_indices.size * 0.01)))
    top = dark_indices[np.argpartition(a[dark_indices], -top_count)[-top_count:]]
    loss += float(0.25 * np.mean(a[top] ** 2))
    d_a[top] += 0.5 * a[top] / top_count

    # Keep thin strokes from disappearing to reduce the background score.
    weak = bright & (a < 0.9 * t)
    low = a[weak] - 0.9 * t[weak]
    loss += float(4.0 * np.dot(low, low) / bright.sum())
    d_a[weak] += 8.0 * low / bright.sum()

    if texture_weight:
        first, second = stroke_neighbors(t) if neighbors is None else neighbors
        if first.size:
            delta = a[first] - a[second]
            loss += float(texture_weight * np.mean(delta ** 2))
            contribution = 2.0 * texture_weight * delta / first.size
            np.add.at(d_a, first, contribution)
            np.add.at(d_a, second, -contribution)

    v = d_a * np.conj(field) / (magnitude * scale)
    gradient = -np.imag(x * (G @ v))
    return loss, gradient


def refine_image_phases(G, target, initial, steps=100, divisions=32,
                        texture_weight=0.0):
    """Refine GS phases; only return a candidate if its quantized score improves."""
    scale = reference_scale(G, initial, target)
    neighbors = stroke_neighbors(target) if texture_weight else None
    objective = lambda phases: loss_and_gradient(
        phases, G, target, scale, texture_weight, neighbors)
    checkpoints = []

    def checkpoint(phases):
        if len(checkpoints) < 12:
            checkpoints.append(np.asarray(phases).copy())

    result = minimize(objective, np.asarray(initial, dtype=float), jac=True,
                      method="L-BFGS-B", callback=checkpoint,
                      options={"maxiter": steps})

    def score(phases):
        return objective(quantize_phases(phases, divisions))[0]

    baseline_score = score(initial)
    refined_score = score(result.x)
    chosen = result.x if refined_score < baseline_score else initial
    return np.mod(chosen, 2 * np.pi), {
        "baseline_score": baseline_score,
        "refined_score": refined_score,
        "selected_score": min(baseline_score, refined_score),
        "scale": scale,
        "iterations": int(result.nit),
        "used_refinement": bool(refined_score < baseline_score),
        "candidates": checkpoints + [result.x.copy()],
    }
