"""Experimental phase-only suppression of near-field off-axis peaks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

import my_patterns as patterns


def _quantized(phases):
    bins = np.rint(patterns._wrap(phases) * patterns.PHASE_DIVS /
                   (2.0 * np.pi)) % patterns.PHASE_DIVS
    return bins * (2.0 * np.pi / patterns.PHASE_DIVS)


def _evaluation_grid(height, half=0.09, step=0.004,
                     height_offsets=(-0.015, 0.0, 0.015)):
    axis = np.arange(-half, half + step / 2, step)
    x, z = np.meshgrid(axis, axis)
    return np.vstack([np.column_stack(
        (x.ravel(), np.full(x.size, height + offset), z.ravel()))
        for offset in height_offsets if height + offset >= 0.05])


def _metrics(pos, phases, grid, propagators, exclusion=0.02):
    field = np.abs(propagators.T @ np.exp(1j * phases))
    target = abs(patterns.field_at(np.asarray(pos)[None, :], phases)[0])
    remote = np.hypot(grid[:, 0] - pos[0], grid[:, 2] - pos[2]) > exclusion
    index = np.flatnonzero(remote)[np.argmax(field[remote])]
    return {"target_pressure": float(target),
            "remote_pressure": float(field[index]),
            "remote_ratio": float(field[index] / target),
            "remote_position": grid[index].tolist(),
            "global_peak_position": grid[int(np.argmax(field))].tolist()}


def _peak_positions(pos, grid, amplitude, count=5):
    remote = np.hypot(grid[:, 0] - pos[0], grid[:, 2] - pos[2]) > 0.02
    ranked = np.argsort(np.where(remote, amplitude, 0.0))[::-1]
    selected = []
    for index in ranked:
        point = grid[index]
        if all(np.linalg.norm(point - previous) >= 0.014
               for previous in selected):
            selected.append(point)
        if len(selected) == count:
            break
    return selected


def _solve(pos, null_points, start, strength, maxiter):
    points = np.vstack((pos, null_points))
    G = patterns._propagators(points)
    reference = abs(G[:, 0] @ np.exp(1j * patterns.phases_focus(pos)))
    scale = reference ** 2

    def objective(phi):
        emit = np.exp(1j * phi)
        field = G.T @ emit
        alias_power = np.abs(field[1:]) ** 2 / scale
        beta = 30.0
        shifted = beta * (alias_power - alias_power.max())
        weights = np.exp(shifted)
        weights /= weights.sum()
        alias_cost = (alias_power.max() + np.log(np.exp(shifted).mean()) / beta)
        target_power = abs(field[0]) ** 2 / scale
        cost = -target_power + strength * alias_cost
        coefficients = np.concatenate(([-1.0], strength * weights))
        gradient = 2.0 * np.real(
            1j * emit * (G @ (coefficients * np.conj(field)))) / scale
        return cost, gradient

    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-10})
    return patterns._wrap(result.x)


def suppress_off_axis_peaks(pos, min_target_fraction=0.75, max_rounds=3,
                            maxiter=120):
    """Iteratively null predicted hotspots; return quantized phases and report.

    Only use after comparing the full quantized plane to a normal focus.
    This is a model experiment, not a measurement of the physical array.
    """
    pos = np.asarray(pos, float)
    if pos.shape != (3,) or not np.isfinite(pos).all() or not 0.07 <= pos[1] <= 0.20:
        raise ValueError("focus must be a finite (x,y,z) in the supported height range")
    grid = _evaluation_grid(pos[1])
    G_eval = patterns._propagators(grid)
    baseline = _quantized(patterns.phases_focus(pos))
    reference = _metrics(pos, baseline, grid, G_eval)
    if reference["remote_ratio"] < 0.25:
        return baseline, {"baseline": reference, "optimized": reference,
                          "improved": False, "null_points": 0}
    chosen = baseline
    best = reference
    nulls = []
    for _ in range(max_rounds):
        amplitude = np.abs(G_eval.T @ np.exp(1j * chosen))
        for peak in _peak_positions(pos, grid, amplitude):
            if all(np.linalg.norm(peak - previous) >= 0.007 for previous in nulls):
                for dx, dz in ((0, 0), (-0.004, 0), (0.004, 0),
                               (0, -0.004), (0, 0.004)):
                    sample = peak + (dx, 0, dz)
                    if max(abs(sample[0]), abs(sample[2])) <= 0.09:
                        nulls.append(sample)
        if not nulls:
            break
        for strength in (3.0, 8.0, 20.0):
            candidate = _quantized(_solve(pos, nulls, chosen, strength, maxiter))
            metrics = _metrics(pos, candidate, grid, G_eval)
            if (metrics["target_pressure"] >=
                    min_target_fraction * reference["target_pressure"]
                    and metrics["remote_pressure"] < reference["remote_pressure"]
                    and metrics["remote_ratio"] < best["remote_ratio"]):
                chosen, best = candidate, metrics
    # A denser, wider volume catches peaks displaced beyond the training grid.
    check_grid = _evaluation_grid(
        pos[1], half=0.12, step=0.0035,
        height_offsets=(-0.02, -0.01, 0, 0.01, 0.02))
    check_G = patterns._propagators(check_grid)
    checked_reference = _metrics(pos, baseline, check_grid, check_G)
    checked_best = _metrics(pos, chosen, check_grid, check_G)
    peak = np.asarray(checked_best["global_peak_position"])
    peak_stays_on_target = (np.hypot(peak[0] - pos[0], peak[2] - pos[2])
                            <= 0.008 and abs(peak[1] - pos[1]) <= 0.015)
    improved = (best is not reference and peak_stays_on_target
                and checked_best["target_pressure"] >=
                min_target_fraction * checked_reference["target_pressure"]
                and checked_best["remote_pressure"] <
                checked_reference["remote_pressure"]
                and checked_best["remote_ratio"] <
                0.9 * checked_reference["remote_ratio"])
    if not improved:
        chosen, checked_best = baseline, checked_reference
    return chosen, {"baseline": checked_reference, "optimized": checked_best,
                    "improved": improved, "null_points": len(nulls)}


def predict_sidelobes(pos, phases=None, half=0.15, step=0.002,
                      count=6, exclusion=0.02, separation=0.025):
    """Predict distinct pressure peaks on the target-height x-z plane.

    The result uses the ideal piston model and quantized board phases. It is
    useful for choosing receiver measurement positions, not a measured field.
    """
    pos = np.asarray(pos, float)
    if pos.shape != (3,) or not np.isfinite(pos).all() or not 0.07 <= pos[1] <= 0.20:
        raise ValueError("focus must be a finite (x,y,z) at height 0.07-0.20 m")
    if (not np.isfinite([half, step, exclusion, separation]).all()
            or not 0.08 <= half <= 0.25 or not 0.001 <= step <= 0.01
            or not 0.01 <= exclusion < half or not 0.005 <= separation <= 0.05
            or not 1 <= count <= 20):
        raise ValueError("invalid prediction grid or peak selection parameters")
    if phases is None:
        phases = patterns.phases_focus(pos)
    phases = np.asarray(phases, float)
    if phases.shape != (patterns.N_EMITTERS,) or not np.isfinite(phases).all():
        raise ValueError("prediction needs 256 finite phases")
    phases = _quantized(phases)
    axis = np.arange(-half, half + step / 2, step)
    x, z = np.meshgrid(axis, axis)
    grid = np.column_stack((x.ravel(), np.full(x.size, pos[1]), z.ravel()))
    emit = np.exp(1j * phases)
    amplitude = np.empty(len(grid))
    for start in range(0, len(grid), 8192):
        end = min(start + 8192, len(grid))
        amplitude[start:end] = np.abs(patterns._propagators(grid[start:end]).T @ emit)
    amplitude_2d = amplitude.reshape(x.shape)
    target_pressure = float(abs(patterns.field_at(pos[None, :], phases)[0]))
    remote = np.hypot(x - pos[0], z - pos[2]) > exclusion
    candidates = np.flatnonzero(remote &
        (amplitude_2d == maximum_filter(amplitude_2d, size=5)))
    candidates = candidates[np.argsort(amplitude[candidates])[::-1]]
    peaks = []
    for index in candidates:
        point = grid[index]
        if all(np.linalg.norm(point[[0, 2]] -
               np.asarray(peak["position_m"])[[0, 2]]) >= separation for peak in peaks):
            ratio = float(amplitude[index] / target_pressure)
            peaks.append({"position_m": point.tolist(),
                          "relative_pressure": ratio,
                          "relative_intensity": ratio * ratio})
        if len(peaks) == count:
            break
    return {"model": "ideal_piston_40kHz_quantized_32_phase_bins",
            "measurement": False,
            "target_m": pos.tolist(),
            "target_pressure_au": target_pressure,
            "plane_y_m": float(pos[1]),
            "grid_half_width_m": float(half),
            "grid_step_m": float(step),
            "excluded_radius_m": float(exclusion),
            "peaks": peaks}


def save_prediction(report, path):
    """Save a model prediction as JSON or a receiver scan point CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    elif path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("point", "x_m", "y_m", "z_m",
                             "model_relative_pressure", "model_relative_intensity"))
            writer.writerow(("target", *report["target_m"], 1.0, 1.0))
            for index, peak in enumerate(report["peaks"], 1):
                writer.writerow((f"side_peak_{index}", *peak["position_m"],
                                 peak["relative_pressure"], peak["relative_intensity"]))
    else:
        raise ValueError("prediction file must end in .json or .csv")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline model-predicted receiver scan points")
    parser.add_argument("x", type=float, help="target x [m]")
    parser.add_argument("y", type=float, help="target height [m]")
    parser.add_argument("z", type=float, help="target z [m]")
    parser.add_argument("--algorithm", choices=("normal", "suppress"), default="normal")
    parser.add_argument("--half", type=float, default=0.15,
                        help="x/z scan half-width [m]")
    parser.add_argument("--step", type=float, default=0.002,
                        help="model grid spacing [m]")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--save", type=Path, help="save .json report or .csv scan points")
    args = parser.parse_args(argv)
    pos = (args.x, args.y, args.z)
    phases = (patterns.phases_focus(pos) if args.algorithm == "normal"
              else suppress_off_axis_peaks(pos)[0])
    report = predict_sidelobes(pos, phases, half=args.half,
                               step=args.step, count=args.count)
    report["algorithm"] = args.algorithm
    print("MODEL ONLY: predicted 40 kHz pressure peaks; no receiver or board data")
    print(f"target x={args.x:+.3f} y={args.y:.3f} z={args.z:+.3f} m")
    for index, peak in enumerate(report["peaks"], 1):
        x, y, z = peak["position_m"]
        print(f"  {index}: x={x:+.3f} y={y:.3f} z={z:+.3f} m, "
              f"relative pressure={peak['relative_pressure']:.1%}")
    if args.save:
        save_prediction(report, args.save)
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
