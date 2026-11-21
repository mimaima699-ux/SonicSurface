"""Model-only pressure slices for the live hand and board diagnostic view."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import my_patterns


@dataclass(frozen=True)
class FieldPlane:
    name: str
    xyz: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class FieldFrame:
    active: bool
    focus: np.ndarray | None
    planes: dict[str, FieldPlane]
    target_pressure: float
    max_pressure: float
    peak_position: np.ndarray | None


class FieldSlices:
    """Cache three intersecting piston-model slices in board XYZ coordinates.

    The pressure is relative model amplitude, not measured sound pressure.
    Planes cover off-axis lobes across the array's x/z surroundings. Supply
    the phases actually sent to the board; NaN marks a disabled emitter.
    """

    def __init__(self, half_width=0.12, lateral_step=0.005,
                 height_min=0.05, height_max=0.25, height_step=0.01):
        if (not np.isfinite([half_width, lateral_step, height_min,
                             height_max, height_step]).all()
                or not 0.08 <= half_width <= 0.20
                or not 0.003 <= lateral_step <= 0.02
                or not 0.02 <= height_min < height_max <= 0.35
                or not 0.005 <= height_step <= 0.03):
            raise ValueError("invalid field sampling bounds or steps")
        self.lateral = np.linspace(-half_width, half_width,
                                   int(round(2 * half_width / lateral_step)) + 1)
        self.heights = np.linspace(height_min, height_max,
                                   int(round((height_max-height_min) / height_step)) + 1)
        self._focus_key = None
        self._points = None
        self._shapes = None
        self._propagators = None
        self._phase_key = None
        self._last_frame = None

    @staticmethod
    def _quantized_sources(phases):
        phases = np.asarray(phases, float)
        if phases.shape != (my_patterns.N_EMITTERS,):
            raise ValueError("field prediction needs 256 phases")
        if np.isinf(phases).any():
            raise ValueError("phase values cannot be infinite")
        enabled = np.isfinite(phases)
        bins = np.zeros(my_patterns.N_EMITTERS, np.uint8)
        bins[enabled] = (np.rint(my_patterns._wrap(phases[enabled])
                                  * my_patterns.PHASE_DIVS / (2 * np.pi))
                         .astype(np.uint8) % my_patterns.PHASE_DIVS)
        sources = np.zeros(my_patterns.N_EMITTERS, complex)
        sources[enabled] = np.exp(2j * np.pi * bins[enabled]
                                  / my_patterns.PHASE_DIVS)
        return sources, bytes(np.where(enabled, bins, my_patterns.PHASE_DIVS)
                              .astype(np.uint8))

    def _build_planes(self, focus):
        axis = self.lateral
        heights = self.heights
        x, z = np.meshgrid(axis, axis, indexing="xy")
        horizontal = np.stack((x, np.full_like(x, focus[1]), z), axis=-1)
        x, y = np.meshgrid(axis, heights, indexing="xy")
        along_x = np.stack((x, y, np.full_like(x, focus[2])), axis=-1)
        z, y = np.meshgrid(axis, heights, indexing="xy")
        along_z = np.stack((np.full_like(z, focus[0]), y, z), axis=-1)
        grids = {"xz": horizontal, "xy": along_x, "zy": along_z}
        self._shapes = {name: grid.shape[:2] for name, grid in grids.items()}
        self._points = grids
        flat = np.concatenate([grid.reshape(-1, 3) for grid in grids.values()])
        self._propagators = my_patterns._propagators(flat)

    def sample(self, phases, focus):
        """Return a FieldFrame, or an inactive empty frame for an OFF pattern."""
        sources, phase_key = self._quantized_sources(phases)
        if not np.any(sources):
            self._phase_key = None
            self._last_frame = None
            return FieldFrame(False, None, {}, 0.0, 0.0, None)
        focus = np.asarray(focus, float)
        if (focus.shape != (3,) or not np.isfinite(focus).all()
                or not 0.03 <= focus[1] <= 0.30):
            raise ValueError("active field needs a finite focus above the board")
        focus_key = tuple(float(v) for v in focus)
        if self._focus_key != focus_key or self._propagators is None:
            self._build_planes(focus)
            self._focus_key = focus_key
            self._phase_key = None
        if phase_key == self._phase_key and self._last_frame is not None:
            return self._last_frame

        pressure = np.abs(self._propagators.T @ sources)
        planes = {}
        offset = 0
        for name, xyz in self._points.items():
            count = xyz.shape[0] * xyz.shape[1]
            planes[name] = FieldPlane(
                name, xyz, pressure[offset:offset+count].reshape(self._shapes[name]))
            offset += count
        target_pressure = float(np.abs(my_patterns._propagators(focus[None, :]).T
                                       @ sources)[0])
        peak_index = int(np.argmax(pressure))
        sampled_max = float(pressure[peak_index])
        if target_pressure >= sampled_max:
            peak_position = focus.copy()
            max_pressure = target_pressure
        else:
            flat_points = np.concatenate([grid.reshape(-1, 3)
                                          for grid in self._points.values()])
            peak_position = flat_points[peak_index].copy()
            max_pressure = sampled_max
        frame = FieldFrame(True, focus.copy(), planes, target_pressure,
                           max_pressure, peak_position)
        self._phase_key = phase_key
        self._last_frame = frame
        return frame
