"""Interactive board-frame view of measured geometry and predicted sound."""

from __future__ import annotations

import time

import numpy as np

import hand_track
import my_patterns
from diagnostic_field_3d import FieldSlices


def _board_points(camera_points, rotation, translation):
    points = np.asarray(camera_points, float).reshape(-1, 3)
    return points @ np.asarray(rotation, float).T + np.asarray(translation, float)


def _hand_landmarks(cap, observation, rotation, translation):
    points = np.full((21, 3), np.nan)
    if observation is None:
        return points
    for index, uv in enumerate(observation.landmarks):
        if not np.isfinite(uv).all():
            continue
        camera = cap.point_3d(uv, observation.mask)
        if camera is not None:
            points[index] = _board_points([camera], rotation, translation)[0]
    return points


class LiveDiagnostic3D:
    """Show the RealSense scene and model pressure slices without blocking tracking.

    RealSense geometry is measured; skeleton depth and acoustic pressure are
    estimates. Closing this window asks the tracking loop to stop and turn off.
    """

    def __init__(self, dry=False, freq=200.0, duty=0.7, no_mod=False,
                 scene_interval=0.33, field_interval=0.5,
                 render_interval=0.2, plotter=None, sampler=None,
                 clock=time.perf_counter):
        import pyvista as pv

        self.pv = pv
        self.plotter = plotter or pv.Plotter(window_size=(1150, 850))
        self.sampler = sampler or FieldSlices()
        self.clock = clock
        self.scene_interval = scene_interval
        self.field_interval = field_interval
        self.render_interval = render_interval
        self._scene_time = -np.inf
        self._field_time = -np.inf
        self._render_time = -np.inf
        self._closed = False
        self._field_visible = False
        self._paused = False
        self._pause_toggle_pending = False
        self._mode = ("DRY: predicted phases" if dry else
                      "Board static ON envelope" if no_mod else
                      f"Board ON envelope: {freq:g} Hz, {duty:.0%} duty")

        self.plotter.set_background("#101821")
        self.plotter.add_mesh(
            pv.Plane(center=(0, -0.003, 0), direction=(0, 1, 0),
                     i_size=0.168, j_size=0.168),
            color="#688395", opacity=0.30, name="board-plane",
            render=False)
        self.plotter.add_points(
            my_patterns.EMITTERS, color="#d8e4ec", point_size=5,
            render_points_as_spheres=True, name="emitters", render=False)
        self.plotter.add_axes()
        self.plotter.camera_position = (
            (0.26, 0.31, 0.26), (0.0, 0.11, 0.0), (0.0, 1.0, 0.0))
        self.plotter.add_text(
            "RealSense geometry | acoustic pressure: MODEL PREDICTION",
            position="upper_left", font_size=10, color="white",
            name="legend", render=False)
        self.plotter.add_key_event("q", self.close)
        self.plotter.add_key_event("space", self._request_pause_toggle)
        self.plotter.iren.add_observer("ExitEvent", self._mark_closed)
        self.plotter.show(interactive_update=True, auto_close=False)

    def _mark_closed(self, *_args):
        self._closed = True

    def _request_pause_toggle(self):
        if not self.plotter.iren.interactor.GetRepeatCount():
            self._pause_toggle_pending = not self._pause_toggle_pending

    def consume_pause_toggle(self):
        """Return and clear a Space key request from the 3D window."""
        pending = self._pause_toggle_pending
        self._pause_toggle_pending = False
        return pending

    def set_paused(self, paused):
        """Keep the last geometry and model field available for inspection."""
        paused = bool(paused)
        if (self._paused == paused or self._closed
                or getattr(self.plotter, "_closed", False)):
            return
        self._paused = paused
        if paused:
            self.plotter.add_text(
                "PAUSED / transmitter OFF\nFrozen last frame: drag to rotate, Space to resume",
                position="upper_right", font_size=11, color="white",
                name="pause-status", render=False)
        else:
            self._remove("pause-status")
        self.plotter.render()

    def poll(self):
        """Process window input without sampling the camera or acoustic field."""
        if self._closed or getattr(self.plotter, "_closed", False):
            return False
        try:
            self.plotter.iren.process_events()
        except (RuntimeError, AttributeError):
            if self._closed or getattr(self.plotter, "_closed", False):
                return False
            raise
        return not (self._closed or getattr(self.plotter, "_closed", False))

    def _remove(self, name):
        self.plotter.remove_actor(name, render=False)

    def _points(self, name, xyz, color, size):
        self._remove(name)
        points = np.asarray(xyz, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            self.plotter.add_points(points, color=color, point_size=size,
                                    render_points_as_spheres=True,
                                    name=name, render=False,
                                    reset_camera=False)

    def _scene(self, cap, rotation, translation, observation):
        palm_uv = observation.point("palm") if observation is not None else None
        vertices, colors = cap.point_cloud(palm_uv)
        world = _board_points(vertices, rotation, translation)
        colors = np.asarray(colors, float).reshape(-1, 3)
        if len(colors) != len(world):
            raise ValueError("RealSense point cloud color count does not match vertices")
        valid = np.isfinite(world).all(axis=1)
        valid &= (np.abs(world[:, 0]) < 0.22) & (np.abs(world[:, 2]) < 0.22)
        valid &= (world[:, 1] > -0.03) & (world[:, 1] < 0.32)
        self._remove("realsense-scene")
        if np.any(valid):
            mesh = self.pv.PolyData(world[valid])
            mesh.point_data["RGB"] = np.uint8(
                np.rint(np.clip(colors[valid], 0, 1) * 255))
            self.plotter.add_mesh(
                mesh, scalars="RGB", rgb=True, style="points",
                point_size=3, name="realsense-scene", render=False,
                reset_camera=False)

        landmarks = _hand_landmarks(cap, observation, rotation, translation)
        self._points("hand-landmarks", landmarks, "#49d8ff", 9)
        self._remove("hand-bones")
        edges = [(a, b) for a, b in hand_track.HAND_CONNECTIONS
                 if np.isfinite(landmarks[[a, b]]).all()
                 and np.linalg.norm(landmarks[a] - landmarks[b]) < 0.08]
        if edges:
            used = sorted({index for edge in edges for index in edge})
            remap = {index: new for new, index in enumerate(used)}
            lines = self.pv.PolyData(landmarks[used])
            lines.lines = np.asarray(
                [[2, remap[a], remap[b]] for a, b in edges]).ravel()
            self.plotter.add_mesh(lines, color="#49d8ff", line_width=3,
                                  name="hand-bones", render=False,
                                  reset_camera=False)

    def _clear_field(self):
        for name in ("xz", "xy", "zy"):
            self._remove(f"pressure-{name}")
        self._remove("model-peak")
        self._field_visible = False

    def _field(self, phases, focus):
        if phases is None or focus is None:
            if self._field_visible:
                self._clear_field()
            return "OFF: no predicted emitted field"
        frame = self.sampler.sample(phases, focus)
        if not frame.active:
            self._clear_field()
            return "OFF: no predicted emitted field"
        scale = max(frame.max_pressure, 1e-9)
        for name, plane in frame.planes.items():
            grid = self.pv.StructuredGrid(
                plane.xyz[:, :, 0], plane.xyz[:, :, 1], plane.xyz[:, :, 2])
            grid.point_data["relative pressure"] = (plane.amplitude / scale).ravel(
                order="F")
            self.plotter.add_mesh(
                grid, scalars="relative pressure", cmap="inferno",
                clim=(0, 1), opacity=0.38, show_scalar_bar=False,
                name=f"pressure-{name}", render=False,
                reset_camera=False)
        self._points("model-peak", [frame.peak_position], "#ffd339", 13)
        self._field_visible = True
        ratio = frame.target_pressure / scale
        return (f"MODEL peak/target relative pressure: {scale:.1f}/{frame.target_pressure:.1f} "
                f"(target {ratio:.0%} of shown peak)")

    def update(self, cap, rotation, translation, observation,
               observed, focus, phases, gate):
        """Return False after the 3D window closes so the caller can stop."""
        if not self.poll():
            return False
        if self._paused:
            return True
        now = self.clock()
        if now - self._render_time < self.render_interval:
            return True
        try:
            if now - self._scene_time >= self.scene_interval:
                self._scene(cap, rotation, translation, observation)
                self._scene_time = now
            self._points("measured-tip", [] if observed is None else [observed],
                         "#3dff68", 18)
            self._points("commanded-focus", [] if focus is None else [focus],
                         "#ff3e86", 18)
            if phases is None or focus is None:
                status = self._field(phases, focus)
                self._field_time = -np.inf
            elif now - self._field_time >= self.field_interval:
                status = self._field(phases, focus)
                self._field_time = now
            else:
                status = "MODEL prediction held until next field refresh"
            self.plotter.add_text(
                f"{self._mode} | gate: {gate}\n"
                f"Green: measured tip | Pink: commanded focus | Yellow: model peak\n"
                f"{status} | axes: board X/Y/Z (m)",
                position="lower_left", font_size=10, color="white",
                name="status", render=False)
            self.plotter.update(stime=1, force_redraw=True)
            self._render_time = self.clock()
        except (RuntimeError, AttributeError, ValueError) as exc:
            if getattr(self.plotter, "_closed", False):
                self._closed = True
                return False
            raise RuntimeError(f"3D diagnostic view failed: {exc}") from exc
        return not (self._closed or getattr(self.plotter, "_closed", False))

    def close(self):
        self._closed = True
        if not getattr(self.plotter, "_closed", False):
            self.plotter.close()
