"""Offline checks for the nonblocking live 3D diagnostic viewer."""

import unittest
from unittest.mock import Mock

import numpy as np

from live_diagnostic_3d import LiveDiagnostic3D, _board_points
from my_patterns import phases_focus


class FakeCamera:
    def point_cloud(self, uv):
        self.sample_uv = uv
        return (np.array([[0, 0.12, 0], [0.01, 0.1, 0]]),
                np.array([[1, 0, 0], [0, 1, 0]]))

    def point_3d(self, uv, mask):
        return np.array([0.0, 0.12, 0.0])


class LiveDiagnostic3DTests(unittest.TestCase):
    def test_camera_points_transform_to_board_frame(self):
        rotation = np.diag([1.0, -1.0, 1.0])
        translated = _board_points([[0.01, 0.2, -0.03]], rotation,
                                   [0, 0.3, 0])
        np.testing.assert_allclose(translated, [[0.01, 0.1, -0.03]])

    def test_field_clears_immediately_when_focus_turns_off(self):
        import pyvista as pv

        viewer = LiveDiagnostic3D(plotter=pv.Plotter(off_screen=True),
                                  render_interval=0)
        camera = FakeCamera()
        try:
            focus = [0.0, 0.12, 0.0]
            self.assertTrue(viewer.update(camera, np.eye(3), np.zeros(3),
                                          None, focus, focus,
                                          phases_focus(focus), "inside"))
            self.assertTrue(viewer._field_visible)
            self.assertTrue(viewer.update(camera, np.eye(3), np.zeros(3),
                                          None, None, None, None, "no_hand"))
            self.assertFalse(viewer._field_visible)
        finally:
            viewer.close()

    def test_closing_window_requests_tracking_stop(self):
        import pyvista as pv

        viewer = LiveDiagnostic3D(plotter=pv.Plotter(off_screen=True))
        viewer.close()
        self.assertFalse(viewer.update(FakeCamera(), np.eye(3), np.zeros(3),
                                       None, None, None, None, "no_hand"))

    def test_native_exit_event_requests_tracking_stop(self):
        import pyvista as pv

        viewer = LiveDiagnostic3D(plotter=pv.Plotter(off_screen=True))
        try:
            viewer.plotter.iren.interactor.InvokeEvent("ExitEvent")
            self.assertFalse(viewer.update(FakeCamera(), np.eye(3), np.zeros(3),
                                           None, None, None, None, "no_hand"))
        finally:
            viewer.close()

    def test_render_is_throttled_but_window_closure_is_checked(self):
        import pyvista as pv

        times = iter([0.0, 0.0, 0.05])
        viewer = LiveDiagnostic3D(plotter=pv.Plotter(off_screen=True),
                                  clock=lambda: next(times))
        camera = FakeCamera()
        try:
            self.assertTrue(viewer.update(camera, np.eye(3), np.zeros(3),
                                          None, None, None, None, "no_hand"))
            camera.point_cloud = Mock(side_effect=AssertionError("extra cloud"))
            self.assertTrue(viewer.update(camera, np.eye(3), np.zeros(3),
                                          None, None, None, None, "no_hand"))
            camera.point_cloud.assert_not_called()
            viewer._mark_closed()
            self.assertFalse(viewer.update(camera, np.eye(3), np.zeros(3),
                                           None, None, None, None, "no_hand"))
        finally:
            viewer.close()

    def test_scene_focus_cloud_sampling_uses_hand_palm(self):
        import pyvista as pv

        camera = FakeCamera()
        observation = Mock()
        observation.point.return_value = (120, 80)
        observation.landmarks = np.full((21, 2), np.nan)
        viewer = LiveDiagnostic3D(plotter=pv.Plotter(off_screen=True))
        try:
            self.assertTrue(viewer.update(camera, np.eye(3), np.zeros(3),
                                          observation, None, None, None,
                                          "no_hand"))
            self.assertEqual(camera.sample_uv, (120, 80))
            observation.point.assert_called_with("palm")
        finally:
            viewer.close()

    def test_space_requests_pause_toggle_only_once_until_consumed(self):
        import pyvista as pv

        viewer = LiveDiagnostic3D(plotter=pv.Plotter(off_screen=True))
        try:
            interactor = viewer.plotter.iren.interactor
            interactor.SetKeyEventInformation(0, 0, " ", 0, "space")
            interactor.InvokeEvent("KeyPressEvent")
            self.assertTrue(viewer.consume_pause_toggle())
            self.assertFalse(viewer.consume_pause_toggle())
            interactor.InvokeEvent("KeyPressEvent")
            self.assertTrue(viewer.consume_pause_toggle())
        finally:
            viewer.close()

    def test_pause_keeps_last_scene_and_field_while_window_polls(self):
        import pyvista as pv
        from diagnostic_field_3d import FieldFrame, FieldPlane

        xyz = np.array([[[0, 0.12, 0], [0.01, 0.12, 0]],
                        [[0, 0.12, 0.01], [0.01, 0.12, 0.01]]])
        frame = FieldFrame(True, np.array([0, 0.12, 0]),
                           {"xz": FieldPlane("xz", xyz, np.ones((2, 2)))},
                           1.0, 1.0, np.array([0, 0.12, 0]))
        sampler = Mock()
        sampler.sample.return_value = frame
        viewer = LiveDiagnostic3D(plotter=pv.Plotter(off_screen=True),
                                  sampler=sampler, render_interval=0)
        camera = FakeCamera()
        try:
            focus = np.array([0, 0.12, 0])
            self.assertTrue(viewer.update(camera, np.eye(3), np.zeros(3),
                                          None, focus, focus,
                                          phases_focus(focus), "inside"))
            scene = viewer.plotter.actors["realsense-scene"]
            field = viewer.plotter.actors["pressure-xz"]
            viewer.set_paused(True)
            self.assertIn("PAUSED / transmitter OFF",
                          viewer.plotter.actors["pause-status"].GetText(3))
            camera.point_cloud = Mock(side_effect=AssertionError("camera sampled while paused"))
            sampler.sample.reset_mock()
            process_events = viewer.plotter.iren.process_events
            viewer.plotter.iren.process_events = Mock(wraps=process_events)
            self.assertTrue(viewer.poll())
            self.assertTrue(viewer.update(camera, np.eye(3), np.zeros(3),
                                          None, None, None, None, "no_hand"))
            self.assertGreaterEqual(viewer.plotter.iren.process_events.call_count, 2)
            camera.point_cloud.assert_not_called()
            sampler.sample.assert_not_called()
            self.assertIs(viewer.plotter.actors["realsense-scene"], scene)
            self.assertIs(viewer.plotter.actors["pressure-xz"], field)
            viewer.set_paused(False)
            self.assertNotIn("pause-status", viewer.plotter.actors)
        finally:
            viewer.close()

    def test_poll_reports_native_exit_while_paused(self):
        import pyvista as pv

        viewer = LiveDiagnostic3D(plotter=pv.Plotter(off_screen=True))
        try:
            viewer.set_paused(True)
            viewer.plotter.iren.interactor.InvokeEvent("ExitEvent")
            self.assertFalse(viewer.poll())
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
