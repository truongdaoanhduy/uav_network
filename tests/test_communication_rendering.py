"""Successful report-hop overlays retain event-time coordinates and sampled events."""
import unittest
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np

from uav_network.experiments import rendering


class CommunicationRenderingTests(unittest.TestCase):
    def event(self, time=2.25, destination=1):
        return dict(event="report_received", time_s=time, source=0, destination=destination,
                    target=2, origin=0, created_s=1., hops=1, power_w=.15,
                    source_xy=[10.,20.], destination_xy=[30.,40.])

    def test_frame_drains_all_unsampled_events_and_preserves_payload(self):
        pending = [self.event(1.25), self.event(4.75, 6)]
        def drain():
            result = list(pending)
            pending.clear()
            return result
        env = SimpleNamespace(steps=5, positions=np.zeros((6,2)), battery_fraction=np.ones(6),
                              reports=SimpleNamespace(known=np.zeros((6,3),dtype=bool),
                                                      delivered=np.zeros(3,dtype=bool)),
                              drain_communication_events=drain,
                              info=lambda: {"buffer_reports":2,"energy":{"active_uavs":6,
                                  "propulsion_energy_j":1000.},"network":{"uav_communication_energy_j":.01}})
        frame = rendering.animation_frame(env)
        self.assertEqual([e["time_s"] for e in frame["communication_events"]], [1.25,4.75])
        self.assertEqual(frame["buffer_reports"],2)
        self.assertEqual(frame["flight_energy_j"],1000.)
        self.assertEqual(rendering.animation_frame(env)["communication_events"],[])

    def test_arrows_only_show_recent_successful_hops_at_event_positions(self):
        events = [self.event(1.), self.event(23.), self.event(24.,6)]
        fig, ax = plt.subplots()
        try:
            rendering._draw_communication(ax, events, now=25., num_uavs=6, window_s=20.)
            arrows = [text for text in ax.texts if getattr(text,"arrow_patch",None) is not None]
            self.assertEqual(len(arrows),2)
            for arrow in arrows:
                self.assertEqual(tuple(arrow.xy),(30.,40.))
                self.assertEqual(tuple(arrow.xyann),(10.,20.))
            labels = " ".join(text.get_text() for text in ax.texts)
            self.assertIn("T2",labels)
            self.assertIn("0.15 W",labels)
        finally:
            plt.close(fig)

    def test_boundary_labels_remain_near_uav_on_small_legal_maps(self):
        for size in (50., 100., 1000.):
            with self.subTest(map_size=size):
                env = SimpleNamespace(config=SimpleNamespace(world=SimpleNamespace(map_size_m=size),
                                      sensor=SimpleNamespace(footprint_radius_m=10.)),
                                      obstacles=[], targets=np.array([[1.,1.]]), gcs=np.array([0.,0.]))
                positions = np.array([[size,size]])
                fig, ax = plt.subplots()
                try:
                    rendering._draw(ax,env,[positions],positions,np.ones(1),
                                    np.zeros(1,dtype=bool),np.zeros(1,dtype=bool),"test")
                    label = next(text for text in ax.texts if text.get_text().startswith("0:"))
                    for coordinate in label.get_position():
                        self.assertGreater(coordinate,.75*size)
                        self.assertLess(coordinate,size)
                finally:
                    plt.close(fig)

    def test_empty_trace_creates_no_fake_connectivity_arrows(self):
        fig, ax = plt.subplots()
        try:
            rendering._draw_communication(ax, [], now=25., num_uavs=6, window_s=20.)
            self.assertEqual(len(ax.texts),0)
        finally:
            plt.close(fig)
