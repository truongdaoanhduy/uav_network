"""Opt-in successful report hops survive frame sampling without changing a mission."""
import importlib.util
import json
import unittest

import numpy as np

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv


def trace_config(**network):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 0
    c.world.max_steps = 20
    c.sensor.altitude_m = 20
    c.network.peer_range_m = c.network.gcs_range_m = 160
    c.runtime.device = "cpu"
    for name, value in network.items():
        setattr(c.network, name, value)
    return c


def action(destination, power=0):
    continuous = np.zeros((2, 3), dtype=np.float32)
    continuous[:, 2] = power
    return Action(continuous, np.asarray(destination, dtype=np.int64))


class CommunicationTraceTests(unittest.TestCase):
    def backends(self):
        yield "native"
        if importlib.util.find_spec("warp"):
            yield "warp_packet"

    def prepare(self, backend, c, enabled=True):
        positions = np.array([[500, 300], [500, 150]], dtype=float)
        targets = np.array([[500, 300], [900, 900]], dtype=float)
        if backend == "native":
            env = SwarmSearchEnv(c)
            if enabled:
                self.assertTrue(callable(getattr(env, "enable_communication_trace", None)),
                                "single-env communication trace opt-in API missing")
                env.enable_communication_trace()
            env.reset(seed=44, options={"positions": positions, "targets": targets, "obstacles": []})
        else:
            import warp as wp
            from uav_network.environment.gpu.vector import WarpMissionEnv
            from uav_network.environment.gpu.geometry import sense
            from uav_network.communication.gpu.mac import reset_network, sense_reports
            c.runtime.environment_backend = backend
            env = WarpMissionEnv(c)
            if enabled:
                self.assertTrue(callable(getattr(env, "enable_communication_trace", None)),
                                "single-env communication trace opt-in API missing")
                env.enable_communication_trace()
            env.reset(seed=44)
            s, p = env.vector.s, env.vector.p
            s.positions.assign(positions[None])
            s.targets.assign(targets[None])
            wp.launch(reset_network, dim=1, inputs=[s, p, wp.array([1], dtype=wp.int32, device="cpu")], device="cpu")
            wp.launch(sense, dim=(1, 2, 2), inputs=[s, p], device="cpu")
            wp.launch(sense_reports, dim=1, inputs=[s, p], device="cpu")
            env._refresh()
        self.addCleanup(env.close)
        return env

    def test_relay_then_gcs_hops_accumulate_until_drain(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                env = self.prepare(backend, trace_config())
                env.step(action([1, 0], -1))
                env.step(action([0, 2], 1))
                for _ in range(4):
                    env.step(action([0, 0]))
                events = env.drain_communication_events()
                self.assertEqual(len(events), 2)
                encoded = json.dumps({"events": events}, allow_nan=False)
                self.assertEqual(json.loads(encoded)["events"], events)
                self.assertEqual([(e["source"], e["destination"], e["hops"]) for e in events],
                                 [(0, 1, 1), (1, 2, 2)])
                for e, power in zip(events, (.1, .4)):
                    self.assertEqual(e["event"], "report_received")
                    self.assertEqual((e["target"], e["origin"], e["created_s"]), (0, 0, 0.))
                    self.assertAlmostEqual(e["power_w"], power)
                    self.assertEqual(e["power_scope"], "last_fragment")
                self.assertLess(events[0]["time_s"], events[1]["time_s"])
                np.testing.assert_array_equal(events[0]["source_xy"], [500, 300])
                np.testing.assert_array_equal(events[0]["destination_xy"], [500, 150])
                np.testing.assert_array_equal(events[1]["destination_xy"], [500, 0])
                self.assertTrue(env.reports.delivered[0])
                self.assertEqual(env.drain_communication_events(), [])
                env.reset(seed=44)
                self.assertEqual(env.drain_communication_events(), [])
                env.close()

    def test_failed_phy_emits_no_report_hop(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                env = self.prepare(backend, trace_config(sinr_threshold_db=200))
                _, _, _, _, info = env.step(action([1, 0]))
                self.assertGreater(info["network"]["phy_failures"], 0)
                self.assertGreater(info["network"]["uav_communication_energy_j"], 0)
                self.assertEqual(env.drain_communication_events(), [])
                env.close()

    def test_partial_fragments_do_not_emit_whole_report_hops(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                c = trace_config(report_bytes=2048, payload_bytes=512, max_queue_packets=1)
                c.world.dt_s = 0.005
                env = self.prepare(backend, c)
                for _ in range(2):
                    env.step(action([1, 0]))
                    self.assertEqual(env.drain_communication_events(), [])
                env.step(action([1, 0]))
                events = env.drain_communication_events()
                self.assertEqual(len(events), 1)
                self.assertEqual((events[0]["source"], events[0]["destination"]), (0, 1))
                env.close()

    def test_opt_in_does_not_change_observations_rewards_or_energy(self):
        for backend in self.backends():
            trajectories = []
            for enabled in (False, True):
                env = self.prepare(backend, trace_config(), enabled)
                if backend == "warp_packet":
                    self.assertGreater(env.vector.allocated_bytes, 0)
                    if not enabled:
                        self.assertEqual(env.vector.s.trace_report.size, 0)
                trajectory = []
                for command in (action([1, 0]), action([0, 2]), action([0, 0])):
                    obs, reward, ended, truncated, info = env.step(command)
                    trajectory.append((obs.copy(), reward, ended, truncated, info))
                if not enabled:
                    self.assertEqual(env.drain_communication_events(), [])
                env.close()
                trajectories.append(trajectory)
            for untraced, traced in zip(*trajectories):
                np.testing.assert_array_equal(untraced[0], traced[0])
                self.assertEqual(untraced[1:], traced[1:])


if __name__ == "__main__":
    unittest.main()
