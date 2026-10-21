"""Cross-step 1 MB Scenario V3 report delivery on native and Warp CPU backends."""
import importlib.util
from pathlib import Path
import unittest
from copy import deepcopy

import numpy as np
import torch

from uav_network.configuration import load_config
from uav_network.environment.env import Action, SwarmSearchEnv


ROOT = Path(__file__).resolve().parents[1]


def v3_fixture(num_uavs=3):
    c = load_config(ROOT / "configs/comparison-25k-scenario-v3.yaml")
    c.world.num_uavs = num_uavs
    c.world.num_targets, c.world.num_obstacles = 1, 0
    c.world.max_steps = 80
    c.sensor.altitude_m = 20
    c.sensor.obstacle_occlusion = False
    c.runtime.num_envs = 1
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.tracking.enabled = False
    return c.validate()


def command(num_uavs, selected=None, power=1.0):
    continuous = np.zeros((num_uavs, 3), dtype=np.float32)
    destination = np.zeros(num_uavs, dtype=np.int64)
    if selected is not None:
        source, index = selected
        destination[source] = index
        continuous[source, 2] = power
    return Action(continuous, destination)


class ScenarioV3IntegrationTests(unittest.TestCase):
    def backends(self):
        yield "native"
        if importlib.util.find_spec("warp"):
            yield "warp_gen1"
            yield "warp_gen2"

    def prepare(self, backend, positions):
        c = v3_fixture(len(positions))
        targets = np.asarray([positions[0]], dtype=float)
        positions = np.asarray(positions, dtype=float)
        if backend == "native":
            env = SwarmSearchEnv(c)
            env.enable_communication_trace()
            env.reset(seed=44, options={"positions": positions, "targets": targets, "obstacles": []})
        else:
            import warp as wp
            from uav_network.environment.gpu.vector import WarpMissionEnv, WarpVectorEnv
            from uav_network.environment.gpu.geometry import sense
            from uav_network.communication.gpu.mac import reset_network, sense_reports
            c.runtime.environment_backend = "warp_packet"
            env = WarpMissionEnv(c)
            scheduler = "gen2" if backend == "warp_gen2" else "gen1"
            if scheduler != "gen1":
                env.vector.close()
                selected = deepcopy(c)
                selected.runtime.num_envs = 1
                env.vector = WarpVectorEnv(selected, device="cpu", packet_scheduler=scheduler)
            env.enable_communication_trace()
            env.reset(seed=44)
            s, p = env.vector.s, env.vector.p
            s.positions.assign(positions[None])
            s.targets.assign(targets[None])
            wp.launch(reset_network, dim=1, inputs=[s, p, wp.array([1], dtype=wp.int32)], device="cpu")
            wp.launch(sense, dim=(1, len(positions), 1), inputs=[s, p], device="cpu")
            wp.launch(sense_reports, dim=1, inputs=[s, p], device="cpu")
            env._refresh()
        self.addCleanup(env.close)
        return env

    def run_until_hop(self, env, selected, expected_hops, limit=20, power=1.0):
        events = []
        last = None
        for _ in range(limit):
            _, _, ended, truncated, last = env.step(command(env.config.world.num_uavs, selected, power))
            events.extend(env.drain_communication_events())
            if any(event["hops"] == expected_hops for event in events):
                return events, last
            if ended or truncated:
                break
        self.fail(f"report hop {expected_hops} did not complete; events={events}, info={last}")

    def test_one_megabyte_direct_delivery_streams_across_steps(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                env = self.prepare(backend, [[500, 300], [950, 950], [50, 950]])
                events, info = self.run_until_hop(env, (0, 3), 1)
                self.assertTrue(env.reports.delivered[0])
                self.assertEqual([(e["source"], e["destination"], e["hops"]) for e in events], [(0, 3, 1)])
                self.assertGreaterEqual(info["network"]["generated_packets"], 977)
                self.assertGreater(info["network"]["uav_tx_energy_j"], 0)
                self.assertGreater(info["network"]["gcs_rx_energy_j"], 0)

    def test_one_megabyte_two_hop_delivery(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                env = self.prepare(backend, [[500, 700], [500, 350], [950, 950]])
                first, _ = self.run_until_hop(env, (0, 1), 1)
                second, info = self.run_until_hop(env, (1, 3), 2)
                events = first + second
                self.assertTrue(env.reports.delivered[0])
                self.assertEqual([(e["source"], e["destination"], e["hops"]) for e in events],
                                 [(0, 1, 1), (1, 3, 2)])
                self.assertEqual(info["network"]["report_average_hops"], 2.0)

    def test_one_megabyte_three_hop_delivery(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                env = self.prepare(backend, [[500, 900], [500, 600], [500, 300]])
                a, _ = self.run_until_hop(env, (0, 1), 1)
                b, _ = self.run_until_hop(env, (1, 2), 2)
                c, info = self.run_until_hop(env, (2, 3), 3)
                events = a + b + c
                self.assertTrue(env.reports.delivered[0])
                self.assertEqual([(e["source"], e["destination"], e["hops"]) for e in events],
                                 [(0, 1, 1), (1, 2, 2), (2, 3, 3)])
                self.assertEqual(info["network"]["report_average_hops"], 3.0)

    def test_failed_low_power_attempt_costs_energy_then_high_power_recovers(self):
        for backend in self.backends():
            with self.subTest(backend=backend):
                env = self.prepare(backend, [[500, 450], [950, 950], [50, 950]])
                _, _, ended, _, failed = env.step(command(3, (0, 3), -1.0))
                self.assertFalse(ended)
                self.assertFalse(env.reports.delivered[0])
                self.assertGreater(failed["network"]["phy_failures"], 0)
                self.assertGreater(failed["network"]["uav_tx_energy_j"], 0)
                spent = failed["network"]["uav_communication_energy_j"]
                events, final = self.run_until_hop(env, (0, 3), 1, limit=25, power=1.0)
                self.assertTrue(env.reports.delivered[0])
                self.assertEqual(len(events), 1)
                self.assertGreater(final["network"]["uav_communication_energy_j"], spent)

    def test_gen1_gen2_v3_contention_matches_every_snapshot_array_and_metric(self):
        if not importlib.util.find_spec("warp"):
            self.skipTest("optional Warp dependency not installed")
        positions = [[500, 300], [520, 300], [950, 950]]
        left = self.prepare("warp_gen1", positions)
        right = self.prepare("warp_gen2", positions)
        actions = [command(3, (0, 3), 1.0), command(3, (1, 3), 1.0),
                   command(3, (0, 3), -1.0), command(3, None)]
        network_keys = ("generated_packets", "received_packets", "phy_successes", "phy_failures",
                        "collisions", "communication_energy_j", "uav_communication_energy_j",
                        "gcs_communication_energy_j", "report_average_hops", "report_e2e_delay_s")
        for action in actions:
            a = left.step(action); b = right.step(action)
            self.assertEqual(a[2:4], b[2:4])
            self.assertEqual(a[1], b[1])
            for key in network_keys:
                self.assertEqual(a[4]["network"][key], b[4]["network"][key], key)
            snap_a, snap_b = left.vector.snapshot_state(), right.vector.snapshot_state()
            self.assertEqual(snap_a["arrays"].keys(), snap_b["arrays"].keys())
            for name in snap_a["arrays"]:
                torch.testing.assert_close(snap_a["arrays"][name], snap_b["arrays"][name],
                                           rtol=0, atol=0, equal_nan=True, msg=name)
            self.assertEqual(left.drain_communication_events(), right.drain_communication_events())



if __name__ == "__main__":
    unittest.main()
