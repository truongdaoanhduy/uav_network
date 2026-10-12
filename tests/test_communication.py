"""Real UavNetSim tests; no simulated-success mocks or skipped integration."""
import unittest
from dataclasses import replace

import numpy as np

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.communication.backend import UavNetSimNetwork


def network_config(**network):
    c = Config()
    return replace(c, world=replace(c.world, num_uavs=2, num_targets=1,
                                    num_obstacles=0, max_steps=20),
                   sensor=replace(c.sensor, altitude_m=20),
                   network=replace(c.network, peer_range_m=160, gcs_range_m=160, **network))


def actions(dest0=0, dest1=0, power=0):
    values = np.zeros((2, 3))
    values[:, 2] = power
    return Action(values, np.array([dest0, dest1]))


class CommunicationTests(unittest.TestCase):
    def setUp(self):
        self.env = None

    def tearDown(self):
        if self.env:
            self.env.close()

    def prepare(self, config=None, positions=None):
        self.env = SwarmSearchEnv(config or network_config())
        positions = positions or [[500, 300], [500, 150]]
        self.env.reset(seed=44, options=dict(positions=positions,
                       targets=[positions[0]], obstacles=[]))
        return self.env

    def test_real_direct_gcs_delivery(self):
        env = self.prepare(positions=[[500, 100], [600, 100]])
        _, reward, terminated, _, info = env.step(actions(2))
        self.assertTrue(terminated)
        self.assertEqual(info["targets_delivered"], 1)
        self.assertGreater(reward, 40)
        self.assertEqual(info["network"]["backend"], "uavnetsim")
        self.assertGreater(info["network"]["generated_packets"], 0)
        self.assertEqual(info["network"]["packet_pdr_percent"], 100)
        self.assertGreater(info["network"]["communication_energy_j"], 0)

    def test_selected_peer_then_gcs_is_two_mission_hops(self):
        env = self.prepare()
        env.step(actions(1))
        self.assertNotIn(0, env.reports.buffers[0])
        self.assertIn(0, env.reports.buffers[1])
        self.assertFalse(env.reports.delivered[0])
        self.assertEqual(env.reports.buffers[1][0].hops, 1)
        env.step(actions(0, 2))
        self.assertTrue(env.reports.delivered[0])
        self.assertEqual(env.network.metrics()["report_average_hops"], 2)

    def test_silent_and_out_of_range_actions_do_not_auto_route(self):
        env = self.prepare()
        env.step(actions())
        env.step(actions(2))
        self.assertIn(0, env.reports.buffers[0])
        self.assertEqual(env.network.metrics()["generated_packets"], 0)

    def test_failed_phy_attempt_consumes_action_power_and_retains_report(self):
        energies = []
        for power in [-1, 1]:
            env = self.prepare(network_config(sinr_threshold_db=200),
                               positions=[[500, 100], [600, 100]])
            env.step(actions(2, power=power))
            stats = env.network.metrics()
            self.assertEqual(stats["targets_delivered"], 0)
            self.assertGreater(stats["phy_failures"], 0)
            self.assertGreater(stats["communication_energy_j"], 0)
            self.assertIn(0, env.reports.buffers[0])
            energies.append(stats["communication_energy_j"])
            env.close()
        self.assertAlmostEqual(energies[1] / energies[0], 4, places=6)

    def test_fragmented_report_needs_all_payload_fragments(self):
        env = self.prepare(network_config(report_bytes=2048, payload_bytes=512),
                           positions=[[500, 100], [600, 100]])
        env.step(actions(2))
        self.assertTrue(env.reports.delivered[0])
        self.assertEqual(env.network.metrics()["generated_packets"], 4)
        self.assertEqual(env.network.metrics()["received_packets"], 4)

    def test_original_ttl_survives_relay_and_no_stale_delivery(self):
        env = self.prepare(network_config(report_ttl_s=1.5))
        env.step(actions(1))
        self.assertEqual(env.reports.buffers[1][0].created_s, 0)
        env.step(actions())
        self.assertFalse(env.reports.delivered[0])
        self.assertFalse(env.reports.buffers[1])
        self.assertEqual(env.reports.expired, 1)

    def test_network_same_seed_replays_metrics(self):
        env = self.prepare()
        env.step(actions(1))
        env.step(actions(0, 2))
        first = env.network.metrics()
        env.reset(seed=44, options=dict(positions=[[500, 300], [500, 150]],
                                       targets=[[500, 300]], obstacles=[]))
        env.step(actions(1))
        env.step(actions(0, 2))
        self.assertEqual(first, env.network.metrics())

    def test_two_simultaneous_instances_reject_global_config_sharing(self):
        self.prepare()
        other = SwarmSearchEnv(network_config())
        with self.assertRaisesRegex(RuntimeError, "one.*process"):
            other.reset(seed=8)
        other.close()

    def test_six_uavs_keep_original_destination_index_mapping(self):
        c = Config()
        c.world.num_targets = 1
        c.world.num_obstacles = 0
        c.sensor.altitude_m = 20
        self.env = SwarmSearchEnv(c)
        self.env.reset(seed=44, options=dict(positions=[[500, 80], [550, 80], [450, 80],
                              [500, 130], [550, 130], [450, 130]],
                              targets=[[500, 80]], obstacles=[]))
        a = Action(np.zeros((6, 3)), np.array([6, 0, 0, 0, 0, 0]))
        self.env.step(a)
        self.assertTrue(self.env.reports.delivered[0])
        self.assertEqual(len(self.env.network.simulator.drones), 7)


if __name__ == "__main__":
    unittest.main()
