"""Battery budgets affect real movement, sensing and packet transmission."""
import unittest

import numpy as np

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.communication.link_budget import report_reference_tx_energy_j


def config(network=False, capacity=1000):
    c = Config()
    c.world.num_uavs = 2
    c.world.num_targets = 1
    c.world.num_obstacles = 0
    c.sensor.altitude_m = 20
    c.network.enabled = network
    c.energy.battery_capacity_j = capacity
    return c


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.env = None

    def tearDown(self):
        if self.env:
            self.env.close()

    def prepare(self, c, positions=None, target=None):
        self.env = SwarmSearchEnv(c)
        self.env.reset(seed=44, options={
            "positions": positions or [[500, 100], [600, 100]],
            "targets": [target or [900, 900]], "obstacles": []})
        return self.env

    def test_report_reference_energy_matches_data_fragment_airtime_at_pmax(self):
        c = Config()
        c.network.report_bytes = 2048
        c.network.payload_bytes = 512
        c.network.bit_rate_bps = 2_000_000.0
        c.network.tx_power_max_w = 0.4
        expected = 4 * (512 * 8 + 464) / 2_000_000.0 * 0.4
        self.assertAlmostEqual(report_reference_tx_energy_j(c.network), expected, places=15)

    def test_radio_hardware_config_validation_and_reference_formula(self):
        c = Config()
        for name, value in (("pa_efficiency", 0.0), ("pa_efficiency", 1.01),
                            ("tx_circuit_power_w", -0.1), ("rx_circuit_power_w", -0.1)):
            with self.subTest(name=name, value=value):
                bad = Config()
                setattr(bad.network, name, value)
                with self.assertRaisesRegex(ValueError, "radio|efficiency|circuit"):
                    bad.validate()
        c.network.report_bytes = 2048
        c.network.payload_bytes = 512
        c.network.bit_rate_bps = 2_000_000.0
        c.network.tx_power_max_w = 0.4
        c.network.pa_efficiency = 0.5
        c.network.tx_circuit_power_w = 0.2
        c.network.rx_circuit_power_w = 0.1
        airtime = 4 * (512 * 8 + 464) / 2_000_000.0
        self.assertAlmostEqual(report_reference_tx_energy_j(c.network),
                               airtime * (0.2 + 0.4 / 0.5), places=15)

    def test_native_radio_energy_splits_tx_rx_and_battery_uses_total_uav_radio(self):
        c = config(network=True, capacity=10)
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        c.network.tx_power_min_w = c.network.tx_power_max_w = 0.4
        c.network.pa_efficiency = 0.5
        c.network.tx_circuit_power_w = 0.2
        c.network.rx_circuit_power_w = 0.1
        env = self.prepare(c, positions=[[500,100],[900,900]], target=[500,100])
        before = env.battery_j.copy()
        _, _, _, _, info = env.step(Action(np.zeros((2, 3)), np.array([2,0])))
        net = info["network"]
        self.assertGreater(net["uav_tx_energy_j"], 0)
        self.assertGreater(net["uav_rx_energy_j"], 0)  # ACK reception
        self.assertGreater(net["gcs_tx_energy_j"], 0)  # ACK transmission
        self.assertGreater(net["gcs_rx_energy_j"], 0)  # DATA reception
        self.assertAlmostEqual(net["uav_communication_energy_j"],
                               net["uav_tx_energy_j"] + net["uav_rx_energy_j"], places=12)
        self.assertAlmostEqual(net["gcs_communication_energy_j"],
                               net["gcs_tx_energy_j"] + net["gcs_rx_energy_j"], places=12)
        self.assertAlmostEqual(float((before-env.battery_j).sum()),
                               net["uav_communication_energy_j"], places=10)

    def test_failed_decode_consumes_receiver_listening_energy(self):
        c = config(network=True, capacity=10)
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        c.network.sinr_threshold_db = 200
        c.network.pa_efficiency = 0.5
        c.network.tx_circuit_power_w = 0.2
        c.network.rx_circuit_power_w = 0.1
        env = self.prepare(c, positions=[[500,100],[600,100]], target=[500,100])
        before = env.battery_j.copy()
        _, _, _, _, info = env.step(Action(np.zeros((2, 3)), np.array([1,0])))
        net = info["network"]
        self.assertGreater(net["phy_failures"], 0)
        self.assertGreater(net["uav_tx_energy_j"], 0)
        self.assertGreater(net["uav_rx_energy_j"], 0)
        self.assertFalse(env.reports.delivered[0])
        self.assertGreater(before[1]-env.battery_j[1], 0)

    def test_communication_efficiency_changes_reward_not_physical_battery(self):
        def run(weight):
            c = config(network=True, capacity=10)
            c.energy.idle_power_w = c.energy.hover_power_w = 0
            c.reward.communication_efficiency = weight
            env = self.prepare(c, positions=[[500,100],[900,900]], target=[500,100])
            action = Action(np.array([[0.,0.,1.],[0.,0.,0.]]), np.array([2,0]))
            _, reward, _, _, info = env.step(action)
            result = (reward, env.battery_j.copy(), info["network"]["uav_communication_energy_j"],
                      dict(info["reward_components"]), report_reference_tx_energy_j(c.network))
            env.close(); self.env = None
            return result
        base = run(0.0)
        shaped = run(1.0)
        np.testing.assert_allclose(base[1], shaped[1], rtol=0, atol=1e-12)
        self.assertAlmostEqual(base[2], shaped[2], places=15)
        expected_cost = shaped[2] / shaped[4]
        self.assertAlmostEqual(shaped[0] - base[0], -expected_cost, places=10)
        self.assertAlmostEqual(shaped[3]["communication_efficiency"], -expected_cost, places=10)
        self.assertEqual(base[3]["communication_efficiency"], 0.0)

    def test_retry_rf_energy_is_fully_charged_to_normalized_communication_cost(self):
        c = config(network=True, capacity=10)
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        c.network.tx_power_min_w, c.network.tx_power_max_w = 0.001, 0.4
        c.network.tx_power_mapping = "log"
        c.network.power_limited_admission = True
        c.reward.communication_efficiency = 1.0
        env = self.prepare(c, positions=[[500,400],[900,900]], target=[500,400])
        action = Action(np.array([[0.,0.,-1.],[0.,0.,0.]]), np.array([2,0]))
        _, _, _, _, info = env.step(action)
        rf = info["network"]["uav_communication_energy_j"]
        d = info["network"]["diagnostics"]
        self.assertGreater(d["data_retries"], 0)
        self.assertGreater(rf, 0)
        expected = -rf / report_reference_tx_energy_j(c.network)
        self.assertAlmostEqual(info["reward_components"]["communication_efficiency"], expected, places=10)

    def test_realized_motion_debits_propulsion_and_observation_charge(self):
        env = self.prepare(config())
        obs, _, _, _, info = env.step(Action(np.array([[1., 0, 0], [0, 0, 0]]),
                                           np.array([0, 0])))
        np.testing.assert_allclose(env.positions[0], [502, 100])
        # (20 idle + 120 hover + .5 * speed^2 + 2 * accel^2) * 1 second.
        np.testing.assert_allclose(env.battery_j, [850, 860])
        np.testing.assert_allclose(obs[:, 8], [.85, .86])
        self.assertEqual(info["energy"]["propulsion_energy_j"], 290)
        self.assertEqual(info["energy"]["active_uavs"], 2)

    def test_unfunded_step_stops_and_all_empty_is_terminal_failure(self):
        env = self.prepare(config(capacity=100))
        start = env.positions.copy()
        _, _, terminated, truncated, info = env.step(
            Action(np.ones((2, 3)), np.array([0, 0])))
        np.testing.assert_array_equal(env.positions, start)
        np.testing.assert_array_equal(env.battery_j, [0, 0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])
        self.assertEqual(info["termination_reason"], "battery_exhausted")

    def test_safety_is_rechecked_against_peer_stopped_by_battery(self):
        env = self.prepare(config(), positions=[[100, 500], [130, 500]])
        env.battery_j[0] = 1
        env.velocities[:] = [[20, 0], [-20, 0]]
        env.step(Action(np.zeros((2, 3)), np.array([0, 0])))
        np.testing.assert_allclose(env.positions, [[100, 500], [130, 500]])
        self.assertEqual(env.battery_j[0], 0)
        self.assertGreater(env.battery_j[1], 0)

    def test_dead_uav_cannot_sense_move_or_choose_destination(self):
        env = self.prepare(config())
        env.battery_j[0] = 0
        env.targets[0] = env.positions[0]
        obs, _, terminated, _, _ = env.step(Action(np.ones((2, 3)), np.array([0, 0])))
        self.assertFalse(env.reports.known[0, 0])
        np.testing.assert_allclose(env.positions[0], [500, 100])
        self.assertFalse(terminated)
        np.testing.assert_array_equal(obs[0, -3:], [1, 0, 0])
        self.assertEqual(obs[0, 9], 0)

    def test_native_rf_attempt_is_debited_exactly_once_and_gcs_is_separate(self):
        c = config(network=True, capacity=10)
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        env = self.prepare(c, target=[500, 100])
        before = env.battery_j.copy()
        _, _, terminated, _, info = env.step(Action(np.zeros((2, 3)), np.array([2, 0])))
        self.assertTrue(terminated)
        stats = info["network"]
        self.assertGreater(stats["uav_communication_energy_j"], 0)
        self.assertGreater(stats["gcs_communication_energy_j"], 0)
        self.assertAlmostEqual(float((before - env.battery_j).sum()),
                               stats["uav_communication_energy_j"], places=10)

    def test_native_phy_cannot_emit_an_unaffordable_frame(self):
        c = config(network=True, capacity=1e-7)
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        env = self.prepare(c, target=[500, 100])
        _, _, _, _, info = env.step(Action(np.zeros((2, 3)), np.array([2, 0])))
        self.assertEqual(info["network"]["uav_communication_energy_j"], 0)
        self.assertEqual(info["network"]["phy_successes"], 0)
        self.assertGreater(info["network"]["energy_blocked_transmissions"], 0)
        self.assertFalse(env.reports.delivered[0])
        self.assertIn(0, env.reports.buffers[0])
        np.testing.assert_allclose(env.battery_j, [1e-7, 1e-7])

    def test_dead_peer_cannot_acquire_report_custody(self):
        env = self.prepare(config(network=True), positions=[[500, 300], [500, 150]],
                           target=[500, 300])
        env.battery_j[1] = 0
        obs, _, _, _, _ = env.step(Action(np.zeros((2, 3)), np.array([1, 0])))
        self.assertIn(0, env.reports.buffers[0])
        self.assertFalse(env.reports.buffers[1])
        self.assertEqual(obs[0, -2], 0)
        self.assertEqual(env.network.metrics()["generated_packets"], 0)

    def test_energy_and_runtime_config_errors_are_explicit(self):
        c = Config()
        c.energy.battery_capacity_j = -1
        with self.assertRaises(ValueError):
            c.validate()


if __name__ == "__main__":
    unittest.main()
