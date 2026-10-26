"""Actual radio decisions and mission custody, native versus Warp CPU."""
import importlib.util
import unittest

import numpy as np

from uav_network.environment.env import Action, SwarmSearchEnv
import test_gpu_packets as packet_tests
from test_gpu_packets import config


class NetworkTelemetryTests(unittest.TestCase):
    def native(self, c=None, positions=None, targets=None, obstacles=None, seed=44):
        c = c or config()
        positions = positions or [[500, 100], [600, 100]]
        env = SwarmSearchEnv(c)
        self.addCleanup(env.close)
        env.reset(seed=seed, options=dict(positions=positions,
                  targets=targets if targets is not None else [positions[0]],
                  obstacles=obstacles if obstacles is not None else []))
        return env

    def step(self, env, dest, power=0):
        values = np.zeros((env.config.world.num_uavs, 3), dtype=np.float32)
        values[:, 2] = power
        return env.step(Action(values, np.asarray(dest, dtype=np.int64)))[-1]

    def test_actual_failed_emissions_not_action_selection_are_power_denominator(self):
        env = self.native(config(sinr_threshold_db=200))
        before = env.info()["network"]["diagnostics"]
        self.assertIsNone(before["data_tx_power_w"]["mean"])
        info = self.step(env, [2, 0], -1)
        d = info["network"]["diagnostics"]
        self.assertEqual(d["data_attempts"], env.config.network.max_retries)
        self.assertEqual(d["data_retries"], d["data_attempts"] - 1)
        self.assertEqual(d["data_phy_failures"], d["data_attempts"])
        self.assertEqual(d["data_weak_signal_failures"], d["data_attempts"])
        self.assertEqual(d["data_interference_failures"], 0)
        self.assertEqual(d["retry_limit"], 1)
        self.assertEqual(d["ack_attempts"], 0)
        self.assertIsNone(d["ack_tx_power_w"]["mean"])
        self.assertAlmostEqual(d["data_tx_power_w"]["mean"], 0.1)
        self.assertEqual(sum(d["data_tx_power_histogram"]), d["data_attempts"])
        self.assertEqual(d["data_decode_count"], d["data_snr_db"]["count"])
        self.assertEqual(d["data_peer_tx_power_w"]["count"], 0)
        self.assertEqual(d["data_gcs_tx_power_w"]["count"], d["data_attempts"])

    def test_v2_high_power_nominally_feasible_links_can_still_fail_from_interference(self):
        from uav_network.communication.link_budget import nominal_required_power_w
        c = config(cca_threshold_dbm=-30)
        c.world.num_targets = 2
        c.network.tx_power_min_w, c.network.tx_power_max_w = 0.001, 0.4
        c.network.tx_power_mapping = "log"
        c.network.power_limited_admission = True
        positions = [[500,100],[550,100]]
        targets = [positions[0],positions[1]]
        for source in positions:
            required = nominal_required_power_w(source, [500,0], [], c.network,
                                                c.sensor.altitude_m, True)
            self.assertLess(required, c.network.tx_power_max_w)
        env = self.native(c, positions=positions, targets=targets)
        d = self.step(env, [2,2], 1)["network"]["diagnostics"]
        self.assertGreater(d["data_interference_failures"], 0)
        self.assertGreater(d["data_attempts"], 0)
        self.assertAlmostEqual(d["data_tx_power_w"]["mean"], 0.4)

    def test_v2_attempts_record_required_power_margin_and_below_requirement(self):
        c = config()
        c.network.tx_power_min_w, c.network.tx_power_max_w = 0.001, 0.4
        c.network.tx_power_mapping = "log"
        c.network.gcs_range_m = 500
        env = self.native(c, positions=[[500,400],[900,900]], targets=[[500,400]])
        d = self.step(env, [2,0], -1)["network"]["diagnostics"]
        self.assertEqual(d["data_required_power_w"]["count"], d["data_attempts"])
        self.assertGreater(d["data_required_power_w"]["mean"], 0.001)
        self.assertLess(d["data_power_margin_db"]["mean"], 0.0)
        self.assertEqual(d["data_below_required_power_attempts"], d["data_attempts"])

    def test_v2_peer_relay_progress_counts_toward_and_away_but_not_gcs_delivery(self):
        cases = [([[500,300],[500,150]], "relay_toward_gcs_transfers", 1),
                 ([[500,150],[500,300]], "relay_away_from_gcs_transfers", -1)]
        for positions, key, sign in cases:
            with self.subTest(key=key):
                c = config(); c.world.num_targets = 2
                env = self.native(c, positions=positions, targets=[positions[0],[900,900]])
                d = self.step(env, [1,0])["network"]["diagnostics"]
                self.assertEqual(d["custody_peer_transfers"], 1)
                self.assertEqual(d[key], 1)
                self.assertEqual(np.sign(d["relay_gcs_distance_improvement_m"]["mean"]), sign)
                env.close()
        c = config(); c.world.num_targets = 2
        env = self.native(c, positions=[[500,100],[900,900]], targets=[[500,100],[900,900]])
        d = self.step(env, [2,0])["network"]["diagnostics"]
        self.assertEqual(d["direct_gcs_reports"], 1)
        self.assertEqual(d["relay_toward_gcs_transfers"], 0)
        self.assertEqual(d["relay_away_from_gcs_transfers"], 0)

    def test_conditional_delivery_hops_and_report_age_are_undefined_until_delivery(self):
        env = self.native()
        r = env.info()["report_diagnostics"]
        self.assertEqual(r["delivery_count"], 0)
        self.assertIsNone(r["delivery_hops_mean"])
        self.assertIsNone(r["delivered_report_age_mean_s"])
        self.step(env, [2, 0])
        r = env.info()["report_diagnostics"]
        self.assertEqual(r["delivery_count"], 1)
        self.assertEqual(r["delivery_hops_mean"], 1)
        self.assertAlmostEqual(r["delivered_report_age_mean_s"], env.reports.delivery_times_s[0])

    def test_energy_denial_is_counted_without_emitted_power_samples(self):
        c = config()
        c.energy.enabled = True
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        c.energy.battery_capacity_j = 1e-6
        env = self.native(c)
        d = self.step(env, [2, 0])["network"]["diagnostics"]
        self.assertGreater(d["data_energy_denials"], 0)
        self.assertEqual(d["energy_denials"], d["data_energy_denials"])
        self.assertEqual(d["data_attempts"], 0)
        self.assertEqual(d["data_tx_power_w"]["count"], 0)
        self.assertIsNone(d["data_tx_power_w"]["mean"])

    def test_silent_and_ineligible_slots_do_not_masquerade_as_opportunities(self):
        env = self.native(positions=[[500, 300], [500, 150]])
        d = self.step(env, [0, 2])["network"]["diagnostics"]
        self.assertEqual(d["action_slots"], 2)
        self.assertEqual(d["action_active"], 2)
        self.assertEqual(d["action_report_bearing"], 1)
        self.assertEqual(d["action_receiver_eligible"], 1)  # Peer, irrespective of chosen action.
        self.assertEqual(d["action_silent"], 1)
        self.assertEqual(d["action_no_opportunity"], 1)
        self.assertEqual(d["data_attempts"], 0)

    def test_relay_delivery_first_discovery_delay_and_cumulative_energy(self):
        c = config()
        c.world.num_targets = 2  # Keep the episode live after the first delivery.
        env = self.native(c, positions=[[500, 300], [500, 150]],
                          targets=[[500, 300], [900, 900]])
        self.step(env, [1, 0])
        info = self.step(env, [0, 2])
        d = info["network"]["diagnostics"]
        self.assertEqual((d["custody_peer_transfers"], d["direct_gcs_reports"],
                          d["relayed_gcs_reports"]), (1, 0, 1))
        self.assertEqual(d["delivery_hop_histogram"], [0, 1, 0, 0, 0, 0, 0, 0])
        r = info["report_diagnostics"]
        self.assertEqual(r["first_discovery_s"], [0., None])
        self.assertGreater(r["first_delivery_s"][0], 1.)
        self.assertEqual(r["first_discovery_to_delivery_s"][0], r["first_delivery_s"][0])
        self.assertIsNone(r["first_discovery_to_delivery_s"][1])
        self.assertEqual(r["discovered_undelivered"], 0)
        self.assertEqual(r["buffer_peak_bytes"], [1024, 1024])
        energy = info["energy"]
        np.testing.assert_allclose(energy["per_uav_total_energy_j"],
                                  np.asarray(energy["per_uav_propulsion_energy_j"]) +
                                  np.asarray(energy["per_uav_rf_energy_j"]), rtol=0, atol=1e-12)
        self.assertEqual(energy["per_uav_rf_energy_j"], env.network.energy_j[:-1].tolist())

    def test_expired_discovery_without_custody_and_queue_lifetime_decisions(self):
        env = self.native(config(report_ttl_s=0.0005))
        d = self.step(env, [2, 0])["network"]["diagnostics"]
        self.assertEqual(d["packet_lifetime"], 1)
        self.assertEqual(d["data_attempts"], 0)
        r = env.info()["report_diagnostics"]
        self.assertEqual((r["discovered_undelivered_with_custody"],
                          r["discovered_undelivered_without_custody"]), (0, 1))

    def test_native_reset_and_reading_telemetry_never_advance_mac_rng(self):
        env = self.native()
        rng_before = [drone.mac_protocol.rng_mac.getstate() for drone in env.network.simulator.drones]
        for _ in range(3):
            env.info()
        self.assertEqual(rng_before, [drone.mac_protocol.rng_mac.getstate()
                                     for drone in env.network.simulator.drones])
        first = self.step(env, [2, 0])
        env.reset(seed=44, options=dict(positions=[[500, 100], [600, 100]],
                                       targets=[[500, 100]], obstacles=[]))
        self.assertEqual(env.info()["network"]["diagnostics"]["data_attempts"], 0)
        self.assertEqual(self.step(env, [2, 0]), first)

    def test_existing_link_budget_calibration_seeded_los_nlos_and_range_admission(self):
        helper = packet_tests.GPUPacketTests()
        outcomes = []
        for blocked, power, distance, expected in ((False, -1, 100, True),
                                                   (True, -1, 100, False),
                                                   (True, 1, 100, True),
                                                   (False, 1, 501, False)):
            c = config()
            c.world.num_targets = 2
            c.world.num_obstacles = int(blocked)
            c.network.peer_range_m = c.network.gcs_range_m = 500
            positions = [[250, 500], [250 + distance, 500]]
            targets = [positions[0], [950, 950]]
            obstacles = [[300, 500, 20]] if blocked else []
            actions = [([1, 0], power)]
            native = helper.oracle(c, positions, actions, targets, obstacles, 44)[0]["diagnostics"]
            repeated = helper.oracle(c, positions, actions, targets, obstacles, 44)[0]["diagnostics"]
            self.assertEqual(native, repeated)
            self.assertEqual(native["data_phy_successes"] > 0, expected)
            if distance > 500:
                self.assertEqual(native["data_attempts"], 0)
                self.assertEqual(native["action_receiver_eligible"], 0)
            else:
                self.assertAlmostEqual(native["data_tx_power_w"]["mean"], 0.1 if power < 0 else 0.4)
                self.assertEqual(native["data_nlos_decodes"] > 0, blocked)
            if importlib.util.find_spec("warp"):
                wp, state, params, mac = helper.prepare(c, positions, targets, obstacles, 44)
                actual = helper.step(c, state, params, mac, wp, [1, 0], power)["diagnostics"]
                self.assertEqual(actual["data_attempts"], native["data_attempts"])
                self.assertEqual(actual["data_phy_successes"], native["data_phy_successes"])
            outcomes.append(native["data_phy_successes"] > 0)
        self.assertEqual(outcomes, [True, False, True, False])

    @unittest.skipUnless(importlib.util.find_spec("warp"), "optional Warp unavailable")
    def test_meaningful_native_warp_diagnostics_and_failure_partition(self):
        helper = packet_tests.GPUPacketTests()
        cases = [
            (config(), [[500, 100], [600, 100]], [([2, 0], 0)], None, None, 44),
            (config(sinr_threshold_db=200), [[500, 100], [600, 100]], [([2, 0], -1)], None, None, 44),
            (config(), [[500, 300], [500, 150]], [([1, 0], 0), ([0, 2], 0)], None, None, 44),
            (config(ack_uses_data_power=False, sinr_threshold_db=35),
             [[500, 100], [600, 100]], [([2, 0], 1)], None, None, 44),
            (config(report_bytes=2048, payload_bytes=512, max_queue_packets=1),
             [[500, 100], [600, 100]], [([2, 0], 0)] * 4, None, None, 44),
            (config(report_ttl_s=0.0005), [[500, 100], [600, 100]], [([2, 0], 0)], None, None, 44),
        ]
        collision = config(cca_threshold_dbm=-30)
        collision.world.num_targets = 2
        cases.append((collision, [[500, 100], [550, 100]], [([2, 2], 0)],
                      [[500, 100], [550, 100]], None, 44))
        nlos = config()
        nlos.world.num_obstacles = 1
        cases.append((nlos, [[500, 100], [600, 100]], [([2, 0], 0)], None, [[500, 50, 15]], 44))
        for c, positions, actions, targets, obstacles, seed in cases:
            with self.subTest(network=vars(c.network), actions=actions):
                expected = helper.oracle(c, positions, actions, targets, obstacles, seed)[0]["diagnostics"]
                wp, s, p, mac = helper.prepare(c, positions, targets, obstacles, seed)
                for destination, power in actions:
                    actual = helper.step(c, s, p, mac, wp, destination, power)["diagnostics"]
                    if int(s.terminated.numpy()[0]):
                        break
                for key, value in expected.items():
                    if isinstance(value, dict):
                        self.assertEqual(actual[key]["count"], value["count"], key)
                        for stat in ("sum", "mean", "min", "max"):
                            if value[stat] is None:
                                self.assertIsNone(actual[key][stat], key)
                            else:
                                self.assertAlmostEqual(actual[key][stat], value[stat], places=8, msg=key + stat)
                    else:
                        self.assertEqual(actual[key], value, key)
                self.assertEqual(actual["data_phy_failures"],
                                 actual["data_weak_signal_failures"] + actual["data_interference_failures"])
                self.assertEqual(actual["data_decode_count"],
                                 actual["data_los_decodes"] + actual["data_nlos_decodes"])
                self.assertEqual(actual["data_decode_count"],
                                 actual["data_peer_decodes"] + actual["data_gcs_decodes"])

    @unittest.skipUnless(importlib.util.find_spec("warp"), "optional Warp unavailable")
    def test_report_energy_summaries_and_rng_reads_match_native_after_relay_and_expiry(self):
        from uav_network.communication.gpu.metrics import world_info
        helper = packet_tests.GPUPacketTests()
        for ttl in (120, 0.0005):
            c = config(report_ttl_s=ttl)
            c.world.num_targets = 2
            c.energy.enabled = True
            c.energy.idle_power_w = c.energy.hover_power_w = 0
            positions = [[500, 300], [500, 150]]
            targets = [[500, 300], [900, 900]]
            native = self.native(c, positions, targets)
            self.step(native, [1, 0])
            expected = self.step(native, [0, 2])
            native.close()
            wp, state, params, mac = helper.prepare(c, positions, targets)
            helper.step(c, state, params, mac, wp, [1, 0])
            helper.step(c, state, params, mac, wp, [0, 2])
            rng_before = [state.mt.numpy().copy(), state.mt_index.numpy().copy(), state.pcg_state.numpy().copy()]
            actual = world_info(state, params)
            for _ in range(3):
                world_info(state, params)
            for left, right in zip(rng_before, (state.mt.numpy(), state.mt_index.numpy(), state.pcg_state.numpy())):
                np.testing.assert_array_equal(left, right)
            for group in ("report_diagnostics", "energy"):
                for key, expected_value in expected[group].items():
                    actual_value = actual[group][key]
                    if isinstance(expected_value, list):
                        for left, right in zip(actual_value, expected_value):
                            if right is None:
                                self.assertIsNone(left, group + key)
                            else:
                                self.assertAlmostEqual(left, right, places=10, msg=group + key)
                    elif expected_value is None:
                        self.assertIsNone(actual_value, group + key)
                    else:
                        self.assertAlmostEqual(actual_value, expected_value, places=10, msg=group + key)

    @unittest.skipUnless(importlib.util.find_spec("warp"), "optional Warp unavailable")
    def test_warp_partial_reset_preserves_other_world_and_resets_every_telemetry_array(self):
        import warp as wp
        from uav_network.environment.gpu.state import allocate_state
        from uav_network.communication.gpu.mac import reset_network
        s, p = allocate_state(config(), 2, seeds=[44, 45])
        for name in ("diagnostic_counts", "diagnostic_stats", "power_histogram",
                     "delivery_hop_histogram", "first_discovery", "first_delivery", "buffer_peak"):
            value = getattr(s, name)
            value.fill_(7)
        wp.launch(reset_network, dim=2, inputs=[s, p, wp.array([1, 0], dtype=wp.int32, device="cpu")], device="cpu")
        for name in ("diagnostic_counts", "diagnostic_stats", "power_histogram",
                     "delivery_hop_histogram", "buffer_peak"):
            self.assertTrue(np.all(getattr(s, name).numpy()[0] == 0), name)
            self.assertTrue(np.all(getattr(s, name).numpy()[1] == 7), name)
        self.assertTrue(np.all(s.first_discovery.numpy()[0] == -1))
        self.assertTrue(np.all(s.first_delivery.numpy()[0] == -1))
        wp.launch(reset_network, dim=2, inputs=[s, p, wp.array([1, 1], dtype=wp.int32, device="cpu")], device="cpu")
        self.assertTrue(np.all(s.diagnostic_counts.numpy() == 0))
        self.assertTrue(np.all(s.first_discovery.numpy() == -1))
        self.assertTrue(np.all(s.first_delivery.numpy() == -1))


if __name__ == "__main__":
    unittest.main()
