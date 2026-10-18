"""Retry exhaustion releases failed fragments for the next policy action."""
import importlib.util
import unittest
import numpy as np

from uav_network.environment.env import Action, SwarmSearchEnv
import test_gpu_packets as packets


POSITIONS = [[500, 400], [900, 900]]


def retry_config(**network):
    return packets.config(peer_range_m=500, gcs_range_m=500,
                          tx_power_min_w=0.001, tx_power_mapping="log",
                          power_limited_admission=True, **network)


class NativeRetryRecoveryTests(unittest.TestCase):
    def prepare(self, config=None, positions=POSITIONS):
        env = SwarmSearchEnv(config or retry_config())
        self.addCleanup(env.close)
        env.reset(seed=44, options=dict(positions=positions,
                  targets=[positions[0]], obstacles=[]))
        return env

    def step(self, env, power, destination=2):
        continuous = np.zeros((2, 3))
        continuous[:, 2] = power
        return env.step(Action(continuous, np.array([destination, 0])))[-1]["network"]

    def test_next_policy_power_recovers_same_receiver_after_retry_exhaustion(self):
        env = self.prepare()
        failed = self.step(env, -1)
        self.assertEqual(failed["diagnostics"]["data_attempts"], 5)
        self.assertEqual(env.reports.buffers[0][0].created_s, 0)
        recovered = self.step(env, 1)
        self.assertTrue(env.reports.delivered[0])
        self.assertEqual(recovered["generated_packets"], 2)
        self.assertEqual(recovered["diagnostics"]["data_attempts"], 6)
        self.assertGreater(recovered["report_e2e_delay_s"], 1)
        self.assertLess(recovered["report_e2e_delay_s"], 2)
        expected_rf = (5 * 0.001 + 0.4) * (1024 * 8 + 464) / 2_000_000
        self.assertAlmostEqual(recovered["uav_communication_energy_j"], expected_rf)

    def test_same_power_restarts_once_per_action_and_silence_has_no_ghost_retry(self):
        env = self.prepare()
        for power, destination, attempts in [(-1, 2, 5), (-1, 2, 10),
                                              (-1, 0, 10), (-1, 2, 15)]:
            result = self.step(env, power, destination)
            self.assertEqual(result["diagnostics"]["data_attempts"], attempts)
            self.assertEqual(result["diagnostics"]["data_tx_power_w"]["max"], 0.001)
            self.assertEqual(env.reports.buffers[0][0].created_s, 0)
        result = self.step(env, 1)
        self.assertTrue(env.reports.delivered[0])
        self.assertEqual(result["generated_packets"], 4)
        self.assertEqual(result["diagnostics"]["data_attempts"], 16)

    def test_fragmented_report_recovers_all_failed_fragments(self):
        env = self.prepare(retry_config(report_bytes=2048))
        failed = self.step(env, -1)
        self.assertEqual(failed["generated_packets"], 2)
        self.assertEqual(failed["diagnostics"]["data_attempts"], 10)
        recovered = self.step(env, 1)
        self.assertTrue(env.reports.delivered[0])
        self.assertEqual(recovered["generated_packets"], 4)
        self.assertEqual(recovered["received_packets"], 2)
        self.assertEqual(recovered["diagnostics"]["data_attempts"], 12)
        self.assertEqual(env.reports.delivery_hops, [1])

    def test_recovery_preserves_a_fragment_received_before_the_link_changes(self):
        c = retry_config(report_bytes=2048)
        c.world.dt_s = 0.006
        env = self.prepare(c, [[500, 100], [900, 900]])
        first = self.step(env, -1)
        self.assertEqual(first["received_packets"], 1)
        # Change the radio fixture between fragments; the queued one keeps low power.
        env.positions[:] = POSITIONS
        for _ in range(12):
            failed = self.step(env, -1, destination=0)
        self.assertEqual(failed["diagnostics"]["data_attempts"], 6)
        self.assertEqual(env.reports.buffers[0][0].created_s, 0)
        recovered = self.step(env, 1)
        self.assertTrue(env.reports.delivered[0])
        self.assertEqual(recovered["generated_packets"], 3)
        self.assertEqual(recovered["received_packets"], 2)
        self.assertEqual(recovered["diagnostics"]["data_attempts"], 7)

    def test_retry_recovery_does_not_refresh_report_ttl(self):
        env = self.prepare(retry_config(report_ttl_s=1.5, rediscovery_recovery=False))
        self.step(env, -1)
        self.step(env, -1, destination=0)
        result = self.step(env, 1)
        self.assertFalse(env.reports.delivered[0])
        self.assertEqual(env.reports.expired, 1)
        self.assertEqual(result["generated_packets"], 1)
        self.assertEqual(result["diagnostics"]["data_attempts"], 5)


@unittest.skipUnless(importlib.util.find_spec("warp"), "optional warp_packet dependency not installed")
class WarpRetryRecoveryTests(unittest.TestCase):
    def compare(self, actions, config=None):
        wp, state, params, mac = packets.GPUPacketTests().compare(
            config or retry_config(), POSITIONS, actions)
        return state, mac.metrics_at(state, params)

    def test_next_policy_power_recovers_same_receiver_after_retry_exhaustion(self):
        state, result = self.compare([([2, 0], -1), ([2, 0], 1)])
        self.assertEqual(state.delivered.numpy().tolist(), [[1]])
        self.assertEqual(result["generated_packets"], 2)
        self.assertEqual(result["diagnostics"]["data_attempts"], 6)

    def test_same_power_restarts_once_per_action_and_silence_has_no_ghost_retry(self):
        state, result = self.compare(
            [([2, 0], -1), ([2, 0], -1), ([0, 0], -1),
             ([2, 0], -1), ([2, 0], 1)])
        self.assertEqual(state.delivered.numpy().tolist(), [[1]])
        self.assertEqual(result["generated_packets"], 4)
        self.assertEqual(result["diagnostics"]["data_attempts"], 16)

    def test_fragmented_report_recovers_all_failed_fragments(self):
        state, result = self.compare([([2, 0], -1), ([2, 0], 1)],
                                     retry_config(report_bytes=2048))
        self.assertEqual(state.delivered.numpy().tolist(), [[1]])
        self.assertEqual(result["generated_packets"], 4)
        self.assertEqual(result["received_packets"], 2)
        self.assertEqual(result["diagnostics"]["data_attempts"], 12)

    def test_recovery_preserves_a_fragment_received_before_the_link_changes(self):
        c = retry_config(report_bytes=2048)
        c.world.dt_s = 0.006
        probe = packets.GPUPacketTests()
        wp, state, params, mac = probe.prepare(c, [[500, 100], [900, 900]])
        first = probe.step(c, state, params, mac, wp, [2, 0], -1)
        self.assertEqual(first["received_packets"], 1)
        state.positions.assign(np.asarray([POSITIONS], dtype=np.float64))
        for _ in range(12):
            failed = probe.step(c, state, params, mac, wp, [0, 0], -1)
        self.assertEqual(failed["diagnostics"]["data_attempts"], 6)
        recovered = probe.step(c, state, params, mac, wp, [2, 0], 1)
        self.assertEqual(state.delivered.numpy().tolist(), [[1]])
        self.assertEqual(recovered["generated_packets"], 3)
        self.assertEqual(recovered["received_packets"], 2)
        self.assertEqual(recovered["diagnostics"]["data_attempts"], 7)

    def test_failed_sender_recovers_while_another_report_streams(self):
        c = retry_config(report_bytes=4096, buffer_bytes=8192, payload_bytes=512,
                         max_queue_packets=2, max_retries=1)
        c.world.num_targets = 2
        positions = [[500, 450], [500, 100]]
        actions = [([2, 2], [-1, 1]), ([2, 2], [1, 1])]
        wp, state, params, mac = packets.GPUPacketTests().compare(
            c, positions, actions, targets=positions)
        self.assertEqual(state.delivered.numpy().tolist(), [[1, 1]])
        result = mac.metrics_at(state, params)
        self.assertEqual(result["generated_packets"], 18)
        self.assertEqual(result["received_packets"], 16)
        self.assertEqual(result["diagnostics"]["data_attempts"], 18)

    def test_retry_recovery_does_not_refresh_report_ttl(self):
        state, result = self.compare([([2, 0], -1), ([0, 0], -1), ([2, 0], 1)],
            retry_config(report_ttl_s=1.5, rediscovery_recovery=False))
        self.assertEqual(state.delivered.numpy().tolist(), [[0]])
        self.assertEqual(result["generated_packets"], 1)
        self.assertEqual(result["diagnostics"]["data_attempts"], 5)
