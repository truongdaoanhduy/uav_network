"""CPU contention and HELLO airtime must follow the active Warp contracts."""
from __future__ import annotations

import unittest

import numpy as np

from zhang_xue.communication.common import (
    HELLO_FRAME_BYTES, INF_TIME_US, PKT_DATA, PKT_E2E_ACK, PKT_HELLO_DV,
    Packet, ST_BACKOFF, ST_DIFS, ST_TX_DATA, US_PER_BYTE,
)
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.gpu.state_v67 import HELLO_V89_TX_US


def _contention(*, difs=False, hidden=False, second_deadline=100):
    net = SevenRadioCsmaCaNetwork(
        num_uavs=2, buildings=[], blackout_centers=[], seed=44,
    )
    net.hello_next_us = [INF_TIME_US] * net.radio_n
    net.chan_gain.fill(1.0e-4)
    np.fill_diagonal(net.chan_gain, 0.0)
    if hidden:
        net.chan_gain[0, 1] = net.chan_gain[1, 0] = 0.0
    for tx, deadline in ((0, 100), (1, second_deadline)):
        r = net.radios[tx]
        r.active_pkt = Packet(
            tx + 1, 0, PKT_HELLO_DV, tx, -1, 0, 0, 500_000,
        )
        r.mac_state = ST_DIFS if difs else ST_BACKOFF
        r.state_end_us = deadline
        r.backoff_frozen = difs
        r.backoff_rem_us = 0
    return net


class TestV121NetworkParity(unittest.TestCase):
    def _assert_equal_slot_collision(self, *, difs=False, hidden=False):
        net = _contention(difs=difs, hidden=hidden)
        self.assertTrue(net.run_until_us(101))
        self.assertEqual([r.mac_state for r in net.radios[:2]], [ST_TX_DATA] * 2)
        self.assertGreater(net.radios[net.gcs_id].rx_max_interf_w, 0.0)
        self.assertTrue(net.run_until_us(501))
        self.assertEqual(net.routes[net.gcs_id], {}, "Equal-power HELLO collision must not create a route")
        self.assertEqual(net.step_tx_airtime_us[:2].tolist(), [360, 360])

    def test_equal_backoff_slot_is_a_collision(self):
        self._assert_equal_slot_collision()

    def test_equal_zero_backoff_after_difs_is_a_collision(self):
        self._assert_equal_slot_collision(difs=True)

    def test_hidden_equal_slot_still_collides_at_common_receiver(self):
        self._assert_equal_slot_collision(hidden=True)

    def test_later_counter_freezes_and_resumes_its_remaining_slot(self):
        net = _contention(second_deadline=120)
        self.assertTrue(net.run_until_us(101))
        self.assertEqual(net.radios[0].mac_state, ST_TX_DATA)
        self.assertNotEqual(net.radios[1].mac_state, ST_TX_DATA)
        self.assertTrue(net.radios[1].backoff_frozen)
        self.assertEqual(net.radios[1].backoff_rem_us, 20)
        # First HELLO ends at 460; the second waits DIFS 50 plus the frozen 20.
        self.assertTrue(net.run_until_us(531))
        self.assertEqual(net.radios[1].mac_state, ST_TX_DATA)
        self.assertEqual(net.radios[1].state_end_us, 890)

    def test_later_difs_restarts_after_busy_channel(self):
        net = _contention(second_deadline=120)
        net.radios[1].mac_state = ST_DIFS
        self.assertTrue(net.run_until_us(101))
        self.assertNotEqual(net.radios[1].mac_state, ST_DIFS)
        self.assertFalse(net.radios[1].backoff_frozen)
        self.assertTrue(net.run_until_us(461))
        self.assertEqual(net.radios[1].mac_state, ST_DIFS)
        self.assertEqual(net.radios[1].state_end_us, 510)

    def test_schema_v2_hello_includes_its_existing_wire_telemetry_on_every_radio(self):
        net = SevenRadioCsmaCaNetwork(obs_schema_v2=True)
        net.hello_next_us = [INF_TIME_US] * net.radio_n
        for tx in range(net.radio_n):
            with self.subTest(tx=tx):
                net._enqueue_hello_dv(tx)
                pkt = net.radios[tx].pending_hello
                self.assertIsNotNone(pkt.peer_pose)
                self.assertEqual(pkt.tx_duration_us, HELLO_V89_TX_US)
                self.assertEqual(pkt.frame_bytes * US_PER_BYTE, HELLO_V89_TX_US)
                self.assertTrue(net.run_until_us(net.sim_time_us + 10_000))
                self.assertEqual(int(net.step_tx_airtime_us[tx]), HELLO_V89_TX_US)

    def test_legacy_hello_data_and_e2e_ack_airtimes_are_unchanged(self):
        net = SevenRadioCsmaCaNetwork(obs_schema_v2=False)
        net._enqueue_hello_dv(0)
        self.assertEqual(net.radios[0].pending_hello.frame_bytes, HELLO_FRAME_BYTES)
        self.assertEqual(net.radios[0].pending_hello.tx_duration_us, 360)
        for count in (1, 62):
            pkt = Packet(1, 0, PKT_DATA, 0, net.gcs_id, 1, 0, 10_000_000,
                         packed_entries=((0, 0),) * count)
            self.assertEqual(pkt.frame_bytes, 70 + 8 * count)
        ack = Packet(2, 0, PKT_E2E_ACK, net.gcs_id, 0, 1, 0, 10_000_000)
        self.assertEqual(ack.tx_duration_us, 360)


if __name__ == "__main__":
    unittest.main()
