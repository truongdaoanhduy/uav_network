"""Regressions discovered while reviewing the v3.2 reference simulator.

These tests intentionally exercise the reference adapter only: they are
not proof of SimPy UavNetSim <-> Warp parity or GPU training correctness.
"""
import unittest

import numpy as np

from uav_adapter.ds_belief import BeliefTable
from uav_adapter.packets_and_queues import (
    MiniDiscreteEventNet,
    PacketQueueManager,
    SearchDataPacket,
)
from uav_adapter.reward_engine import RewardEngine


def make_network():
    belief = BeliefTable(3, 12, fusion_mode="MAX_PLUS_SUM")
    positions = {0: (0., 0.), 1: (400., 0.), 2: (800., 0.)}
    return MiniDiscreteEventNet(positions, belief, tx_power_w=0.5)


def packet(packet_id, seq=None, cell=7, now=0.):
    return SearchDataPacket(
        packet_id=packet_id, episode_id=1, src_id=0, dst_id=2,
        seq_id=packet_id if seq is None else seq,
        creation_time=now, deadline=5.0,
        entries=((0, cell, 1, 0),),
    )


class TestV34Regressions(unittest.TestCase):
    def test_queue_full_does_not_transmit_rejected_packet(self):
        net = make_network()
        net.queues[0] = PacketQueueManager(max_size=1)
        self.assertTrue(net.send_data(0., packet(1, cell=7)))
        self.assertFalse(net.send_data(0., packet(2, cell=8)))
        net.run_until(1.)
        self.assertEqual(net.watermarks.get((0, 2)), 1)
        self.assertEqual(net.belief.pos[2, 0, 7], 1)
        self.assertEqual(net.belief.pos[2, 0, 8], 0)
        self.assertEqual(net.metrics.data_generated, 2)
        self.assertAlmostEqual(net.metrics.data_pdr, 0.5)

    def test_no_route_waiting_list_retries_after_radio_recovery(self):
        net = make_network()
        net.radio_enabled[1] = False
        self.assertTrue(net.send_data(0., packet(3)))
        net.run_until(0.2)
        self.assertEqual(net.watermarks.get((0, 2)), None)
        self.assertEqual(len(net.queues[0].waiting_list), 1)
        net.radio_enabled[1] = True
        net.run_until(1.)
        self.assertEqual(net.watermarks.get((0, 2)), 3)
        self.assertEqual(len(net.queues[0].waiting_list), 0)

    def test_sender_blackout_keeps_packet_for_later_retry(self):
        net = make_network()
        net.radio_enabled[0] = False
        self.assertTrue(net.send_data(0., packet(4)))
        net.run_until(0.1)
        self.assertEqual(net.watermarks.get((0, 2)), None)
        net.radio_enabled[0] = True
        net.run_until(1.)
        self.assertEqual(net.watermarks.get((0, 2)), 4)

    def test_successful_transmissions_release_queue_capacity(self):
        net = make_network()
        net.queues[0] = PacketQueueManager(max_size=2)
        for i in range(1, 8):
            now = i * 0.3
            self.assertTrue(net.send_data(now, packet(i, now=now, cell=i)))
            net.run_until(now + 0.1)
        self.assertEqual(len(net.queues[0].transmitting_queue), 0)
        self.assertEqual(net.metrics.data_generated, 7)
        self.assertAlmostEqual(net.metrics.data_pdr, 1.)

    def test_safety_and_boundary_penalty_are_costs_not_bonuses(self):
        reward = RewardEngine(3, 2, np.array([True, False]))
        mf = np.zeros(2, np.float32)
        value = reward.compute_unified_step_reward(
            0., 0., mf, 0., safety_penalty=2., boundary_penalty=3.)
        self.assertAlmostEqual(value, -5.)
        with self.assertRaises(ValueError):
            reward.compute_unified_step_reward(
                0., 0., mf, 0., safety_penalty=-1.)

    def test_false_positive_information_has_no_positive_credit(self):
        reward = RewardEngine(6, 2, np.array([True, False]))
        score = reward.compute_unified_step_reward(
            np.array([0., .2], dtype=np.float32),
            np.array([0., .3], dtype=np.float32),
            np.array([0., .96], dtype=np.float32),
            airtime_fraction=0.,
        )
        self.assertAlmostEqual(float(score), -5.)

    def test_true_target_information_and_latches_reward_once(self):
        reward = RewardEngine(6, 2, np.array([True, False]))
        search = np.array([.2, .3], np.float32)
        share = np.array([.4, .5], np.float32)
        mf = np.array([.96, .96], np.float32)
        value = reward.compute_unified_step_reward(search, share, mf, 0.)
        self.assertAlmostEqual(float(value), 12.2, places=5)
        second = reward.compute_unified_step_reward(
            np.zeros(2), np.zeros(2), mf, 0.)
        self.assertEqual(float(second), 0.)

    def test_data_packet_rejects_mutable_or_out_of_wire_range_fields(self):
        with self.assertRaises((TypeError, ValueError)):
            SearchDataPacket(70, 1, 0, 2, 70, 0., 5.,
                             entries=[(0, 7, 1, 0)])
        with self.assertRaises(ValueError):
            SearchDataPacket(71, 1, 0, 2, 71, 0., 5.,
                             entries=((0, 7, 65536, 0),))
        with self.assertRaises(ValueError):
            SearchDataPacket(72, 1, 0, 2, 72, 0., -1.,
                             entries=((0, 7, 1, 0),))

    def test_source_must_send_to_a_different_valid_peer(self):
        net = make_network()
        self_packet = SearchDataPacket(
            73, 1, src_id=0, dst_id=0, seq_id=73,
            creation_time=0., deadline=2., entries=((0, 7, 1, 0),))
        with self.assertRaises(ValueError):
            net.send_data(0., self_packet)


if __name__ == "__main__":
    unittest.main()
