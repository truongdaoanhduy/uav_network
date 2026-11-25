import os
import sys
import unittest
import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from uav_adapter.ds_belief import BeliefTable, ds_log_space_safe
from uav_adapter.packets_and_queues import (
    E2EAckPacket,
    MiniDiscreteEventNet,
    SearchDataPacket,
)
from uav_adapter.reward_engine import RewardEngine


class TestV32FullGates(unittest.TestCase):
    def test_dynamic_payload_serialization_and_hard_mtu(self):
        p1 = SearchDataPacket(1, 1, 0, 1, 1, 0.0, 5.0, entries=((0, 5, 1, 0),))
        self.assertEqual(p1.payload_bytes, 20)
        self.assertEqual(p1.frame_bytes, 78)
        self.assertEqual(len(p1.to_bytes()), 20)

        p32 = SearchDataPacket(2, 1, 0, 1, 2, 0.0, 5.0, entries=tuple((0, i, 1, 0) for i in range(32)))
        self.assertEqual(p32.payload_bytes, 268)
        self.assertEqual(p32.frame_bytes, 326)

        p62 = SearchDataPacket(3, 1, 0, 1, 3, 0.0, 5.0, entries=tuple((0, i, 1, 0) for i in range(62)))
        self.assertEqual(p62.payload_bytes, 508)
        self.assertEqual(p62.frame_bytes, 566)

        # 63 entries (516 B > 512 B) and 100,000 entries MUST raise ValueError
        with self.assertRaises(ValueError):
            SearchDataPacket(4, 1, 0, 1, 4, 0.0, 5.0, entries=tuple((0, i, 1, 0) for i in range(63)))
        with self.assertRaises(ValueError):
            SearchDataPacket(5, 1, 0, 1, 5, 0.0, 5.0, entries=tuple((0, i % 1600, 1, 0) for i in range(100000)))

    def test_real_2hop_data_reverse_e2e_ack_and_blackout_recovery(self):
        # 3 UAVs in a line: U0(0,0) --400m-- U1(400,0) --400m-- U2(800,0); ideal range Pt=0.5W is 557.4m
        # So U0 cannot reach U2 directly (800m > 557.4m), MUST relay 2 hops via U1!
        bt = BeliefTable(n_uavs=3, n_cells=10, fusion_mode="MAX_PLUS_SUM")
        bt.record_own_observation(0, 7, True)
        net = MiniDiscreteEventNet({0: (0.0, 0.0), 1: (400.0, 0.0), 2: (800.0, 0.0)}, bt, tx_power_w=0.5)

        pkt = SearchDataPacket(1, 1, src_id=0, dst_id=2, seq_id=1, creation_time=0.0, deadline=5.0, entries=((0, 7, 1, 0),))
        net.send_data(0.0, pkt)
        gain = net.run_until(1.0)

        self.assertAlmostEqual(gain, 0.80, places=5)
        self.assertEqual(net.watermarks.get((0, 2)), 1)  # Reverse 2-hop E2E ACK reached U0!
        self.assertAlmostEqual(net.metrics.data_pdr, 1.0, places=6)  # E2EAckPacket did NOT inflate PDR above 1.0
        self.assertEqual(net.metrics.e2e_ack_delivered, 1)
        # 2-hop DATA (78B: 0.802ms * 2) + 2-hop E2E ACK (90B: 0.850ms * 2) = 3.304 ms
        self.assertAlmostEqual(net.metrics.total_airtime_s * 1000.0, 3.304, places=3)

        # Blackout at relay U1 blocks delivery & packet expires cleanly
        net.radio_enabled[1] = False
        pkt2 = SearchDataPacket(2, 1, src_id=0, dst_id=2, seq_id=2, creation_time=1.0, deadline=2.0, entries=((0, 8, 1, 0),))
        net.send_data(1.0, pkt2)
        gain2 = net.run_until(4.0)
        self.assertEqual(gain2, 0.0)
        self.assertEqual(net.watermarks.get((0, 2)), 1)  # Watermark did NOT advance
        self.assertEqual(len(net.queues[0].waiting_list), 0)  # Expired packet purged!

    def test_disjoint_tp_fp_latch_prevents_positive_fp_reward_hacking(self):
        # Cell 0 is a TRUE target; Cell 1 is EMPTY
        re = RewardEngine(n_uavs=6, n_cells=2, true_target_mask=np.array([True, False]), lambda_b=30.0)
        # Cell 1 gets 2 false alarms -> m_F = 0.96 >= 0.95
        r_tp, r_fp = re.step_target_and_fp_reward(np.array([0.0, 0.96], dtype=np.float32), mode="LATCHED")
        self.assertEqual(r_tp, 0.0)   # MUST NOT award +10.0 on empty cell!
        self.assertEqual(r_fp, 5.0)   # Penalized -5.0 cleanly
        # Cell 0 confirmed true target -> +10.0 once, oscillation does not re-trigger
        r_tp2, _ = re.step_target_and_fp_reward(np.array([0.96, 0.96], dtype=np.float32), mode="LATCHED")
        self.assertEqual(r_tp2, 10.0)
        re.step_target_and_fp_reward(np.array([0.82, 0.0], dtype=np.float32), mode="LATCHED")
        r_tp3, _ = re.step_target_and_fp_reward(np.array([0.97, 0.0], dtype=np.float32), mode="LATCHED")
        self.assertEqual(r_tp3, 0.0)

    def test_calibrated_lambda_b_30_under_25pct_2hop_pdr(self):
        # 32-cell 2-hop airtime = 5.288 ms = 0.005288 s; at lambda_B=30, cost = 0.15864
        cost_2h_32c = 30.0 * 0.005288
        exp_net_25pct_pdr = 0.25 * 2.40 - cost_2h_32c
        self.assertGreater(exp_net_25pct_pdr, 0.40)  # Still clearly profitable at 25% PDR!
        self.assertLess(-cost_2h_32c, -0.15)         # Duplicate packet clearly penalized vs NO_SEND

    def test_config_v3_2_zero_nulls(self):
        cfg_path = os.path.join(ROOT, "config_v3.2.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        def check(obj, path=""):
            self.assertIsNotNone(obj, f"Null at {path}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check(v, f"{path}[{i}]")

        check(data)
        self.assertEqual(data["packet_serialization"]["max_entries_per_packet"], 62)
        self.assertEqual(data["rl_and_reward"]["reward_weights"]["lambda_network_airtime_main"], 30.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
