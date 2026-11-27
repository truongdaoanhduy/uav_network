"""Additional failure-first contract tests for the v5.4 CPU reference patch.

Run with: python3 -m unittest discover -s tests -p 'test_v54*.py' -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from zhang_xue.compat.env import EnvConfigV54, ZhangXue7NodePacketEnv, action_to_dest_radio_id
from zhang_xue.compat.network import (
    SevenRadioCsmaCaNetwork, RADIO_N, GCS_ID, GCS_POS_M, PACKET_TTL_US,
    Packet, PKT_DATA, pack_entry_2x_uint32
)


def near_pos():
    p = np.full((RADIO_N, 3), [950.0, 950.0, 50.0])
    p[0] = [120.0,120.0,50.0]
    p[GCS_ID] = GCS_POS_M
    return p

class Gate2Regressions(unittest.TestCase):
    def test_10_monotonic_hello_warmup_and_1s_step(self):
        env=ZhangXue7NodePacketEnv(seed=123)
        env.reset(seed=123)
        self.assertEqual(env.net.sim_time_us, 600_000)
        self.assertEqual(env.epoch_us, 600_000)
        v=np.zeros((6,2),dtype=float)
        env.step(v,np.zeros(6,dtype=np.int32))
        self.assertEqual(env.net.sim_time_us, 1_600_000)

    def test_11_invalid_comm_action_rejected_not_silently_mapped(self):
        env=ZhangXue7NodePacketEnv(seed=123)
        env.reset(seed=123)
        for bad in (-1,7,100):
            with self.assertRaises(ValueError):
                action_to_dest_radio_id(0,bad)
            with self.assertRaises(ValueError):
                env.step(np.zeros((6,2)),np.full(6,bad,dtype=np.int32))

    def test_12_pending_e2e_expiry_is_bounded(self):
        net=SevenRadioCsmaCaNetwork(seed=123)
        p=near_pos(); p[0]=[775.0,275.0,50.0]
        net.reset(p,episode_id=123)
        net.record_local_observation(0,23,True,1)
        for _ in range(20): net.enqueue_app_data(0,GCS_ID,net.get_dirty_entries_non_oracle(0,GCS_ID))
        self.assertEqual(len(net.e2e_pending),20)
        self.assertTrue(net.run_until_us(PACKET_TTL_US+1_000_000))
        self.assertEqual(len(net.e2e_pending),0)
        self.assertEqual(len(net.e2e_pending_expires_us),0)

    def test_13_final_receiver_app_dedup_across_hop_local_ids(self):
        net=SevenRadioCsmaCaNetwork(seed=123)
        net.reset(near_pos(),episode_id=123)
        net.run_until_us(600_000)
        net.record_local_observation(0,47,True,t_s=1)
        entry=pack_entry_2x_uint32(0,47,1,1,0)
        # Emulate two legitimate MAC-forwarded frames with distinct hop-local pkt IDs.
        for i in range(2):
            net.radios[0].active_next_hop = GCS_ID
            p=Packet(pkt_id=900+i,episode_id=123,pkt_type=PKT_DATA,app_src=0,app_dst=GCS_ID,
                     app_seq=55,create_us=net.sim_time_us,expire_us=net.sim_time_us+PACKET_TTL_US,
                     packed_entries=(entry,))
            net._handle_rx_frame(GCS_ID,prev_hop=0,pkt=p)
        self.assertEqual(net.stats['data_delivered_gcs'],1)
        self.assertGreaterEqual(net.stats['dedup_hits'],1)
        self.assertEqual(int(net.ledger_pos[GCS_ID,0,47]),1)

    def test_14_stale_reception_does_not_refresh_freshness_time(self):
        net=SevenRadioCsmaCaNetwork(seed=123)
        net.reset(near_pos(),episode_id=123)
        net.run_until_us(600_000)
        net.sim_time_us=2_000_000
        net.radios[0].active_next_hop=GCS_ID
        a=Packet(pkt_id=990,episode_id=123,pkt_type=PKT_DATA,app_src=0,app_dst=GCS_ID,
                 app_seq=100,create_us=0,expire_us=12_000_000,
                 packed_entries=(pack_entry_2x_uint32(0,47,2,1,0),))
        net._handle_rx_frame(GCS_ID,prev_hop=0,pkt=a)
        prev_rx = int(net.gcs_last_rx_us[47]); self.assertEqual(prev_rx,2_000_000)
        net.sim_time_us=3_000_000
        b=Packet(pkt_id=991,episode_id=123,pkt_type=PKT_DATA,app_src=0,app_dst=GCS_ID,
                 app_seq=101,create_us=0,expire_us=12_000_000,
                 packed_entries=(pack_entry_2x_uint32(0,47,1,1,0),))
        net._handle_rx_frame(GCS_ID,prev_hop=0,pkt=b)
        self.assertEqual(int(net.gcs_last_rx_us[47]),prev_rx)
        self.assertEqual(int(net.gcs_latest_u_s[47]),2)

    def test_15_global_stats_not_in_decentralized_actor_obs(self):
        env=ZhangXue7NodePacketEnv(seed=123)
        env.reset(seed=123)
        obs1,_,_=env._build_obs_state_mask()
        env.net.stats['e2e_ack_rx']=123456
        obs2,_,_=env._build_obs_state_mask()
        np.testing.assert_array_equal(obs1,obs2)

    def test_16_incomplete_event_window_must_not_mutate_rl_step(self):
        env=ZhangXue7NodePacketEnv(seed=123)
        env.reset(seed=123)
        prev=env.t
        net_step=env.net.run_until_us
        env.net.run_until_us = lambda target_us: False
        try:
            with self.assertRaises(RuntimeError):
                env.step(np.zeros((6,2)),np.zeros(6,dtype=np.int32))
            self.assertEqual(env.t,prev)
        finally:
            env.net.run_until_us=net_step

if __name__=='__main__': unittest.main()

class ControlPlaneRegressions(unittest.TestCase):
    def test_17_hello_not_blocked_by_no_route_data_head(self):
        net=SevenRadioCsmaCaNetwork(seed=1337)
        p=np.full((7,3),[960.0,960.0,50.0]);
        p[0]=[600,850,50];p[1]=[630,850,50];p[6]=GCS_POS_M
        net.reset(p,episode_id=1)
        # All initial UAV data are addressed to a currently unreachable GCS.
        for u in (0,1):
            net.record_local_observation(u,10+u,True,t_s=1)
            net.enqueue_app_data(u,GCS_ID,net.get_dirty_entries_non_oracle(u,GCS_ID))
        self.assertTrue(net.run_until_us(2_000_000))
        # Wireless peers MUST discover each other even with unrouteable DATA queued.
        self.assertIn(1,net.routes[0])
        self.assertIn(0,net.routes[1])

class MotionSafetyRegressions(unittest.TestCase):
    def test_19_initial_uavs_have_minimum_separation(self):
        env=ZhangXue7NodePacketEnv(seed=123)
        env.reset(seed=123)
        for i in range(6):
            for j in range(i+1,6):
                self.assertGreaterEqual(
                    np.linalg.norm(env.positions[i,:2]-env.positions[j,:2]),
                    env.cfg.uav_safe_distance_m)

    def test_20_simultaneous_headon_motion_respects_accel_and_separation(self):
        env=ZhangXue7NodePacketEnv(seed=123)
        env.reset(seed=123)
        # Only first pair: use a safe free-space region and close at 5 m/s each.
        env.positions[0,:2]=[450.0,200.0]
        env.positions[1,:2]=[505.0,200.0]
        for i in range(2,6):
            env.positions[i,:2]=[100.0+100.0*(i-2),800.0]
        env.net.update_positions(env.positions)
        before=env.positions[:2,:2].copy()
        cmds=np.zeros((6,2))
        cmds[0]=[25,0];cmds[1]=[-25,0]
        _,_,_,_,_,_,info=env.step(cmds,np.zeros(6,dtype=np.int32))
        # An instantaneous stop/reject from 25 m/s is not physically valid.
        # The shield must satisfy acceleration and the *swept* pairwise safety.
        displacement = env.positions[:2,:2] - before
        self.assertTrue(np.all(np.linalg.norm(displacement, axis=1) <= env.cfg.a_max_mps2 * env.cfg.dt_s + 1e-6))
        self.assertGreaterEqual(float(np.linalg.norm(env.positions[0,:2]-env.positions[1,:2])),
                                env.cfg.uav_safe_distance_m - 1e-6)
        self.assertGreaterEqual(info['rejected_motion_count'],1)
