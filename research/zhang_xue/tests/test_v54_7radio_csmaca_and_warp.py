"""9-test original verification suite for v5.4 7-Radio/GCS CSMA/CA, DS, Battery, AoI & Warp SoA."""

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zhang_xue.compat.env import EnvConfigV54, ZhangXue7NodePacketEnv
from zhang_xue.compat.network import (
    ACTOR_N,
    GCS_ID,
    GCS_POS_M,
    MAX_RETX,
    PACKET_TTL_US,
    QUEUE_CAP_PKTS,
    RADIO_N,
    SevenRadioCsmaCaNetwork,
    pack_entry_2x_uint32,
    unpack_entry_2x_uint32,
)
from zhang_xue.compat.belief_mobility import (
    ZengBatteryConfig,
    clamp_velocity_with_accel,
    counts_to_canonical_ds_stable,
    uav_step_energy_j,
    zeng_propulsion_power_w,
)
from zhang_xue.gpu.kernels.scaffold import compute_exact_80array_soa_bytes


class TestV54SevenRadioAndWarp(unittest.TestCase):
    def test_01_ds_stable_large_counts_and_total_conflict(self) -> None:
        self.assertEqual(counts_to_canonical_ds_stable(0, 0), (0.0, 0.0, 1.0))
        m18 = counts_to_canonical_ds_stable(18, 18)
        self.assertAlmostEqual(sum(m18), 1.0, places=12)
        self.assertAlmostEqual(m18[0], 0.5, places=6)
        m30_20 = counts_to_canonical_ds_stable(30, 20)
        self.assertGreater(m30_20[0], 0.999999)
        m500 = counts_to_canonical_ds_stable(500, 499)
        self.assertAlmostEqual(m500[0], 5.0 / 6.0, places=9)
        self.assertAlmostEqual(m500[1], 1.0 / 6.0, places=9)
        m65k = counts_to_canonical_ds_stable(65535, 65534)
        self.assertAlmostEqual(m65k[0], 5.0 / 6.0, places=9)
        with self.assertRaises(ValueError):
            counts_to_canonical_ds_stable(1, 1, mf=1.0, me=1.0)

    def test_02_per_uav_battery_accel_clamp_and_hover(self) -> None:
        v_clamped = clamp_velocity_with_accel([0.0, 0.0], [25.0, 0.0], dt_s=1.0, v_max_mps=25.0, a_max_mps2=5.0)
        self.assertAlmostEqual(float(np.linalg.norm(v_clamped)), 5.0, places=6)
        # Stationary hover (168.49 + 15 = 183.49 W) > cruise 15 m/s (138.55 + 15 = 153.55 W)
        e_hover = uav_step_energy_j(0.0, 0.0)
        e_cruise = uav_step_energy_j(15.0, 0.0)
        self.assertGreater(e_hover, e_cruise)
        self.assertAlmostEqual(e_hover, 183.49, places=1)

    def test_03_1hop_uav_to_gcs_data_mac_ack_and_e2e_ack(self) -> None:
        net = SevenRadioCsmaCaNetwork(tx_power_w=0.5, seed=101)
        pos = np.full((RADIO_N, 3), [950.0, 950.0, 50.0])
        pos[0] = [120.0, 120.0, 50.0]  # Within 100 m of GCS (50,50,15), clear LoS
        pos[GCS_ID] = GCS_POS_M
        net.reset(pos, episode_id=1)
        net.run_until_us(600_000)  # HELLO discovery
        self.assertIn(GCS_ID, net.routes[0])

        net.record_local_observation(0, 42, True, t_s=1)
        dirty = net.get_dirty_entries_non_oracle(0, GCS_ID)
        self.assertEqual(len(dirty), 1)
        net.enqueue_app_data(0, GCS_ID, dirty)
        net.run_until_us(1_200_000)

        self.assertEqual(net.stats["data_delivered_gcs"], 1)
        self.assertGreaterEqual(net.stats["mac_ack_ok"], 2)  # DATA MAC ACK + E2E ACK MAC ACK
        self.assertEqual(net.stats["e2e_ack_rx"], 1)
        self.assertEqual(int(net.ledger_pos[GCS_ID, 0, 42]), 1)
        # Non-oracle watermark advanced -> cell 42 is no longer dirty
        self.assertEqual(len(net.get_dirty_entries_non_oracle(0, GCS_ID)), 0)
        # GCS TX airtime is recorded on index 6, NOT billed to UAVs 1..5
        self.assertGreater(net.step_tx_airtime_us[GCS_ID], 0)

    def test_04_2hop_relay_does_not_prematurely_fuse_and_preserves_gen_ts(self) -> None:
        net = SevenRadioCsmaCaNetwork(tx_power_w=0.5, seed=202)
        # Place UAV 0 at (50, 750, 50) -> 700m from GCS (out of 557m LoS); UAV 1 at (50, 400, 50) bridges both!
        pos = np.full((RADIO_N, 3), [950.0, 950.0, 50.0])
        pos[0] = [50.0, 750.0, 50.0]
        pos[1] = [50.0, 400.0, 50.0]
        pos[GCS_ID] = GCS_POS_M
        net.reset(pos, episode_id=1)
        net.run_until_us(1_600_000)  # 2 rounds of HELLO/DV so UAV 0 learns 2-hop route via UAV 1
        self.assertIn(GCS_ID, net.routes[0])
        self.assertEqual(net.routes[0][GCS_ID][0], 1)

        net.record_local_observation(0, 77, True, t_s=2)
        net.enqueue_app_data(0, GCS_ID, net.get_dirty_entries_non_oracle(0, GCS_ID))
        net.run_until_us(2_200_000)

        self.assertEqual(net.stats["data_delivered_gcs"], 1)
        self.assertEqual(net.stats["e2e_ack_rx"], 1)
        self.assertEqual(int(net.ledger_pos[GCS_ID, 0, 77]), 1)
        self.assertEqual(int(net.gcs_latest_u_s[77]), 2)  # Preserved immutable gen_ts_s
        # Relay (UAV 1) MUST NOT have prematurely fused UAV 0's destination evidence
        self.assertEqual(int(net.ledger_pos[1, 0, 77]), 0)

    def test_05_data_received_mac_ack_lost_deduplicates_retry(self) -> None:
        net = SevenRadioCsmaCaNetwork(tx_power_w=0.5, seed=303)
        pos = np.full((RADIO_N, 3), [950.0, 950.0, 50.0])
        pos[0] = [120.0, 120.0, 50.0]
        pos[GCS_ID] = GCS_POS_M
        net.reset(pos, episode_id=1)
        net.run_until_us(600_000)

        net.record_local_observation(0, 12, True, t_s=1)
        net.force_drop_next_mac_ack_from = GCS_ID  # Force GCS MAC ACK loss!
        net.enqueue_app_data(0, GCS_ID, net.get_dirty_entries_non_oracle(0, GCS_ID))
        net.run_until_us(1_500_000)

        self.assertGreaterEqual(net.stats["mac_ack_timeout"], 1)
        self.assertGreaterEqual(net.stats["dedup_hits"], 1)  # Retry deduplicated at GCS
        self.assertEqual(net.stats["data_delivered_gcs"], 1)  # Delivered ONCE
        self.assertEqual(net.stats["e2e_ack_rx"], 1)          # Only ONE E2E ACK

    def test_06_queue_full_drop_and_ttl_expiry(self) -> None:
        net = SevenRadioCsmaCaNetwork(tx_power_w=0.5, queue_cap=4, seed=404)
        # Put UAV 0 inside blackout zone (775, 275) -> no route, packets sit in queue
        pos = np.full((RADIO_N, 3), [950.0, 950.0, 50.0])
        pos[0] = [775.0, 275.0, 50.0]
        pos[GCS_ID] = GCS_POS_M
        net.reset(pos, episode_id=1)
        net.record_local_observation(0, 5, True, t_s=1)
        dirty = net.get_dirty_entries_non_oracle(0, GCS_ID)
        for _ in range(6):
            net.enqueue_app_data(0, GCS_ID, dirty)
        self.assertEqual(net.stats["drop_qfull"], 2)
        # Advance past 10 s TTL -> queued packets expire cleanly
        net.run_until_us(PACKET_TTL_US + 500_000)
        self.assertGreaterEqual(net.stats["drop_ttl"], 4)

    def test_07_blackout_store_carry_forward_recovery(self) -> None:
        net = SevenRadioCsmaCaNetwork(tx_power_w=0.5, seed=505)
        pos = np.full((RADIO_N, 3), [950.0, 950.0, 50.0])
        pos[0] = [775.0, 275.0, 50.0]  # Inside blackout zone 0
        pos[GCS_ID] = GCS_POS_M
        net.reset(pos, episode_id=1)
        net.record_local_observation(0, 88, True, t_s=1)
        net.enqueue_app_data(0, GCS_ID, net.get_dirty_entries_non_oracle(0, GCS_ID))
        net.run_until_us(1_000_000)
        self.assertEqual(net.stats["data_delivered_gcs"], 0)  # Held in queue (store-carry)

        # UAV 0 exits blackout & flies near GCS before 10 s TTL expires
        pos[0] = [150.0, 150.0, 50.0]
        net.update_positions(pos)
        net.run_until_us(2_200_000)
        self.assertEqual(net.stats["data_delivered_gcs"], 1)  # Forwarded after recovery!

    def test_08_full_7node_env_csmaca_full_comm_beats_zero_comm(self) -> None:
        def run_policy(comm: bool, steps: int = 80):
            env = ZhangXue7NodePacketEnv(EnvConfigV54(safety_horizon_s=1500), seed=20261125)
            obs, state, mask = env.reset(seed=20261125)
            self.assertEqual(obs.shape, (ACTOR_N, 514))
            self.assertEqual(mask.shape, (ACTOR_N, 7))
            tot_r = 0.0
            last_info = {}
            dirs = np.array([[15.0, 4.0], [12.0, 10.0], [6.0, 14.0], [14.0, 5.0], [8.0, 13.0], [10.0, 11.0]])
            for _ in range(steps):
                act_d = np.full(ACTOR_N, 6, dtype=np.int32) if comm else np.zeros(ACTOR_N, dtype=np.int32)
                obs, state, mask, r, term, trunc, info = env.step(dirs, act_d)
                tot_r += r
                last_info = info
                if term or trunc:
                    break
            return tot_r, last_info

        r_zero, info_zero = run_policy(comm=False, steps=80)
        r_comm, info_comm = run_policy(comm=True, steps=80)
        self.assertEqual(info_zero["gcs_tp_confirmed"], 0)
        self.assertEqual(info_zero["data_delivered_gcs"], 0)
        self.assertGreater(info_comm["data_delivered_gcs"], 0)
        self.assertGreater(info_comm["e2e_ack_rx"], 0)
        self.assertGreater(r_comm, r_zero)
        # Map-wide age uses the explicitly censored never-delivered convention
        self.assertGreater(info_comm["mean_gcs_aoi_mapwide_censored_s"], 0.0)
        self.assertGreater(info_comm["delivered_cell_coverage_pct"], 0.0)

    def test_09_warp_80array_soa_vram_exact_and_syntax_valid(self) -> None:
        v2048 = compute_exact_80array_soa_bytes(2048)
        v4096 = compute_exact_80array_soa_bytes(4096)
        self.assertEqual(v2048["array_count"], 80)
        self.assertEqual(v2048["bytes_per_env"], 490_001)
        self.assertAlmostEqual(v2048["total_gib"], 0.9346, places=4)
        self.assertAlmostEqual(v4096["total_gib"], 1.8692, places=4)
        # Syntax-verify Warp kernel file via AST
        src = (ROOT / "src/zhang_xue/gpu/kernels/scaffold.py").read_text(encoding="utf-8")
        ast.parse(src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
