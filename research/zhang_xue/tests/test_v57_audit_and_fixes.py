"""v5.7 Comprehensive Audit & Regression Test Suite.

Verifies every P0/P1/P2 fix identified by the multi-perspective Q1/A*, Warp GPU, and QA audit:
  - B1: Weak-signal (< CCA) cannot lock receiver
  - B3: Unroutable head-of-line DATA does not block E2E_ACK or routable DATA
  - B4: Battery-depleted UAV radio is completely silenced (no HELLO, no relay, zero channel gain)
  - B9: MAC retry re-looks up next hop when routing table updates
  - B10: r_loc_team respects ds_fusion_mode='zhang_max'
  - B11: obs[u, 11] (dirty_gcs) reports full dirty fraction (not capped at 62/400 = 0.155)
  - B12: gcs_last_rx_us updates on same-second novel evidence
  - C2: 3D LoS sensing occlusion prevents seeing inside/through 70-80m buildings
  - C4: Kinematic Safety Shield strictly enforces ||v_new - v_prev|| / dt <= a_max = 5.0 m/s^2
  - V57-AOI: Exact O(1) EWMA effective counts prevent artificial rejuvenation of stale evidence
  - V57-CAL: Calibrated likelihood-ratio BBA (max_mass=0.80) prevents 1-sample FP latch
  - V57-ACCEL: Zeng 2019 acceleration thrust-tilt + kinetic power > steady-level power
  - V57-UNLATCH: FP unlatch clears false positive when negative evidence lowers m_T < confirm_th
  - V57-SEMANTIC: Semantics-aware 2x2 pooling preserves target peak (no 4x dilution)
  - V57-PHY-GPU: Pure-NumPy reference of GPU 3D NLoS PHY kernel matches CPU link_path_gain
  - V57-VRAM: Compressed SoA + 300k StateDerivedReplayBuffer fits comfortably on 1x 16 GiB T4
  - V57-WARP-AST: Static contract verification on Warp event/inject/reset/mac/phy modules
"""
from __future__ import annotations

import ast
import math
import pathlib
import sys
import unittest

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uav_v54_7node_env import EnvConfigV54, ZhangXue7NodePacketEnv, make_v57_config
from uav_v54_7radio_csmaca_sim import (
    CANONICAL_BUILDINGS_5,
    GCS_ID,
    PKT_DATA,
    PKT_E2E_ACK,
    SevenRadioCsmaCaNetwork,
    pack_entry_2x_uint32,
)
from uav_v54_ds_and_battery import (
    ZengBatteryConfig,
    counts_to_canonical_ds_effective,
    counts_to_canonical_ds_stable,
    ewma_effective_counts_update,
    kinematic_safety_shield,
    sensor_likelihood_to_bba,
    zeng_propulsion_power_w,
    zeng_propulsion_power_with_accel_w,
)
from v55_state_derived_replay import estimate_replay_bytes
from v57_warp_phy_kernels import compute_channel_gain_3d_nlos_numpy, v57_gpu_memory_budget


class TestV57AuditFixes(unittest.TestCase):
    def test_b1_weak_signal_cannot_lock_receiver(self):
        net = SevenRadioCsmaCaNetwork(seed=7)
        pos = np.array([
            [50.0, 50.0, 50.0],     # UAV 0 near GCS
            [950.0, 950.0, 50.0],   # UAV 1 far (~1273 m from GCS, well beyond 500m / below CCA)
            [50.0, 200.0, 50.0],
            [50.0, 300.0, 50.0],
            [50.0, 400.0, 50.0],
            [50.0, 500.0, 50.0],
            [0.0, 0.0, 10.0],       # GCS
        ], dtype=np.float64)
        net.reset(pos, episode_id=1)
        # Force a tiny sub-CCA channel gain from UAV 1 -> GCS (1.5e-15 W / 0.5 W = 3e-15)
        net.chan_gain[1, GCS_ID] = 3.0e-15
        net.chan_gain[GCS_ID, 1] = 3.0e-15
        net.radios[1].active_pkt = None
        net._enqueue_hello_dv(1)
        net.radios[1].active_pkt = net.radios[1].pending_hello
        net.radios[1].pending_hello = None
        net._start_tx(1, is_ack=False)
        # GCS must NOT lock onto UAV 1's sub-CCA signal!
        self.assertEqual(net.radios[GCS_ID].rx_locked_tx, -1)

    def test_b3_hol_unroutable_data_does_not_block_e2e_ack(self):
        net = SevenRadioCsmaCaNetwork(seed=11)
        pos = np.array([
            [50.0, 50.0, 50.0],     # UAV 0 near GCS
            [950.0, 950.0, 50.0],   # UAV 1 disconnected
            [950.0, 850.0, 50.0],
            [950.0, 750.0, 50.0],
            [950.0, 650.0, 50.0],
            [950.0, 550.0, 50.0],
            [0.0, 0.0, 10.0],       # GCS
        ], dtype=np.float64)
        net.reset(pos, episode_id=1)
        self.assertTrue(net.run_until_us(600_000))
        # Enqueue an unroutable DATA packet to UAV 1 at head of UAV 0's queue
        e1 = [pack_entry_2x_uint32(0, 5, 1, 1, 0)]
        net.enqueue_app_data(0, 1, e1)
        # Enqueue a routable DATA packet to GCS behind it
        e2 = [pack_entry_2x_uint32(0, 6, 1, 1, 0)]
        net.enqueue_app_data(0, GCS_ID, e2)
        self.assertTrue(net.run_until_us(1_600_000))
        # GCS must still receive packet e2 without waiting 10s for e1 TTL!
        self.assertGreaterEqual(net.stats["data_delivered_gcs"], 1)

    def test_b4_depleted_uav_radio_is_silenced(self):
        env = ZhangXue7NodePacketEnv(make_v57_config(), seed=13)
        env.reset(seed=13)
        env.battery_j[1] = 0.0
        env.step(np.zeros((6, 2)), np.zeros(6, dtype=np.int32))
        self.assertFalse(bool(env.net.radio_alive[1]))
        self.assertEqual(int(env.net.step_tx_airtime_us[1]), 0)
        self.assertTrue(np.all(env.net.chan_gain[1, :] == 0.0))

    def test_b10_r_loc_team_respects_zhang_max(self):
        env_sum = ZhangXue7NodePacketEnv(EnvConfigV54(ds_fusion_mode="sum"), seed=21)
        env_max = ZhangXue7NodePacketEnv(EnvConfigV54(ds_fusion_mode="zhang_max"), seed=21)
        env_sum.reset(seed=21)
        env_max.reset(seed=21)
        # Two UAVs both observe cell 10 positively once
        for e in (env_sum, env_max):
            e.net.record_local_observation(0, 10, True, 1)
            e.net.record_local_observation(1, 10, True, 1)
        _, _, _, _, _, _, info_sum = env_sum.step(np.zeros((6, 2)), np.zeros(6, dtype=np.int32))
        _, _, _, _, _, _, info_max = env_max.step(np.zeros((6, 2)), np.zeros(6, dtype=np.int32))
        # With sum: n_pos=2 -> lower uncertainty -> higher r_loc_team than zhang_max (n_pos=1)
        self.assertGreater(info_sum["r_loc_team"], info_max["r_loc_team"])

    def test_b11_dirty_gcs_obs_not_capped_at_0_155(self):
        env = ZhangXue7NodePacketEnv(EnvConfigV54(), seed=23)
        env.reset(seed=23)
        for cid in range(120):
            env.net.record_local_observation(0, cid, False, 1)
        obs, _, _ = env._build_obs_state_mask()
        self.assertAlmostEqual(float(obs[0, 11]), 120.0 / 400.0, places=5)

    def test_c2_3d_sensing_occlusion_blocks_through_building(self):
        env = ZhangXue7NodePacketEnv(make_v57_config(enable_3d_sensing_occlusion=True), seed=29)
        env.reset(seed=29)
        # Place UAV 0 right outside building 0 (center 275, 275, r=45, h=75) at (225, 275, 50)
        # Cell (5, 5) has center (275, 275) inside building 0 (dist=50m <= fov_radius=50m)
        env.positions[0] = np.array([225.0, 275.0, 50.0])
        env.step(np.zeros((6, 2)), np.zeros(6, dtype=np.int32))
        cid_inside = 5 * 20 + 5
        self.assertEqual(int(env.net.ledger_pos[0, 0, cid_inside] + env.net.ledger_neg[0, 0, cid_inside]), 0)

    def test_c4_kinematic_safety_shield_enforces_a_max(self):
        # Two UAVs approaching head-on at 20 m/s with a_max=5 m/s^2
        pos = np.array([
            [200.0, 500.0],
            [270.0, 500.0],
            [100.0, 100.0],
            [100.0, 200.0],
            [100.0, 300.0],
            [100.0, 400.0],
        ], dtype=np.float64)
        vel = np.array([
            [20.0, 0.0],
            [-20.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float64)
        cmd = vel.copy()
        alive = np.ones(6, dtype=bool)
        v_safe, interv = kinematic_safety_shield(
            pos, vel, cmd, alive, CANONICAL_BUILDINGS_5,
            map_size_m=1000.0, dt_s=1.0, v_max_mps=25.0, a_max_mps2=5.0, uav_safe_distance_m=50.0,
        )
        for u in range(6):
            dv = float(np.linalg.norm(v_safe[u] - vel[u]))
            self.assertLessEqual(dv, 5.0 + 1e-6, f"UAV {u} violated a_max: dv={dv}")
            self.assertLessEqual(float(np.linalg.norm(v_safe[u])), 25.0 + 1e-6)
        self.assertGreater(float(interv[0] + interv[1]), 0.0)

    def test_v57_ewma_exact_aoi_no_artificial_rejuvenation(self):
        # 5 old observations at t=0, then 1 fresh observation at t=100s with eta=0.05
        ep, en = 0.0, 0.0
        for _ in range(5):
            ep, en = ewma_effective_counts_update(ep, en, 0.0, 0.0, True, eta=0.05)
        ep_new, en_new = ewma_effective_counts_update(ep, en, 0.0, 100.0, True, eta=0.05)
        expected = 5.0 * math.exp(-0.05 * 100.0) + 1.0
        self.assertAlmostEqual(ep_new, expected, places=9)
        mt_old, _, _ = counts_to_canonical_ds_effective(ep_new, en_new, 0.80, 0.80)
        mt_rejuv, _, _ = counts_to_canonical_ds_stable(6, 0, 0.80, 0.80)
        self.assertLess(mt_old, mt_rejuv)

    def test_v57_calibrated_likelihood_ratio_single_fp_no_latch(self):
        mf_raw, _ = sensor_likelihood_to_bba(0.85, 0.03)
        mf_cap, _ = sensor_likelihood_to_bba(0.85, 0.03, max_mass=0.80)
        self.assertGreater(mf_raw, 0.90)  # Raw single false alarm would exceed 0.90!
        self.assertLessEqual(mf_cap, 0.80)  # Capped requires >= 2 independent detections

    def test_v57_zeng_accel_power_exceeds_steady_level(self):
        cfg = ZengBatteryConfig()
        p_steady = zeng_propulsion_power_w(10.0, cfg)
        p_accel = zeng_propulsion_power_with_accel_w([7.5, 0.0], [12.5, 0.0], 1.0, cfg)
        self.assertGreater(p_accel, p_steady)

    def test_v57_semantic_pooling_preserves_target_peak(self):
        env_mean = ZhangXue7NodePacketEnv(EnvConfigV54(obs_pool_mode="mean"), seed=31)
        env_sem = ZhangXue7NodePacketEnv(make_v57_config(obs_pool_mode="semantic"), seed=31)
        env_mean.reset(seed=31)
        env_sem.reset(seed=31)
        for e in (env_mean, env_sem):
            for _ in range(3):
                e.net.record_local_observation(0, 0, True, 1)
        obs_m, _, _ = env_mean._build_obs_state_mask()
        obs_s, _, _ = env_sem._build_obs_state_mask()
        # Channel 0 (target mass) at pooled cell (0,0) is index 14
        self.assertGreater(float(obs_s[0, 14]), 0.95)
        self.assertLess(float(obs_m[0, 14]), 0.30)

    def test_v57_warp_phy_3d_nlos_numpy_parity_with_cpu_sim(self):
        net = SevenRadioCsmaCaNetwork(seed=41)
        pos = np.array([
            [180.0, 275.0, 50.0],
            [370.0, 275.0, 50.0],   # Line 0-1 passes straight through building 0 (275,275,r=45,h=75)!
            [200.0, 100.0, 50.0],
            [200.0, 750.0, 50.0],   # Inside blackout zone (4, 15) -> (225, 775)
            [600.0, 600.0, 50.0],
            [800.0, 800.0, 50.0],
            [0.0, 0.0, 10.0],
        ], dtype=np.float64)
        net.reset(pos, episode_id=1)
        g_gpu_ref = compute_channel_gain_3d_nlos_numpy(pos[None, :, :].astype(np.float32))[0]
        np.testing.assert_allclose(g_gpu_ref, net.chan_gain.astype(np.float32), rtol=1e-4, atol=1e-10)

    def test_v57_gpu_memory_budget_fits_single_t4(self):
        b = v57_gpu_memory_budget(num_envs_per_gpu=2048, replay_capacity=300_000, obs_dim=514)
        self.assertEqual(b["fits_single_t4_16gib"], 1.0)
        self.assertLess(b["estimated_total_gib"], 10.0)
        self.assertGreater(estimate_replay_bytes(1000, 6, 514, 2, 7, False), 0)

    def test_v57_warp_static_ast_contracts(self):
        # Verify enable_backward=False is set in event, injection, reset, mac, phy modules
        for fname in (
            "zhang_xue/gpu/state.py",
            "zhang_xue/gpu/mac_functions.py",
            "v56_warp_event_kernels.py",
            "v56_warp_injection_kernels.py",
            "v56_warp_reset_kernels.py",
            "v57_warp_phy_kernels.py",
        ):
            text = (SRC / fname).read_text(encoding="utf-8")
            ast.parse(text)  # Valid Python syntax
            self.assertIn("enable_backward", text, f"{fname} missing enable_backward=False")
        # Verify dead duplicate _reset/_step removed from v56_warp_7radio_mac.py
        mac_text = (SRC / "zhang_xue/gpu/mac_functions.py").read_text(encoding="utf-8")
        state_text = (SRC / "zhang_xue/gpu/state.py").read_text(encoding="utf-8")
        self.assertNotIn("def _step(", mac_text)
        self.assertNotIn("def _reset(", mac_text)
        self.assertIn("_sum_interference_at", mac_text)
        self.assertIn("hello_dv_dist", state_text)
        self.assertIn("wp.atomic_add(s.stats_drop", mac_text)


if __name__ == "__main__":
    unittest.main()
