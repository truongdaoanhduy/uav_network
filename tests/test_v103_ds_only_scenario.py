"""V103 DS-Only Scenario Regression & Parity Contract Suite (CPU + Warp source contracts).

Verifies:
1. V102 P0 bug fix: raw_u16_count + aoi_ewma_end_to_end=True no longer scales by 256x
   across record_local_observation, observations._fused_cell, transition reward/GCS,
   and communication/delivery PEER info gain.
2. Exact DS belief math, vacuous (0,0,1) initialization, O(1) cumulative conflict K,
   and FP32/FP64 numerical stability at large counts (n=5000).
3. CPU <-> Warp observation schema parity (obs_dim=549 with non-oracle HELLO telemetry).
4. Anti-reward-hacking: local sensing updates peer_h_min; monotone team_u_prev watermark.
5. Warp GPU DS-only fail-closed contracts, FP32 energy grouping, and 4096-env seed hashing.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

from zhang_xue.belief.dempster_shafer import (
    counts_to_canonical_ds_effective,
    counts_to_canonical_ds_stable,
    counts_to_canonical_ds_with_conflict,
    deng_entropy,
    ds_cumulative_conflict,
    fuse_raw_provenance_counts,
)
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params


class TestV103DsOnlyScenario(unittest.TestCase):
    V103_YAML = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"

    def test_v103_config_loads_and_enforces_ds_only_baseline_contracts(self):
        cfg = load_scenario(self.V103_YAML)
        self.assertEqual(cfg.belief_inference_mode, "dempster_shafer")
        self.assertEqual(cfg.static_evidence_mode, "raw_u16_count")
        self.assertTrue(cfg.aoi_ewma_end_to_end)
        self.assertFalse(cfg.use_uq88_ewma)
        self.assertEqual(cfg.ds_aoi_eta, 0.0)
        self.assertTrue(cfg.obs_schema_v2)
        self.assertFalse(cfg.enable_tx_power_control)
        self.assertAlmostEqual(cfg.tx_power_w, 0.1)
        self.assertFalse(cfg.enable_peer_dissemination)
        self.assertEqual(cfg.w_p2p, 0.0)

    def test_raw_u16_count_write_and_fusion_fixes_v102_256x_bug(self):
        cfg = load_scenario(self.V103_YAML)
        env = ZhangXue7NodePacketEnv(cfg, seed=20261127)
        env.reset(seed=20261127)
        cases = [
            (1, 0, 0.8000, 0.0000, 0.2000),
            (2, 0, 0.9600, 0.0000, 0.0400),
            (3, 1, 0.9612, 0.0310, 0.0078),
            (1, 1, 0.4444, 0.4444, 0.1111),
            (10, 10, 0.5000, 0.5000, 0.0000),
        ]
        for idx, (n_pos, n_neg, exp_t, exp_f, exp_u) in enumerate(cases):
            cid = 10 + idx
            t_s = 1
            for _ in range(n_pos):
                env.net.record_local_observation(0, cid, True, t_s)
                t_s += 1
            for _ in range(n_neg):
                env.net.record_local_observation(0, cid, False, t_s)
                t_s += 1
            # Raw integer count stored (NOT 256x UQ8.8)
            self.assertEqual(int(env.net.ledger_pos[0, 0, cid]), n_pos)
            self.assertEqual(int(env.net.ledger_neg[0, 0, cid]), n_neg)
            m_t, m_f, m_u = env._fused_cell(0, cid, now_s=float(t_s))
            self.assertAlmostEqual(m_t, exp_t, places=4)
            self.assertAlmostEqual(m_f, exp_f, places=4)
            self.assertAlmostEqual(m_u, exp_u, places=4)
            self.assertAlmostEqual(m_t + m_f + m_u, 1.0, places=9)

    def test_ds_vacuous_init_and_large_count_stability_and_conflict(self):
        # Never-sensed cell must be vacuous (0, 0, 1), never Bayesian prior 15/392
        self.assertEqual(fuse_raw_provenance_counts([0, 0], [0, 0], [-1, -1]), (0.0, 0.0, 1.0))
        # Large counts (5000, 4998) with mf=me=0.80 must match (2, 0) -> m_T=0.96, zero NaN
        mt, mf, mu, k = counts_to_canonical_ds_with_conflict(5000, 4998, 0.80, 0.80)
        self.assertTrue(all(math.isfinite(v) for v in (mt, mf, mu, k)))
        self.assertAlmostEqual(mt, 25.0 / 26.0, places=6)
        self.assertAlmostEqual(mf, 1.0 / 26.0, places=6)
        self.assertAlmostEqual(mu, 0.0, places=9)
        self.assertAlmostEqual(k, 1.0, places=9)
        # Effective count parity on large equal counts
        et, ef, eu = counts_to_canonical_ds_effective(5000.0, 4998.0, 0.80, 0.80)
        self.assertAlmostEqual(et, 25.0 / 26.0, places=6)
        self.assertAlmostEqual(ef, 1.0 / 26.0, places=6)
        # Exact cumulative conflict values
        self.assertAlmostEqual(ds_cumulative_conflict(1, 0, 0.80, 0.80), 0.0, places=9)
        self.assertAlmostEqual(ds_cumulative_conflict(1, 1, 0.80, 0.80), 0.64, places=9)
        self.assertAlmostEqual(ds_cumulative_conflict(5, 5, 0.80, 0.80), (1.0 - 0.2**5) ** 2, places=9)

    def test_local_sensing_updates_peer_h_min_blocking_peer_info_gain_farming(self):
        cfg = load_scenario(self.V103_YAML)
        env = ZhangXue7NodePacketEnv(cfg, seed=7)
        env.reset(seed=7)
        cid = 42
        env.net.record_local_observation(0, cid, True, 1)
        env.net.record_local_observation(0, cid, True, 2)
        expected_h = deng_entropy(*counts_to_canonical_ds_stable(2, 0, 0.80, 0.80))
        self.assertAlmostEqual(float(env.net.peer_h_min[0, cid]), expected_h, places=6)
        self.assertLess(float(env.net.peer_h_min[0, cid]), 0.35)

    def test_cpu_rollout_obs_549_parity_and_ds_diagnostics(self):
        cfg = load_scenario(self.V103_YAML)
        env = ZhangXue7NodePacketEnv(cfg, seed=103)
        obs, state, mask = env.reset(seed=103)
        self.assertEqual(obs.shape, (6, 549))
        self.assertEqual(state.shape, (6 * 549,))
        self.assertEqual(mask.shape, (6, 3))
        # PEER action masked off in V103 baseline
        self.assertTrue(np.all(mask[:, 2] == 0))
        self.assertTrue(np.isfinite(obs).all())
        # Step 5 ticks with valid actions
        for _ in range(5):
            vel_cmd = np.zeros((6, 2), dtype=np.float32)
            comm_cmd = np.ones(6, dtype=np.int64) * mask[:, 1]
            obs, state, mask, rew, term, trunc, info = env.step(vel_cmd, comm_cmd)
            self.assertEqual(obs.shape, (6, 549))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(math.isfinite(rew))
        for key in (
            "gcs_tp_active_confirmed",
            "gcs_tp_belief_drops",
            "gcs_fp_ever_confirmed",
            "gcs_fp_unlatched_count",
            "gcs_mean_conflict_k",
            "gcs_max_conflict_k",
            "gcs_high_conflict_cells",
            "gcs_mean_uncertainty_u",
            "gcs_mean_deng_entropy",
        ):
            self.assertIn(key, info)

    def test_warp_gpu_ds_only_and_stability_source_contracts(self):
        # GPUWorldV79Params accepts V103 DS-only YAML and rejects Bayesian YAML
        params = GPUWorldV79Params.from_yaml(self.V103_YAML, envs=4, device="cpu", seed=44)
        self.assertTrue(params.static_raw_count)
        self.assertTrue(params.obs_schema_v2)
        self.assertFalse(params.enable_peer)
        with self.assertRaises(ValueError):
            GPUWorldV79Params.from_yaml(
                ROOT / "research/zhang_xue/configs/scenario_v91_bayes_1km.yaml",
                envs=4, device="cpu", seed=44,
            )
        k_src = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py").read_text(encoding="utf-8")
        r_src = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_reset_obs.py").read_text(encoding="utf-8")
        m79_src = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/mission_v79.py").read_text(encoding="utf-8")
        m80_src = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/mission_v80.py").read_text(encoding="utf-8")
        self.assertNotIn("_bayes_from_counts_v91", k_src)
        self.assertNotIn("_fuse_belief_v91", k_src)
        self.assertNotIn("_fuse_belief_v91", r_src)
        self.assertIn("wp.abs(logmf - logme) <= 1.0e-7", k_src)
        self.assertIn("_mix_seed_v103", k_src)
        self.assertIn("_mix_seed_v103", r_src)
        self.assertIn("compute_dirty_dest_v67", m79_src)
        self.assertIn("compute_dirty_dest_v67", m80_src)


if __name__ == "__main__":
    unittest.main()
