"""V120 automated CPU conflict-resolution regression gates for C15, C16, C27, C39, C40.

Verifies:
  - C27: Multi-trajectory non-oracle ground-truth sentinel taint test (obs, state, mask bit-exact)
  - C39: Latch (ever-confirmed) vs active belief-drop contract on positive-then-negative evidence
  - C40: Blackout-exit route-gated sync (inside blackout mask[0,1]==0 & no GCS delivery; on exit
         HELLO restores route and syncs exactly-once via ack_wm with zero double counting)
  - C16: Active DS-only runtime lock & pinned SHA256 hashes of both active YAMLs
  - C15: Underlying 20x20 DS belief preserves simplex sum=1.0 while pooled 5x10x10 is derived summary
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np
import yaml

from zhang_xue.belief.dempster_shafer import fuse_raw_provenance_counts
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv


ROOT = Path(__file__).resolve().parents[1]
BASE_YAML = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"
RF_YAML = ROOT / "research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml"


def _load_cfg(path: Path, **overrides) -> EnvConfigV54:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.update(overrides)
    return EnvConfigV54(**data)


class TestV120ConflictResolutionGates(unittest.TestCase):
    def test_c16_active_yamls_ds_only_and_pinned_hashes(self) -> None:
        base_sha = hashlib.sha256(BASE_YAML.read_bytes()).hexdigest()
        rf_sha = hashlib.sha256(RF_YAML.read_bytes()).hexdigest()
        self.assertEqual(
            base_sha,
            "3fcfe434b35c29310835c2aa99f982a96c4af58375dc5e916e1e4d017a832fc5",
            "Active Base YAML SHA256 must stay pinned to V119 GPU-verified hash",
        )
        self.assertEqual(
            rf_sha,
            "3ee084e793716baa9ea83d842af0b51345690fdb8156ab0e5c527739ad033b76",
            "Active RF-Stress YAML SHA256 must stay pinned to V119 GPU-verified hash",
        )
        for path in (BASE_YAML, RF_YAML):
            cfg = _load_cfg(path)
            self.assertEqual(cfg.belief_inference_mode, "dempster_shafer")
            self.assertFalse(cfg.enable_tx_power_control)
            self.assertAlmostEqual(cfg.tx_power_w, 0.1)

    def test_c27_multi_trajectory_non_oracle_sentinel_taint(self) -> None:
        """Swapping true_targets on cells outside initial FoV must produce bit-exact obs/state/mask."""
        cfg = _load_cfg(BASE_YAML)
        for seed in (44, 45, 46):
            env_a = ZhangXue7NodePacketEnv(cfg)
            env_b = ZhangXue7NodePacketEnv(cfg)
            env_a.reset(seed=seed)
            env_b.reset(seed=seed)

            far_cids = []
            for cid in range(env_a.num_cells):
                cx = (cid % env_a.grid_w + 0.5) * env_a.cell_size_m
                cy = (cid // env_a.grid_w + 0.5) * env_a.cell_size_m
                dists = np.linalg.norm(env_a.positions[: env_a.actor_n, :2] - np.array([cx, cy]), axis=1)
                if float(np.min(dists)) > 250.0:
                    far_cids.append(cid)
            self.assertGreater(len(far_cids), 50)

            env_b.true_targets[far_cids] = ~env_b.true_targets[far_cids]
            self.assertFalse(np.array_equal(env_a.true_targets, env_b.true_targets))

            act_xy = np.zeros((cfg.num_uavs, 2), dtype=np.float32)
            act_comm = np.ones((cfg.num_uavs,), dtype=np.int32)
            for _ in range(5):
                obs_a, state_a, mask_a, _, _, _, _ = env_a.step(act_xy, act_comm)
                obs_b, state_b, mask_b, _, _, _, _ = env_b.step(act_xy, act_comm)
                self.assertEqual(float(np.max(np.abs(obs_a - obs_b))), 0.0)
                self.assertEqual(float(np.max(np.abs(state_a - state_b))), 0.0)
                self.assertTrue(np.array_equal(mask_a, mask_b))

    def test_c39_gcs_tp_latch_vs_active_belief_drop_contract(self) -> None:
        """2 positive hits confirm (m_T=0.96>=0.90), then 2 negative hits drop active belief while latch stays 1."""
        cfg = _load_cfg(BASE_YAML, enable_tp_unlatch=False)
        env = ZhangXue7NodePacketEnv(cfg)
        env.reset(seed=44)
        tgt_cid = int(np.flatnonzero(env.true_targets)[0])
        gcs_id = env.net.gcs_id

        env.net.ledger_pos[gcs_id, 0, tgt_cid] = 2
        env.net.ledger_neg[gcs_id, 0, tgt_cid] = 0
        env.net.ledger_gen_ts_s[gcs_id, 0, tgt_cid] = 0
        _, _, _, _, _, _, info1 = env.step(np.zeros((6, 2), dtype=np.float32), np.zeros((6,), dtype=np.int32))
        self.assertGreaterEqual(int(info1["gcs_tp_confirmed"]), 1)
        self.assertGreaterEqual(int(info1["gcs_tp_active_confirmed"]), 1)

        env.net.ledger_neg[gcs_id, 0, tgt_cid] = 2
        _, _, _, _, _, _, info2 = env.step(np.zeros((6, 2), dtype=np.float32), np.zeros((6,), dtype=np.int32))
        self.assertGreaterEqual(int(info2["gcs_tp_confirmed"]), 1)
        self.assertEqual(int(info2["gcs_tp_active_confirmed"]), 0)
        self.assertGreaterEqual(int(info2["gcs_tp_belief_drops"]), 1)

    def test_c40_blackout_exit_route_gated_exact_once_sync(self) -> None:
        """Inside blackout cell center, DATA cannot reach GCS; after exit, HELLO restores route and syncs once."""
        cfg = _load_cfg(BASE_YAML)
        env = ZhangXue7NodePacketEnv(cfg)
        env.reset(seed=44)
        bx_cell, by_cell = env.net.blackout_centers[0]
        env.positions[0, 0] = (bx_cell + 0.5) * env.cell_size_m
        env.positions[0, 1] = (by_cell + 0.5) * env.cell_size_m
        env.net.update_positions(env.positions)
        env.net.routes[0].clear()
        env.net.record_local_observation(0, 10, True, 0.0)

        act_comm = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
        _, _, mask_in, _, _, _, _ = env.step(np.zeros((6, 2), dtype=np.float32), act_comm)
        self.assertEqual(int(mask_in[0, 1]), 0, "Inside blackout with no route, mask[0,1] must be 0")
        self.assertEqual(int(env.net.ledger_pos[env.net.gcs_id, 0, 10]), 0)

        # Exit blackout near GCS; after HELLO restores route, cell 10 syncs exactly once (ack_wm=1)
        env.positions[0, 0], env.positions[0, 1] = 500.0, 500.0
        env.net.update_positions(env.positions)
        for _ in range(3):
            _, _, mask_out, _, _, _, _ = env.step(np.zeros((6, 2), dtype=np.float32), act_comm)
        self.assertEqual(int(mask_out[0, 1]), 1, "After exiting blackout near GCS, route must recover")
        self.assertEqual(int(env.net.ledger_pos[env.net.gcs_id, 0, 10]), 1)
        self.assertEqual(int(env.net.ack_wm_pos[0, env.net.gcs_id, 10]), 1)

    def test_c15_underlying_ds_grid_preserves_simplex(self) -> None:
        """Underlying 20x20 DS belief triples always sum to 1.0 within float tolerance."""
        cfg = _load_cfg(BASE_YAML)
        env = ZhangXue7NodePacketEnv(cfg)
        env.reset(seed=44)
        for _ in range(10):
            env.step(np.zeros((6, 2), dtype=np.float32), np.ones((6,), dtype=np.int32))
        gcs_id = env.net.gcs_id
        for cid in range(env.num_cells):
            m_t, m_f, u = fuse_raw_provenance_counts(
                env.net.ledger_pos[gcs_id, : env.actor_n, cid],
                env.net.ledger_neg[gcs_id, : env.actor_n, cid],
                env.net.ledger_gen_ts_s[gcs_id, : env.actor_n, cid],
                mf=env.evidence_mf,
                me=env.evidence_me,
                mode=cfg.ds_fusion_mode,
            )
            self.assertAlmostEqual(float(m_t + m_f + u), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
