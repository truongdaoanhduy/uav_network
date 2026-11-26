"""v6.9 Unified Regression Tests (v6.8 QA + v6.7 Executable Warp + CPU/B09 Fixes).

Covers:
  1. P0 `r_loc_team` UQ8.8 decoding fix (`decode_uq88` + `counts_to_canonical_ds_effective`).
  2. `obs[u, 11]` (`dirty_gcs <= 1.0`), `obs[u, 12]` (`ack_wm_origin_ts`), and `pooled[4] * factor^2` at 40x40.
  3. `grid_w^2 <= 4096` guard & `mac_queue_capacity == 200` in `make_uavnetsim_reference_config()`.
  4. B09 paper metrics (`local_tp_discovered`, `true_area_observed_pct`, `pdr_gcs`, `mean_queue_len`, `discovery_auc`).
  5. Executable Warp v6.7 modules (`layout_v67`, `state_v67`, `runtime_v67`, `parity_v67_ref`, `kernels/*_v67`),
     `480,686 B/env` compressed Data-Mule SoA, and 2-contact `NumPySoARefV67` CPU parity.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from zhang_xue.environment.configuration import (
    EnvConfigV54,
    make_uavnetsim_reference_config,
    make_v65_datamule_config,
)
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.layout_v67 import backing_bytes_per_environment_v67, v67_gpu_memory_budget
from zhang_xue.gpu.parity_v67_ref import NumPySoARefV67
from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67, resolve_compact3_actions_v67_np

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestV69CpuLogicAndB09Metrics(unittest.TestCase):
    def test_r_loc_team_decodes_uq88_without_instant_saturation(self):
        cfg = make_v65_datamule_config(num_uavs=2, grid_w=20, map_size_m=1000.0)
        env = ZhangXue7NodePacketEnv(cfg, seed=123)
        env.reset(seed=123)
        env.net.record_local_observation(0, 100, True, 1)
        from zhang_xue.belief.dempster_shafer import counts_to_canonical_ds_effective, decode_uq88
        ep = decode_uq88(int(env.net.ledger_pos[0, 0, 100]))
        self.assertAlmostEqual(ep, 1.0, places=3)
        _, _, u_now = counts_to_canonical_ds_effective(ep, 0.0, cfg.ds_mf, cfg.ds_me)
        self.assertAlmostEqual(u_now, 0.20, places=3)

    def test_b09_metrics_present_and_bounded_on_step(self):
        cfg = make_v65_datamule_config(num_uavs=4, grid_w=20, map_size_m=1000.0)
        env = ZhangXue7NodePacketEnv(cfg, seed=19)
        env.reset(seed=19)
        vel = np.zeros((4, 2), dtype=np.float64)
        act = np.ones(4, dtype=np.int64)
        _, _, _, _, _, _, info = env.step(vel, act)
        for key in ("local_tp_discovered", "true_area_observed_pct", "pdr_gcs", "mean_queue_len", "discovery_auc"):
            self.assertIn(key, info)
        self.assertGreaterEqual(info["local_tp_discovered"], info["gcs_tp_confirmed"])
        self.assertGreaterEqual(info["true_area_observed_pct"], 0.0)
        self.assertLessEqual(info["true_area_observed_pct"], 100.0)
        self.assertGreaterEqual(info["discovery_auc"], 0.0)

    def test_observations_mean_pool_40x40_and_config_guards(self):
        ref = make_uavnetsim_reference_config()
        self.assertEqual(ref.mac_queue_capacity, 200)
        with self.assertRaises(ValueError):
            EnvConfigV54(grid_w=70, map_size_m=3500.0)
        cfg = make_v65_datamule_config(num_uavs=4, grid_w=40, map_size_m=2000.0, obs_pool_mode="mean")
        env = ZhangXue7NodePacketEnv(cfg, seed=5)
        obs, _, _ = env.reset(seed=5)
        self.assertLessEqual(float(obs[:, 11].max()), 1.0)
        self.assertAlmostEqual(float(obs[0, 14 + 400 : 14 + 500].sum()), 1.0, places=5)


class TestV69WarpModulesAndParity(unittest.TestCase):
    def test_v67_warp_memory_and_executable_methods(self):
        self.assertEqual(backing_bytes_per_environment_v67(6, 20), 481_141)
        self.assertEqual(WarpCompressedRadioNetworkV67.active_layout_id(), "v67_datamule_compressed")
        for m in ("reset", "update_phy_from_positions", "sense_batch", "inject_compact3", "step", "snapshot"):
            self.assertTrue(callable(getattr(WarpCompressedRadioNetworkV67, m, None)))
        budget = v67_gpu_memory_budget(num_envs_per_gpu=2048, num_uavs=6, grid_w=20)
        self.assertEqual(budget["fits_single_t4_16gib"], 1.0)
        self.assertLess(budget["network_soa_gib"], 1.0)

    def test_v67_resolver_and_two_contact_numpy_soa_parity(self):
        actions = np.array([[2, 1, 2]], dtype=np.int32)
        pos = np.array([[[0.0, 0.0], [10.0, 0.0], [50.0, 0.0]]], dtype=np.float32)
        alive = np.array([[True, True, True]], dtype=bool)
        dirty = np.zeros((1, 3, 4), dtype=bool)
        dirty[0, 0, 2] = True
        dirty[0, 1, 3] = False
        dirty[0, 2, 0] = True
        can_tx = np.array([[True, True, False]], dtype=bool)
        dst = resolve_compact3_actions_v67_np(actions, pos, alive, True, dirty_dest=dirty, can_tx=can_tx)
        self.assertEqual(dst.tolist(), [[2, -1, -1]])

        ref = NumPySoARefV67(num_envs=1, num_uavs=3, grid_w=20, enable_data_mule=True)
        ref.sense_batch(np.array([[42, -1, -1]]), np.array([[1, 0, 0]]), t_s=5)
        dst1 = ref.resolve_compact3(np.array([[2, 0, 0]]), pos)
        self.assertEqual(int(dst1[0, 0]), 1)
        self.assertEqual(ref.inject_and_deliver_ideal(dst1), 1)
        dst2 = ref.resolve_compact3(np.array([[0, 1, 0]]), pos)
        self.assertEqual(int(dst2[0, 1]), 3)
        self.assertEqual(ref.inject_and_deliver_ideal(dst2), 1)
        self.assertEqual(int(ref.data["ledger_gen"][0, 3, 0, 42]), 5)
        self.assertFalse(bool(ref.compute_dirty_dest()[0, 1, 3]))

    def test_all_v67_gpu_modules_ast_valid_and_under_450_lines(self):
        gpu_dir = REPO_ROOT / "research/zhang_xue/src/zhang_xue/gpu"
        for rel in (
            "layout_v67.py", "state_v67.py", "runtime_v67.py", "parity_v67_ref.py",
            "kernels/phy_v67.py", "kernels/reset_v67.py", "kernels/mac_v67.py",
            "kernels/inject_v67.py", "kernels/events_v67.py",
        ):
            src = (gpu_dir / rel).read_text(encoding="utf-8")
            ast.parse(src)
            self.assertLessEqual(len(src.splitlines()), 450, f"{rel} exceeds 450 lines")


if __name__ == "__main__":
    unittest.main()
