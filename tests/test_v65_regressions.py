"""v6.5 Regression & Static Verification Tests.

Covers all 7 v6.5 fixes:
  1. MASAC learner.py: never zeroes vy when continuous_dim == 2 across all 4 _expected_q* paths.
  2. factorized_candidate.py: supports compact_3 (k=3) and never zeroes vy when c == 2.
  3. matd3/diagnostics.py: imports valid_destinations and guards RF metrics for 2D motion-only actor.
  4. schema.py: auto-aligns target_continuous_entropy with continuous_dim when None.
  5. 2-3 km obstacle/blackout proportional scaling (bit-exact at 1000 m, scaled at 2000/3000 m).
  6. True Data-Mule multi-origin forwarding (UAV A -> UAV B -> GCS) with zero double-counting & peer ablation.
  7. Warp v6.5 compressed schema (379,848 B/env, -40.3% VRAM), UavNetSim PHY defaults, and phy.py z>=0 parity.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest

import numpy as np

from zhang_xue.communication.common import (
    CANONICAL_BLACKOUT_CENTERS_2,
    CANONICAL_BLACKOUT_R,
    CANONICAL_BUILDINGS_5,
    scale_canonical_geometry,
)
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.environment.configuration import EnvConfigV54, make_v65_datamule_config
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.kernels.phy import compute_channel_gain_3d_nlos_numpy
from zhang_xue.gpu.runtime_v65 import (
    WarpCompressedRadioNetwork,
    resolve_compact3_actions_np,
    v65_gpu_memory_budget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestV65AlgorithmStaticGuards(unittest.TestCase):
    def test_masac_learner_never_unconditionally_zeroes_last_continuous_dim(self):
        src = (REPO_ROOT / "uav_network/algorithms/masac/learner.py").read_text(encoding="utf-8")
        self.assertNotIn("\n        choices[:, 0, -1] = 0\n", src)
        self.assertNotIn("\n        choices[:, :, 0, -1] = 0\n", src)
        self.assertGreaterEqual(src.count("if self.continuous_dim > self.config.motion_dim:"), 4)
        self.assertIn("learn_power=self.learn_power", src)

    def test_factorized_candidate_supports_compact3_and_2d_motion(self):
        src = (REPO_ROOT / "uav_network/algorithms/core/factorized_candidate.py").read_text(encoding="utf-8")
        self.assertNotIn("k == n + 1", src)
        self.assertIn("k >= 2", src)
        self.assertGreaterEqual(src.count("if c > 2:"), 2)

    def test_matd3_diagnostics_imports_valid_destinations_and_guards_rf(self):
        src = (REPO_ROOT / "uav_network/algorithms/matd3/diagnostics.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn("valid_destinations", imported)
        self.assertIn("has_rf = self.continuous_dim > self.config.motion_dim", src)

    def test_schema_defaults_target_continuous_entropy_for_2d_motion(self):
        from uav_network.algorithms.configuration.schema import LearnerConfig

        cfg2 = LearnerConfig(continuous_dim=2).validate()
        self.assertEqual(cfg2.masac.target_continuous_entropy, -2.0)


class TestV65ScalingAndPeerAblation(unittest.TestCase):
    def test_canonical_geometry_bit_exact_at_1km_and_scaled_at_2km(self):
        b1, c1, r1 = scale_canonical_geometry(1000.0, 20, scale_obstacles=True)
        self.assertEqual(b1, list(CANONICAL_BUILDINGS_5))
        self.assertEqual(c1, list(CANONICAL_BLACKOUT_CENTERS_2))
        self.assertEqual(r1, CANONICAL_BLACKOUT_R)

        b2, c2, r2 = scale_canonical_geometry(2000.0, 40, scale_obstacles=True)
        self.assertAlmostEqual(b2[0][0], 560.0)
        self.assertAlmostEqual(b2[3][0], 1480.0)
        self.assertEqual(c2, [(30, 10), (28, 30)])
        self.assertEqual(r2, 6)

    def test_per_peer_mode_blocks_peers_when_peer_dissemination_disabled(self):
        cfg = make_v65_datamule_config(
            comm_action_mode="per_peer",
            enable_peer_dissemination=False,
        )
        env = ZhangXue7NodePacketEnv(cfg, seed=11)
        env.reset(seed=11)
        # Action 1..num_uavs-1 are peers; action num_uavs is GCS
        self.assertEqual(env._resolve_comm_dest(0, 1), -1)
        self.assertEqual(env._resolve_comm_dest(0, env.actor_n), env.gcs_id)

    def test_2km_env_step_runs_cleanly_with_scaled_obstacles(self):
        cfg = make_v65_datamule_config(num_uavs=4, grid_w=40, map_size_m=2000.0)
        env = ZhangXue7NodePacketEnv(cfg, seed=7)
        obs, state, mask = env.reset(seed=7)
        self.assertEqual(obs.shape, (4, 514))
        self.assertEqual(mask.shape, (4, 3))
        self.assertAlmostEqual(env.buildings[0][0], 560.0)
        vel = np.zeros((4, 2), dtype=np.float64)
        act = np.zeros(4, dtype=np.int64)
        _, _, _, r, term, trunc, _ = env.step(vel, act)
        self.assertTrue(math.isfinite(r))
        self.assertFalse(term or trunc)


class TestV65TrueDataMuleForwarding(unittest.TestCase):
    def test_store_carry_forward_relay_to_gcs_without_double_counting(self):
        net = SevenRadioCsmaCaNetwork(
            tx_power_w=0.1,
            noise_power_w=4.38957709293155e-13,
            cca_threshold_dbm=-82.0,
            aoi_ewma_end_to_end=True,
            enable_data_mule_forwarding=True,
            num_uavs=4,
            seed=42,
        )
        # Position UAV0 & UAV1 near each other, UAV1 & GCS in range (all LoS)
        pos = np.array([
            [150.0, 150.0, 50.0],
            [180.0, 150.0, 50.0],
            [210.0, 150.0, 50.0],
            [240.0, 150.0, 50.0],
            [50.0, 50.0, 15.0],
        ])
        net.reset(pos, episode_id=1)
        self.assertTrue(net.run_until_us(600_000))

        # UAV 0 observes cell 42 at t=10
        net.record_local_observation(0, 42, True, 10)
        entries_0_to_1 = net.get_dirty_entries_non_oracle(0, 1)
        self.assertEqual(len(entries_0_to_1), 1)
        net.enqueue_app_data(0, 1, entries_0_to_1)
        self.assertTrue(net.run_until_us(1_600_000))

        # UAV 1 now holds origin=0 evidence and MUST consider it dirty for GCS (id=4)
        self.assertGreater(int(net.ledger_pos[1, 0, 42]), 0)
        self.assertEqual(net.count_all_dirty_entries_non_oracle(1, 4), 1)
        # UAV 1 must NOT send origin=0 evidence back to UAV 0
        self.assertEqual(net.count_all_dirty_entries_non_oracle(1, 0), 0)

        entries_1_to_gcs = net.get_dirty_entries_non_oracle(1, 4)
        self.assertEqual(len(entries_1_to_gcs), 1)
        net.enqueue_app_data(1, 4, entries_1_to_gcs)
        self.assertTrue(net.run_until_us(2_600_000))

        # GCS received origin=0 evidence via UAV 1, and UAV 1's 4D watermark is updated!
        self.assertEqual(int(net.ledger_pos[4, 0, 42]), int(net.ledger_pos[0, 0, 42]))
        self.assertEqual(net.count_all_dirty_entries_non_oracle(1, 4), 0)


class TestV65WarpAndPhyParity(unittest.TestCase):
    def test_warp_v65_compressed_budget_and_uavnetsim_phy_defaults(self):
        self.assertEqual(WarpCompressedRadioNetwork.active_layout_id(), "v65_compressed_param")
        b_per_env = WarpCompressedRadioNetwork.active_soa_bytes_per_environment(6, 20)
        self.assertEqual(b_per_env, 379_848)
        phy = WarpCompressedRadioNetwork.default_phy_params()
        self.assertAlmostEqual(phy["tx_power_w"], 0.1)
        self.assertAlmostEqual(phy["cca_threshold_w"], 10.0 ** ((-82.0 - 30.0) / 10.0), places=18)
        self.assertAlmostEqual(phy["noise_w"], 4.38957709293155e-13, places=20)
        budget = v65_gpu_memory_budget(num_envs_per_gpu=2048, num_uavs=6, grid_w=20)
        self.assertEqual(budget["fits_single_t4_16gib"], 1.0)
        self.assertLess(budget["network_soa_gib"], 0.8)

    def test_compact3_gpu_helper_skips_dead_peers(self):
        actions = np.array([[2, 1, 0]], dtype=np.int32)
        pos = np.array([[[0.0, 0.0], [10.0, 0.0], [50.0, 0.0]]], dtype=np.float32)
        alive = np.array([[True, False, True]], dtype=bool)
        dst = resolve_compact3_actions_np(actions, pos, alive, enable_peer_dissemination=True)
        # Peer 1 is closer (10 m) but dead -> must pick peer 2 (50 m)
        self.assertEqual(int(dst[0, 0]), 2)
        self.assertEqual(int(dst[0, 1]), 3)  # GCS
        self.assertEqual(int(dst[0, 2]), -1)

    def test_phy_numpy_underground_segment_does_not_hit_building(self):
        pos = np.zeros((1, 7, 3), dtype=np.float32)
        pos[0, 0] = [240.0, 280.0, -5.0]
        pos[0, 1] = [320.0, 280.0, -5.0]
        g = compute_channel_gain_3d_nlos_numpy(pos)
        d = 80.0
        g0 = (299_792_458.0 / (4.0 * math.pi * 2.4e9)) ** 2
        self.assertAlmostEqual(float(g[0, 0, 1]), g0 / (d * d), places=12)


if __name__ == "__main__":
    unittest.main()
