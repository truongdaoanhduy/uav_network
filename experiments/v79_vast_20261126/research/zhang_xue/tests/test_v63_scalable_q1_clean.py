"""v6.3 Q1-Clean & Scalable (N-UAV, Grid, Map, Compact-3 Action, 2D Velocity, UavNetSim PHY) Tests."""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhang_xue.environment.configuration import make_v63_q1_config
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.layout_v62 import backing_bytes_per_environment_v63
from zhang_xue.learners.actor_adapter import decode_policy_action_2d, encode_actor_input_scalable


class TestV63ScalableQ1Clean(unittest.TestCase):
    def test_v63_scales_cleanly_across_4_6_8_10_uavs(self):
        for n_uav in (4, 6, 8, 10):
            cfg = make_v63_q1_config(num_uavs=n_uav)
            env = ZhangXue7NodePacketEnv(cfg, seed=100 + n_uav)
            obs, state, mask = env.reset(seed=100 + n_uav)
            self.assertEqual(obs.shape, (n_uav, 514))
            self.assertEqual(state.shape, (n_uav * 514,))
            # compact_3 action space is N-invariant (always 3!) -> zero-shot transferable across swarm sizes!
            self.assertEqual(mask.shape, (n_uav, 3))
            enc = encode_actor_input_scalable(obs, mask)
            self.assertEqual(enc.shape, (n_uav, 518))
            vel, comm = decode_policy_action_2d(
                np.zeros((n_uav, 2)), np.zeros(n_uav, dtype=np.int64), mask, v_max_mps=cfg.v_max_mps
            )
            obs2, _, _, rew, term, trunc, info = env.step(vel, comm)
            self.assertEqual(obs2.shape, (n_uav, 514))
            self.assertTrue(np.isfinite(rew))
            self.assertFalse(term)
            self.assertFalse(trunc)

    def test_v63_scales_to_40x40_grid_and_2km_map(self):
        cfg = make_v63_q1_config(num_uavs=8, grid_w=40, map_size_m=2000.0, num_targets=25)
        env = ZhangXue7NodePacketEnv(cfg, seed=202)
        obs, _, mask = env.reset(seed=202)
        self.assertEqual(obs.shape, (8, 514))
        self.assertEqual(env.num_cells, 1600)
        self.assertEqual(int(env.true_targets.sum()), 25)
        obs2, _, _, rew, _, _, _ = env.step(np.zeros((8, 2)), np.zeros(8, dtype=np.int32))
        self.assertTrue(np.isfinite(obs2).all() and np.isfinite(rew))

    def test_v63_uavnetsim_phy_and_no_bloat_defaults(self):
        cfg = make_v63_q1_config()
        self.assertAlmostEqual(cfg.tx_power_w, 0.1, places=6)
        self.assertAlmostEqual(cfg.mac_cca_threshold_dbm, -82.0, places=6)
        self.assertFalse(cfg.enable_tx_power_control)
        self.assertFalse(cfg.enable_accel_energy)
        self.assertEqual(cfg.evidence_calibration, "paper_fixed")
        self.assertAlmostEqual(cfg.lambda_b, 0.0, places=6)
        self.assertTrue(cfg.aoi_ewma_end_to_end)
        self.assertEqual(cfg.comm_action_mode, "compact_3")

    def test_v63_parameterized_gpu_soa_bytes(self):
        b4 = backing_bytes_per_environment_v63(num_uavs=4, grid_w=20)
        b6 = backing_bytes_per_environment_v63(num_uavs=6, grid_w=20)
        b10 = backing_bytes_per_environment_v63(num_uavs=10, grid_w=20)
        self.assertLess(b4, b6)
        self.assertLess(b6, b10)
        self.assertLess(b6, 400_000)


if __name__ == "__main__":
    unittest.main()
