"""V115 pure-unittest suite for C38 (unified normalized XY velocity ABI) and C42 (canonical launcher defaults)."""
from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/zhang_xue/src"))

from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.learners.actor_adapter import (
    scale_normalized_xy_to_mps,
    step_normalized_xy_2d,
)


class TestV115ActionAbiAndDefaults(unittest.TestCase):
    BASE_YAML = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"
    LAUNCHER = ROOT / "scripts/gpu/run_v115_canonical_ds_baseline.py"

    def test_c38_normalized_xy_scaling_matches_warp_unit_disk_and_steps_cpu_env(self):
        # Diagonal (1, 1) must be normalized onto unit disk so ||v|| == v_max_mps (15.0 m/s)
        norm = np.array([[1.0, 1.0], [0.5, 0.0], [-2.0, 0.0], [0.0, 0.0], [0.0, -1.0], [0.6, 0.8]], dtype=np.float32)
        vel = scale_normalized_xy_to_mps(norm, v_max_mps=15.0)
        self.assertEqual(vel.shape, (6, 2))
        speeds = np.linalg.norm(vel, axis=1)
        self.assertTrue(np.all(speeds <= 15.0 + 1e-6))
        self.assertAlmostEqual(float(speeds[0]), 15.0, places=5)
        self.assertAlmostEqual(float(speeds[1]), 7.5, places=5)
        self.assertAlmostEqual(float(speeds[2]), 15.0, places=5)
        self.assertAlmostEqual(float(speeds[3]), 0.0, places=6)

        cfg = load_scenario(self.BASE_YAML)
        env = ZhangXue7NodePacketEnv(cfg, seed=44)
        obs, _st, mask = env.reset(seed=44)
        out = step_normalized_xy_2d(env, np.zeros((6, 2), dtype=np.float32), np.zeros(6, dtype=np.int64), mask)
        self.assertEqual(out[0].shape, obs.shape)
        self.assertGreater(float(out[-1]["r_loc_team"]), 0.0)

    def test_c42_canonical_launcher_locks_ds_only_v103_and_canonical_continuous(self):
        spec = importlib.util.spec_from_file_location("run_v115_canonical_ds_baseline", self.LAUNCHER)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)

        args = argparse.Namespace(
            algorithm="matd3",
            scenario=mod.DEFAULT_SCENARIO,
            learner_config=mod.DEFAULT_LEARNER_CFG,
            envs=64,
            steps=300,
            batch_size=256,
            replay_capacity=65536,
            world_device="cuda:0",
            learner_device="cuda:0",
            seed=44,
            eval_episodes=0,
            allow_experimental_physics=True,
            output=Path("outputs/tmp_v115_dry.json"),
        )
        cmd = mod.build_command(args)
        self.assertIn("scenario_v103_ds_only_1km.yaml", " ".join(cmd))
        self.assertIn("v85_canonical_matd3_masac_baseline.yaml", " ".join(cmd))
        self.assertIn("canonical_continuous", cmd)
        self.assertIn("--allow-experimental-physics", cmd)
        self.assertIn("256", cmd)


if __name__ == "__main__":
    unittest.main()
