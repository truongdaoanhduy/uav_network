"""V87 tests: pre-reset completed-episode tracking (IQM), grad_clip_norm=0 canonical support, checkpoint contracts."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/gpu")]

from episode_metrics_v87 import CompletedEpisodeTracker, compute_iqm
from uav_network.algorithms.configuration.loader import load_learner_config
from uav_network.algorithms.configuration.schema import LearnerConfig

EP_MOD = ROOT / "scripts/gpu/episode_metrics_v87.py"
CANONICAL_MOD = ROOT / "uav_network/algorithms/baselines/canonical_continuous.py"
RUNNER = ROOT / "scripts/gpu/run_v81_real_rollout.py"
SCHEMA = ROOT / "uav_network/algorithms/configuration/schema.py"
CANONICAL_YAML = ROOT / "configs/learners/v85_canonical_matd3_masac_baseline.yaml"


class TestV87EpisodeMetricsAndCheckpointContracts(unittest.TestCase):
    def test_v87_iqm_matches_middle_50_percent_mean(self):
        vals = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 1000.0]
        # Middle 50% (indices 2..6): [20, 30, 40, 50] -> mean 35.0
        self.assertAlmostEqual(compute_iqm(vals), 35.0, places=9)
        self.assertAlmostEqual(compute_iqm([]), 0.0, places=9)

    def test_v87_completed_episode_tracker_captures_metrics_before_reset(self):
        """Regression for V86 E64x900 blind spot: captures delivery/coverage/return before auto-reset."""
        tr = CompletedEpisodeTracker(num_envs=2, e_max_j=180_000.0, num_targets=15)
        # Step 1: neither env finishes
        c1 = tr.record_step_np([1.5, -0.5], [0, 0], [0, 0], [3, 1], [0, 0], [20.0, 10.0], [90.0, 90.0])
        self.assertEqual(c1, 0)
        # Step 2: env 0 finishes with all 15 targets confirmed (bit 4); env 1 truncates at horizon (bit 8)
        c2 = tr.record_step_np([25.0, 0.5], [1, 1], [4, 8], [15, 6], [0, 1], [85.0, 45.0], [72.0, 25.0])
        self.assertEqual(c2, 2)
        # Running accumulators reset cleanly to 0 after completion
        np.testing.assert_array_equal(tr.running_return, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(tr.running_length, np.array([0, 0]))
        summary = tr.summary()
        self.assertEqual(summary["completed_episodes"], 2)
        self.assertAlmostEqual(summary["metrics"]["return"]["mean"], 13.25, places=6)
        self.assertAlmostEqual(summary["metrics"]["gcs_delivery_pct"]["mean"], 70.0, places=6)
        self.assertAlmostEqual(summary["metrics"]["full15_success"]["mean"], 0.5, places=6)
        self.assertAlmostEqual(summary["metrics"]["safety_failed"]["mean"], 0.0, places=6)

    def test_v87_schema_allows_zero_grad_clip_for_canonical_baselines(self):
        cfg = load_learner_config(CANONICAL_YAML)
        self.assertEqual(cfg.matd3.grad_clip_norm, 0.0)
        self.assertEqual(cfg.masac.grad_clip_norm, 0.0)
        bad = LearnerConfig()
        bad.matd3.grad_clip_norm = -1.0
        with self.assertRaises(ValueError):
            bad.validate()

    def test_v87_source_contracts_and_500_line_policy(self):
        for path in (EP_MOD, CANONICAL_MOD, RUNNER, SCHEMA):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            self.assertLessEqual(len(text.splitlines()), 500, f"{path.name} exceeded 500 lines")
        r_src = RUNNER.read_text(encoding="utf-8")
        self.assertIn("ep_tracker.record_step_world", r_src)
        self.assertLess(r_src.rindex("ep_tracker.record_step_world"), r_src.rindex("world.auto_reset_gpu()"))
        self.assertIn("--save-checkpoint", r_src)
        self.assertIn("--eval-episodes", r_src)
        c_src = CANONICAL_MOD.read_text(encoding="utf-8")
        self.assertIn("def save(self, path)", c_src)
        self.assertIn("def load(cls, path", c_src)


if __name__ == "__main__":
    unittest.main()
