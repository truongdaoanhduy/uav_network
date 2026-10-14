"""Real checkpoint comparisons preserve policy bytes and use a disjoint test bank."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from uav_network.configuration import Config
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate


class ComparisonEvaluationTests(unittest.TestCase):
    def test_real_checkpoint_evaluation_emits_observed_metrics_and_seed_leakage(self):
        with TemporaryDirectory() as folder:
            c = Config()
            c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 1
            c.world.max_steps = 3
            c.network.enabled = False
            c.runtime.algorithm, c.runtime.device, c.runtime.replay_device = "matd3", "cpu", "cpu"
            c.runtime.num_envs, c.runtime.episodes = 1, 2
            c.runtime.output_dir = str(Path(folder) / "train")
            c.matd3.hidden_sizes, c.matd3.replay_capacity = [16, 16], 16
            c.matd3.batch_size = c.matd3.learning_starts = 2
            result = train(c)
            checkpoint = Path(result["checkpoint"])
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report = evaluate(checkpoint, episodes=2, seed=2000000044, output_dir=Path(folder) / "evaluation")
            self.assertIn("protocol", report, "Missing scientific evaluation protocol")
            self.assertFalse(report["protocol"]["training_seed_overlap"])
            self.assertIn("metric_summary", report)
            self.assertEqual(report["records"][0]["comparison_metrics"]["safety_violation_steps"], 0)
            self.assertEqual(len(report["records"][0]["comparison_metrics"]["first_discovery_observed_s"]), 2)
            leaked = evaluate(checkpoint, episodes=1, seed=44, output_dir=Path(folder) / "leaked")
            self.assertTrue(leaked["protocol"]["training_seed_overlap"])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)

    def test_compare_cli_runs_on_cpu_and_exports_paired_json_and_csv(self):
        import subprocess
        import sys
        with TemporaryDirectory() as folder:
            c = Config()
            c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 2
            c.network.enabled = False
            c.runtime.algorithm, c.runtime.device, c.runtime.replay_device = "matd3", "cpu", "cpu"
            c.runtime.episodes = 1
            c.runtime.output_dir = str(Path(folder) / "train")
            c.matd3.hidden_sizes, c.matd3.replay_capacity = [16, 16], 16
            c.matd3.batch_size = c.matd3.learning_starts = 2
            checkpoint = train(c)["checkpoint"]
            output = Path(folder) / "comparison"
            p = subprocess.run([sys.executable, "-m", "uav_network", "compare", "--left", checkpoint,
                                "--right", checkpoint, "--episodes", "2", "--output-dir", str(output)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr[-1500:])
            report = json.loads((output / "comparison.json").read_text())
            self.assertEqual(report["status"], "paired_scenario_comparison")
            self.assertEqual(report["metrics"]["gcs_delivery_fraction"]["difference_left_minus_right"], 0.)
            self.assertTrue((output / "comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
