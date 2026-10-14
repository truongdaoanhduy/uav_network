"""Pinned comparison jobs measure legacy failures without launching production training."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.configuration import load_config
from uav_network.cloud import cli, remote


class ComparisonCloudTests(unittest.TestCase):
    def test_comparison_dispatch_uses_exact_profile_and_removes_launch_credentials(self):
        self.assertTrue(callable(getattr(remote, "comparison_benchmarks", None)),
                        "Missing cloud matched comparison runner")
        c = load_config("configs/motion-pql.yaml")
        spec = {"stage": "benchmark", "config": "configs/motion-pql.yaml", "overrides": [],
                "runs": {"masac": {}, "matd3": {}}}
        def child(args, **kwargs):
            self.assertIn("scripts/compare_training_gpu.py", args)
            self.assertEqual(args[args.index("--algorithms") + 1:args.index("--algorithms") + 3],
                             ["masac", "matd3"])
            self.assertEqual(args[args.index("--steps") + 1], "40")
            self.assertNotIn("KAGGLE_API_TOKEN", kwargs["env"])
            Path(args[args.index("--output") + 1]).write_text(json.dumps(
                {"results": [{"case": case, "algorithm": algorithm,
                              "status": "oom" if (case, algorithm) == ("legacy", "masac") else "passed"}
                             for algorithm in ("masac", "matd3")
                             for case in ("legacy", "pql-single", "pql-dual")]}))
            return SimpleNamespace(returncode=1)
        with TemporaryDirectory() as folder, patch.object(remote.subprocess, "run", side_effect=child), \
             patch.dict("os.environ", {"KAGGLE_API_TOKEN": "test-launch-token"}):
            report = remote.comparison_benchmarks(spec, c, Path(folder))
        self.assertEqual(report["results"][0]["status"], "oom")
        self.assertEqual(c.masac.batch_size, 8192)

    def test_incomplete_report_is_preserved_but_cannot_complete_cloud_benchmark(self):
        c = load_config("configs/motion-pql.yaml")
        spec = {"stage": "benchmark", "config": "configs/motion-pql.yaml", "overrides": [],
                "runs": {"masac": {}, "matd3": {}}}
        def child(args, **kwargs):
            Path(args[args.index("--output") + 1]).write_text(json.dumps(
                {"status": "passed", "gpu_comparison_verified": True,
                 "results": [{"case": case, "algorithm": "masac", "status": "passed"}
                             for case in ("legacy", "pql-single", "pql-dual")]}))
            return SimpleNamespace(returncode=-9)
        with TemporaryDirectory() as folder, patch.object(remote.subprocess, "run", side_effect=child):
            with self.assertRaisesRegex(RuntimeError, "incomplete|interrupted"):
                remote.comparison_benchmarks(spec, c, Path(folder))
            saved = json.loads(Path(folder, "job-result.json").read_text())
            self.assertFalse(saved["comparison_completed"])
            self.assertFalse(saved["gpu_comparison_verified"])

    def test_complete_passed_cases_with_runner_failure_are_not_accepted(self):
        c = load_config("configs/motion-pql.yaml")
        spec = {"stage": "benchmark", "config": "configs/motion-pql.yaml", "overrides": [],
                "runs": {"masac": {}}}
        def child(args, **kwargs):
            Path(args[args.index("--output") + 1]).write_text(json.dumps(
                {"status": "passed", "gpu_comparison_verified": True,
                 "results": [{"case": case, "algorithm": "masac", "status": "passed"}
                             for case in ("legacy", "pql-single", "pql-dual")]}))
            return SimpleNamespace(returncode=1)
        with TemporaryDirectory() as folder, patch.object(remote.subprocess, "run", side_effect=child):
            with self.assertRaisesRegex(RuntimeError, "incomplete|interrupted"):
                remote.comparison_benchmarks(spec, c, Path(folder))

    def test_cli_accepts_explicit_comparison_only_for_benchmark(self):
        def launch(args, credentials):
            self.assertTrue(args.compare_legacy)
            self.assertEqual(args.stage, "benchmark")
            return {"comparison": True}
        with patch.object(cli, "load_credentials", return_value=None), patch.object(cli, "launch", side_effect=launch):
            report = cli.main(["launch", "--stage", "benchmark", "--compare-legacy",
                               "--credential-dataset", "owner/private"])
        self.assertTrue(report["comparison"])


if __name__ == "__main__":
    unittest.main()
