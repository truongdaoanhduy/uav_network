"""PQL uses both visible GPUs per learner and validates its own architecture."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from uav_network.configuration import load_config
from uav_network.cloud import remote


class PQLCloudTests(unittest.TestCase):
    def test_allocation_keeps_both_gpus_for_pql_and_preserves_baseline_partitioning(self):
        self.assertTrue(callable(getattr(remote, "gpu_allocation", None)), "Missing PQL GPU allocation")
        c = load_config("configs/motion-pql.yaml")
        self.assertEqual(remote.gpu_allocation(c, 2, 0), (1, "0,1"))
        self.assertEqual(remote.gpu_allocation(c, 2, 1), (1, "0,1"))
        c.pql.enabled = False
        self.assertEqual(remote.gpu_allocation(c, 2, 0), (2, "0"))
        self.assertEqual(remote.gpu_allocation(c, 2, 1), (2, "1"))

    def test_pql_benchmark_gate_uses_exact_scenario_rates_batch_and_all_devices(self):
        self.assertTrue(callable(getattr(remote, "pql_benchmarks", None)), "Missing PQL benchmark gate")
        c = load_config("configs/motion-pql.yaml")
        spec = {"config": "configs/motion-pql.yaml", "overrides": [], "runs": {"masac": {}, "matd3": {}}}
        seen = []
        def child(args, **kwargs):
            seen.append(args)
            self.assertIn("scripts/benchmark_pql.py", args)
            self.assertEqual(args[args.index("--envs") + 1], "4096")
            self.assertEqual(args[args.index("--batches") + 1], "8192")
            self.assertGreater(int(args[args.index("--steps") + 1]), c.pql.warmup_collection_steps)
            output = Path(args[args.index("--output") + 1])
            output.write_text(json.dumps({"results": [{"status": "passed", "cuda_pipeline_verified": True}]}))
        with TemporaryDirectory() as folder, patch.object(remote.subprocess, "run", side_effect=child):
            result = remote.pql_benchmarks(spec, c, Path(folder))
        self.assertEqual(len(seen), 2)
        self.assertEqual(set(result), {"masac", "matd3"})

    def test_oom_gate_refuses_to_launch_training_without_changing_batch(self):
        self.assertTrue(callable(getattr(remote, "pql_benchmarks", None)), "Missing PQL benchmark gate")
        c = load_config("configs/motion-pql.yaml")
        spec = {"config": "configs/motion-pql.yaml", "overrides": [], "runs": {"masac": {}}}
        def child(args, **kwargs):
            Path(args[args.index("--output") + 1]).write_text(json.dumps(
                {"results": [{"status": "oom", "cuda_pipeline_verified": False}]}))
        with TemporaryDirectory() as folder, patch.object(remote.subprocess, "run", side_effect=child):
            with self.assertRaisesRegex(RuntimeError, "benchmark"):
                remote.pql_benchmarks(spec, c, Path(folder))
        self.assertEqual(c.masac.batch_size, 8192)


if __name__ == "__main__":
    unittest.main()
