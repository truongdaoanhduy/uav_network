"""Real probe, resource-limit classification, and stop-at-first-failure contract."""
import importlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


class EnvironmentScaleTests(unittest.TestCase):
    def module(self):
        try:
            module = importlib.import_module("uav_network.experiments.env_scale")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "The measured env-scale runner is missing")
        return module

    def test_cuda_oom_stops_before_sibling_and_larger_counts_and_preserves_results(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            def runner(count, algorithm, folder):
                Path(folder, "attempted").touch()
                return {"num_envs": count, "algorithm": algorithm,
                        "status": "cuda_oom" if count == 256 else "ok"}
            result = module.sweep([128, 256, 512, 1024, 2048, 4096], ["masac", "matd3"],
                                  Path(directory) / "sweep", runner)
            self.assertEqual([(r["num_envs"], r["algorithm"]) for r in result["results"]],
                             [(128, "masac"), (128, "matd3"), (256, "masac")])
            self.assertEqual(result["stop_reason"], "cuda_oom")
            self.assertEqual(result["skipped_counts"], [512, 1024, 2048, 4096])
            saved = json.loads(Path(directory, "sweep/results.json").read_text())
            self.assertEqual(saved["results"], result["results"])
            self.assertFalse(Path(directory, "sweep/256-matd3").exists())
            self.assertFalse(Path(directory, "sweep/512-masac").exists())

    def test_cpu_limit_stops_with_a_cpu_reason_instead_of_cuda_oom(self):
        module = self.module()
        self.assertEqual(module.guard_reason({"memory_used_bytes": 900, "memory_limit_bytes": 1000}, .85),
                         "cpu_ram_guard")
        self.assertEqual(module.guard_reason({"pids_current": 91, "pids_limit": 100}, .85), "cpu_pid_guard")
        self.assertIsNone(module.guard_reason({"memory_used_bytes": 50, "memory_limit_bytes": 100,
                                              "pids_current": 10, "pids_limit": 100}, .85))

    def test_existing_output_is_not_overwritten_or_reexecuted(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            Path(directory, "results.json").write_text("prior evidence")
            def runner(*args):
                raise AssertionError("An existing report must prevent execution")
            with self.assertRaises(FileExistsError):
                module.sweep([128], ["masac"], Path(directory), runner)
            self.assertEqual(Path(directory, "results.json").read_text(), "prior evidence")

    def test_timeout_terminates_a_real_child_process(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            result = module.supervise([sys.executable, "-c", "import time; time.sleep(5)"],
                                      Path(directory), timeout=.2, memory_fraction=.99)
            self.assertEqual(result["status"], "case_timeout")
            self.assertNotEqual(result["exit_code"], 0)

    def test_native_two_worker_probe_runs_real_policy_updates_for_both_algorithms(self):
        from uav_network.configuration import Config
        module = self.module()
        with TemporaryDirectory() as directory:
            for algorithm in ("masac", "matd3"):
                c = Config()
                c.runtime.algorithm = algorithm
                c.runtime.num_envs = 2
                c.runtime.device = c.runtime.replay_device = "cpu"
                c.runtime.vectorize_critics = True
                c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 100
                settings = c.algorithm_config
                settings.hidden_sizes, settings.batch_size = [32, 32], 8
                settings.replay_capacity, settings.learning_starts = 32, 8
                result = module.probe(c, Path(directory) / algorithm, steps=2)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["num_envs"], 2)
                self.assertEqual(result["global_step"], 12)
                self.assertEqual(result["measured_updates"], 2)
                self.assertEqual(result["total_updates"], 4)
                self.assertEqual(len(set(result["native_worker_pids"])), 2)
                self.assertNotIn(os.getpid(), result["native_worker_pids"])
                self.assertGreater(result["joint_transitions_per_s"], 0)
                self.assertEqual(result["network_backend"], "uavnetsim")


if __name__ == "__main__":
    unittest.main()
