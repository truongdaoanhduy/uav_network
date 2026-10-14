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

    def test_nested_cgroup_and_ancestor_limits_are_enforced(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            proc, group = root / "proc", root / "cgroup"
            (proc / "self").mkdir(parents=True)
            (group / "team/job").mkdir(parents=True)
            (proc / "meminfo").write_text("MemTotal: 16000 kB\nMemAvailable: 12000 kB\n")
            (proc / "stat").write_text("cpu 1 2 3 4 5 6 7 8\n")
            (proc / "self/cgroup").write_text("0::/team/job\n")
            (proc / "self/mountinfo").write_text(f"1 0 0:1 / {group} rw - cgroup2 cgroup2 rw\n")
            for relative, value in {"team/memory.max": 1000000, "team/memory.current": 950000,
                                    "team/job/memory.max": 500000, "team/job/memory.current": 10000,
                                    "team/pids.max": 64, "team/pids.current": 60,
                                    "team/job/pids.max": "max", "team/job/pids.current": 2}.items():
                (group / relative).write_text(str(value))
            sample = module.host_resources(proc)
            self.assertEqual(sample["memory_limit_bytes"], 500000)
            self.assertEqual(sample["pids_limit"], 64)
            self.assertEqual(module.guard_reason(sample, .85), "cpu_ram_guard")
            (group / "team/memory.current").write_text("1000")
            self.assertEqual(module.guard_reason(module.host_resources(proc), .85), "cpu_pid_guard")

    def test_only_intervals_wholly_inside_measurement_have_utilization_means(self):
        module = self.module()
        samples = [dict(phase="measure", sample_end_mono=11., gpu_interval_start_mono=9.,
                        cpu_interval_start_mono=9., gpu_util_percent=99., cpu_util_percent=99.),
                   dict(phase="measure", sample_end_mono=12., gpu_interval_start_mono=11.,
                        cpu_interval_start_mono=11., gpu_util_percent=50., cpu_util_percent=60.),
                   dict(phase="done", sample_end_mono=21., gpu_interval_start_mono=19.,
                        cpu_interval_start_mono=19., gpu_util_percent=99., cpu_util_percent=99.)]
        result = module.summarize(samples, measurement=(10., 20.))
        self.assertEqual(result["gpu_util_percent_mean_measure"], 50.)
        self.assertEqual(result["cpu_util_percent_mean_measure"], 60.)
        self.assertEqual(result["gpu_util_percent_samples_measure"], 1)
        self.assertEqual(result["measurement_status"], "incomplete")
        missing = module.summarize([{"phase": "measure"}], measurement=(10., 20.))
        self.assertEqual(missing["measurement_status"], "unavailable")
        self.assertNotIn("gpu_util_percent_mean_measure", missing)

    def test_unfinished_child_exit_is_a_worker_error(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            code = ("import json, pathlib; pathlib.Path(" + repr(str(Path(directory, "case.json"))) +
                    ").write_text(json.dumps({'status':'running','phase':'vector_startup'}))")
            result = module.supervise([sys.executable, "-c", code], Path(directory),
                                      timeout=5, memory_fraction=.99)
            self.assertEqual(result["status"], "worker_error")

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
