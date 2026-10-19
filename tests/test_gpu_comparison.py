"""Matched schedules, real CPU collectors, physical GPU samples and isolated cases."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
import yaml
from uav_network.configuration import Config


def tiny_config(algorithm="masac"):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 1
    c.world.max_steps = 5
    c.runtime.algorithm = algorithm
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.environment_backend = "warp_packet"
    c.runtime.num_envs = 4
    c.runtime.torch_threads = 1
    c.runtime.log_interval_steps = 32
    c.tracking.enabled = False
    c.pql.enabled, c.pql.mode = True, "async"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.pql.warmup_collection_steps, c.pql.critic_sample_ratio = 32, 8
    for settings in (c.masac, c.matd3):
        settings.hidden_sizes = [16, 16]
        settings.batch_size, settings.replay_capacity = 8, 128
        settings.learning_starts, settings.updates_per_step = 0, 1
    return c.validate()


class GPUResourceTests(unittest.TestCase):
    def module(self):
        name = "uav_network.experiments.gpu_resources"
        self.assertIsNotNone(importlib.util.find_spec(name), "Physical GPU sampler is missing")
        return importlib.import_module(name)

    def samples(self, module, values):
        result = []
        for timestamp, power in values:
            result.extend(module.parse_gpu_samples(
                f"GPU-a, 0, 50, 100, {power}\nGPU-b, 1, 80, 200, {power}\n", timestamp))
        return result

    def test_parse_all_physical_gpu_rows_and_missing_power(self):
        module = self.module()
        samples = module.parse_gpu_samples(
            "GPU-a, 0, 15, 1024, 70.5\nGPU-b, 3, N/A, 2048, [Not Supported]\n", 2.5)
        self.assertEqual([row["physical_index"] for row in samples], [0, 3])
        self.assertEqual([row["uuid"] for row in samples], ["GPU-a", "GPU-b"])
        self.assertEqual(samples[0]["timestamp"], 2.5)
        self.assertEqual(samples[0]["memory_used_bytes"], 1024 ** 3)
        self.assertEqual(samples[0]["power_w"], 70.5)
        self.assertIsNone(samples[1]["power_w"])
        self.assertIsNone(samples[1]["utilization_gpu_percent"])

    def test_power_integrates_only_between_samples_inside_the_run_window(self):
        module = self.module()
        samples = self.samples(module, [(0, 100), (2, 200), (4, 100)])
        report = module.summarize_gpu_samples(samples, 1, 3)
        self.assertEqual(report["scope"], "physical_whole_gpu")
        self.assertEqual(len(report["gpus"]), 2)
        for gpu in report["gpus"]:
            self.assertAlmostEqual(gpu["power"]["energy_joules"], 350)
            self.assertAlmostEqual(gpu["power"]["coverage_seconds"], 2)
            self.assertAlmostEqual(gpu["power"]["coverage_fraction"], 1)
            self.assertEqual(gpu["power"]["status"], "available")
        wide = module.summarize_gpu_samples(samples, -1, 5)
        self.assertAlmostEqual(wide["gpus"][0]["power"]["energy_joules"], 600)
        self.assertAlmostEqual(wide["gpus"][0]["power"]["coverage_fraction"], 4 / 6)

    def test_missing_power_is_unavailable_and_is_never_interpolated_across(self):
        module = self.module()
        samples = self.samples(module, [(0, 100), (1, "N/A"), (2, 200)])
        report = module.summarize_gpu_samples(samples, 0, 2)
        for gpu in report["gpus"]:
            self.assertEqual(gpu["power"]["status"], "unavailable")
            self.assertIsNone(gpu["power"]["energy_joules"])
            self.assertEqual(gpu["power"]["coverage_seconds"], 0)

    def test_sampler_does_not_interpolate_power_across_a_long_poll_gap(self):
        module = self.module()
        sampler = module.GPUResourceSampler(interval_seconds=.25)
        sampler.samples = self.samples(module, [(0, 100), (10, 100)])
        with patch.object(sampler, "_sample"):
            report = sampler.finish(3, 7)
        for gpu in report["gpus"]:
            self.assertEqual(gpu["sample_count"], 0)
            self.assertEqual(gpu["power"]["status"], "unavailable")
            self.assertEqual(gpu["power"]["coverage_fraction"], 0)
            self.assertIsNone(gpu["power"]["energy_joules"])

    def test_sampler_reports_missing_nvidia_smi_without_failing_the_benchmark(self):
        module = self.module()
        with patch.object(module.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi")):
            sampler = module.GPUResourceSampler()
            sampler.start()
            report = sampler.finish(0, 1)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["gpus"], [])
        self.assertIn("nvidia-smi", report["error"])


@unittest.skipUnless(importlib.util.find_spec("warp"), "Warp optional dependency unavailable")
class LegacyComparisonTests(unittest.TestCase):
    def module(self):
        name = "uav_network.experiments.legacy_benchmark"
        self.assertIsNotNone(importlib.util.find_spec(name), "Fixed legacy scheduler benchmark is missing")
        return importlib.import_module(name)

    def test_real_legacy_and_async_pql_complete_the_same_matched_update_budget(self):
        module = self.module()
        from uav_network.experiments.pql_benchmark import benchmark_pql
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for algorithm in ("masac", "matd3"):
                expected_actors = 72 if algorithm == "masac" else 36
                c = tiny_config(algorithm)
                c.runtime.output_dir = str(root / (algorithm + "-pql"))
                pql = benchmark_pql(c, steps=40)
                self.assertEqual(pql["status"], "passed", pql)
                self.assertEqual(pql["critic_updates"], 72)
                self.assertEqual(pql["actor_updates"], expected_actors)
                for overlap in (True, False):
                    with self.subTest(algorithm=algorithm, collector_overlap=overlap):
                        c.runtime.collector_overlap = overlap
                        c.runtime.output_dir = str(root / f"{algorithm}-legacy-{overlap}")
                        report = module.benchmark_legacy(c, steps=40)
                        self.assertEqual(report["status"], "passed", report)
                        self.assertEqual(report["critic_updates"], pql["critic_updates"])
                        self.assertEqual(report["actor_updates"], pql["actor_updates"])
                        self.assertEqual(report["learning_starts"], 32 * 4)
                        self.assertEqual(report["updates_per_step"], 8)
                        self.assertEqual(report["collector_overlap"], overlap)
                        self.assertEqual(report["devices"], dict(collector="cpu", critic="cpu", policy="cpu"))
                        self.assertGreater(report["environment_bytes"], 0)
                        self.assertGreater(report["replay_bytes"], 0)
                        self.assertGreater(report["completed_episodes"], 0)
                        self.assertEqual(report["completed_episodes"], pql["completed_episodes"])
                        self.assertEqual(report["reset_collections"], pql["reset_collections"])
                        self.assertEqual(pql["reset_collections"], 7)
                        self.assertGreater(report["joint_transitions_per_s"], 0)
                        self.assertGreaterEqual(report["total_wall_seconds"], report["elapsed_s"])
                        self.assertEqual(report["legacy_reference_commit"],
                                         "afeaa3dba27edd736be13f84b602f84c268e3a22")
                        for artifact in ("resolved_config.yaml", "diagnostics.jsonl", "episodes.jsonl",
                                         "benchmark.json"):
                            self.assertTrue(Path(c.runtime.output_dir, artifact).is_file())
                        self.assertTrue(Path(c.runtime.output_dir, "diagnostics.jsonl").read_text().strip())
                self.assertTrue(c.pql.enabled)
                self.assertEqual(c.algorithm_config.learning_starts, 0)

    def test_final_collection_does_not_reset_worlds_or_lose_episode_metrics(self):
        module = self.module()
        from uav_network.experiments.pql_benchmark import benchmark_pql
        with TemporaryDirectory() as folder:
            c = tiny_config("matd3")
            c.world.max_steps = 3
            c.pql.warmup_collection_steps, c.pql.critic_sample_ratio = 2, 1
            c.runtime.output_dir = str(Path(folder) / "pql")
            result = benchmark_pql(c, steps=6)
            self.assertEqual(result.get("reset_collections"), 1)
            self.assertEqual(result.get("completed_episodes"), 8)
            self.assertEqual(len(Path(c.runtime.output_dir, "episodes.jsonl").read_text().splitlines()), 8)
            c.runtime.output_dir = str(Path(folder) / "legacy")
            legacy = module.benchmark_legacy(c, steps=6)
            self.assertEqual(legacy.get("reset_collections"), 1)
            self.assertEqual(legacy.get("completed_episodes"), result["completed_episodes"])

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_legacy_device_resolution_can_split_collector_from_learner(self, _count, _available):
        module = self.module()
        c = tiny_config("matd3")
        c.runtime.device = c.runtime.replay_device = "cuda"
        c.pql.collector_device = "cuda:0"
        c.pql.learner_device = c.pql.policy_device = "cuda:1"
        collector, learner = module.resolve_legacy_devices(c)
        self.assertEqual(str(collector), "cuda:0")
        self.assertEqual(str(learner), "cuda:1")

    def test_legacy_refuses_fractional_update_ratio(self):
        module = self.module()
        c = tiny_config()
        c.pql.critic_sample_ratio = 2.5
        with self.assertRaisesRegex(ValueError, "integer"):
            module.benchmark_legacy(c, steps=40)

    def test_legacy_reports_oom_without_changing_requested_batch(self):
        module = self.module()
        with TemporaryDirectory() as folder:
            c = tiny_config()
            c.runtime.output_dir = folder
            with patch.object(module, "make_learner", side_effect=torch.OutOfMemoryError("test allocation")):
                result = module.benchmark_legacy(c, steps=40)
            self.assertEqual(result["status"], "oom")
            self.assertEqual(result["batch_size"], 8)
            self.assertEqual(result["error_type"], "OutOfMemoryError")
            self.assertTrue(Path(folder, "benchmark.json").is_file())


class ComparisonRunnerTests(unittest.TestCase):
    def module(self):
        name = "scripts.compare_training_gpu"
        self.assertIsNotNone(importlib.util.find_spec(name), "Isolated comparison runner is missing")
        return importlib.import_module(name)

    def test_algorithm_flag_and_case_roles_preserve_the_selected_algorithm_settings(self):
        module = self.module()
        c = tiny_config()
        c.masac.batch_size, c.matd3.batch_size = 4, 12
        c.tracking.enabled = True
        for case in ("legacy", "pql-single"):
            with self.subTest(case=case):
                selected = module.case_config(c, case, "matd3", "unused-output", allow_cpu=True)
                self.assertEqual(selected.runtime.algorithm, "matd3")
                self.assertEqual(selected.algorithm_config.batch_size, 12)
                self.assertEqual(selected.algorithm_config.learning_starts, 128)
                self.assertEqual(selected.algorithm_config.updates_per_step, 8)
                self.assertFalse(selected.tracking.enabled)
                self.assertEqual(selected.pql.enabled, case != "legacy")
                self.assertEqual(selected.pql.mode, "async")
                self.assertEqual(selected.pql.learner_device, "cpu")
        self.assertEqual(c.runtime.algorithm, "masac")
        self.assertTrue(c.tracking.enabled)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_base_dual_routes_collector_and_learner_to_separate_gpus(self, _count, _available):
        module = self.module()
        c = tiny_config()
        c.runtime.device = c.runtime.replay_device = "cuda"
        selected = module.case_config(c, "base-dual", "matd3", "unused-output")
        self.assertFalse(selected.pql.enabled)
        self.assertEqual(selected.pql.collector_device, "cuda:0")
        self.assertEqual(selected.pql.learner_device, "cuda:1")
        self.assertEqual(selected.pql.policy_device, "cuda:1")

    def test_cpu_requires_explicit_permission_and_dual_gpu_is_explicitly_unavailable(self):
        module = self.module()
        c = tiny_config()
        with self.assertRaisesRegex(module.UnavailableCase, "CUDA"):
            module.case_config(c, "legacy", "masac", "unused-output")
        with self.assertRaisesRegex(module.UnavailableCase, "two.*CUDA"):
            module.case_config(c, "pql-dual", "masac", "unused-output", allow_cpu=True)

    @unittest.skipUnless(importlib.util.find_spec("warp"), "Warp optional dependency unavailable")
    def test_parent_launches_each_case_in_a_fresh_process_and_continues_after_unavailable_dual(self):
        self.module()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            c = tiny_config()
            c.pql.warmup_collection_steps, c.pql.critic_sample_ratio = 2, 2
            c.tracking.enabled = True
            profile = root / "profile.yaml"
            profile.write_text(yaml.safe_dump(c.to_dict()))
            target = root / "report.json"
            process = subprocess.run([sys.executable, "scripts/compare_training_gpu.py",
                "--config", str(profile), "--algorithms", "matd3", "--steps", "4",
                "--set", "runtime.algorithm=masac", "--allow-cpu", "--output", str(target)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stderr[-3000:])
            report = json.loads(target.read_text())
            self.assertEqual(report["status"], "partial")
            cases = report["results"]
            self.assertEqual([case["case"] for case in cases], ["legacy", "pql-single", "pql-dual"])
            self.assertEqual([case["status"] for case in cases], ["passed", "passed", "unavailable"])
            self.assertEqual(len({case["process_id"] for case in cases}), 3)
            for case in cases[:2]:
                self.assertEqual(case["algorithm"], "matd3")
                self.assertEqual(case["critic_updates"], 6)
                self.assertEqual(case["actor_updates"], 3)
                self.assertEqual(case["batch_size"], 8)
                self.assertFalse(case["cuda_pipeline_verified"])
                self.assertFalse(case["tracking_enabled"])
                self.assertEqual(case["updates_per_step"], 2)
                self.assertEqual(case["learning_starts"], 8)
            self.assertFalse(report["gpu_comparison_verified"])


if __name__ == "__main__":
    unittest.main()
