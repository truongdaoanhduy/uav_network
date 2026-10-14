"""Real CPU Warp collector, deterministic debug, resume and device failures."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from uav_network.experiments.training import train
from test_pql_system import pql_config


@unittest.skipUnless(importlib.util.find_spec("warp"), "Warp optional dependency unavailable")
class PQLTrainingTests(unittest.TestCase):
    def config(self, directory, algorithm="masac", mode="async"):
        c = pql_config(algorithm, mode)
        c.runtime.environment_backend = "warp_packet"
        c.runtime.output_dir = str(directory)
        c.runtime.episodes = c.runtime.num_envs
        c.world.max_steps = 4
        c.runtime.checkpoint_interval = 2
        c.runtime.log_interval_steps = 4
        return c

    def test_real_gpu_backend_collector_uses_pql_workers_without_cpu_env_processes(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as folder:
                result = train(self.config(Path(folder) / "run", algorithm))
                self.assertEqual(result.get("training_architecture"), "pql-inspired")
                self.assertEqual(result["num_cpu_env_workers"], 0)
                self.assertEqual(result["episodes"], 4)
                self.assertEqual(result["updates"], 7)
                self.assertEqual(result["pql"]["pql/actor_updates"], 7 if algorithm == "masac" else 3)
                self.assertTrue(Path(result["resume_checkpoint"]).is_file())
                policy = torch.load(result["checkpoint"], weights_only=True)
                self.assertEqual(policy["version"], 2)
                self.assertNotIn("training", policy)

    def test_synchronous_mode_repeats_scientific_outcomes_and_weights(self):
        with TemporaryDirectory() as folder:
            results = [train(self.config(Path(folder) / name, mode="synchronous"))
                       for name in ("first", "second")]
            for key in ("model_hash", "episode_hash", "settings_hash"):
                self.assertEqual(results[0]["reproducibility"][key], results[1]["reproducibility"][key])
            self.assertTrue(results[0]["reproducibility"]["scheduling_deterministic"])

    def test_resume_retains_episode_prefix_optimizer_replay_and_rate_counters(self):
        with TemporaryDirectory() as folder:
            first = train(self.config(Path(folder) / "first", "matd3", "synchronous"))
            c = self.config(Path(folder) / "second", "matd3", "synchronous")
            self.assertIn("resume_checkpoint", first, "Missing complete PQL training checkpoint")
            c.runtime.warm_start_checkpoint = first["resume_checkpoint"]
            c.runtime.episodes = 8
            second = train(c)
            self.assertEqual(second["initial_episodes"], 4)
            self.assertEqual(second["episodes"], 8)
            self.assertEqual(second["initial_global_step"], 16)
            self.assertEqual(second["updates"], 17)
            self.assertEqual(second["state_restore"], "training_state_environment_reset")
            self.assertTrue(second["optimizer_replay_restored"])
            saved = torch.load(second["resume_checkpoint"], weights_only=True)
            self.assertEqual(len(saved["progress"]["records"]), 8)
            self.assertEqual(saved["training"]["collections"], 8)

    def test_target_profile_and_real_small_benchmark(self):
        from uav_network.configuration import load_config
        self.assertTrue(Path("configs/motion-pql.yaml").is_file(), "Missing target PQL profile")
        c = load_config("configs/motion-pql.yaml")
        self.assertEqual(c.runtime.num_envs, 4096)
        self.assertEqual(c.pql.critic_sample_ratio, 8)
        self.assertEqual(c.pql.warmup_collection_steps, 32)
        for learner in (c.masac, c.matd3):
            self.assertEqual(learner.batch_size, 8192)
            self.assertEqual(learner.replay_capacity, 500000)
        from uav_network.experiments.pql_benchmark import benchmark_pql
        with TemporaryDirectory() as folder:
            report = benchmark_pql(self.config(Path(folder) / "benchmark"), steps=4)
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["cuda_pipeline_verified"])
            self.assertFalse(report["target_t4_scenario_verified"])
            self.assertEqual(report["critic_updates"], 7)
            self.assertGreater(report["joint_transitions_per_s"], 0)

    def test_benchmark_cannot_pass_without_both_learner_updates(self):
        from uav_network.experiments.pql_benchmark import benchmark_pql
        with TemporaryDirectory() as folder:
            c = self.config(Path(folder) / "empty-benchmark")
            c.pql.critic_sample_ratio = .01
            with self.assertRaisesRegex(ValueError, "critic.*policy"):
                benchmark_pql(c, steps=4)

    def test_explicit_absent_cuda_is_rejected_without_silent_cpu_fallback(self):
        from uav_network.experiments.pql_system import resolve_devices
        c = pql_config()
        c.runtime.device = "cuda"
        with patch("torch.cuda.device_count", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                resolve_devices(c)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_worker_snapshots_and_gpu_collector_smoke(self):
        with TemporaryDirectory() as folder:
            c = self.config(Path(folder) / "gpu")
            c.runtime.device = c.runtime.replay_device = "cuda"
            c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "auto"
            result = train(c)
            self.assertTrue(result["devices"]["collector"].startswith("cuda"))
            self.assertTrue(result["devices"]["critic"].startswith("cuda"))
            self.assertEqual(result["num_cpu_env_workers"], 0)


if __name__ == "__main__":
    unittest.main()
