"""Runtime reports actual devices and benchmarks the real small scenario."""
from copy import deepcopy
import unittest

import torch

from uav_network.configuration import load_config
from uav_network.experiments.hardware import hardware_info
from uav_network.experiments.training import choose_device
from uav_network.experiments.benchmark import benchmark


class RuntimeTests(unittest.TestCase):
    def test_device_selection_does_not_silently_ignore_explicit_cuda(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(choose_device("auto"), expected)
        self.assertEqual(hardware_info()["cuda_available"], torch.cuda.is_available())
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                choose_device("cuda")

    def test_profiles_validate_for_both_learners(self):
        for name in ("default", "smoke", "smoke-parallel", "workstation"):
            c = load_config(f"configs/{name}.yaml")
            for algorithm in ("masac", "matd3"):
                copy = deepcopy(c)
                copy.runtime.algorithm = algorithm
                copy.validate()

    def test_motion_gpu_profile_uses_parallel_off_policy_training_budget(self):
        c = load_config("configs/motion-gpu.yaml")
        self.assertEqual(c.world.num_targets, 25)
        self.assertEqual(c.world.num_obstacles, 15)
        self.assertEqual(c.runtime.num_envs, 512)
        self.assertEqual(c.runtime.log_interval_steps, 51200)
        for learner in (c.masac, c.matd3):
            self.assertEqual(learner.replay_capacity, 200000)
            self.assertEqual(learner.batch_size, 512)
            self.assertEqual(learner.learning_starts, 16384)
            self.assertEqual(learner.updates_per_step, 8)

    def test_benchmark_reports_separate_collection_and_learning_rates(self):
        c = load_config("configs/smoke.yaml")
        c.world.max_steps = 4
        c.masac.batch_size = 4
        report = benchmark(c, env_counts=[1], steps=4, learner_updates=2)
        result = report["results"][0]
        self.assertTrue(report["network_enabled"])
        self.assertEqual(result["joint_transitions"], 4)
        self.assertEqual(result["learner_batch_size"], 4)
        self.assertGreater(result["joint_transitions_per_s"], 0)
        self.assertGreater(result["learner_updates_per_s"], 0)


if __name__ == "__main__":
    unittest.main()
