"""Real device-vector preflight must collect, reset and update both learners."""
import math
import unittest
import torch

from uav_network.configuration import Config
from uav_network.experiments.benchmark import benchmark


def check_warp_benchmark(device):
    for algorithm in ("masac", "matd3"):
        c = Config()
        c.runtime.algorithm = algorithm
        c.runtime.device = c.runtime.replay_device = device
        c.runtime.environment_backend = "warp_packet"
        c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 1
        settings = c.algorithm_config
        settings.hidden_sizes, settings.replay_capacity = [16, 16], 16
        settings.batch_size = settings.learning_starts = 2
        report = benchmark(c, env_counts=(2,), steps=2, learner_updates=1)["results"][0]
        assert report["joint_transitions"] == 4
        assert report["learner_updates"] == 1
        assert report["reset_s"] > 0
        assert math.isfinite(report["joint_transitions_per_s"])
        assert report["joint_transitions_per_s"] > 0


class BenchmarkDeviceTests(unittest.TestCase):
    def test_warp_tensor_preflight_collects_resets_and_updates_both_learners(self):
        check_warp_benchmark("cpu")

    def test_dual_gpu_evidence_requires_roles_actor_update_and_memory_on_both_boards(self):
        from uav_network.experiments.benchmark import dual_gpu_verified
        devices = {"collector": "cuda:0", "learner": "cuda:1", "replay": "cuda:1"}
        peaks = [{"device": 0, "peak_allocated_bytes": 10}, {"device": 1, "peak_allocated_bytes": 20}]
        self.assertTrue(dual_gpu_verified(devices, actor_updates=1, peak_memory=peaks))
        self.assertFalse(dual_gpu_verified(devices, actor_updates=0, peak_memory=peaks))
        self.assertFalse(dual_gpu_verified({**devices, "learner": "cuda:0"}, actor_updates=1, peak_memory=peaks))
        self.assertFalse(dual_gpu_verified(devices, actor_updates=1,
                                           peak_memory=[peaks[0], {"device": 1, "peak_allocated_bytes": 0}]))

class BenchmarkOomReportingTests(unittest.TestCase):
    def test_base_benchmark_reports_cuda_oom_instead_of_raising(self):
        from unittest.mock import patch
        from uav_network.configuration import Config
        from uav_network.experiments.benchmark import benchmark
        c = Config()
        c.runtime.device = c.runtime.replay_device = "cpu"
        c.runtime.environment_backend = "native"
        with patch("uav_network.experiments.benchmark.make_vector", side_effect=torch.OutOfMemoryError("probe oom")):
            report = benchmark(c, env_counts=(1,), steps=1, learner_updates=1)
        row = report["results"][0]
        self.assertEqual(row["status"], "oom")
        self.assertEqual(row["error_type"], "OutOfMemoryError")
