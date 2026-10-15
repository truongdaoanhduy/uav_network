"""Real device-vector preflight must collect, reset and update both learners."""
import math
import unittest

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
