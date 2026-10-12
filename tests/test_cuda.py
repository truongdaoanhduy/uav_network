"""Real CUDA checks, explicitly skipped when this host cannot use a GPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from uav_network.algorithms.optimization import tensor_batch
from uav_network.algorithms.replay import TensorReplayBuffer
from uav_network.configuration import Config
from uav_network.environment.env import Action
from uav_network.experiments.training import train


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable on this host")
class CUDATests(unittest.TestCase):
    def test_gpu_replay_to_cpu_is_ready_for_immediate_cpu_consumption(self):
        # A blocking D2H copy is required even if the CPU learner runs quickly.
        source = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1024, 1024).cuda()
        expected = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1024, 1024)
        for _ in range(4):
            copied = tensor_batch({"state": source}, "cpu")["state"]
            torch.testing.assert_close(copied, expected, rtol=0, atol=0)

    def test_pinned_cpu_replay_samples_transfer_to_cuda_with_correct_values(self):
        replay = TensorReplayBuffer(8, (2, 10), 5, pin_memory=True)
        action = Action(np.zeros((8, 2, 3)), np.zeros((8, 2), dtype=np.int64))
        replay.add_batch(np.ones((8, 2, 10)), np.ones((8, 5)), action, np.ones(8),
                         np.ones((8, 2, 10)), np.ones((8, 5)), np.zeros(8))
        sampled = replay.sample(4)
        self.assertTrue(all(value.is_pinned() for value in sampled.values()))
        batch = tensor_batch(sampled, "cuda")
        self.assertTrue(all(value.is_cuda for value in batch.values()))
        torch.testing.assert_close(batch["reward"], torch.ones(4, device="cuda"))

    def test_both_cuda_learners_train_from_spawned_real_network_environments(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c = Config()
                c.runtime.algorithm = algorithm
                c.runtime.device = c.runtime.replay_device = "cuda"
                c.runtime.num_envs = 2
                c.runtime.episodes = 2
                c.runtime.output_dir = str(Path(directory) / "train")
                c.world.max_steps = 4
                c.world.num_targets = 2
                c.world.num_obstacles = 1
                a = c.algorithm_config
                a.hidden_sizes = [32, 32]
                a.replay_capacity = 32
                a.batch_size = a.learning_starts = 4
                result = train(c)
                self.assertEqual(result["device"], "cuda")
                self.assertEqual(result["replay_device"], "cuda")
                self.assertGreater(result["updates"], 0)
                saved = torch.load(result["checkpoint"], map_location="cuda", weights_only=True)
                self.assertTrue(all(v.is_cuda for v in saved["model"].values()))
                self.assertTrue(all(torch.isfinite(v).all() for v in saved["model"].values()))


if __name__ == "__main__":
    unittest.main()
