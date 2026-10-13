"""Real CUDA checks, explicitly skipped when this host cannot use a GPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from uav_network.algorithms.optimization import tensor_batch
from uav_network.algorithms.replay import TensorReplayBuffer
from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.algorithms.masac import MASAC
from uav_network.experiments.training import train


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable on this host")
class CUDATests(unittest.TestCase):
    def test_vectorized_cuda_critics_match_legacy_and_repeat_native_training(self):
        from test_vectorized_critics import check_vectorized_learners
        from test_vector import small_parallel_config
        torch.set_num_threads(1)
        check_vectorized_learners("cuda")
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c = small_parallel_config(algorithm)
                c.runtime.device = c.runtime.replay_device = "cuda"
                c.runtime.vectorize_critics = True
                hashes = []
                for name in ("first", "repeat"):
                    c.runtime.output_dir = str(Path(directory) / name)
                    result = train(c)
                    self.assertGreater(result["updates"], 0)
                    hashes.append(result["reproducibility"])
                self.assertEqual(hashes[0]["episode_hash"], hashes[1]["episode_hash"])
                self.assertEqual(hashes[0]["model_hash"], hashes[1]["model_hash"])

    def test_grouped_cuda_actor_updates_match_separate_backward_and_adam(self):
        from test_performance import check_learner_equivalence
        torch.set_num_threads(1)
        check_learner_equivalence("cuda")

    def test_fused_cuda_adam_retains_finite_real_learner_updates(self):
        from test_vector import small_parallel_config
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c = small_parallel_config(algorithm)
                c.runtime.device = c.runtime.replay_device = "cuda"
                c.runtime.fused_optimizers = True
                c.runtime.output_dir = str(Path(directory) / "fused")
                result = train(c)
                self.assertGreater(result["updates"], 0)
                saved = torch.load(result["checkpoint"], weights_only=True)
                self.assertTrue(all(torch.isfinite(v).all() for v in saved["model"].values()))

    def test_repeated_cuda_training_same_seed_has_identical_scientific_hashes(self):
        from test_vector import small_parallel_config
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as folder:
                c=small_parallel_config(algorithm)
                c.runtime.device=c.runtime.replay_device="cuda"
                c.runtime.episodes=4
                hashes=[]
                for name in ("first","repeat"):
                    c.runtime.output_dir=str(Path(folder)/name)
                    hashes.append(train(c)["reproducibility"])
                self.assertEqual(hashes[0]["episode_hash"],hashes[1]["episode_hash"])
                self.assertEqual(hashes[0]["model_hash"],hashes[1]["model_hash"])

    def test_cuda_warm_restart_preserves_targets_counters_and_real_episode_prefix(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c = Config()
                c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 4
                c.runtime.algorithm = algorithm
                c.runtime.device = c.runtime.replay_device = "cuda"
                c.runtime.num_envs, c.runtime.episodes = 2, 2
                c.runtime.output_dir = str(Path(directory) / "first")
                a = c.algorithm_config
                a.hidden_sizes, a.replay_capacity = [32, 32], 32
                a.batch_size = a.learning_starts = 4
                first = train(c)
                c.runtime.episodes = 4
                c.runtime.output_dir = str(Path(directory) / "continued")
                c.runtime.warm_start_checkpoint = first["checkpoint"]
                second = train(c)
                self.assertEqual(second["initial_global_step"], first["global_step"])
                self.assertEqual(second["episodes"], 4)
                self.assertEqual(second["updates"], first["updates"] * 2)
                saved = torch.load(second["checkpoint"], map_location="cuda", weights_only=True)
                self.assertEqual(len(saved["progress"]["records"]), 4)
                self.assertTrue(all(torch.isfinite(v).all() for v in saved["model"].values()))

    def test_masked_entropy_stays_finite_with_alpha_above_one_without_metrics(self):
        c = Config()
        c.world.num_targets, c.world.num_obstacles = 2, 1
        c.masac.hidden_sizes = [32, 32]
        env = SwarmSearchEnv(c)
        try:
            observations, _ = env.reset(seed=44)
            obs = np.repeat(observations[None], 8, axis=0)
            obs[:, :, -7:] = 0
            obs[:, :, -7] = 1
            obs[::2, :, -1] = 1
            state = np.repeat(env.state()[None], 8, axis=0)
            action = Action(np.zeros((8, 6, 3)), np.zeros((8, 6), dtype=np.int64))
            replay = TensorReplayBuffer(8, observations.shape, env.state_dim, device="cuda")
            replay.add_batch(obs, state, action, np.ones(8), obs, state, np.zeros(8))
            for alpha in (1.01, 2., float(np.exp(5))):
                learner = MASAC(c, env.observation_dim, env.state_dim, "cuda")
                with torch.no_grad():
                    learner.log_alpha_discrete.fill_(np.log(alpha))
                for _ in range(2):
                    learner.update(replay.sample(8), collect_metrics=False)
                self.assertTrue(all(torch.isfinite(p).all() for p in learner.parameters()))
                actions = learner.act_batch(obs)
                self.assertTrue(np.isfinite(actions.continuous).all())
                self.assertTrue(np.isin(actions.destination, [0, 6]).all())
        finally:
            env.close()

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

    def test_cuda_mission_deadline_stops_bootstrapping_for_both_learners(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as folder:
                c = Config()
                c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 1, 0, 1
                c.runtime.algorithm = algorithm
                c.runtime.device = c.runtime.replay_device = "cuda"
                c.runtime.num_envs, c.runtime.episodes = 1, 1
                c.runtime.output_dir = str(Path(folder) / "train")
                a = c.algorithm_config
                a.hidden_sizes, a.replay_capacity = [16, 16], 2
                a.batch_size = a.learning_starts = 1
                record = train(c)["final_episode"]
                self.assertTrue(record["terminated"])
                self.assertFalse(record["truncated"])
                self.assertEqual(record["learner"]["batch_terminated_fraction"], 1)
                self.assertAlmostEqual(record["learner"]["target_mean"], record["return"], places=6)

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
