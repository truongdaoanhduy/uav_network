"""Spawned real network workers preserve terminal data before partial reset."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from uav_network.configuration import Config
from uav_network.environment.vector import make_vector, unpack_info, random_actions
from uav_network.algorithms.replay import TensorReplayBuffer
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate


def small_parallel_config(algorithm="masac"):
    c = Config()
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.world.max_steps = 4
    c.network.enabled = True
    c.runtime.num_envs = 2
    c.runtime.episodes = 2
    c.runtime.algorithm = algorithm
    c.runtime.device = "cpu"
    c.runtime.log_interval_steps = 1
    for a in (c.masac, c.matd3):
        a.hidden_sizes = [32, 32]
        a.batch_size = 4
        a.replay_capacity = 32
        a.learning_starts = 4
    return c


class VectorTests(unittest.TestCase):
    def test_real_workers_are_isolated_and_masked_reset_keeps_other_state(self):
        c = small_parallel_config()
        vector = make_vector(c)
        try:
            obs, infos = vector.reset(seed=[44, 45])
            pids = [unpack_info(infos, i)["worker_pid"] for i in range(2)]
            self.assertEqual(len(set(pids)), 2)
            self.assertNotIn(os.getpid(), pids)
            rng = np.random.default_rng(44)
            action = random_actions(obs["observation"], rng, c.world.num_uavs + 1)
            for _ in range(4):
                final, reward, terminated, truncated, infos = vector.step({
                    "continuous": action.continuous, "destination": action.destination})
            self.assertTrue(terminated.all())
            self.assertFalse(truncated.any())
            self.assertEqual(unpack_info(infos, 0)["network"]["backend"], "uavnetsim")
            terminal_obs = final["observation"].copy()
            terminal_state = final["state"].copy()
            replay = TensorReplayBuffer(8, terminal_obs.shape[1:], terminal_state.shape[1])
            replay.add_batch(obs["observation"], obs["state"], action, reward,
                             terminal_obs, terminal_state, terminated)
            reset, _ = vector.reset(seed=[46, None], options={"reset_mask": np.array([True, False])})
            np.testing.assert_array_equal(reset["state"][1], terminal_state[1])
            self.assertFalse(np.array_equal(reset["state"][0], terminal_state[0]))
            np.testing.assert_array_equal(replay.data["next_state"][:2].numpy(), terminal_state)
            self.assertEqual(float(replay.data["terminated"][:2].sum()), 2)
        finally:
            vector.close()
        self.assertTrue(all(not p.is_alive() for p in vector.processes))

    def test_tensor_ring_batch_wrap_preserves_latest_joint_transitions(self):
        replay = TensorReplayBuffer(3, (2, 10), 5)
        from uav_network.environment.env import Action
        action = Action(np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=np.int64))
        for rewards in ([1, 2], [3, 4]):
            replay.add_batch(np.zeros((2, 2, 10)), np.zeros((2, 5)), action, rewards,
                             np.zeros((2, 2, 10)), np.zeros((2, 5)), [False, True])
        self.assertEqual(len(replay), 3)
        self.assertEqual(set(replay.data["reward"].tolist()), {2, 3, 4})
        sample = replay.sample(2)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in sample.values()))
        self.assertGreater(replay.allocated_bytes, 0)

    def test_both_algorithms_train_and_evaluate_with_real_parallel_networks(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c = small_parallel_config(algorithm)
                c.runtime.output_dir = str(Path(directory) / "train")
                result = train(c)
                self.assertEqual(result["algorithm"], algorithm)
                self.assertEqual(result["num_envs"], 2)
                self.assertEqual(result["global_step"], 8)
                self.assertGreater(result["updates"], 0)
                report = evaluate(result["checkpoint"], 1, Path(directory) / "evaluate")
                self.assertEqual(report["algorithm"], algorithm)
                self.assertTrue(Path(report["render_path"]).is_file())

    def test_final_ready_batch_is_learned_with_and_without_overlap(self):
        models = []
        for overlap in (True, False):
            with TemporaryDirectory() as directory:
                c = small_parallel_config()
                c.world.max_steps = 1
                c.network.enabled = False
                c.masac.learning_starts = c.masac.batch_size = 2
                c.runtime.collector_overlap = overlap
                c.runtime.output_dir = str(Path(directory) / "train")
                result = train(c)
                self.assertEqual(result["updates"], 1)
                models.append(torch.load(result["checkpoint"], weights_only=True)["model"])
        for name in models[0]:
            torch.testing.assert_close(models[0][name], models[1][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
