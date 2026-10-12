"""Hybrid policy, centralized learning and evaluation checkpoint checks."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.algorithms.masac import MASAC, soft_bellman_target
from uav_network.algorithms.replay import ReplayBuffer
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate


def small_config(network=False):
    c = Config()
    return replace(c, world=replace(c.world, num_targets=2, num_obstacles=1, max_steps=12),
                   network=replace(c.network, enabled=network),
                   masac=replace(c.masac, hidden_sizes=[32, 32], replay_capacity=64,
                                 batch_size=8, learning_starts=8),
                   runtime=replace(c.runtime, episodes=1, device="cpu"))


class MASACTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(44)
        self.config = small_config()
        self.env = SwarmSearchEnv(self.config)
        self.obs, _ = self.env.reset(seed=44)
        self.learner = MASAC(self.config, self.env.observation_dim, self.env.state_dim, "cpu")

    def tearDown(self):
        self.env.close()

    def batch(self):
        replay = ReplayBuffer(self.config.masac.replay_capacity, self.obs.shape,
                              self.env.state_dim, seed=44)
        rng = np.random.default_rng(44)
        obs = self.obs
        for _ in range(10):
            state = self.env.state()
            action = Action(rng.uniform(-1, 1, (6, 3)), rng.integers(0, 7, 6))
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            replay.add(obs, state, action, reward, next_obs, self.env.state(), terminated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = self.env.reset(seed=45)
        batch = replay.sample(8)
        # Explicit candidate beacons isolate gradients in the destination head.
        batch["observation"][:, :, -7:] = 1
        batch["next_observation"][:, :, -7:] = 1
        return batch

    def test_six_actors_have_separate_parameters_and_valid_actions(self):
        self.assertEqual(len(self.learner.actors), 6)
        self.assertEqual(len(self.learner.critics), 6)
        identities = [next(a.parameters()).data_ptr() for a in self.learner.actors]
        self.assertEqual(len(set(identities)), 6)
        action = self.learner.act(self.obs)
        action.validate(6)
        self.assertEqual(action.continuous.shape, (6, 3))
        self.assertEqual(action.destination.shape, (6,))

    def test_batched_masked_actions_zero_silent_power_and_dead_motion(self):
        obs = np.repeat(self.obs[None], 4, axis=0)
        obs[:, :, -7:] = 0
        obs[:, :, -7] = 1
        obs[:, 0, -1] = 1
        obs[:, 1, 9] = 0
        action = self.learner.act_batch(obs)
        self.assertEqual(action.continuous.shape, (4, 6, 3))
        self.assertTrue(np.isin(action.destination[:, 0], [0, 6]).all())
        self.assertTrue((action.destination[:, 1:] == 0).all())
        self.assertTrue((action.continuous[:, 1] == 0).all())
        self.assertTrue((action.continuous[:, :, 2][action.destination == 0] == 0).all())

    def test_no_destination_choice_has_zero_categorical_entropy(self):
        batch = self.batch()
        for key in ("observation", "next_observation"):
            batch[key][:, :, -7:] = 0
            batch[key][:, :, -7] = 1
        stats = self.learner.update(batch)
        self.assertEqual(stats["discrete_entropy"], 0)

    def test_zero_entropy_targets_still_update_temperatures_for_available_controls(self):
        a = self.learner.config.masac
        a.target_continuous_entropy = 0
        a.target_discrete_entropy_ratio = 0
        before_c = self.learner.log_alpha_continuous.detach().clone()
        before_d = self.learner.log_alpha_discrete.detach().clone()
        self.learner.update(self.batch())
        self.assertFalse(torch.equal(before_c, self.learner.log_alpha_continuous))
        self.assertFalse(torch.equal(before_d, self.learner.log_alpha_discrete))

    def test_real_adam_update_changes_both_action_branches_and_critics(self):
        actor = self.learner.actors[0]
        before_move = actor.mean.weight.detach().clone()
        before_dest = actor.destination.weight.detach().clone()
        before_q = next(self.learner.critics.parameters()).detach().clone()
        stats = self.learner.update(self.batch())
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertFalse(torch.equal(before_move, actor.mean.weight))
        self.assertFalse(torch.equal(before_dest, actor.destination.weight))
        self.assertFalse(torch.equal(before_q, next(self.learner.critics.parameters())))
        for p in self.learner.parameters():
            self.assertTrue(torch.isfinite(p).all())

    def test_tanh_log_probability_is_finite_at_saturation(self):
        obs = torch.full((8, self.env.observation_dim), 10000.0)
        action, logp, probs, log_probs = self.learner.actors[0].sample(obs)
        self.assertTrue(torch.isfinite(action).all())
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue(torch.isfinite(log_probs).all())
        torch.testing.assert_close(probs.sum(-1), torch.ones(8))

    def test_bellman_target_bootstraps_timeout_but_not_true_termination(self):
        reward = torch.tensor([1., 1.])
        terminated = torch.tensor([0., 1.])
        target = soft_bellman_target(reward, terminated, torch.tensor([10., 10.]), .9)
        torch.testing.assert_close(target, torch.tensor([10., 1.]))

    def test_checkpoint_preserves_exact_deterministic_actor_output(self):
        self.learner.update(self.batch())
        before = self.learner.act(self.obs, deterministic=True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            self.learner.save(path)
            restored = MASAC.load(path, device="cpu")
            after = restored.act(self.obs, deterministic=True)
        np.testing.assert_array_equal(before.continuous, after.continuous)
        np.testing.assert_array_equal(before.destination, after.destination)

    def test_ring_replay_retains_recent_transitions(self):
        replay = ReplayBuffer(2, self.obs.shape, self.env.state_dim, seed=44)
        action = Action(np.zeros((6, 3)), np.zeros(6, dtype=int))
        for reward in [1, 2, 3]:
            replay.add(self.obs, self.env.state(), action, reward,
                       self.obs, self.env.state(), False)
        self.assertEqual(len(replay), 2)
        self.assertEqual(set(replay.sample(2)["reward"].tolist()), {2, 3})

    def test_train_evaluate_produces_checkpoint_metrics_and_png(self):
        with TemporaryDirectory() as directory:
            self.config.runtime.output_dir = str(Path(directory) / "train")
            result = train(self.config)
            self.assertGreater(result["updates"], 0)
            self.assertTrue(Path(result["checkpoint"]).is_file())
            report = evaluate(result["checkpoint"], episodes=1,
                              output_dir=Path(directory) / "evaluation")
            self.assertTrue(Path(report["render_path"]).is_file())
            self.assertEqual(report["episodes"], 1)
            self.assertEqual(report["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
