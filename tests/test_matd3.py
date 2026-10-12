"""MATD3 double critics, delayed policies, categorical gradients and checkpoints."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from uav_network.algorithms.matd3 import MATD3, td3_target
from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.observations import observation_dim, state_dim


class MATD3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(44)
        c = self.config = Config()
        c.runtime.algorithm = "matd3"
        c.network.enabled = False
        c.matd3.hidden_sizes = [32, 32]
        c.matd3.batch_size = 8
        c.matd3.replay_capacity = 64
        self.env = SwarmSearchEnv(c)
        self.obs, _ = self.env.reset(seed=44)
        self.learner = MATD3(c, observation_dim(c), state_dim(c))

    def batch(self):
        b, n, o, s = 8, 6, observation_dim(self.config), state_dim(self.config)
        rng = np.random.default_rng(44)
        obs = np.repeat(self.obs[None], b, axis=0)
        # Explicit valid next hops exercise both policy branches in this unit check.
        obs[:, :, -7:] = 1
        return dict(observation=obs, next_observation=obs.copy(),
                    state=rng.normal(size=(b, s)), next_state=rng.normal(size=(b, s)),
                    continuous=rng.uniform(-1, 1, (b, n, 3)),
                    destination=rng.integers(0, 7, (b, n)), reward=np.arange(b),
                    terminated=np.array([0, 1] * 4))

    def test_target_uses_smaller_twin_q_and_true_termination(self):
        result = td3_target(torch.tensor([1., 1.]), torch.tensor([0., 1.]),
                            torch.tensor([4., 8.]), torch.tensor([3., 9.]), .9)
        torch.testing.assert_close(result, torch.tensor([3.7, 1.]))

    def test_actor_and_targets_wait_for_policy_delay(self):
        before_actor = [p.detach().clone() for p in self.learner.actors.parameters()]
        before_target = [p.detach().clone() for p in self.learner.target_critics.parameters()]
        before_critic = next(self.learner.critics.parameters()).detach().clone()
        first = self.learner.update(self.batch())
        self.assertFalse(first["policy_updated"])
        self.assertFalse(torch.equal(before_critic, next(self.learner.critics.parameters())))
        for before, after in zip(before_actor, self.learner.actors.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for before, after in zip(before_target, self.learner.target_critics.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        actor = self.learner.actors[0]
        move, dest = actor.continuous.weight.detach().clone(), actor.destination.weight.detach().clone()
        second = self.learner.update(self.batch())
        self.assertTrue(second["policy_updated"])
        self.assertFalse(torch.equal(move, actor.continuous.weight))
        self.assertFalse(torch.equal(dest, actor.destination.weight))
        self.assertTrue(all(np.isfinite(v) for v in second.values()))

    def test_target_noise_clips_continuous_actions_and_respects_dead_mask(self):
        obs = torch.tensor(self.batch()["observation"], dtype=torch.float32)
        obs[:, 0, 9] = 0
        obs[:, 0, -7:] = 0
        obs[:, 0, -7] = 1
        joint = self.learner.target_actions(obs)
        self.assertLessEqual(float(joint[:, :, :3].abs().max()), 1)
        torch.testing.assert_close(joint[:, 0, :3], torch.zeros((8, 3)))
        self.assertTrue(torch.all(joint[:, 0, 3:].argmax(-1) == 0))

    def test_batched_exploration_samples_only_valid_candidates(self):
        obs = np.repeat(self.obs[None], 4, axis=0)
        obs[:, :, -7:] = 0
        obs[:, :, -7] = 1
        obs[:, 0, -1] = 1
        action = self.learner.act_batch(obs, deterministic=False, step=0)
        self.assertEqual(action.continuous.shape, (4, 6, 3))
        self.assertTrue(np.isin(action.destination[:, 0], [0, 6]).all())
        self.assertTrue((action.destination[:, 1:] == 0).all())
        self.assertTrue((action.continuous[:, :, 2][action.destination == 0] == 0).all())

    def test_checkpoint_round_trip_and_algorithm_mismatch_error(self):
        before = self.learner.act(self.obs, deterministic=True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            self.learner.save(path)
            restored = MATD3.load(path)
            after = restored.act(self.obs, deterministic=True)
            from uav_network.algorithms.masac import MASAC
            with self.assertRaisesRegex(ValueError, "algorithm"):
                MASAC.load(path)
        np.testing.assert_array_equal(before.continuous, after.continuous)
        np.testing.assert_array_equal(before.destination, after.destination)


if __name__ == "__main__":
    unittest.main()
