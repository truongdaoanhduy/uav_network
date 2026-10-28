"""Mathematical and gradient regression checks for hybrid MASAC/MATD3.

These tests intentionally exercise algorithm properties rather than comparing
against the frozen legacy implementation.  They protect the hybrid-action
adaptations used by the UAV training line.
"""
import unittest

import torch
from torch import nn

from uav_network.algorithms.masac import MASAC, soft_bellman_target
from uav_network.algorithms.matd3 import MATD3, td3_target
from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim


def small_config(algorithm="masac"):
    c = Config()
    c.world.num_uavs = 2
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.runtime.algorithm = algorithm
    c.runtime.vectorize_critics = False
    c.runtime.amp_mode = "off"
    c.runtime.compile_models = False
    c.masac.hidden_sizes = [16, 16]
    c.matd3.hidden_sizes = [16, 16]
    return c


def replay_batch(config, count=8):
    n = config.world.num_uavs
    k = n + 1
    obs_dim = observation_dim(config)
    state_size = state_dim(config)
    generator = torch.Generator().manual_seed(111)
    obs = torch.randn(count, n, obs_dim, generator=generator)
    obs[:, :, 9] = 1
    obs[:, :, -k:] = 1
    state = torch.randn(count, state_size, generator=generator)
    return {
        "observation": obs,
        "next_observation": obs.clone(),
        "state": state,
        "next_state": state.clone(),
        "continuous": torch.zeros(count, n, 3),
        "destination": torch.zeros(count, n, dtype=torch.long),
        "reward": torch.linspace(-1, 1, count),
        "terminated": torch.zeros(count),
    }


class DestinationIndexCritic(nn.Module):
    """Return the selected destination index for one chosen UAV."""

    def __init__(self, index, destinations):
        super().__init__()
        self.index = index
        self.destinations = destinations

    def forward(self, state, joint_actions):
        one_hot = joint_actions[:, self.index, 3:]
        values = torch.arange(self.destinations, device=joint_actions.device,
                              dtype=joint_actions.dtype)
        q = (one_hot * values).sum(-1)
        return q, q


class SumCritic(nn.Module):
    """Simple differentiable centralized critic used to expose action gradients."""

    def forward(self, state, joint_actions):
        q = joint_actions.sum(dim=(1, 2))
        return q, q


class AlgorithmMathRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(44)

    def _masac_alpha_after_update(self, *, continuous_target, discrete_ratio,
                                  skew_destination=False):
        c = small_config("masac")
        c.masac.target_continuous_entropy = continuous_target
        c.masac.target_discrete_entropy_ratio = discrete_ratio
        learner = MASAC(c, observation_dim(c), state_dim(c))
        if skew_destination:
            with torch.no_grad():
                for actor in learner.actors:
                    actor.destination.weight.zero_()
                    actor.destination.bias.zero_()
                    actor.destination.bias[0] = 10
        before_c = learner.log_alpha_continuous.detach().clone()
        before_d = learner.log_alpha_discrete.detach().clone()
        learner.update_actor(replay_batch(c), collect_metrics=False)
        return (before_c, learner.log_alpha_continuous.detach().clone(),
                before_d, learner.log_alpha_discrete.detach().clone())

    def test_masac_alpha_moves_toward_entropy_targets(self):
        # H > target => gradient descent lowers alpha.
        before_c, after_c, before_d, after_d = self._masac_alpha_after_update(
            continuous_target=-100.0, discrete_ratio=0.0)
        self.assertTrue(torch.all(after_c < before_c))
        self.assertTrue(torch.all(after_d < before_d))

        # H < target => gradient descent raises alpha.  A strongly skewed
        # categorical policy guarantees H_discrete < log(number of choices).
        before_c, after_c, before_d, after_d = self._masac_alpha_after_update(
            continuous_target=100.0, discrete_ratio=1.0, skew_destination=True)
        self.assertTrue(torch.all(after_c > before_c))
        self.assertTrue(torch.all(after_d > before_d))

    def test_masac_enumerates_each_destination_exactly(self):
        c = small_config("masac")
        learner = MASAC(c, observation_dim(c), state_dim(c))
        batch = replay_batch(c, count=3)
        state = batch["state"]
        with torch.no_grad():
            joint, _, _, _, continuous = learner._joint_sample(batch["observation"])
        critic = DestinationIndexCritic(index=0, destinations=learner.destinations)
        q = learner._expected_q_full(critic, state, joint, continuous[:, 0], index=0)
        expected = torch.arange(learner.destinations, dtype=q.dtype).expand_as(q)
        torch.testing.assert_close(q, expected)

        probabilities = torch.tensor([0.05, 0.15, 0.80], dtype=q.dtype).expand_as(q)
        exact_expectation = (probabilities * q).sum(-1)
        torch.testing.assert_close(exact_expectation,
                                   torch.full((len(q),), 1.75, dtype=q.dtype))

    def test_masac_actor_gradient_isolated_from_peer_actor(self):
        c = small_config("masac")
        learner = MASAC(c, observation_dim(c), state_dim(c))
        batch = replay_batch(c)
        # Deliberately keep the sampled peer actions attached.  The MASAC
        # expectation helper itself must detach peers and reconnect only the
        # current actor's continuous action.
        base_joint, *_ = learner._joint_sample(batch["observation"])
        continuous, _, _, _ = learner._sample(learner.actors[0], batch["observation"][:, 0])
        q = learner._expected_q_full(SumCritic(), batch["state"], base_joint,
                                     continuous, index=0)
        loss = -q.mean()
        own = list(learner.actors[0].parameters())
        peer = list(learner.actors[1].parameters())
        gradients = torch.autograd.grad(loss, own + peer, allow_unused=True)
        own_gradients = gradients[:len(own)]
        peer_gradients = gradients[len(own):]
        self.assertTrue(any(g is not None and torch.count_nonzero(g) > 0 for g in own_gradients))
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in peer_gradients))

    def test_matd3_straight_through_is_hard_forward_soft_backward(self):
        c = small_config("matd3")
        learner = MATD3(c, observation_dim(c), state_dim(c))
        actor = learner.actors[0]
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
        obs = replay_batch(c)["observation"][:, 0]
        action = learner._actor_action(actor, obs, straight_through=True)
        discrete = action[:, 3:]
        torch.testing.assert_close(discrete, discrete.round(), rtol=0, atol=1e-6)
        torch.testing.assert_close(discrete.sum(-1), torch.ones(len(discrete)), rtol=0, atol=1e-6)

        weights = torch.arange(learner.destinations, dtype=discrete.dtype)
        loss = (discrete * weights).sum()
        actor.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(actor.destination.bias.grad)
        self.assertGreater(float(actor.destination.bias.grad.abs().sum()), 0.0)

    def test_matd3_target_smoothing_changes_only_continuous_branch(self):
        c = small_config("matd3")
        learner = MATD3(c, observation_dim(c), state_dim(c))
        with torch.no_grad():
            for actor in learner.target_actors:
                for parameter in actor.parameters():
                    parameter.zero_()
        obs = replay_batch(c)["next_observation"]

        learner.config.matd3.target_noise_std = 0.0
        baseline = learner.target_actions(obs)
        learner.config.matd3.target_noise_std = 0.2
        learner.config.matd3.target_noise_clip = 0.5
        torch.manual_seed(123)
        smoothed = learner.target_actions(obs)

        # Zero logits choose destination 0 (silent); smoothing must not change
        # the categorical branch, and silent TX power remains canonical zero.
        expected_discrete = torch.zeros_like(baseline[..., 3:])
        expected_discrete[..., 0] = 1
        torch.testing.assert_close(baseline[..., 3:], expected_discrete, rtol=0, atol=0)
        torch.testing.assert_close(smoothed[..., 3:], expected_discrete, rtol=0, atol=0)
        torch.testing.assert_close(smoothed[..., 2], torch.zeros_like(smoothed[..., 2]), rtol=0, atol=0)
        self.assertGreater(float((smoothed[..., :2] - baseline[..., :2]).abs().sum()), 0.0)
        self.assertLessEqual(float(smoothed[..., :3].abs().max()), 1.0)

    def test_team_reward_broadcasts_to_every_critic_target(self):
        reward = torch.tensor([[1.0], [-2.0], [3.5]])
        terminated = torch.tensor([[0.0], [1.0], [0.0]])
        q1 = torch.tensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
        q2 = torch.tensor([[3.0, 6.0], [5.0, 8.0], [7.0, 10.0]])
        gamma = 0.9

        td3 = td3_target(reward, terminated, q1, q2, gamma)
        expected_td3 = reward.expand_as(q1) + gamma * (1 - terminated).expand_as(q1) * torch.minimum(q1, q2)
        torch.testing.assert_close(td3, expected_td3)
        torch.testing.assert_close(td3[1], reward[1].expand_as(td3[1]))

        soft_value = torch.tensor([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]])
        masac = soft_bellman_target(reward, terminated, soft_value, gamma)
        expected_masac = reward.expand_as(soft_value) + gamma * (1 - terminated).expand_as(soft_value) * soft_value
        torch.testing.assert_close(masac, expected_masac)
        torch.testing.assert_close(masac[1], reward[1].expand_as(masac[1]))


if __name__ == "__main__":
    unittest.main()
