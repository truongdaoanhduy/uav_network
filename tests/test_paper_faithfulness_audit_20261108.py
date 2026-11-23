"""Executable paper-alignment audits for the frozen UAV MASAC/MATD3 baselines.

References:
- SAC / TD3: OpenAI Spinning Up PyTorch implementations.
- Ackermann et al. 2019 (arXiv:1910.01465), Eq. 5: smoothing of *all* agents.
- Hybrid MASAC: expectation over own categorical destination; peer actions sampled.
These tests deliberately do not change an algorithmic objective or a training recipe.
"""
from itertools import product

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from uav_network.algorithms.action_utils import joint_action
from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.replay import ReplayBuffer
from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.environment.observations import (
    centralized_state,
    local_observations,
    observation_dim,
    state_dim,
)


def _config(algorithm, shared=False):
    config = Config()
    config.world.num_uavs = 2
    config.world.num_targets = 2
    config.world.num_obstacles = 1
    config.world.motion_dimensions = 3
    config.sensor.model = "bayesian"
    config.world.motion_safety = "cbf"
    config.world.map_size_m = 1000.0
    config.network.enabled = False
    config.runtime.algorithm = algorithm
    config.runtime.device = "cpu"
    config.runtime.vectorize_critics = False
    config.runtime.critic_chunk_size = 0
    for settings in (config.masac, config.matd3):
        settings.hidden_sizes = [16, 16]
        settings.share_actor_parameters = shared
        settings.share_critic_parameters = shared
    return config


class JointCategoricalCritic(nn.Module):
    """Non-additive 2-UAV Q: catches an accidental factorization of joint Q."""

    def __init__(self):
        super().__init__()
        self.register_buffer("q_values", torch.tensor([
            [0.5, -2.0, 1.25],
            [3.0, 0.1, -4.0],
            [2.75, 5.5, -0.75],
        ], dtype=torch.float32))

    def forward(self, state, joint):
        continuous_dim = joint.shape[-1] - self.q_values.shape[0]
        indices = joint[..., continuous_dim:].argmax(-1)
        value = self.q_values[indices[:, 0], indices[:, 1]]
        return value, value + 1.0  # The minimum Q must remain the first value.


def test_shared_masac_conditional_enumeration_matches_exact_joint_expectation():
    """Mean of conditional exact-Q estimates is unbiased under independent policies."""
    c = _config("masac", shared=True)
    learner = MASAC(c, observation_dim(c), state_dim(c))
    k = learner.destinations
    assert k == 3
    choices = torch.tensor(list(product(range(k), repeat=2)), dtype=torch.long)
    continuous = torch.zeros(len(choices), 2, c.continuous_dim)
    state = torch.zeros(len(choices), state_dim(c))
    joint = joint_action(continuous, choices, k)
    critic = JointCategoricalCritic()
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]])

    with torch.no_grad():
        conditional_q = learner._expected_q_shared_all_full(
            critic, state, joint, continuous
        )  # [every peer sample, each agent, every candidate destination]

    matrix = critic.q_values
    assert conditional_q.shape == (k * k, 2, k)
    for row, (d0, d1) in enumerate(choices.tolist()):
        torch.testing.assert_close(conditional_q[row, 0], matrix[:, d1])
        torch.testing.assert_close(conditional_q[row, 1], matrix[d0, :])

    estimate_by_joint_sample = (
        conditional_q * probs[None, :, :]
    ).sum(-1).mean(-1)
    joint_sample_prob = probs[0, choices[:, 0]] * probs[1, choices[:, 1]]
    estimated_expectation = (joint_sample_prob * estimate_by_joint_sample).sum()
    exact_expectation = torch.einsum("i,j,ij->", probs[0], probs[1], matrix)
    torch.testing.assert_close(estimated_expectation, exact_expectation, rtol=0, atol=1e-6)


@pytest.mark.parametrize("shared", [False, True])
def test_matd3_target_smoothing_applies_to_all_agents_as_in_paper(shared):
    """Ackermann Eq. (5), unlike a one-agent-only implementation variant."""
    c = _config("matd3", shared=shared)
    learner = MATD3(c, observation_dim(c), state_dim(c))
    observations = torch.zeros(24, c.world.num_uavs, observation_dim(c))
    observations[..., 9] = 1.0
    observations[..., -learner.destinations:] = 1.0

    for actor in learner.target_actors:
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()

    # No categorical sampling noise: isolate TD3 Gaussian target smoothing.
    learner._relaxed_destination = lambda prob: F.one_hot(
        torch.ones(prob.shape[:-1], dtype=torch.long, device=prob.device),
        num_classes=prob.shape[-1],
    ).to(prob.dtype)
    learner.config.matd3.target_noise_std = 0.0
    torch.manual_seed(44)
    clean = learner.target_actions(observations)

    learner.config.matd3.target_noise_std = 0.2
    learner.config.matd3.target_noise_clip = 0.5
    torch.manual_seed(44)
    smoothed = learner.target_actions(observations)

    torch.testing.assert_close(
        smoothed[..., c.continuous_dim:], clean[..., c.continuous_dim:],
        rtol=0, atol=0,
    )
    displacement = smoothed[..., :c.motion_dim] - clean[..., :c.motion_dim]
    assert displacement.shape[:2] == (24, 2)
    assert torch.all(displacement.abs().sum(dim=(0, 2)) > 0)
    assert torch.isfinite(smoothed).all()
    assert float(displacement.abs().max()) <= 0.5 + 1e-7


def test_matd3_soft_categorical_q_input_differs_from_executed_hard_onehot():
    """Document known interface mismatch without silently changing source policy."""
    torch.manual_seed(44)
    logits = torch.zeros(32, 3, requires_grad=True)
    with torch.no_grad():
        logits[:, -1] = -40.0  # Very unlikely category, not an invalid mask.
    probs = logits.softmax(-1)
    soft = MATD3._relaxed_destination(probs)
    hard = F.one_hot(soft.argmax(-1), 3).to(soft.dtype)

    torch.testing.assert_close(soft.sum(-1), torch.ones(32), atol=1e-6, rtol=0)
    assert bool(((soft > 0) & (soft < 1)).any())
    assert not torch.allclose(soft, hard)
    # A nonlinear critic can assign systematically different values to each.
    q_soft = soft.square().sum(-1)
    q_hard = hard.square().sum(-1)
    assert bool(torch.all(q_soft < q_hard))
    q_soft.mean().backward()
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_replay_records_issued_command_not_cbf_corrected_velocity():
    """Replay action is the MDP's issued command; shield remains in dynamics."""
    c = _config("matd3")
    env = SwarmSearchEnv(c)
    try:
        env.reset(seed=44)
        env.positions[0] = [500.0, 500.0, 0.5]
        env.positions[1] = [700.0, 700.0, 80.0]
        env.velocities[:] = 0.0
        observation = local_observations(env)
        state = centralized_state(env)
        commands = np.zeros((2, c.continuous_dim), dtype=np.float32)
        commands[0, 2] = -1.0  # Would cross lower altitude boundary unshielded.
        action = Action(commands.copy(), np.zeros(2, dtype=np.int64))

        next_observation, reward, terminated, truncated, _ = env.step(action)
        assert not truncated
        assert env.motion_counts["shield_interventions"] >= 1
        assert env.positions[0, 2] >= c.world.altitude_min_m
        assert env.velocities[0, 2] != -c.world.max_accel_mps2

        replay = ReplayBuffer(4, observation.shape, len(state),
                              seed=44, continuous_dim=c.continuous_dim)
        replay.add(observation, state, action, reward, next_observation,
                   centralized_state(env), terminated)
        np.testing.assert_array_equal(replay.data["continuous"][0], commands)
        np.testing.assert_array_equal(replay.data["destination"][0], action.destination)
    finally:
        env.close()
