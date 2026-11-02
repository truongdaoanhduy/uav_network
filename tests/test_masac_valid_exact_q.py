"""MASAC exact-Q must skip probability-zero destinations without changing weighted values."""
from copy import deepcopy
from unittest.mock import patch

import torch
from torch import nn

from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.optimization import compile_models
from uav_network.configuration import Config


class CountingCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows = 0

    def forward(self, state, joint):
        self.rows += len(state)
        value = state.sum(-1) + joint.square().sum(dim=(-1, -2))
        return value, value + 0.25


def learner(num_uavs=2):
    c = Config()
    c.world.num_uavs = num_uavs
    c.masac.hidden_sizes = [16, 16]
    c.runtime.vectorize_critics = True
    c.runtime.critic_chunk_size = 0
    return MASAC(c, observation_dim=16, state_dim=5)


def dense_expected_q(critic, state, joint, continuous, index, destinations):
    b, k = len(state), destinations
    alternatives = joint.detach()[:, None].expand(-1, k, -1, -1).clone()
    choices = continuous[:, None].expand(-1, k, -1).clone()
    choices[:, 0, 2] = 0
    alternatives[:, :, index, :3] = choices
    alternatives[:, :, index, 3:] = torch.eye(k, dtype=alternatives.dtype)[None]
    repeated_state = state[:, None].expand(-1, k, -1).reshape(b * k, -1)
    q1, q2 = critic(repeated_state, alternatives.reshape(b * k, joint.shape[1], -1))
    return torch.minimum(q1, q2).reshape(b, k)


def test_sparse_exact_q_only_evaluates_valid_rows_and_matches_dense_valid_values():
    model = learner()
    b, n, k = 4, model.num_uavs, model.destinations
    state = torch.randn(b, model.state_dim)
    joint = torch.randn(b, n, 3 + k)
    continuous = torch.randn(b, 3)
    valid = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=torch.bool)
    dense_critic = CountingCritic()
    sparse_critic = CountingCritic()

    expected = dense_expected_q(dense_critic, state, joint, continuous, 0, k)
    observed = model._expected_q(sparse_critic, state, joint, continuous, 0, valid)

    torch.testing.assert_close(observed[valid], expected[valid])
    torch.testing.assert_close(observed[~valid], torch.zeros_like(observed[~valid]))
    assert sparse_critic.rows == int(valid.sum())
    assert sparse_critic.rows < dense_critic.rows == b * k


def test_sparse_exact_q_preserves_probability_weighted_value_and_continuous_gradient():
    torch.manual_seed(44)
    model = learner()
    b, n, k = 5, model.num_uavs, model.destinations
    state = torch.randn(b, model.state_dim)
    joint = torch.randn(b, n, 3 + k)
    valid = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    raw = torch.rand(b, k) * valid
    probs = raw / raw.sum(-1, keepdim=True)
    dense_continuous = torch.randn(b, 3, requires_grad=True)
    sparse_continuous = dense_continuous.detach().clone().requires_grad_()
    critic = deepcopy(model.critics[0])

    dense = dense_expected_q(critic, state, joint, dense_continuous, 0, k)
    sparse = model._expected_q(critic, state, joint, sparse_continuous, 0, valid)
    dense_value = (probs * dense).sum(-1).mean()
    sparse_value = (probs * sparse).sum(-1).mean()
    dense_grad, = torch.autograd.grad(dense_value, dense_continuous)
    sparse_grad, = torch.autograd.grad(sparse_value, sparse_continuous)

    torch.testing.assert_close(sparse_value, dense_value, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(sparse_grad, dense_grad, rtol=2e-6, atol=2e-7)


def test_all_critics_skip_each_agents_invalid_destinations_with_mixed_masks():
    model = learner(num_uavs=3)
    b, n, k = 4, model.num_uavs, model.destinations
    state = torch.randn(b, model.state_dim)
    joint = torch.randn(b, n, 3 + k)
    continuous = torch.randn(b, n, 3)
    valid = torch.tensor([
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]],
        [[1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]],
        [[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0]],
    ], dtype=torch.bool)
    critics = nn.ModuleList(CountingCritic() for _ in range(n))

    observed = model._expected_q_all(critics, state, joint, continuous, valid)

    assert observed.shape == (b, n, k)
    for i, critic in enumerate(critics):
        expected = dense_expected_q(CountingCritic(), state, joint, continuous[:, i], i, k)
        torch.testing.assert_close(observed[:, i][valid[:, i]], expected[valid[:, i]])
        torch.testing.assert_close(observed[:, i][~valid[:, i]], torch.zeros_like(observed[:, i][~valid[:, i]]))
        assert critic.rows == int(valid[:, i].sum())


def test_compile_models_installs_compact_exact_q_callable_without_changing_state_keys():
    model = learner()
    keys = set(model.state_dict())
    compiled = []

    def fake_compile(function=None, **kwargs):
        compiled.append((function, kwargs))
        return function

    with patch("torch.compile", side_effect=fake_compile):
        compile_models(model)

    assert hasattr(model, "_compiled_exact_q_values")
    assert set(model.state_dict()) == keys
    assert any(kwargs.get("dynamic") is True and kwargs.get("backend") == "inductor"
               for _, kwargs in compiled)
