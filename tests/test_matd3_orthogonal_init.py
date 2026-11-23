"""Paired initialization-only MATD3 ablation regression checks."""
import math

import pytest
import torch
from torch import nn

from uav_network.algorithms.matd3 import MATD3
from uav_network.configuration import Config, load_config


def config(mode):
    c = load_config("configs/comparison-reference-masac-matd3-xyz-sensing.yaml",
                    ["runtime.algorithm=matd3", "runtime.critic_chunk_size=0",
                     f"matd3.initialization={mode}"])
    c.matd3.hidden_sizes = [16, 16]  # CPU-only construction smoke
    return c


def assert_semiorthogonal(layer, gain):
    w = layer.weight.detach() / gain
    gram = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
    torch.testing.assert_close(gram, torch.eye(len(gram)), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(layer.bias.detach(), torch.zeros_like(layer.bias))


def test_config_modes_are_validated():
    with pytest.raises(ValueError, match="matd3.initialization"):
        load_config("configs/comparison-reference-masac-matd3-xyz-sensing.yaml",
                    ["matd3.initialization=wrong-mode"])
    assert config("default").matd3.initialization == "default"
    assert config("orthogonal").matd3.initialization == "orthogonal"


def test_default_mode_retains_bitwise_default_weights_and_rng():
    c = config("default")
    torch.manual_seed(44)
    original = MATD3(c, 198, 565)
    saved_rng = torch.get_rng_state().clone()
    torch.manual_seed(44)
    repeated = MATD3(c, 198, 565)
    assert torch.equal(saved_rng, torch.get_rng_state())
    for p, q in zip(original.parameters(), repeated.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_orthogonal_initialization_covers_actor_and_all_twin_critic_layers():
    c = config("orthogonal")
    torch.manual_seed(44)
    trained = MATD3(c, 198, 565)
    assert len(trained.actors) == len(trained.critics) == 1
    actor = trained.actors[0]
    for layer in actor.encoder.modules():
        if isinstance(layer, nn.Linear):
            assert_semiorthogonal(layer, math.sqrt(2))
    for head in (actor.continuous, actor.destination):
        assert_semiorthogonal(head, .01)
    for tower in (trained.critics[0].q1, trained.critics[0].q2):
        linear = [layer for layer in tower.modules() if isinstance(layer, nn.Linear)]
        for layer in linear[:-1]:
            assert_semiorthogonal(layer, math.sqrt(2))
        assert_semiorthogonal(linear[-1], 1.)
    for target, source in ((trained.target_actors, trained.actors),
                           (trained.target_critics, trained.critics)):
        for a, b in zip(target.parameters(), source.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(not p.requires_grad for p in trained.target_actors.parameters())
    assert all(not p.requires_grad for p in trained.target_critics.parameters())
    obs = torch.zeros(2, 6, 198)
    obs[:, :, -7:] = 1
    actions = trained.act_batch(obs, deterministic=True)
    assert torch.isfinite(torch.tensor(actions.continuous)).all()
    assert actions.continuous.shape == (2, 6, c.continuous_dim)


def test_orthogonal_does_not_advance_rng_beyond_default_model_creation():
    torch.manual_seed(44)
    base = MATD3(config("default"), 198, 565)
    expected = torch.get_rng_state().clone()
    torch.manual_seed(44)
    orthogonal = MATD3(config("orthogonal"), 198, 565)
    actual = torch.get_rng_state().clone()
    assert torch.equal(expected, actual)
    assert not torch.equal(base.actors[0].continuous.weight, orthogonal.actors[0].continuous.weight)
    assert base.config.runtime.critic_chunk_size == orthogonal.config.runtime.critic_chunk_size == 0
