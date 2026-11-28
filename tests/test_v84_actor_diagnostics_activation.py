"""Diagnostics must describe the executed Actor for each supported activation."""
import pytest
import torch
from uav_network.algorithms.configuration.schema import LearnerConfig
from uav_network.algorithms.matd3.learner import MATD3
from uav_network.algorithms.core.action_utils import valid_destinations


@pytest.mark.parametrize("activation", ["relu", "silu"])
@pytest.mark.parametrize("shared", [False, True])
def test_sampled_diagnostics_match_direct_actor_outputs(activation, shared):
    torch.manual_seed(73)
    cfg = LearnerConfig()
    cfg.motion_dim = cfg.continuous_dim = 2
    cfg.network.power_control = "fixed"
    cfg.network.destination_mode = "compact_3"
    cfg.matd3.hidden_sizes = (8, 8)
    cfg.matd3.actor_layer_norm = True
    cfg.matd3.activation = activation
    cfg.matd3.share_actor_parameters = shared
    learner = MATD3(cfg, observation_dim=21, state_dim=126)
    obs = torch.randn(40, 6, 21)
    obs[..., 9] = 1
    continuous = []
    probabilities = []
    for i in range(6):
        actor = learner._actor_at(i)
        c, p = actor(learner._actor_input(obs[:32, i], i),
                     valid_destinations(obs[:32, i], 3))
        continuous.append(c)
        probabilities.append(p)
    c = torch.stack(continuous, 1)
    p = torch.stack(probabilities, 1)
    # The small output values avoid inversion rounding near tanh saturation.
    expected_pretanh = torch.atanh(c).abs().mean()
    expected_derivative = (1 - c.square()).mean()
    expected_entropy = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
    before = {k: v.clone() for k, v in learner.state_dict().items()}
    rng = torch.get_rng_state().clone()
    metrics = learner._actor_output_diagnostics(obs)
    torch.testing.assert_close(metrics["actor_motion_pretanh_abs_mean"], expected_pretanh)
    torch.testing.assert_close(metrics["actor_motion_tanh_derivative_mean"], expected_derivative)
    torch.testing.assert_close(metrics["actor_destination_entropy_mean"], expected_entropy)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        assert torch.equal(value, learner.state_dict()[key]), key
