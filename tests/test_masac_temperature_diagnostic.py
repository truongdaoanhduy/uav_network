"""MASAC alpha-vs-log-alpha audit: isolate exactly one profile variable."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from scripts.diagnose_masac_temperature_35t25o import make_config
from uav_network.environment.observations import observation_dim, state_dim


CONFIG = Path(__file__).resolve().parents[1] / "configs/comparison-independent-35t25o-h512-xyz-sensing.yaml"


def test_paired_modes_only_differ_temperature_loss():
    c = make_config(CONFIG, 64)
    other = deepcopy(c)
    other.masac.temperature_loss = "alpha"
    assert c.world.num_targets == 35 and c.world.num_obstacles == 25
    assert c.motion_dim == 3
    assert c.masac.share_actor_parameters and c.masac.share_critic_parameters
    assert c.masac.actor_layer_norm is True
    assert c.masac.mask_aware_discrete_entropy_target is True
    assert observation_dim(c) == observation_dim(other)
    assert state_dim(c) == state_dim(other)
    assert c.masac.temperature_loss == "log_alpha"
    assert other.masac.temperature_loss == "alpha"
    left, right = c.to_dict(), other.to_dict()
    right["masac"]["temperature_loss"] = left["masac"]["temperature_loss"]
    assert left == right


@pytest.mark.parametrize("log_alpha", [0.0, -2.0, 1.0])
def test_temperature_gradient_relation(log_alpha):
    """The alpha dual has alpha-scaled gradient wrt log_alpha; same at alpha==1."""
    log_param = torch.tensor(log_alpha, requires_grad=True)
    gap = torch.tensor(-0.2)
    g_log = torch.autograd.grad(log_param * gap, log_param, retain_graph=True)[0]
    g_alpha = torch.autograd.grad(log_param.exp() * gap, log_param)[0]
    torch.testing.assert_close(g_alpha, g_log * log_param.detach().exp())
    if log_alpha == 0:
        torch.testing.assert_close(g_log, g_alpha)


def test_only_silent_destination_has_zero_attainable_discrete_entropy():
    k = 7
    probs = torch.zeros(k)
    probs[0] = 1.0
    h = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    target_maskaware = 0.2 * torch.log(torch.tensor(1.0))
    target_fixed = 0.2 * torch.log(torch.tensor(float(k)))
    torch.testing.assert_close(h, target_maskaware)
    assert target_fixed > 0
