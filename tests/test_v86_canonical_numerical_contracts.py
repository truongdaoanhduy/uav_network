"""V86 reproducible numerical failures: masked targets, routes, canonical gradient defaults."""
from pathlib import Path
import numpy as np
import pytest
import torch

from uav_network.algorithms.baselines.canonical_continuous import (
    CanonicalContinuousMATD3, CanonicalContinuousMASAC,
    _assert_canonical_config, event_driven_comm_action_np, event_driven_comm_action_torch,
)
from uav_network.algorithms.configuration.loader import load_learner_config

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "configs/learners/v85_canonical_matd3_masac_baseline.yaml"


def config(alg):
    cfg = load_learner_config(YAML)
    cfg.world.num_uavs = 2
    getattr(cfg, alg).hidden_sizes = [16, 16]
    return cfg


def batch():
    # Encoded observation[:, :, 9] is ACTIVE FLAG, not GCS DISTANCE.
    torch.manual_seed(17)
    obs = torch.randn(3, 2, 518)
    obs[..., 9] = 1.
    nex = obs.clone()
    nex[:, 1, 9] = 0.  # The second UAV's battery is depleted.
    return dict(observation=obs, next_observation=nex,
                state=obs.reshape(3, -1), next_state=nex.reshape(3, -1),
                continuous=torch.zeros(3, 2, 2),
                reward=torch.zeros(3), terminated=torch.zeros(3))


@pytest.mark.parametrize("alg,klass", [
    ("matd3", CanonicalContinuousMATD3),
    ("masac", CanonicalContinuousMASAC),
])
def test_target_critic_cannot_receive_motion_from_depleted_uav(alg, klass):
    cfg = config(alg)
    learner = klass(cfg, 518, 1036)
    # Force target 1 to emit nonzero action, detecting missing masking exactly.
    with torch.no_grad():
        for p in learner.actors[1].parameters(): p.zero_()
        for p in (learner.target_actors[1].parameters()
                  if alg == "matd3" else learner.actors[1].parameters()):
            p.zero_()
        if alg == "matd3":
            learner.target_actors[1][-1].bias.fill_(2.)
        else:
            learner.actors[1][-1].bias[:2].fill_(2.)
    seen = []
    handle = learner.target_q1[0].register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    learner.update(batch())
    handle.remove()
    assert seen, "Target Q was never evaluated"
    # Last 4 entries are (uav0_vx, uav0_vy, uav1_vx, uav1_vy).
    assert torch.allclose(seen[0][:, -2:], torch.zeros_like(seen[0][:, -2:])), seen[0][:, -4:]


def test_gcs_route_priority_uses_observed_route_validity():
    obs = np.zeros((1, 2, 518), dtype=np.float32)
    obs[..., -3:] = [1., 1., 1.]
    obs[:, 0, 10] = 0.  # GCS route absent, but peer transmission allowed.
    obs[:, 1, 10] = 1.  # GCS route available.
    expected = np.array([[2, 1]])
    np.testing.assert_array_equal(event_driven_comm_action_np(obs), expected)
    np.testing.assert_array_equal(event_driven_comm_action_torch(torch.tensor(obs)).numpy(), expected)


def test_canonical_gradient_clipping_must_be_explicitly_disabled():
    for alg in ("matd3", "masac"):
        cfg = config(alg)
        assert getattr(cfg, alg).grad_clip_norm == 0
        getattr(cfg, alg).grad_clip_norm = 10.
        with pytest.raises(ValueError, match="grad_clip"):
            _assert_canonical_config(cfg, alg)


def test_matd3_delayed_update_and_target_polyak():
    torch.manual_seed(3)
    learner = CanonicalContinuousMATD3(config("matd3"), 518, 1036)
    before_actor = [p.detach().clone() for p in learner.actors[0].parameters()]
    before_target = [p.detach().clone() for p in learner.target_actors[0].parameters()]
    first = learner.update(batch())
    assert "actor_loss" not in first
    assert all(torch.equal(a, b) for a, b in zip(before_actor, learner.actors[0].parameters()))
    second = learner.update(batch())
    assert "actor_loss" in second
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, learner.actors[0].parameters()))
    for old, current, target in zip(before_target, learner.actors[0].parameters(),
                                    learner.target_actors[0].parameters()):
        expected = old * (1 - learner.tau) + current.detach() * learner.tau
        assert torch.allclose(target, expected, atol=1e-7)


def test_masac_actor_alpha_updates_and_actions_in_bounds():
    torch.manual_seed(8)
    learner = CanonicalContinuousMASAC(config("masac"), 518, 1036)
    before_alpha = learner.log_alpha.detach().clone()
    before_actor = [p.detach().clone() for p in learner.actors[0].parameters()]
    metrics = learner.update(batch())
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(before_alpha, learner.log_alpha.detach())
    assert torch.all(learner.log_alpha.exp() > 0)
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, learner.actors[0].parameters()))
    obs = batch()["observation"]
    obs[:, 1, 9] = 0.
    action = learner.act_batch(obs, return_tensors=True)
    assert action.continuous.shape == (3, 2, 2)
    assert torch.all(action.continuous.abs() <= 1)
    assert torch.count_nonzero(action.continuous[:, 1]) == 0


def test_disabled_clipping_validation_preserves_negative_guard():
    cfg = config("matd3")
    cfg.matd3.grad_clip_norm = -0.1
    with pytest.raises(ValueError):
        cfg.validate()
