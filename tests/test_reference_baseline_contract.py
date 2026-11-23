"""Source-backed invariants with declared shared/LN/chunk execution.

These checks intentionally reject UAV-specific algorithm stabilizers from BASE.
They do not constrain environment/reward choices, which are shared task design.
"""
from pathlib import Path

import pytest

from uav_network.configuration import load_config


XYZ_BASELINES = [Path("configs/comparison-reference-masac-matd3-xyz-sensing.yaml")]


@pytest.mark.parametrize("path", XYZ_BASELINES)
def test_xyz_baseline_uses_source_backed_algorithm_defaults(path):
    c = load_config(path)

    # SAC/SB3/BenchMARL automatic-temperature convention.
    assert c.masac.initial_alpha == 1.0
    # TorchRL/BenchMARL discrete SAC auto target uses 0.2 * log(n_actions).
    # Action masking changes sampling support, not the fixed reference target cardinality.
    assert c.masac.target_discrete_entropy_ratio == 0.2
    assert c.masac.target_discrete_entropy_final_ratio == 0.2
    assert c.masac.target_discrete_entropy_anneal_updates == 0
    # BenchMARL/TorchRL uses fixed num_actions in the auto discrete target;
    # masking changes policy support, not the target cardinality.
    assert c.masac.mask_aware_discrete_entropy_target is False
    # Continuous SAC auto target is -dim(A); XYZ + learned RF power = four controls.
    assert c.masac.target_continuous_entropy == -4.0

    # Fujimoto TD3 defaults/source implementation.
    assert c.matd3.policy_delay == 2
    assert c.matd3.target_noise_std == 0.2
    assert c.matd3.target_noise_clip == 0.5
    assert c.matd3.exploration_noise_std == 0.1
    assert c.matd3.output_normalization is False
    assert c.masac.share_actor_parameters is True
    assert c.masac.share_critic_parameters is True
    assert c.matd3.share_actor_parameters is True
    assert c.matd3.share_critic_parameters is True

    # Runtime mechanics must not change the baseline equations.
    assert c.runtime.critic_chunk_size == 2048
    assert c.runtime.vectorize_critics is False
    assert c.runtime.compile_models is False
    assert c.runtime.amp_mode == "off"
    assert c.runtime.fused_optimizers is False
    assert c.pql.enabled is False


def test_sac_reference_log_std_bounds_are_used():
    from uav_network.algorithms import models
    assert models.LOG_STD_MIN == -20
    assert models.LOG_STD_MAX == 2


def test_masac_temperature_is_not_hard_capped_by_uav_specific_limit():
    import torch
    from uav_network.algorithms.masac import MASAC
    from uav_network.configuration import Config
    from uav_network.environment.observations import observation_dim, state_dim

    c = Config()
    c.world.num_uavs = 2
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.masac.hidden_sizes = [16, 16]
    learner = MASAC(c, observation_dim(c), state_dim(c))
    # A source baseline can learn alpha freely; freeze the optimizer so this test
    # isolates post-step clipping from the temperature objective itself.
    for group in learner.alpha_optimizer.param_groups:
        group["lr"] = 0.0
    with torch.no_grad():
        learner.log_alpha_continuous.fill_(6.0)
        learner.log_alpha_discrete.fill_(6.0)

    b, n = 4, c.world.num_uavs
    obs = torch.randn(b, n, observation_dim(c))
    obs[:, :, 9] = 1
    obs[:, :, -(n + 1):] = 1
    batch = {
        "observation": obs,
        "next_observation": obs.clone(),
        "state": torch.randn(b, state_dim(c)),
        "next_state": torch.randn(b, state_dim(c)),
        "continuous": torch.zeros(b, n, c.continuous_dim),
        "destination": torch.zeros(b, n, dtype=torch.long),
        "reward": torch.zeros(b),
        "terminated": torch.zeros(b),
    }
    learner.update_actor(batch, collect_metrics=False)
    torch.testing.assert_close(learner.log_alpha_continuous, torch.full((n,), 6.0), rtol=0, atol=0)
    torch.testing.assert_close(learner.log_alpha_discrete, torch.full((n,), 6.0), rtol=0, atol=0)


def test_matd3_actor_update_keeps_peer_actions_from_replay():
    import torch
    from torch import nn
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.action_utils import joint_action
    from uav_network.configuration import Config
    from uav_network.environment.observations import observation_dim, state_dim

    class RecordingCritic(nn.Module):
        def __init__(self, own):
            super().__init__()
            self.own = own
            self.seen = None
        def forward(self, state, joint):
            self.seen = joint.detach().clone()
            q = joint[:, self.own, :3].sum(-1)
            return q, q

    c = Config()
    c.world.num_uavs = 2
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.runtime.vectorize_critics = False
    c.matd3.hidden_sizes = [16, 16]
    learner = MATD3(c, observation_dim(c), state_dim(c))
    critics = nn.ModuleList([RecordingCritic(0), RecordingCritic(1)])
    learner.critics = critics

    b, n = 4, c.world.num_uavs
    obs = torch.randn(b, n, observation_dim(c))
    obs[:, :, 9] = 1
    obs[:, :, -(n + 1):] = 1
    replay_continuous = torch.tensor([[[0.81, 0.72, 0.63], [-0.81, -0.72, -0.63]]]).expand(b, -1, -1).clone()
    replay_destination = torch.tensor([[1, 2]]).expand(b, -1).clone()
    batch = {
        "observation": obs,
        "next_observation": obs.clone(),
        "state": torch.randn(b, state_dim(c)),
        "next_state": torch.randn(b, state_dim(c)),
        "continuous": replay_continuous,
        "destination": replay_destination,
        "reward": torch.zeros(b),
        "terminated": torch.zeros(b),
    }
    replay_joint = joint_action(replay_continuous, replay_destination, n + 1)
    torch.manual_seed(123)
    learner.update_actor(batch, collect_metrics=False)

    # Critic i may receive a new action only for agent i. Every peer slot must
    # stay exactly equal to the sampled replay action (Ackermann MATD3 source).
    torch.testing.assert_close(critics[0].seen[:, 1], replay_joint[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(critics[1].seen[:, 0], replay_joint[:, 0], rtol=0, atol=0)


def test_matd3_differentiable_discrete_action_uses_soft_gumbel_sample():
    import torch
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.configuration import Config
    from uav_network.environment.observations import observation_dim, state_dim

    c = Config()
    c.world.num_uavs = 2
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.matd3.hidden_sizes = [16, 16]
    learner = MATD3(c, observation_dim(c), state_dim(c))
    actor = learner.actors[0]
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    obs = torch.randn(8, observation_dim(c))
    obs[:, 9] = 1
    obs[:, -(c.world.num_uavs + 1):] = 1

    torch.manual_seed(123)
    action = learner._actor_action(actor, obs, relaxed_discrete=True, agent_index=0)
    discrete = action[:, c.continuous_dim:]
    torch.testing.assert_close(discrete.sum(-1), torch.ones(len(discrete)), rtol=1e-6, atol=1e-6)
    assert torch.all((discrete > 0) & (discrete < 1))
    # A soft Gumbel sample must not be the deterministic probability vector
    # nor a hard one-hot vector.
    uniform = torch.full_like(discrete, 1.0 / learner.destinations)
    assert not torch.allclose(discrete, uniform)
    assert not torch.allclose(discrete, discrete.round())


def test_masac_uses_per_agent_temperature_per_entropy_component():
    from uav_network.algorithms.masac import MASAC
    from uav_network.configuration import Config
    from uav_network.environment.observations import observation_dim, state_dim

    c = Config()
    c.world.num_uavs = 2
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.masac.hidden_sizes = [16, 16]
    learner = MASAC(c, observation_dim(c), state_dim(c))
    assert learner.log_alpha_continuous.shape == (c.world.num_uavs,)
    assert learner.log_alpha_discrete.shape == (c.world.num_uavs,)


def test_matd3_masked_gumbel_has_zero_invalid_mass_and_finite_gradients():
    import torch
    from uav_network.algorithms.matd3 import MATD3

    probs = torch.tensor([[0.7, 0.3, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
    torch.manual_seed(7)
    sample = MATD3._relaxed_destination(probs)
    torch.testing.assert_close(sample[:, 2], torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(sample[1, 1:], torch.zeros(2), rtol=0, atol=0)
    sample[:, 0].sum().backward()
    assert torch.isfinite(probs.grad).all()


def test_reference_gradient_clipping_matches_primary_sources():
    from uav_network.configuration import load_config
    c = load_config('configs/comparison-reference-masac-matd3-xyz-sensing.yaml')
    # Delalleau Hybrid SAC explicitly reports no gradient clipping.
    assert c.masac.grad_clip_mode == 'none'
    # Ackermann official MATD3 code uses clip_by_norm per variable at 0.5.
    assert c.matd3.grad_clip_mode == 'per_parameter'
    assert c.matd3.grad_clip_norm == 0.5


def test_gradient_clip_modes_execute_source_semantics():
    import torch
    from uav_network.algorithms.optimization import step_optimizer

    def updated(mode):
        p1 = torch.nn.Parameter(torch.tensor(0.0))
        p2 = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([p1, p2], lr=1.0)
        loss = 3.0 * p1 + 4.0 * p2
        step_optimizer(loss, optimizer, [p1, p2], 0.5, grad_clip_mode=mode)
        return torch.stack((p1.detach(), p2.detach()))

    torch.testing.assert_close(updated('none'), torch.tensor([-3.0, -4.0]), rtol=0, atol=0)
    # Official MATD3 clips each variable's gradient norm independently.
    torch.testing.assert_close(updated('per_parameter'), torch.tensor([-0.5, -0.5]), rtol=0, atol=1e-6)


def test_reference_profile_preserves_common_uav_scenario_and_reward():
    from dataclasses import asdict
    reference = load_config('configs/comparison-reference-masac-matd3-xyz-sensing.yaml')
    common = load_config('configs/comparison-20k-hybrid75-1kb-shared-actor-critic-xyz-sensing.yaml')
    groups = ('world', 'sensor', 'network', 'energy', 'reward')
    assert {g: asdict(getattr(reference, g)) for g in groups} == {
        g: asdict(getattr(common, g)) for g in groups
    }


@pytest.mark.parametrize("temperature_loss", ["log_alpha", "alpha"])
@pytest.mark.parametrize("mask_aware", [False, True], ids=["reference", "mask-aware"])
def test_discrete_entropy_target_respects_selected_profile(monkeypatch, mask_aware, temperature_loss):
    """The mask-aware variant updates legal choices and holds forced silence steady."""
    import math
    import types
    import torch
    import uav_network.algorithms.masac as masac_module
    from uav_network.algorithms.masac import MASAC
    from uav_network.environment.observations import observation_dim, state_dim

    c = load_config(
        XYZ_BASELINES[0],
        [f"masac.mask_aware_discrete_entropy_target={str(mask_aware).lower()}",
         f"masac.temperature_loss={temperature_loss}"],
    )
    c.world.num_uavs = 2
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.masac.hidden_sizes = [16, 16]
    c.masac.initial_alpha = 0.5
    learner = MASAC(c, observation_dim(c), state_dim(c))
    learner.alpha_optimizer = torch.optim.SGD(
        [learner.log_alpha_continuous, learner.log_alpha_discrete], lr=0.1
    )

    b, n, k = 4, c.world.num_uavs, c.world.num_uavs + 1
    obs = torch.zeros(b, n, observation_dim(c))
    obs[:, :, 9] = 1
    # Agent 0 has two or three legal destinations. Agent 1 can only stay silent.
    obs[:, 0, -k:] = 1
    obs[::2, 0, -1] = 0
    obs[:, 1, -k] = 1
    batch = {
        'observation': obs, 'next_observation': obs.clone(),
        'state': torch.zeros(b, state_dim(c)), 'next_state': torch.zeros(b, state_dim(c)),
        'continuous': torch.zeros(b, n, c.continuous_dim),
        'destination': torch.zeros(b, n, dtype=torch.long),
        'reward': torch.zeros(b), 'terminated': torch.zeros(b),
    }

    def fixed_sample(self, actor, observation, deterministic=False, agent_index=0):
        count = len(observation)
        mask = observation[:, -k:].bool()
        probs = mask.float() / mask.sum(-1, keepdim=True)
        log_probs = torch.where(mask, probs.clamp_min(1e-30).log(), torch.zeros_like(probs))
        return (torch.zeros(count, self.continuous_dim), torch.full((count,), -2.0), probs, log_probs)

    learner._sample = types.MethodType(fixed_sample, learner)
    monkeypatch.setattr(
        masac_module, 'step_actor_optimizers',
        lambda losses, optimizers, groups, *args, **kwargs: [torch.tensor(0.0)] * len(losses),
    )
    before = learner.log_alpha_discrete.detach().clone()
    before_continuous = learner.log_alpha_continuous.detach().clone()
    learner.update_actor(batch, collect_metrics=False)

    actual_entropy = torch.tensor([(math.log(2) + math.log(3)) / 2, 0.0])
    target_entropy = actual_entropy if mask_aware else math.log(k)
    target = c.masac.target_discrete_entropy_ratio * target_entropy
    gradient_scale = before.exp() if temperature_loss == "alpha" else 1.0
    expected = before - 0.1 * gradient_scale * (actual_entropy - target)
    torch.testing.assert_close(learner.log_alpha_discrete, expected, rtol=0, atol=1e-6)
    expected_controls = torch.tensor([4 - (1 / 2 + 1 / 3) / 2, 3.0])
    continuous_scale = before_continuous.exp() if temperature_loss == "alpha" else 1.0
    expected_continuous = before_continuous - 0.1 * continuous_scale * (2.0 + expected_controls)
    torch.testing.assert_close(learner.log_alpha_continuous, expected_continuous, rtol=0, atol=1e-6)


def test_independent_source_profile_uses_per_agent_plain_mlp_baselines():
    from dataclasses import asdict
    from uav_network.algorithms.masac import MASAC
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.environment.observations import observation_dim, state_dim

    c = load_config("configs/comparison-independent-baseline-xyz-sensing.yaml")
    reference = load_config(XYZ_BASELINES[0])
    for group in ("world", "sensor", "network", "energy", "reward"):
        assert asdict(getattr(c, group)) == asdict(getattr(reference, group))
    assert c.masac.temperature_loss == "alpha"
    assert c.masac.mask_aware_discrete_entropy_target
    assert c.matd3.initialization == "default"
    assert not c.matd3.output_normalization
    assert c.runtime.critic_chunk_size == 0
    for algorithm, cls in (("masac", MASAC), ("matd3", MATD3)):
        options = getattr(c, algorithm)
        assert not options.actor_layer_norm
        assert not options.share_actor_parameters
        assert not options.share_critic_parameters
        options.hidden_sizes = [8, 8]
        model = cls(c, observation_dim(c), state_dim(c))
        assert len(model.actors) == len(model.critics) == c.world.num_uavs


def test_historical_temperature_config_hash_remains_accepted():
    import hashlib
    import json
    from uav_network.experiments.best_checkpoints import _config_hashes

    c = load_config(XYZ_BASELINES[0])
    assert c.masac.temperature_loss == "log_alpha"
    legacy = c.to_dict()
    legacy["masac"].pop("temperature_loss")
    old_hash = hashlib.sha256(json.dumps(legacy, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert old_hash in _config_hashes(c)
    from uav_network.experiments.best_checkpoints import selection_identity
    old_identity = selection_identity(c)
    c.masac.temperature_loss = "alpha"
    assert old_hash not in _config_hashes(c)
    assert selection_identity(c) != old_identity


def test_temperature_loss_rejects_unknown_objective():
    with pytest.raises(ValueError, match="masac.temperature_loss"):
        load_config(XYZ_BASELINES[0], ["masac.temperature_loss=unknown"])
