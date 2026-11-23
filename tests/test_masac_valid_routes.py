"""Compare valid-route MASAC training with the original dense twin critic."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from uav_network.algorithms.action_utils import valid_destinations
from uav_network.algorithms.masac import MASAC
from uav_network.configuration import Config, config_fingerprint
from uav_network.environment.observations import observation_dim, state_dim


@pytest.mark.parametrize("chunk_size", [0, 3])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA device required")),
])
def test_valid_routes_skip_zero_probability_candidates_and_match_dense_updates(chunk_size, device, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(44)
    config = Config()
    config.world.num_uavs = 6
    config.world.num_targets = 35
    config.world.num_obstacles = 25
    config.network.power_control = "learned"
    config.masac.hidden_sizes = [256, 512]
    config.masac.share_actor_parameters = True
    config.masac.share_critic_parameters = True
    config.masac.actor_layer_norm = True
    config.runtime.critic_chunk_size = chunk_size
    obs_dim, global_dim = observation_dim(config), state_dim(config)
    dense = MASAC(config, obs_dim, global_dim, device)
    compact_config = deepcopy(config)
    compact_config.masac.shared_critic_mode = "valid_routes"
    compact = MASAC(compact_config, obs_dim, global_dim, device)
    compact.load_state_dict(dense.state_dict())
    batch_size, drones, routes = 6, dense.num_uavs, dense.destinations
    obs, next_obs = (torch.rand(batch_size, drones, obs_dim) for _ in range(2))
    for observations in (obs, next_obs):
        observations[..., 9] = 1
        observations[..., -routes:] = 0
    obs[0, 1, -routes:] = 1
    obs[1, 0, -2] = 1
    next_obs[-1, 4, -1] = 1
    # Inactive UAVs and replay actions are both independent of route enumeration.
    obs[-1, -1, 9] = 0
    next_obs[0, -1, 9] = 0
    batch = {
        "observation": obs,
        "next_observation": next_obs,
        "state": torch.randn(batch_size, global_dim),
        "next_state": torch.randn(batch_size, global_dim),
        "continuous": torch.randn(batch_size, drones, config.continuous_dim).tanh(),
        "destination": torch.zeros(batch_size, drones, dtype=torch.long),
        "reward": torch.randn(batch_size) * 3,
        "terminated": torch.tensor([False, True, False, False, False, False]),
    }

    # Ordinary critic regression processes batch_size rows per update.
    rows = {"actor_and_regression": 0, "target": 0}

    def count_rows(label, inputs):
        rows[label] += len(inputs[0])

    compact.critics[0].register_forward_pre_hook(
        lambda _module, inputs: count_rows("actor_and_regression", inputs)
    )
    compact.target_critics[0].register_forward_pre_hook(
        lambda _module, inputs: count_rows("target", inputs)
    )
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for model in (dense, compact):
            torch.manual_seed(4401)
            model.update(batch, collect_metrics=True)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)
    assert rows["target"] == int(valid_destinations(next_obs, routes).sum())
    # Frozen 18d756ef hybrid MASAC only skips invalid routes for *target*
    # expectation. Its Actor path remains dense. This test certifies actual
    # byte-pinned semantics rather than silently changing the research model.
    assert rows["actor_and_regression"] >= batch_size + batch_size * drones * routes
    if chunk_size == 0:
        assert rows["actor_and_regression"] == batch_size + batch_size * drones * routes
    for name, expected in dense.state_dict().items():
        torch.testing.assert_close(compact.state_dict()[name], expected, rtol=2e-5, atol=2e-6)


def test_shared_critic_mode_config_validation():
    config = Config()
    assert config.masac.shared_critic_mode == "dense"
    config.masac.shared_critic_mode = "typo"
    with pytest.raises(ValueError, match="masac.shared_critic_mode"):
        config.validate()



def test_explicit_dense_keeps_legacy_identity_and_alternatives_get_distinct_identities():
    config = Config()
    legacy = config.to_dict()
    del legacy["masac"]["shared_critic_mode"]
    # The canonical fingerprint also omits the MATD3 zero-weight default,
    # which was added after this MASAC legacy-identity regression was written.
    assert legacy["matd3"]["actor_output_regularization"] == 0
    del legacy["matd3"]["actor_output_regularization"]
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True, allow_nan=False).encode()).hexdigest()
    assert config_fingerprint(config) == expected
    fingerprints = {expected}
    for mode in ("split", "valid_routes"):
        config.masac.shared_critic_mode = mode
        fingerprints.add(config_fingerprint(config.validate()))
    assert len(fingerprints) == 3
