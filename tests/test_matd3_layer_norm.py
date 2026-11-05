"""MATD3 actor normalization, gradient flow and checkpoint architecture contracts."""
import numpy as np
import pytest
import torch
from torch import nn

from uav_network.algorithms.matd3 import MATD3
from uav_network.configuration import Config, load_config, with_legacy_feature_defaults
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.experiments.evaluation_snapshots import (
    load_evaluation_snapshot, save_evaluation_snapshot,
)


def config():
    c = load_config('configs/comparison-reference-masac-matd3-xyz-sensing.yaml')
    c.runtime.algorithm = 'matd3'
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 1
    c.matd3.hidden_sizes = [32, 32]
    c.matd3.share_actor_parameters = c.matd3.share_critic_parameters = True
    c.tracking.enabled = False
    return c


def observations(c, batch=8):
    x = torch.randn(batch, c.world.num_uavs, observation_dim(c))
    x[..., 9] = 1
    x[..., -(c.world.num_uavs + 1):] = 1
    return x


def test_current_profile_enables_actor_layer_norm_without_output_normalization():
    c = load_config('configs/comparison-reference-masac-matd3-xyz-sensing.yaml')
    assert getattr(c.matd3, 'actor_layer_norm', False) is True
    assert c.matd3.output_normalization is False
    assert getattr(Config().matd3, 'actor_layer_norm', None) is False


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('pql', [False, True])
def test_actor_hidden_norm_keeps_amplified_features_differentiable(shared, pql):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    c = config()
    c.matd3.actor_layer_norm = True
    c.matd3.share_actor_parameters = shared
    c.pql.enabled = pql
    learner = MATD3(c, observation_dim(c), state_dim(c))
    for actor in [*learner.actors, *learner.target_actors]:
        assert sum(isinstance(layer, nn.LayerNorm) for layer in actor.encoder) == 2
        assert actor.output_normalization is False
    assert not any(isinstance(layer, nn.LayerNorm) for layer in learner.critics.modules())
    x = observations(c) * 1e6
    continuous, _ = learner._policy_outputs(x)
    assert torch.isfinite(continuous).all()
    assert continuous.abs().max() < .99
    continuous.square().mean().backward()
    for actor in learner.actors:
        assert torch.isfinite(actor.continuous.weight.grad).all()
        assert actor.continuous.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in learner.actors.parameters() if p.grad is not None)


@pytest.mark.parametrize('kind', ['policy', 'evaluation'])
@pytest.mark.parametrize('legacy', [False, True])
def test_actor_normalization_checkpoint_round_trip(tmp_path, kind, legacy):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    c = config()
    c.matd3.actor_layer_norm = not legacy
    learner = MATD3(c, observation_dim(c), state_dim(c))
    obs = observations(c)
    before = learner.act_batch(obs, deterministic=True)
    if kind == 'policy':
        path = tmp_path / 'policy.pt'
        learner.save(path)
    else:
        path = save_evaluation_snapshot(c, tmp_path, learner.actors.state_dict(), 0,
                                       {'episodes': 0, 'global_step': 0})
    saved = torch.load(path, weights_only=True)
    if legacy:
        saved['config']['matd3'].pop('actor_layer_norm', None)
    else:
        assert saved['config']['matd3'].get('actor_layer_norm') is True
    torch.save(saved, path)
    restored = (MATD3.load(path) if kind == 'policy' else load_evaluation_snapshot(saved))
    assert getattr(restored.config.matd3, 'actor_layer_norm', None) is (not legacy)
    after = restored.act_batch(obs, deterministic=True)
    np.testing.assert_array_equal(before.continuous, after.continuous)
    np.testing.assert_array_equal(before.destination, after.destination)
    for actor in restored.actors:
        assert sum(isinstance(layer, nn.LayerNorm) for layer in actor.encoder) == (0 if legacy else 2)


def test_legacy_resume_config_defaults_keep_layer_norm_disabled():
    old = {'matd3': {}}
    normalized = with_legacy_feature_defaults(old)
    assert normalized['matd3'].get('actor_layer_norm') is False
    assert old == {'matd3': {}}


@pytest.mark.parametrize('algorithm', ['masac', 'matd3'])
@pytest.mark.parametrize('pql', [False, True])
@pytest.mark.parametrize('legacy_xyz_hash', [False, True])
def test_historical_best_state_resumes_without_norm_but_rejects_changed_architecture(
        tmp_path, algorithm, pql, legacy_xyz_hash):
    import hashlib
    import io
    import json
    from copy import deepcopy
    from test_best_checkpoints import config as best_config, candidate
    from uav_network.configuration_xyz import strip_legacy_xyz_defaults
    from uav_network.experiments.best_checkpoints import BestCheckpoint

    c = best_config()
    c.runtime.algorithm, c.pql.enabled = algorithm, pql
    c.matd3.actor_layer_norm = False
    manager = BestCheckpoint(c, tmp_path / 'first')
    result, source = candidate(c, tmp_path / 'snapshots')
    assert manager.consider(result, source)
    state = manager.state_dict()
    saved = torch.load(io.BytesIO(state['checkpoint_bytes'].numpy().tobytes()), weights_only=True)
    saved['config']['matd3'].pop('actor_layer_norm')
    buffer = io.BytesIO()
    torch.save(saved, buffer)
    data = buffer.getvalue()
    state['checkpoint_bytes'] = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    meta = state['metadata']
    meta['checkpoint_sha256'] = hashlib.sha256(data).hexdigest()
    historical = saved['config']
    sha_config = strip_legacy_xyz_defaults(historical) if legacy_xyz_hash else historical
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    meta['config_sha256'] = digest(sha_config)
    # This is the historical serialized identity, computed before the new flag existed.
    before = strip_legacy_xyz_defaults(historical)
    identity = {k: before[k] for k in ('world', 'sensor', 'network', 'energy', 'reward', algorithm, 'pql')}
    identity['runtime'] = {k: before['runtime'][k] for k in
                           ('algorithm', 'seed', 'amp_mode', 'num_envs', 'environment_backend')}
    identity['validation'] = [c.tracking.periodic_evaluation_seed, c.tracking.periodic_evaluation_episodes]
    meta['identity'] = digest(identity)
    restored = BestCheckpoint(c, tmp_path / 'restored', state=state)
    assert restored.path.read_bytes() == data
    if algorithm == 'matd3':
        changed = deepcopy(c)
        changed.matd3.actor_layer_norm = True
        with pytest.raises(ValueError, match='identity/provenance'):
            BestCheckpoint(changed, tmp_path / 'changed', state=state)
