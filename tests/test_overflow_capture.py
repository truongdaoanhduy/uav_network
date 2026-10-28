"""Raw overflow evidence must precede clipping and permit isolated precision replay."""
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
import json
import torch
import pytest
from test_precision import precision, same
from test_split_updates import small_learner, replay_batch


@pytest.mark.parametrize('factor,category', [(1e38, 'nonfinite_gradients'), (1e20, 'norm_reduction_overflow')])
def test_skip_observer_sees_original_unclipped_gradients(factor, category):
    execution = precision(groups=('critic',))
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([p])
    events = []
    def observe(event, parameters):
        events.append((event, [v.grad.detach().clone() for group in parameters for v in group]))
    execution.overflow_observer = SimpleNamespace(on_skip=observe)
    before = p.detach().clone()
    norm = execution.step('critic', (p * factor).sum(), optimizer, [p], 1)
    assert not torch.isfinite(norm)
    assert len(events) == 1
    event, grads = events[0]
    assert event['category'] == category
    assert event['scale_before'] == 128
    assert event['scale_after'] == 64
    assert event['raw_gradients_finite'] == (category == 'norm_reduction_overflow')
    assert bool(torch.isfinite(grads[0]).all()) == (category == 'norm_reduction_overflow')
    same(p, before)
    assert not optimizer.state


def test_observer_failure_does_not_abort_scaler_backoff():
    execution = precision(groups=('critic',))
    def broken(*args): raise OSError('disk unavailable')
    execution.overflow_observer = SimpleNamespace(on_skip=broken)
    p = torch.nn.Parameter(torch.ones(2))
    execution.step('critic', (p * 1e38).sum(), torch.optim.Adam([p]), [p], 1)
    assert execution.counts['critic']['skipped'] == 1
    assert getattr(execution, 'observer_errors', 0) == 1


def capture_module():
    assert find_spec('uav_network.experiments.overflow_capture'), 'Missing first-overflow snapshot/replay'
    return import_module('uav_network.experiments.overflow_capture')


@pytest.mark.parametrize('algorithm', ['masac', 'matd3'])
@pytest.mark.parametrize('group', ['critic', 'actor'])
def test_first_skip_preserves_batch_model_rng_and_replays_fp32_fp16(tmp_path, algorithm, group):
    module = capture_module()
    learner = small_learner(algorithm)
    learner.config.runtime.amp_mode = 'fp16'
    learner.precision = import_module('uav_network.algorithms.precision').MixedPrecision('fp16', 'cpu', learner.precision.groups)
    state = learner.precision.state_dict()
    for scaler in state['scalers'].values(): scaler['scale'] = 1e30
    learner.precision.load_state_dict(state)
    batch = replay_batch(learner)
    before = deepcopy(learner.state_dict())
    recorder = module.attach_overflow_capture(learner, tmp_path, 'test')
    rng = torch.get_rng_state().clone()
    update = learner.update_critic if group == 'critic' else learner.update_actor
    update(batch)
    paths = list(tmp_path.glob('overflow-test-' + group + '.pt'))
    assert len(paths) == 1
    saved = torch.load(paths[0], weights_only=True)
    same(saved['model'], before)
    same(saved['batch'], batch)
    same(saved['rng']['cpu'], rng)
    assert saved['event']['category'] == 'nonfinite_gradients'
    assert saved['event']['raw_gradients_finite'] is False
    assert all(row['finite'] for row in saved['input_stats'].values())
    original_bytes = paths[0].read_bytes()
    update(batch)
    assert paths[0].read_bytes() == original_bytes
    rng = torch.get_rng_state().clone()
    result = module.replay_overflow(paths[0], device='cpu')
    same(torch.get_rng_state(), rng)
    assert result['cases']['off']['losses_finite'] is True
    assert result['cases']['off']['skipped'] == 0
    assert result['cases']['fp16']['skipped'] >= 1
    assert result['cases']['fp16']['first_nonfinite_parameter_gradient'] is not None
    json.dumps(result, allow_nan=False)


def test_capture_storage_failure_keeps_real_update_alive(tmp_path):
    module = capture_module()
    learner = small_learner('matd3')
    learner.config.runtime.amp_mode = 'fp16'
    cls = import_module('uav_network.algorithms.precision').MixedPrecision
    learner.precision = cls('fp16', 'cpu', learner.precision.groups)
    state = learner.precision.state_dict()
    for scaler in state['scalers'].values(): scaler['scale'] = 1e30
    learner.precision.load_state_dict(state)
    blocked = tmp_path / 'blocked'
    blocked.write_text('not a directory')
    module.attach_overflow_capture(learner, blocked, 'test')
    learner.update_critic(replay_batch(learner))
    assert learner.precision.counts['critic']['skipped'] == 1
    assert learner.precision.observer_errors == 1


@pytest.mark.parametrize('mode', ['base', 'synchronous', 'async'])
@pytest.mark.parametrize('algorithm', ['masac', 'matd3'])
def test_training_attaches_capture_to_real_base_and_pql_learners(tmp_path, mode, algorithm):
    from test_overflow_telemetry import config
    from uav_network.experiments.training import train
    from uav_network.algorithms.precision import MixedPrecision
    from unittest.mock import patch
    c = config(tmp_path, algorithm, mode)
    c.runtime.amp_mode = 'fp16'
    original = MixedPrecision.__init__
    def high_scale(self, *args, **kwargs):
        original(self, *args, **kwargs)
        state = self.state_dict()
        for scaler in state['scalers'].values(): scaler['scale'] = 1e30
        self.load_state_dict(state)
    with patch.object(MixedPrecision, '__init__', high_scale):
        result = train(c)
    assert result['episodes'] == 4
    paths = list((tmp_path / 'first').glob('overflow-*.pt'))
    assert paths
    for path in paths:
        assert path.with_suffix('.json').is_file()
        payload = torch.load(path, weights_only=True)
        assert payload['event']['category'] == 'nonfinite_gradients'
        assert all(s['finite'] for s in payload['input_stats'].values())


def test_preclipping_observer_keeps_torch25_compatibility(monkeypatch):
    monkeypatch.delattr(torch.nn.utils, 'get_total_norm', raising=False)
    monkeypatch.delattr(torch.nn.utils, 'clip_grads_with_norm_', raising=False)
    execution = precision(groups=('critic',))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    execution.step('critic', parameter * 100, torch.optim.SGD([parameter], lr=1), [parameter], 2)
    assert parameter.item() == pytest.approx(-1)
