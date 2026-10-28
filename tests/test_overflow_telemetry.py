"""Handled numeric overflow must not destroy episode/history or scalar charts."""
from pathlib import Path
import json
import math
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_pql_system import pql_config
from test_tracking import OnlineRun, StoredArtifact
from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.matd3 import MATD3
from uav_network.experiments.training import train
from uav_network.experiments.tracking import OnlineTracker


def config(tmp_path, algorithm, mode):
    c = pql_config(algorithm, 'async' if mode == 'async' else 'synchronous')
    c.pql.enabled = mode != 'base'
    c.runtime.environment_backend = 'warp_packet'
    c.runtime.output_dir = str(tmp_path / 'first')
    c.runtime.episodes = 4
    c.world.max_steps = 2
    c.algorithm_config.learning_starts = 0
    c.pql.warmup_collection_steps = 0
    c.pql.critic_sample_ratio = 1
    c.tracking.evaluation_interval = 0
    return c


@pytest.mark.parametrize('algorithm', ['masac', 'matd3'])
@pytest.mark.parametrize('mode', ['base', 'synchronous', 'async'])
def test_nonfinite_learner_metrics_survive_episode_summary_and_resume(tmp_path, algorithm, mode):
    c = config(tmp_path, algorithm, mode)
    cls = MASAC if algorithm == 'masac' else MATD3
    update = cls.update_critic
    def with_overflow_metrics(self, *args, **kwargs):
        values = update(self, *args, **kwargs)
        if values:
            values.update(critic_grad_norm=float('inf'), diagnostic_probe=float('nan'))
        return values
    with patch.object(cls, 'update_critic', with_overflow_metrics):
        first = train(c)
    records = [json.loads(s, parse_constant=lambda s: pytest.fail(s))
               for s in (tmp_path / 'first' / 'episodes.jsonl').read_text().splitlines()]
    assert len(records) == 4
    assert records[-1]['learner']['critic_grad_norm'] is None
    assert records[-1]['learner']['critic_grad_norm_nonfinite'] == 1
    assert records[-1]['learner']['diagnostic_probe_nonfinite_kind'] == 'NaN'
    json.dumps(first, allow_nan=False)
    c.runtime.output_dir = str(tmp_path / 'resumed')
    c.runtime.warm_start_checkpoint = first['resume_checkpoint']
    c.runtime.episodes = 8
    second = train(c)
    assert second['initial_episodes'] == 4
    assert second['episodes'] == 8
    assert math.isfinite(second['final_mean_return'])
    json.dumps(second, allow_nan=False)


def test_online_scalar_channels_keep_nonfinite_numeric_values_missing(tmp_path):
    c = config(tmp_path, 'masac', 'base')
    result = train(c)
    record = json.loads((tmp_path / 'first' / 'episodes.jsonl').read_text().splitlines()[-1])
    record['learner'] = {'critic_grad_norm': float('inf'), 'actor_grad_norm': 0.25}
    c.tracking.credentials_file = str(tmp_path / 'absent')
    run = OnlineRun()
    sdk = SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw, Artifact=StoredArtifact)
    with patch.dict(sys.modules, {'wandb': sdk}), patch.dict(os.environ, {
        'WANDB_API_KEY': 'test-key', 'WANDB_ENTITY': 'owner', 'WANDB_PROJECT': 'project'}):
        tracker = OnlineTracker(c, tmp_path / 'tracking')
        tracker.episode(record)
        tracker.learner(record['learner'], 10)
        tracker.learner({'critic_grad_norm': 2.0}, 11)
        tracker.close()
    for values in run.logged:
        json.dumps(values, allow_nan=False)
        if values.get('learner/critic_grad_norm_nonfinite') == 1:
            assert values['learner/critic_grad_norm'] is None
    rows = [v for v in run.logged if 'learner/critic_grad_norm' in v]
    assert rows[-1]['learner/critic_grad_norm_nonfinite'] == 0
    assert rows[-1]['learner/critic_grad_norm'] == 2.0


def test_online_publishes_bounded_overflow_evidence_once_and_isolates_failure(tmp_path):
    c = config(tmp_path, 'matd3', 'base')
    c.tracking.credentials_file = str(tmp_path / 'absent')
    run = OnlineRun()
    calls = []
    def save(path, **kwargs):
        if Path(path).name.startswith('overflow-'):
            calls.append(Path(path).name)
            if Path(path).suffix == '.json': raise OSError('upload unavailable')
    run.save = save
    sdk = SimpleNamespace(init=lambda **kw: run, Settings=lambda **kw: kw, Artifact=StoredArtifact)
    with patch.dict(sys.modules, {'wandb': sdk}), patch.dict(os.environ, {
        'WANDB_API_KEY': 'test-key', 'WANDB_ENTITY': 'owner', 'WANDB_PROJECT': 'project'}):
        tracker = OnlineTracker(c, tmp_path)
        (tmp_path / 'overflow-base-critic.pt').write_bytes(b'diagnostic snapshot')
        (tmp_path / 'overflow-base-critic.json').write_text('{}')
        tracker.learner({'critic_loss': 1.0}, 1)
        tracker.learner({'critic_loss': 0.5}, 2)
        tracker.close()
    assert calls.count('overflow-base-critic.pt') == 1
    assert calls.count('overflow-base-critic.json') == 1


@pytest.mark.parametrize('value,kind', [(float('inf'), 'inf'), (float('-inf'), '-inf'), (float('nan'), 'NaN')])
def test_learner_telemetry_is_idempotent_and_keeps_original_nonfinite_labels(value, kind):
    from uav_network.experiments.evidence import learner_telemetry
    once = learner_telemetry({'critic_grad_norm': value, 'critic_loss': 2.0})
    twice = learner_telemetry(once)
    assert twice == once
    assert twice['critic_grad_norm_nonfinite_kind'] == kind
    assert set(twice) == {'critic_grad_norm', 'critic_loss', 'critic_grad_norm_nonfinite', 'critic_grad_norm_nonfinite_kind'}
