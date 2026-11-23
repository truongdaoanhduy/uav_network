"""TDD: only every100-step Actor grid is validated, not 178 extra candidates.

Diagnosis from public HF checkpoint tar manifest 602a1e82: 76 periodic
snapshots, 178 candidate-only, 254 total -> all 254 were validated. The
FASTGRID switch opts out of EXTRA training-metric candidate Actor snapshots
while preserving every periodic Actor, initial and final, and Eval100/Test1000.
"""
from pathlib import Path
import pytest

from uav_network.configuration import load_config
from uav_network.experiments.training import (
    training_candidate_snapshots_enabled,
    timing_unattributed_host_seconds,
)

CONFIG='configs/matd3-nooracle-posterior-only-map800-h256x256-orthogonal-compile-tx08-hf-fullflow-20261123.yaml'


def test_grid_only_opt_in_never_breaks_legacy_default(monkeypatch):
    c=load_config(CONFIG)
    assert c.tracking.deferred_gpu_eval_keep_all
    monkeypatch.delenv('UAV_PERIODIC_ACTORS_ONLY',raising=False)
    assert training_candidate_snapshots_enabled(c,env_step_limit=0)
    monkeypatch.setenv('UAV_PERIODIC_ACTORS_ONLY','true')
    assert not training_candidate_snapshots_enabled(c,env_step_limit=0)
    monkeypatch.setenv('UAV_PERIODIC_ACTORS_ONLY','1')
    assert not training_candidate_snapshots_enabled(c,env_step_limit=0)
    monkeypatch.setenv('UAV_PERIODIC_ACTORS_ONLY','false')
    assert training_candidate_snapshots_enabled(c,env_step_limit=0)
    assert not training_candidate_snapshots_enabled(c,env_step_limit=4096)
    c.tracking.deferred_gpu_eval_keep_all=False
    assert not training_candidate_snapshots_enabled(c,env_step_limit=0)


def test_new_timing_probe_accounting_does_not_synchronize_cuda():
    assert timing_unattributed_host_seconds(100,15,50,25)==pytest.approx(10)
    assert timing_unattributed_host_seconds(85,15,50,25)==pytest.approx(0)
    assert timing_unattributed_host_seconds(20,0,0,0)==20
    import inspect
    from uav_network.experiments import training
    src=inspect.getsource(training)
    assert 'timing/unattributed_host_seconds' in src
    assert 'torch.cuda.synchronize()' not in inspect.getsource(training.timing_unattributed_host_seconds)


def test_fastgrid_notebook_uses_same_science_and_logs():
    from scripts.build_nooracle_map800_speed_fullflow_ui_20261123 import build_speed_notebook
    from scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 import build_notebook
    old=build_notebook();new=build_speed_notebook()
    assert old.metadata['source_sha256']==new.metadata['source_sha256']
    assert new.metadata['science_config']==old.metadata['science_config']
    assert len(new.cells)==3
    assert len(new.cells[2].source.encode())<825000
    import nbformat
    assert len(nbformat.writes(new).encode())<850000
    assert 'UAV_PERIODIC_ACTORS_ONLY' in new.cells[2].source
    assert "UAV_BATCH_SCALAR_METRICS" in new.cells[2].source
    assert "UAV_FAST_GRAD_CLIP" in new.cells[2].source
    assert "UAV_CUDA_EVENT_PROFILE" in new.cells[2].source
    assert 'UAV_FAST_TRAINING' in new.cells[2].source
    assert 'UAV_FORENSIC_ENABLED' in new.cells[2].source
    assert 'UAV_DIAGNOSTIC_STORE' in new.cells[2].source
    assert 'UAV_WANDB_PLOTS_ONLY' in new.cells[2].source
    assert 'validate_all_snapshots' in new.cells[2].source
    assert 'str(FINAL_TEST_EPISODES // 2)' in new.cells[2].source
    assert 'fastgrid' in new.metadata['speed_profile']


def test_fastgrid_real_warp_cpu_train_retains_periodic_actor_only(tmp_path,monkeypatch):
    """A real MATD3 episode-completion run, not a mocked bank."""
    from uav_network.experiments.training import train
    from uav_network.configuration import load_config
    import json
    c=load_config(CONFIG)
    c.runtime.device='cpu';c.runtime.replay_device='cpu'
    c.runtime.replay_cpu_fraction=.75  # keep the same hybrid 75% CPU layout
    c.runtime.num_envs=2;c.runtime.episodes=6
    c.runtime.compile_models=False;c.runtime.fused_optimizers=False
    c.world.max_steps=4
    c.matd3.hidden_sizes=[32,32]; c.matd3.batch_size=8
    c.matd3.replay_capacity=64;c.matd3.learning_starts=4
    c.matd3.updates_per_step=1
    c.runtime.output_dir=str(tmp_path/'fastgrid')
    c.tracking.enabled=False
    c.tracking.deferred_gpu_eval_enabled=True
    c.tracking.deferred_gpu_eval_keep_all=True
    c.tracking.deferred_gpu_eval_collection_stride=2
    c.validate()
    monkeypatch.setenv('UAV_PERIODIC_ACTORS_ONLY','true')
    monkeypatch.setenv('UAV_BATCH_SCALAR_METRICS','true')
    monkeypatch.setenv('UAV_FAST_GRAD_CLIP','true')
    result=train(c)
    assert result['episodes']==6
    d=json.loads((tmp_path/'fastgrid'/'deferred-gpu-eval'/'manifest.json').read_text())
    assert d['count']>=2 and d['keep_all_distinct_versions']
    assert all('training_metric_best_candidate' not in row['reasons'] for row in d['snapshots'])
    assert 'periodic' in d['snapshots'][0]['reasons']
