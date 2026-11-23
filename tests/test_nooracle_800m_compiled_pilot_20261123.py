"""Bounded MATD3 train must select 800m orthogonal Inductor configuration."""
from pathlib import Path
import os
import pytest

from uav_network.configuration import load_config
from scripts.trial_nooracle_map800_orthogonal_compile_gpu_pilot_20261123 import make_config,run
from uav_network.environment.observations import observation_version


def test_pilot_keep_real_cuda_training_science_and_h256x256(tmp_path):
    c=make_config(device='cuda',output=tmp_path/'out',envs=4096,steps=64,batch=8192,
                  updates=4,horizon=1500,learning_starts=131072)
    assert c.runtime.compile_models is True
    assert c.world.map_size_m==800 and c.sensor.grid_cell_m==20
    assert c.matd3.initialization=='orthogonal'
    assert c.matd3.hidden_sizes==[256,256]
    assert c.runtime.replay_cpu_fraction==.75
    assert c.matd3.replay_capacity==1_000_000
    assert c.world.altitude_max_m==150 and c.sensor.confirmation_threshold==.99
    assert c.network.tx_power_max_w==.4
    assert c.runtime.num_envs==4096 and c.matd3.batch_size==8192
    assert observation_version(c)==10
    assert c.tracking.enabled is False and c.tracking.deferred_gpu_eval_enabled is False


def test_small_cpu_compiled_training_checkpoint_and_new_mdp(tmp_path,monkeypatch):
    import torch
    torch.set_num_threads(1)
    monkeypatch.setenv('TORCHINDUCTOR_COMPILE_THREADS','1')
    result=run(device='cpu',output=tmp_path/'pilot',envs=4,steps=16,
               batch=16,updates=1,horizon=12,learning_starts=16)
    assert result['transitions']==64
    assert result['actual_learner_updates']>0
    assert result['completed_episodes']==4
    assert result['stop_reason']=='env_step_budget'
    assert result['observation_version']==10
    assert result['completed_20k'] is False
    assert result['model_architecture']==[256,256]
    assert result['compile_models'] is True
    assert result['initialization']=='orthogonal'
    assert (tmp_path/'pilot'/'adaptive_pilot_report.json').is_file()
    # Real full-flow handoff: compiled MATD3 training -> checkpoint ->
    # held-out CPU evaluator with the exact new 800m/20m/orthogonal science.
    from uav_network.experiments.evaluation import evaluate
    from uav_network.configuration import load_config
    trained=__import__('torch').load(result['checkpoint_path'],map_location='cpu',weights_only=True)
    assert trained['observation_version']==10
    assert trained['config']['world']['map_size_m']==800
    assert trained['config']['matd3']['initialization']=='orthogonal'
    assert trained['config']['runtime']['compile_models'] is True
    evaluation=evaluate(result['checkpoint_path'],episodes=2,seed=2_000_000_044,
                        output_dir=tmp_path/'eval')
    assert evaluation['episodes']==2
    assert __import__('numpy').isfinite(evaluation['mean_return'])
    assert (tmp_path/'eval'/'metrics.json').exists()
