"""Pilot uses NEW MDP, real learner, strictly bounded GPU step budget."""
from scripts.trial_adaptive_altitude_gpu_pilot_20261123 import make_config
from uav_network.environment.observations import observation_version


def test_adaptive_pilot_gpu_config_uses_original_h256x512_training_dimensions(tmp_path):
    c=make_config(device='cuda', output=tmp_path/'training',envs=4096,steps=64,
                  batch=8192,updates=4,horizon=1500,learning_starts=131072)
    assert c.runtime.num_envs==4096
    assert c.matd3.hidden_sizes==[256,512]
    assert c.matd3.batch_size==8192
    assert c.matd3.updates_per_step==4
    assert c.matd3.replay_capacity==1000000
    assert c.runtime.replay_cpu_fraction==.75
    assert c.world.altitude_max_m==150
    assert c.sensor.confirmation_mode=='likelihood_ratio'
    assert c.sensor.confirmation_threshold==.99
    assert observation_version(c)==9
    assert c.tracking.enabled is False


def test_adaptive_cpu_bounded_actual_train_from_worker_and_mdp_hash(tmp_path,monkeypatch):
    from scripts.trial_adaptive_altitude_gpu_pilot_20261123 import run
    import torch
    monkeypatch.setenv('UAV_DIAGNOSTIC_STORE','off')
    torch.set_num_threads(1)
    data=run(device='cpu',output=tmp_path/'pilot',envs=4,steps=16,
             batch=16,updates=1,horizon=12,learning_starts=16)
    assert data['kind']=='adaptive_altitude_real_matd3_bounded_no_20k'
    assert data['observation_version']==9
    assert data['actual_learner_updates']>0
    assert data['completed_episodes']==4
    assert data['stop_reason']=='env_step_budget'
    assert data['transitions']==64
    assert data['completed_20k'] is False
    assert (tmp_path/'pilot'/'adaptive_pilot_report.json').exists()


def test_bounded_worker_restores_process_environment_after_training(tmp_path):
    """In-process testing must not cap unrelated future train/eval tests."""
    import os
    from scripts.trial_adaptive_altitude_gpu_pilot_20261123 import run
    keys=('UAV_TRAIN_ENV_STEP_LIMIT','UAV_TRAINING_ONLY','UAV_DIAGNOSTIC_STORE',
          'UAV_FORENSIC_ENABLED','UAV_DEFERRED_EXPORT_HF','UAV_FAST_TRAINING',
          'UAV_DIRECT_GPU_COLLECTOR_COPY','WANDB_MODE','OMP_NUM_THREADS',
          'MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
    before={key:os.environ.get(key) for key in keys}
    run(device='cpu',output=tmp_path/'scoped',envs=4,steps=16,
        batch=16,updates=1,horizon=12,learning_starts=16)
    assert {key:os.environ.get(key) for key in keys}==before
