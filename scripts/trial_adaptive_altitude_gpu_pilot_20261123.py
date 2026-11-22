"""Strictly bounded real MATD3 training pilot for adaptive-altitude NEW MDP.

Runs the actual Warp collector, replay and learner. Never 20k episodes; the
hard environment-step cap is enforced by the training runner itself. This is
NOT a GPU proof until users run it on T4x2 and provide output artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from uav_network.configuration import load_config
from uav_network.environment.observations import observation_version
from uav_network.experiments.training import train

CONFIG='configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml'


def make_config(*,device,output,envs,steps,batch,updates,horizon,learning_starts):
    if (device not in ('cpu','cuda') or min(envs,steps,batch,updates,horizon)<1
            or learning_starts<0 or steps*envs <= learning_starts):
        raise ValueError('Pilot requires a positive bounded budget with real optimizer updates')
    c=load_config(CONFIG)
    c.runtime.device=device
    c.runtime.num_envs=envs
    c.runtime.episodes=20_000  # The hard step cap stops well before this.
    c.runtime.output_dir=str(output)
    c.runtime.max_train_seconds=0
    c.runtime.replay_device='cpu'
    c.matd3.batch_size=batch
    c.matd3.updates_per_step=updates
    c.matd3.learning_starts=learning_starts
    c.world.max_steps=horizon
    if device=='cpu':
        # Local CPU testing only: avoid allocating 1M x large transition arrays.
        c.matd3.replay_capacity=max(512,4*batch)
    # Real GPU pilot retains original R1M, replay 75% CPU, 25% CUDA.
    c.tracking.enabled=False
    c.tracking.deferred_gpu_eval_enabled=False
    c.tracking.online_gpu_eval_enabled=False
    c.tracking.evaluation_interval=0
    c.tracking.evaluation_interval_steps=0
    c.tracking.periodic_evaluation_interval_episodes=0
    c.tracking.resumable_checkpoint_interval_seconds=0
    c.validate()
    assert c.world.num_uavs==6 and c.world.num_targets==35 and c.world.num_obstacles==25
    assert c.world.altitude_max_m==150
    assert c.sensor.report_cell_hypotheses
    assert c.sensor.confirmation_mode=='likelihood_ratio'
    assert c.sensor.confirmation_threshold==.99
    assert c.sensor.confirmation_min_likelihood_ratio==4.0
    assert c.matd3.hidden_sizes==[256,512]
    assert c.runtime.replay_cpu_fraction==.75
    assert observation_version(c)==10
    assert c.runtime.deterministic
    return c


def science_sha256(config):
    d=config.to_dict()
    d['runtime'].pop('output_dir',None)
    d['tracking'].pop('run_name',None)
    d['tracking'].pop('group',None)
    return hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest()


def run(*,device,output,envs,steps,batch,updates,horizon,learning_starts):
    c=make_config(device=device,output=output,envs=envs,steps=steps,batch=batch,
                  updates=updates,horizon=horizon,learning_starts=learning_starts)
    if device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()<2):
        raise RuntimeError('Kaggle T4x2 required; cannot silently run on CPU')
    output=Path(output)
    if output.exists():raise FileExistsError('Refuse to overwrite any existing run '+str(output))
    # Scope pilot-only controls to the train() call. Without restoring the
    # environment, a later independent training job inherits the 64-step cap!
    # Even tests importing this function must not mutate the parent process.
    controls={'UAV_TRAIN_ENV_STEP_LIMIT':str(envs*steps),
              'UAV_TRAINING_ONLY':'true','UAV_DIAGNOSTIC_STORE':'off',
              'UAV_FORENSIC_ENABLED':'false','UAV_DEFERRED_EXPORT_HF':'false',
              'UAV_FAST_TRAINING':'false','UAV_DIRECT_GPU_COLLECTOR_COPY':'false',
              'WANDB_MODE':'disabled',
              'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    print('ADAPTIVE_ALTITUDE_TRAIN_START',json.dumps(dict(device=device,envs=envs,
          steps=steps,batch=batch,updates=updates,horizon=horizon,
          science_sha256=science_sha256(c),confirmation_mode=c.sensor.confirmation_mode)),flush=True)
    wall=time.perf_counter()
    from unittest.mock import patch
    with patch.dict(os.environ,controls):
        summary=train(c)
    elapsed=time.perf_counter()-wall
    if (summary['stop_reason']!='env_step_budget'
            or summary['global_step']!=envs*steps
            or summary['requested_episodes']!=20_000
            or summary['episode_budget_complete']):
        raise RuntimeError('Adaptive pilot training did not stop at hard capped environment budget')
    artifact=torch.load(summary['checkpoint'],map_location='cpu',weights_only=True)
    if artifact['observation_version']!=10:
        raise RuntimeError('Adaptive checkpoint schema mismatch')
    digest=hashlib.sha256()
    for name,tensor in sorted(artifact['model'].items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    record={'kind':'adaptive_altitude_real_matd3_bounded_no_20k',
            'science_sha256':science_sha256(c),'device':device,
            'envs':envs,'vector_steps':steps,'transitions':summary['global_step'],
            'horizon':horizon,'batch':batch,'updates_per_vector_step':updates,
            'actual_learner_updates':summary['updates'],'completed_episodes':summary['episodes'],
            'wall_seconds':elapsed,'train_core_seconds':summary['elapsed_s'],
            'joint_transitions_per_second':summary['joint_transitions_per_s'],
            'collector_wait_s':summary['collection_wait_s'],
            'learner_s':summary['learner_s'],
            'replay_sample_host_s':summary['replay_sample_host_s'],
            'optimizer_update_host_s':summary['optimizer_update_host_s'],
            'actor_model_sha256':digest.hexdigest(),
            'observation_version':artifact['observation_version'],
            'checkpoint_path':str(summary['checkpoint']),
            'stop_reason':summary['stop_reason'],
            'completed_20k':False,'episode_metrics_informative':summary['episodes']>0,
            'gpu_science_convergence_unverified':True}
    (output/'adaptive_pilot_report.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    print('ADAPTIVE_ALTITUDE_GPU_PILOT_JSON '+json.dumps(record,allow_nan=False),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--envs',type=int,default=4096)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--batch',type=int,default=8192)
    parser.add_argument('--updates',type=int,default=4)
    parser.add_argument('--horizon',type=int,default=1500)
    parser.add_argument('--learning-starts',type=int,default=131072)
    args=parser.parse_args()
    run(**vars(args))


if __name__=='__main__':main()
