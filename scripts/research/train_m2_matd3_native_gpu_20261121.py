"""Reproducible, explicitly LIMITED full-M2 pilot: Native UavNetSim env + CUDA MATD3.

Not full Warp E4096. The GPU hosts the neural learner; each mission is the
packet-accurate Native M2 model with target-only 2D F/E/U belief and 3D flight.
Train MATD3 first. MASAC remains a separate follow-up; do not mix runs.

For the intended 1000m pilot: 6 UAV, 35 targets, 25 physical obstacles,
1500-step missions, 8 observations/UAV/step physical sensor budget. Updates
use B128 once per 8 new transitions, keeping 16 sampled transitions per
collected transition: 128/8=16 (cf. E4096, U8, B8192 = 16).
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from uav_network.configuration import Config
from uav_network.environment.env import Action
from uav_network.environment.m2_mission import CooperativeM2Env
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.replay import ReplayBuffer


@dataclass(frozen=True)
class PilotSettings:
    episodes: int=120
    scan_cells: int=8
    batch_size: int=128
    replay_capacity: int=20_000
    warmup_steps: int=1024
    update_every: int=8
    eval_every: int=10
    eval_episodes: int=2
    max_train_seconds: float=0.0
    seed: int=44
    device: str='cuda:0'
    output_dir: Path=Path('/kaggle/working/m2_matd3_native_gpu_pilot')

    def validate(self):
        if (self.episodes<1 or self.scan_cells<1 or self.batch_size<1 or
            self.replay_capacity<self.batch_size or self.warmup_steps<0 or
            self.update_every<1 or self.eval_every<1 or self.eval_episodes<1 or
            self.max_train_seconds<0):
            raise ValueError('Invalid MATD3 pilot runtime settings')
        return self


def make_config(*,seed:int=44,episode_horizon:int=1500):
    if episode_horizon<1:raise ValueError('episode_horizon must be positive')
    c=Config()
    c.world.map_size_m=1000.
    c.sensor.grid_cell_m=25.
    c.world.num_uavs=6;c.world.num_targets=35;c.world.num_obstacles=25
    c.world.motion_dimensions=3;c.world.motion_safety='cbf'
    c.world.max_steps=int(episode_horizon)
    c.sensor.model='bayesian'
    c.sensor.obstacle_occlusion=True
    c.network.peer_range_m=500.;c.network.gcs_range_m=500.
    c.network.report_bytes=1024;c.network.payload_bytes=1024
    c.network.power_control='learned'
    c.network.enabled=True
    c.energy.enabled=True
    c.matd3.hidden_sizes=[256,256]
    c.matd3.share_actor_parameters=True
    c.matd3.share_critic_parameters=True
    c.matd3.actor_layer_norm=True
    c.matd3.output_normalization=False
    c.matd3.initialization='default'
    c.runtime.m2_enabled=True
    c.runtime.environment_backend='native'
    c.runtime.algorithm='matd3'
    c.runtime.seed=int(seed)
    c.runtime.coalesce_idle_polls=True
    c.runtime.torch_threads=2
    c.runtime.amp_mode='off'
    c.runtime.fused_optimizers=False
    c.tracking.enabled=False
    c.validate()
    return c


def require_training_device(device:str):
    requested=torch.device(device)
    if requested.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this training device')
    if requested.type=='cuda' and requested.index is not None and requested.index>=torch.cuda.device_count():
        raise RuntimeError('Requested CUDA device is not present')
    return requested


def _metrics(info,*,episode,seed,score,steps,phase,global_steps,critic_updates,actor_updates,elapsed):
    network=info['communication'] or {}
    selections=network.get('selection_counts') or {}
    g=info['gcs_map']
    values={
        'phase':phase,'episode':int(episode),'seed':int(seed),
        'steps':int(steps),'global_steps':int(global_steps),
        'episode_return':float(score),
        'num_targets':35,'backend':'native_uavnetsim',
        'correct_gcs_delivery_recall':float(info['correct_gcs_delivery_recall']),
        'gcs_verified_correct':int(info['gcs_verified_correct']),
        'gcs_verified_false':int(info['gcs_verified_false']),
        'target_map_auprc':float(g['target_auprc']),
        'target_brier':float(g['target_brier']),
        'mean_uncertainty':float(g['mean_uncertainty']),
        'gcs_coverage':float(g['coverage']),
        'normalized_delivery_auc':float(info['normalized_delivery_auc']),
        'energy_total_j':float(info['energy_total_j']),
        'gcs_evidence_events':int(info['gcs_events']),
        'packet_pdr_percent':float(network.get('packet_pdr_percent',0.) or 0.),
        'radio_received_packets':int(network.get('received_packets',0)),
        'radio_selected':int(selections.get('selected',0)),
        'radio_accepted':int(selections.get('accepted',0)),
        'radio_no_buffer':int(selections.get('no_buffer',0)),
        'radio_receiver_rejected':int(selections.get('receiver_rejected',0)),
        'radio_out_of_range':int(selections.get('out_of_range_or_inactive',0)),
        'critic_updates':int(critic_updates),'actor_updates':int(actor_updates),
        'elapsed_s':float(elapsed),
    }
    return values


def _append_csv(path:Path,record:dict):
    create=not path.exists() or path.stat().st_size==0
    with path.open('a',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=list(record))
        if create:w.writeheader()
        w.writerow(record)
        file.flush()


def _write_json(path:Path,data):
    draft=path.with_suffix(path.suffix+'.tmp')
    draft.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    draft.replace(path)


_M2_CHECKPOINT_FORMAT='m2-target-belief-2d-matd3-v1'
_M2_OBSERVATION_SCHEMA='m2-local-5x5-f-e-u-verify-v1'


def _snapshot_policy(path:Path,agent:MATD3,meta):
    # Evaluation-only M2 checkpoint. Do NOT route through legacy MATD3.load:
    # the legacy saved observation ABI uses an entirely different M2-free map.
    if not agent.config.runtime.m2_enabled or agent.continuous_dim!=5:
        raise ValueError('Only opt-in M2 MATD3 may use this checkpoint format')
    draft=path.with_suffix(path.suffix+'.tmp')
    data={'format':_M2_CHECKPOINT_FORMAT,
          'observation_schema':_M2_OBSERVATION_SCHEMA,
          'algorithm':'matd3','metadata':meta,
          'config':agent.config.to_dict(),
          'model_state_dict':agent.state_dict(),
          'observation_dim':agent.observation_dim,'state_dim':agent.state_dim,
          'continuous_dim':agent.continuous_dim,
          'critic_updates':agent.updates,'actor_updates':agent.actor_updates,
          'resume_exactly_supported':False}
    torch.save(data,draft)
    draft.replace(path)


def load_m2_policy_checkpoint(path,*,device='cpu') -> MATD3:
    from uav_network.configuration import _apply
    payload=torch.load(path,map_location=device,weights_only=True)
    if (payload.get('format')!=_M2_CHECKPOINT_FORMAT or
        payload.get('observation_schema')!=_M2_OBSERVATION_SCHEMA or
        payload.get('algorithm')!='matd3'):
        raise ValueError('Not an M2 Target Belief 2D MATD3 policy checkpoint')
    c=Config()
    _apply(c,payload['config'])
    c.validate()
    if not c.runtime.m2_enabled or c.continuous_dim!=5:
        raise ValueError('Incorrect M2 continuous action schema in checkpoint')
    expected_observation=17+25*9+c.world.num_uavs+(c.world.num_uavs+1)
    if (payload['observation_dim']!=expected_observation or
        payload['state_dim']!=expected_observation*c.world.num_uavs):
        raise ValueError('M2 5x5 observation/critic schema mismatch')
    agent=MATD3(c,payload['observation_dim'],payload['state_dim'],device=device)
    agent.load_state_dict(payload['model_state_dict'])
    agent.updates=int(payload['critic_updates'])
    agent.actor_updates=int(payload['actor_updates'])
    return agent


def _plot_progress(output_dir:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    train_path=output_dir/'train_episodes.csv'; eval_path=output_dir/'evaluation.csv'
    if not train_path.exists() or not eval_path.exists():return
    tr=pd.read_csv(train_path);ev=pd.read_csv(eval_path)
    fig,axes=plt.subplots(3,1,figsize=(10,11),sharex=True,constrained_layout=True)
    for ax,train_key,eval_key,lab in (
        (axes[0],'episode_return','mean_return','Return'),
        (axes[1],'correct_gcs_delivery_recall','mean_delivery_recall','GCS verified delivery recall'),
        (axes[2],'gcs_coverage','mean_coverage','GCS target-map coverage')):
        ax.plot(tr['global_steps'],tr[train_key],alpha=.5,label='train episodes')
        ax.plot(ev['global_steps'],ev[eval_key],marker='o',linewidth=2,label='fixed-seed evaluation')
        ax.set_ylabel(lab);ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xlabel('Environment transitions — 1 Native M2 world')
    fig.suptitle('MATD3 on Native UavNetSim M2 (GPU learner) — NOT full Warp E4096')
    fig.savefig(output_dir/'matd3_pilot_learning_curves.png',dpi=130)
    plt.close(fig)


def evaluate_policy(config,agent:MATD3,*,seeds,scan_cells:int):
    results=[]
    for seed in seeds:
        env=CooperativeM2Env(config,correlation_strength=.55,
                             max_scan_cells=scan_cells,map_buffer_bytes=65536)
        start=time.perf_counter()
        try:
            observation,_=env.reset(seed=int(seed))
            done=False;score=0.;info=None
            while not done:
                action=agent.act(observation,deterministic=True)
                observation,reward,done,_,info=env.step(action)
                score+=reward
                if env.steps%300==0:
                    print('EVAL_PROGRESS',json.dumps({'seed':seed,'step':env.steps,
                        'gcs_correct':info['gcs_verified_correct'],
                        'gcs_coverage':info['gcs_map']['coverage']}),flush=True)
            results.append(_metrics(info,episode=-1,seed=seed,score=score,
                                    steps=env.steps,phase='evaluation',global_steps=0,
                                    critic_updates=agent.updates,actor_updates=agent.actor_updates,
                                    elapsed=time.perf_counter()-start))
        finally:env.close()
    return results


def train(config:Config,pilot:PilotSettings):
    pilot.validate()
    cfg=config
    cfg.validate()
    if (not cfg.runtime.m2_enabled or cfg.runtime.environment_backend!='native'
        or cfg.continuous_dim!=5 or cfg.world.map_size_m!=1000
        or cfg.world.num_uavs!=6 or cfg.world.num_targets!=35
        or cfg.world.num_obstacles!=25):
        raise ValueError('Full 1000m target-only M2 Native configuration is required; do not train legacy Warp')
    device=require_training_device(pilot.device)
    # Prevent accidental oversubscription of Kaggle CPU workers.
    torch.set_num_threads(min(4,max(1,cfg.runtime.torch_threads)))
    np.random.seed(pilot.seed);random.seed(pilot.seed);torch.manual_seed(pilot.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(pilot.seed)
    rng=np.random.default_rng(pilot.seed)
    output_dir=Path(pilot.output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    # Avoid loading an already existing run and silently appending rows.
    if (output_dir/'train_episodes.csv').exists():
        raise FileExistsError('Output directory has train_episodes.csv; choose an empty run directory')
    probe=CooperativeM2Env(cfg,correlation_strength=.55,
                           max_scan_cells=pilot.scan_cells,map_buffer_bytes=65536)
    try:
        obs,_=probe.reset(seed=pilot.seed)
        obs_shape=obs.shape;state_dim=probe.state().size
        agent=MATD3(probe.config,obs.shape[-1],state_dim,device=device)
    finally:probe.close()
    replay=ReplayBuffer(pilot.replay_capacity,obs_shape,state_dim,
                        seed=pilot.seed,continuous_dim=5)
    summary={'algorithm':'MATD3','environment_backend':'native_uavnetsim',
             'warp_full_m2_environment':False,'gpu_learner':device.type=='cuda',
             'device':str(device),'num_envs':1,'world_size_m':1000,
             'episode_horizon':cfg.world.max_steps,
             'num_uavs':6,'num_targets':35,'num_obstacles':25,
             'sensor_cells_per_uav_step':pilot.scan_cells,
             'learner_matd3_hidden':[256,256],
             'replay_sample_ratio':pilot.batch_size/pilot.update_every,
             'science_warning':'This single-world GPU-learner pilot is NOT full Warp E4096 and NOT multi-seed convergence',
             'settings':{key:(str(val) if isinstance(val,Path) else val)
                         for key,val in asdict(pilot).items()},
             'episodes_completed':0,'environment_transitions':0,
             'learner_critic_updates':0,'learner_actor_updates':0,'evaluation_count':0}
    started=time.perf_counter()
    _write_json(output_dir/'run_manifest.json',summary)

    def evaluate_at(ep:int,global_steps:int):
        values=evaluate_policy(cfg,agent,
                               seeds=[2000000044+i for i in range(pilot.eval_episodes)],
                               scan_cells=pilot.scan_cells)
        result={'episode':ep,'global_steps':global_steps,
                'mean_return':float(np.mean([x['episode_return'] for x in values])),
                'mean_delivery_recall':float(np.mean([x['correct_gcs_delivery_recall'] for x in values])),
                'mean_coverage':float(np.mean([x['gcs_coverage'] for x in values])),
                'mean_target_auprc':float(np.mean([x['target_map_auprc'] for x in values])),
                'mean_target_brier':float(np.mean([x['target_brier'] for x in values])),
                'mean_false_reports':float(np.mean([x['gcs_verified_false'] for x in values])),
                'mean_delivery_auc':float(np.mean([x['normalized_delivery_auc'] for x in values])),
                'mean_uncertainty':float(np.mean([x['mean_uncertainty'] for x in values])),
                'mean_energy_j':float(np.mean([x['energy_total_j'] for x in values])),
                'eval_seeds':';'.join(str(x['seed']) for x in values),
                'critic_updates':agent.updates,'actor_updates':agent.actor_updates}
        _append_csv(output_dir/'evaluation.csv',result)
        summary['evaluation_count']+=1
        print('EVAL',json.dumps(result),flush=True)
        return result

    evaluate_at(0,0)
    steps=0
    last_update_metrics={}
    for episode in range(1,pilot.episodes+1):
        if pilot.max_train_seconds and time.perf_counter()-started>=pilot.max_train_seconds:
            summary['stopped_by_time_budget']=True
            break
        env=CooperativeM2Env(cfg,correlation_strength=.55,
                             max_scan_cells=pilot.scan_cells,map_buffer_bytes=65536)
        ep_start=time.perf_counter();score=0.
        action_count=0;motion_saturated=0;sum_z=0.;sum_kind=0.;sum_power=0.;silent_count=0;gcs_count=0
        try:
            observation,info=env.reset(seed=pilot.seed+episode-1)
            done=False
            while not done:
                state=env.state()
                if steps<pilot.warmup_steps:
                    action=Action(rng.uniform(-1,1,(6,5)).astype(np.float32),
                                  rng.integers(0,7,6,dtype=np.int64))
                else:
                    action=agent.act(observation,deterministic=False)
                a_cont=np.asarray(action.continuous)
                a_dest=np.asarray(action.destination)
                action_count+=len(a_dest)
                motion_saturated+=int((np.abs(a_cont[:,:3])>=.95).sum())
                sum_z+=float(a_cont[:,2].sum())
                sum_kind+=float(a_cont[:,3].sum())
                sum_power+=float(a_cont[:,4].sum())
                silent_count+=int((a_dest==0).sum())
                gcs_count+=int((a_dest==6).sum())
                next_obs,reward,done,_,info=env.step(action)
                next_state=env.state()
                replay.add(observation,state,action,reward,next_obs,next_state,done)
                steps+=1
                if (steps>=pilot.warmup_steps and
                    steps%pilot.update_every==0 and len(replay)>=pilot.batch_size):
                    last_update_metrics=agent.update(replay.sample(pilot.batch_size),collect_metrics=True)
                    for name,value in last_update_metrics.items():
                        if isinstance(value,(int,float,np.number)) and not math.isfinite(float(value)):
                            raise FloatingPointError(f'MATD3 learner returned non-finite {name}: {value}')
                score+=float(reward)
                observation=next_obs
                if env.steps%300==0:
                    print('TRAIN_PROGRESS',json.dumps({'episode':episode,
                        'step':env.steps,'global_steps':steps,
                        'gcs_correct':info['gcs_verified_correct'],
                        'gcs_coverage':info['gcs_map']['coverage'],
                        'actor_updates':agent.actor_updates,
                        'critic_updates':agent.updates}),flush=True)
        finally:env.close()
        row=_metrics(info,episode=episode,seed=pilot.seed+episode-1,
                     score=score,steps=env.steps,phase='train',global_steps=steps,
                     critic_updates=agent.updates,actor_updates=agent.actor_updates,
                     elapsed=time.perf_counter()-ep_start)
        row['motion_saturation_rate']=motion_saturated/max(1,3*action_count)
        row['action_z_mean']=sum_z/max(1,action_count)
        row['packet_type_action_mean']=sum_kind/max(1,action_count)
        row['tx_power_action_mean']=sum_power/max(1,action_count)
        row['silent_action_fraction']=silent_count/max(1,action_count)
        row['gcs_destination_fraction']=gcs_count/max(1,action_count)
        row['critic_loss']=float(last_update_metrics.get('critic_loss',float('nan')))
        row['actor_loss']=float(last_update_metrics.get('actor_loss',float('nan')))
        row['actor_grad_norm']=float(last_update_metrics.get('actor_grad_norm',float('nan')))
        _append_csv(output_dir/'train_episodes.csv',row)
        summary['episodes_completed']=episode
        summary['environment_transitions']=steps
        summary['learner_critic_updates']=agent.updates
        summary['learner_actor_updates']=agent.actor_updates
        _write_json(output_dir/'run_manifest.json',summary)
        print('TRAIN',json.dumps({key:row[key] for key in (
            'episode','global_steps','episode_return','gcs_verified_correct',
            'correct_gcs_delivery_recall','gcs_coverage','mean_uncertainty',
            'critic_updates','actor_updates','elapsed_s')}),flush=True)
        if episode%pilot.eval_every==0 or episode==pilot.episodes:
            evaluate_at(episode,steps)
            _snapshot_policy(output_dir/f'policy_ep{episode:04d}.pt',agent,summary)
            _plot_progress(output_dir)
    _snapshot_policy(output_dir/'final_policy.pt',agent,summary)
    summary['elapsed_s']=time.perf_counter()-started
    _write_json(output_dir/'run_manifest.json',summary)
    print('M2_MATD3_NATIVE_GPU_PILOT_COMPLETE',json.dumps(summary),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description='Native M2 + GPU MATD3 pilot, not full Warp E4096')
    ap.add_argument('--episodes',type=int,default=120)
    ap.add_argument('--steps',type=int,default=1500)
    ap.add_argument('--eval-every',type=int,default=10)
    ap.add_argument('--eval-episodes',type=int,default=2)
    ap.add_argument('--scan-cells',type=int,default=8)
    ap.add_argument('--max-hours',type=float,default=0.0)
    ap.add_argument('--warmup-steps',type=int,default=1024)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--update-every',type=int,default=8)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--output-dir',type=Path,default=Path('/kaggle/working/m2_matd3_native_gpu_pilot'))
    args=ap.parse_args()
    pilot=PilotSettings(episodes=args.episodes,scan_cells=args.scan_cells,
        batch_size=args.batch_size,warmup_steps=args.warmup_steps,
        update_every=args.update_every,eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,max_train_seconds=args.max_hours*3600,
        seed=44,device=args.device,output_dir=args.output_dir)
    train(make_config(seed=44,episode_horizon=args.steps),pilot)

if __name__=='__main__':main()
