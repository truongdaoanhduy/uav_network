"""CPU-only, research hypothesis test for non-motion yaw action, NOT an RL benchmark.

Same deterministic position-control policy / seed and fixed communication policy.
When --counter-keyed is active, sensor-noise draws are identical for the same
(seed, episode, UAV, cell, tick) even when camera footprints differ.
Never treat as Warp or Q1-paper-validated camera dynamics.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import yaml
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml'


def _sector_policy(env, t):
    pos=env.positions[:env.actor_n,:2]
    n=env.actor_n
    phase=(t//120)%2
    centers=60.0+(np.arange(n)+0.25+0.5*phase)*(880.0/n)
    y_goal=np.where((np.arange(n)+(t//60)%2)%2==0,900.,100.)
    goal=np.stack((centers,y_goal),axis=1)
    diff=goal-pos
    dist=np.maximum(np.linalg.norm(diff,axis=1,keepdims=True),1.)
    return np.clip(diff/dist,-1.,1.)*env.cfg.v_max_mps


def _one(seed,steps,arm,run_file,*,counter_keyed=False,cooldown_steps=0):
    cfg_dict=yaml.safe_load(SRC.read_text())
    cfg_dict['enable_gimbal_azimuth_action']=(arm!='nadir_canonical')
    cfg_dict['sensor_rng_mode'] = 'counter_keyed' if counter_keyed else 'sequential'
    cfg_dict['sensor_revisit_cooldown_steps'] = int(cooldown_steps)
    cfg=EnvConfigV54(**cfg_dict)
    env=ZhangXue7NodePacketEnv(cfg)
    env.reset(seed=seed)
    output=None
    done_at=steps
    # Belief tracker works on local observations, not ground-truth target positions.
    per_step=[]
    for step in range(steps):
        vel=_sector_policy(env,step)
        comm=np.ones(env.actor_n,dtype=np.int32)
        if arm=='nadir_canonical':
            res=env.step(vel,comm)
        elif arm=='fixed_east_camera':
            res=env.step(vel,comm,gimbal_azimuth_cmd=np.full(env.actor_n,0.5))
        elif arm=='time_sweep_camera':
            direction=0.5 if (step//15)%2==0 else -0.5
            res=env.step(vel,comm,gimbal_azimuth_cmd=np.full(env.actor_n,direction))
        else:
            raise ValueError(arm)
        _,_,_,reward,term,trunc,info=res
        rec={'seed':seed,'arm':arm,'step':step+1,'reward':float(reward),
             'gcs_tp':int(info['gcs_tp_confirmed']),
             'gcs_fp_ever':int(info['gcs_fp_ever_confirmed']),
             'delivered_coverage_pct':float(info['delivered_cell_coverage_pct']),
             'uav_xy':np.round(env.positions[:env.actor_n,:2],2).tolist(),
             'gimbal_yaw_rad':np.round(env.gimbal_azimuth_rad,4).tolist(),
             'terminated':bool(term),'truncated':bool(trunc)}
        run_file.write(json.dumps(rec,sort_keys=True)+'\n')
        per_step.append(rec)
        output=info
        if term or trunc:
            done_at=step+1
            break
    assert output is not None
    return {'seed':seed,'arm':arm,'steps':done_at,
            'gcs_tp':int(output['gcs_tp_confirmed']),
            'gcs_fp_ever':int(output['gcs_fp_ever_confirmed']),
            'delivered_coverage_pct':float(output['delivered_cell_coverage_pct']),
            'mean_reward_per_step':round(float(sum(x['reward'] for x in per_step)/len(per_step)),5),
            'min_battery_pct':round(float(np.min(env.battery_j)/env.batt_cfg.e_max_j*100),4),
            'final_uav_xy':np.round(env.positions[:env.actor_n,:2],2).tolist(),
            'canonical_unchanged':arm=='nadir_canonical'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--seeds',type=int,nargs='+',default=[44,45,46])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--counter-keyed', action='store_true', help='CRN by (seed, episode, t, UAV, cell); research CPU only')
    p.add_argument('--cooldown-steps', type=int, default=0)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    arms=['nadir_canonical','fixed_east_camera','time_sweep_camera']
    outcomes=[]
    t0=time.monotonic()
    with a.output.with_suffix('.jsonl').open('w') as log:
        for seed in a.seeds:
            for arm in arms:
                result=_one(seed,a.steps,arm,log,counter_keyed=a.counter_keyed,cooldown_steps=a.cooldown_steps)
                outcomes.append(result)
                print('completed',seed,arm, 'TP',result['gcs_tp'],'FP',result['gcs_fp_ever'],'Cov',result['delivered_coverage_pct'],flush=True)
    by_arm={name:{key:float(np.mean([x[key] for x in outcomes if x['arm']==name]))
                  for key in ['gcs_tp','gcs_fp_ever','delivered_coverage_pct','mean_reward_per_step','min_battery_pct']}
            for name in arms}
    # Bootstrap over master seeds ONLY. Repeated time steps and UAVs are not
    # independent experimental units; n=5 yields broad preliminary CI.
    rng = np.random.default_rng(1242026)
    paired={}
    for other in arms[1:]:
        deltas = {}
        for field in ['gcs_tp','gcs_fp_ever','delivered_coverage_pct','mean_reward_per_step']:
            diffs = np.array([
                next(x[field] for x in outcomes if x['seed']==s and x['arm']==other) -
                next(x[field] for x in outcomes if x['seed']==s and x['arm']==arms[0])
                for s in a.seeds
            ],dtype=np.float64)
            boots=rng.choice(diffs,size=(5000,len(diffs)),replace=True).mean(axis=1)
            deltas[field]={'mean_delta':float(diffs.mean()),'per_seed':diffs.tolist(),
                           'bootstrap_ci95':np.quantile(boots,[0.025,0.975]).tolist()}
        paired[other]=deltas
    obj={'source_yaml':str(SRC.relative_to(ROOT)),'experimental_action':'azimuth only; fixed tilt 25deg half angle25deg, rate 45deg/s',
         'sensing_pd_pf':'unchanged 0.85/0.03; NO oblique-camera calibration',
         'seeds':a.seeds,'sensor_rng_mode':'counter_keyed' if a.counter_keyed else 'sequential',
         'sensor_revisit_cooldown_steps':a.cooldown_steps,
         'paired_delta_vs_nadir':paired,'max_steps':a.steps,'runtime_s':round(time.monotonic()-t0,3),
         'arms':arms,'outcomes':outcomes,'group_mean':by_arm,
         'limitations':['CRN counter keyed only when --counter-keyed; otherwise sensor stream is not paired',
                        'Not proof of learned-policy gain versus deterministic rule',
                        'No full FoV pixel resolution/detector calibration or GPU parity',
                        'Small 3-seed exploratory pilot, not inferential statistical evidence']}
    a.output.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'means':by_arm,'runtime_s':obj['runtime_s']},indent=2))

if __name__=='__main__': main()
