#!/usr/bin/env python3
"""V112 CUDA all-step safety/overflow adversarial stress. No RL, no training.

Controls are valid finite inputs in [-1,1]^2 and communication 0..2.
The V99 sticky monitor records *every* step before device selective reset.
Failure != policy invalid (invalid actions are not injected by design).
Success != proof safety for all continuous policies or trajectories.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/zhang_xue/src'),str(ROOT/'scripts/gpu')]
from v99_endurance_sticky_cuda_gate import sticky_monitor, query_mib

@wp.kernel
def generate_adverse_actions(a:wp.array3d(dtype=wp.float32), d:wp.array2d(dtype=wp.int32),
                             mode:int, tick:int, seed:int):
    e,u=wp.tid()
    if mode==0: # Push toward NE outer boundary for all six vehicles
        a[e,u,0]=1.0
        a[e,u,1]=1.0
        d[e,u]=1
    elif mode==1: # alternate max commands and application destination
        sign=1.0 if (tick%2)==0 else -1.0
        a[e,u,0]=sign
        a[e,u,1]=-sign
        d[e,u]=wp.int32((tick+u)%3)
    else: # independent per-(world,uav,time) hashed RNG
        rng=wp.rand_init(seed+tick*110351, e*6+u)
        a[e,u,0]=2.0*wp.randf(rng)-1.0
        a[e,u,1]=2.0*wp.randf(rng)-1.0
        k=wp.randf(rng)
        if k<0.333333:
            d[e,u]=0
        elif k<0.666667:
            d[e,u]=1
        else:
            d[e,u]=2

def run(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument('--scenario',type=Path,default=ROOT/'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml')
    ap.add_argument('--envs',type=int,default=4096)
    ap.add_argument('--steps',type=int,default=900)
    ap.add_argument('--seed',type=int,default=44)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--mode',choices=['boundary_push','command_churn','random_exploration'],default='random_exploration')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params,WarpParallelMissionV80
    p=GPUWorldV80Params.from_yaml(a.scenario,envs=a.envs,device=a.device,seed=a.seed)
    w=WarpParallelMissionV80(p)
    act=wp.zeros((a.envs,6,2),dtype=wp.float32,device=a.device)
    dest=wp.zeros((a.envs,6),dtype=wp.int32,device=a.device)
    sticky={k:wp.zeros(a.envs,dtype=wp.int32,device=a.device) for k in ('failed','overflow','nonfinite','done')}
    sticky['err']=wp.zeros(a.envs,dtype=wp.float32,device=a.device)
    wp.synchronize_device(a.device)
    mem=[query_mib()]
    t0=time.monotonic()
    mode={'boundary_push':0,'command_churn':1,'random_exploration':2}[a.mode]
    for step in range(1,a.steps+1):
        wp.launch(generate_adverse_actions,dim=(a.envs,6),inputs=[act,dest,mode,step,a.seed],device=a.device)
        w.step(act,dest)
        wp.launch(sticky_monitor,dim=a.envs,inputs=[
            w.buffers['positions'],w.buffers['battery'],w.buffers['reward'],w.buffers['reward_components'],
            w.buffers['failed'],w.network.data['overflow'],w.buffers['done'],
            sticky['failed'],sticky['overflow'],sticky['nonfinite'],sticky['err'],sticky['done']],device=a.device)
        w.auto_reset_gpu()
        if step%150==0:
            wp.synchronize_device(a.device)
            mem.append(query_mib())
            print('STEP',step,'/',a.steps,'GPU_RAM_MiB',mem[-1],flush=True)
    wp.synchronize_device(a.device)
    d={k:v.numpy() for k,v in sticky.items()}
    res={
        'gate':'V112_C14_ADVERSARIAL_CUDA_STICKY',
        'device':a.device,'mode':a.mode,'seed':a.seed,'envs':a.envs,'steps':a.steps,
        'scenario':a.scenario.name,'scenario_sha256':hashlib.sha256(a.scenario.read_bytes()).hexdigest(),
        'all_step_monitor':True,'episode_handling':'GPU masked auto_reset after sampled transition',
        'ever_safety_failure_worlds':int(np.count_nonzero(d['failed'])),
        'ever_overflow_worlds':int(np.count_nonzero(d['overflow'])),
        'ever_nonfinite_worlds':int(np.count_nonzero(d['nonfinite'])),
        'max_reward_component_sum_error':float(d['err'].max()),
        'terminal_events':int(d['done'].sum()),
        'memory_samples_mib':mem,'peak_vram_mib':max([x for x in mem if x is not None],default=None),
        'world_steps_per_sec':round(a.envs*a.steps/max(1e-12,time.monotonic()-t0),2),
        'scenario_frozen':False,
        'limitations':['Not a proof over all policies','worlds can terminate/reset early','No Bayesian inference or RL policy','No event-level SimPy parity'],
    }
    res['status']='PASS_STRESS_RESET_MODE' if (
        res['ever_safety_failure_worlds']==0 and res['ever_overflow_worlds']==0 and
        res['ever_nonfinite_worlds']==0 and res['max_reward_component_sum_error']<1e-4 and
        res['peak_vram_mib'] is not None) else 'FAIL_STRESS'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(res,indent=2,allow_nan=False)+'\n')
    print(json.dumps(res,indent=2,allow_nan=False),flush=True)
    return 0 if res['status'].startswith('PASS') else 2

if __name__=='__main__':raise SystemExit(run())
