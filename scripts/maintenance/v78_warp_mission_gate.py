#!/usr/bin/env python3
"""Honest independent-world validation gate, not proof of CPU model equivalence.

Checks execution of REAL dynamic GPU/CPU Warp kernels for 15 unique targets,
independent world data, motion, sensing, basic radio, and per-world reset.
Does not claim complete CPU parity or publishable full RL results.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
import numpy as np


def run(envs:int,steps:int,device:str, model:str,seed:int):
    import warp as wp
    from zhang_xue.gpu.mission_v78 import WarpParallelMissionV78,GPUWorldV78Params
    w=WarpParallelMissionV78(GPUWorldV78Params(envs=envs,device=device,seed=seed,obstacle_model=model))
    init=w.guard()
    if init['active_truth_per_world_min']!=15 or init['active_truth_per_world_max']!=15:
        raise AssertionError('15 targets per world invariant broken')
    if envs>1 and init['distinct_target_layouts']<2:
        raise AssertionError('independent environments share identical truth')
    previous=w.buffers['positions'].numpy().copy()
    outputs=[]
    for step in range(steps):
        act=np.zeros((envs,6,2),dtype=np.float32)
        act[:,:,0]= 0.30 if step%2==0 else -0.2
        dest=np.full((envs,6),2 if step%2==0 else 1,dtype=np.int32)
        obs,mask,reward,done=w.step(wp.array(act,dtype=wp.float32,device=device),wp.array(dest,dtype=wp.int32,device=device))
        current=w.buffers['positions'].numpy()
        delta=np.sqrt(((current[:,:6,:2]-previous[:,:6,:2])**2).sum(axis=2))
        if float(delta.max())>15.0005:raise AssertionError('UAV exceeds max speed')
        if not np.isfinite(reward.numpy()).all():raise AssertionError('NaN reward')
        if np.any(mask.numpy()[:,:,0]!=1):raise AssertionError('NO_SEND illegally masked')
        g=w.guard()
        outputs.append({'step':step+1,'mean_reward':float(reward.numpy().mean()),'max_motion_m':float(delta.max()),**g})
        previous=current.copy()
        if np.any(done.numpy()):
            selected=done.numpy().copy()
            if selected.sum() == envs:raise AssertionError('all worlds unexpectedly terminal in short smoke test')
            w.reset(selected)
            previous=w.buffers['positions'].numpy()
    if steps>=2 and not any(o['max_motion_m']>0 for o in outputs):
        raise AssertionError('motion kernel never moved UAV')
    return {'status':'PASS','type':'REAL_INDEPENDENT_WARP_WORLDS_NOT_CPU_PARITY',
            'device':device,'envs':envs,'steps':steps,'model':model,'seed':seed,
            'invariants':init,'samples':outputs,
            'full_cpu_warp_parity':False,'full_training_verified':False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--envs',type=int,default=4)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--device',default='cpu')
    p.add_argument('--model',default='multi_obstacle',choices=['binary_nlos','multi_obstacle','rf_opaque'])
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--output',default='')
    args=p.parse_args()
    try:
        result=run(args.envs,args.steps,args.device,args.model,args.seed)
        code=0
    except Exception as e:
        result={'status':'FAIL','error':str(e),'exception_type':type(e).__name__,
                'traceback':traceback.format_exc(),'device':args.device,
                'envs':args.envs,'steps':args.steps,'full_training_verified':False}
        code=1
    if args.output:
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    sys.exit(code)

if __name__=='__main__':main()
