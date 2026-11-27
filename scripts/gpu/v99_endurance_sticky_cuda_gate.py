#!/usr/bin/env python3
"""Experimental cumulative safety/finite/reward gate (non-RL); all Warp checks on-device.

Checks ALL steps, logs real device VRAM through nvidia-smi, handles episode terminations
via selective on-device reset. This is a harder, DIFFERENT run from the V98 script:
900 transitions/world may span several episodes; it does NOT certify one 900-step
flight or prove recursive feasibility in all possible states.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import warp as wp

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/zhang_xue/src')]

@wp.kernel
def sticky_monitor(
    pos: wp.array(dtype=wp.float32,ndim=3),
    battery: wp.array(dtype=wp.float32,ndim=2),
    reward: wp.array(dtype=wp.float32,ndim=1),
    components: wp.array(dtype=wp.float32,ndim=2),
    failed: wp.array(dtype=wp.uint8,ndim=1),
    overflow: wp.array(dtype=wp.uint8,ndim=1),
    done: wp.array(dtype=wp.uint8,ndim=1),
    everfailed: wp.array(dtype=wp.int32,ndim=1),
    everoverflow: wp.array(dtype=wp.int32,ndim=1),
    evernonfinite: wp.array(dtype=wp.int32,ndim=1),
    maxerr: wp.array(dtype=wp.float32,ndim=1),
    donecount: wp.array(dtype=wp.int32,ndim=1),
):
    i=wp.tid()
    if failed[i]!=0: everfailed[i]=1
    if overflow[i]!=0: everoverflow[i]=1
    if done[i]!=0: donecount[i]=donecount[i]+1
    if not wp.isfinite(reward[i]): evernonfinite[i]=1
    ledger=wp.float32(0.0)
    for k in range(8):
        if not wp.isfinite(components[i,k]): evernonfinite[i]=1
        ledger=ledger+components[i,k]
    err=wp.abs(ledger-reward[i])
    if err>maxerr[i]: maxerr[i]=err
    for u in range(7):
        for axis in range(3):
            if not wp.isfinite(pos[i,u,axis]): evernonfinite[i]=1
    for u in range(6):
        if not wp.isfinite(battery[i,u]): evernonfinite[i]=1


def query_mib():
    try:
        v=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--id=0'],timeout=8,text=True).strip().splitlines()[0]
        return float(v.strip().split()[0])
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):return None


def run(args):
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params,WarpParallelMissionV80
    p=GPUWorldV80Params.from_yaml(args.scenario,envs=args.envs,device=args.device,seed=args.seed)
    w=WarpParallelMissionV80(p)
    ang=np.linspace(0,2*np.pi,6,endpoint=False,dtype=np.float32)
    a=np.zeros((args.envs,6,2),dtype=np.float32)
    if args.motion=='gentle_patrol':
        a[:,:,0]=.25*np.cos(ang)[None,:]
        a[:,:,1]=.25*np.sin(ang)[None,:]
    act=wp.array(a,dtype=wp.float32,device=args.device)
    dst=wp.array(np.ones((args.envs,6),dtype=np.int32),dtype=wp.int32,device=args.device)
    sticky={k:wp.zeros(args.envs,dtype=wp.int32,device=args.device) for k in ('failed','overflow','nonfinite','done')}
    sticky['err']=wp.zeros(args.envs,dtype=wp.float32,device=args.device)
    wp.synchronize_device(args.device)
    mem=[query_mib()]
    t0=time.monotonic()
    for step in range(1,args.steps+1):
        if args.motion=='gentle_patrol' and step%60==0:
            a=-a
            wp.copy(act,wp.array(a,dtype=wp.float32,device=args.device))
        w.step(act,dst)
        wp.launch(sticky_monitor,dim=args.envs,inputs=[w.buffers['positions'],w.buffers['battery'],w.buffers['reward'],w.buffers['reward_components'],w.buffers['failed'],w.network.data['overflow'],w.buffers['done'],sticky['failed'],sticky['overflow'],sticky['nonfinite'],sticky['err'],sticky['done']],device=args.device)
        w.auto_reset_gpu()
        if step%150==0:
            wp.synchronize_device(args.device)
            mem.append(query_mib())
    wp.synchronize_device(args.device)
    elapsed=time.monotonic()-t0
    d={k:v.numpy() for k,v in sticky.items()}
    out={
        'status':'UNVERIFIED','gate':'V99_ALL_STEP_STICKY_4096X900',
        'scenario_sha256':hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        'device':args.device,'envs':args.envs,'steps':args.steps,
        'observed_steps':args.steps,'motion':args.motion,
        'episode_handling':'GPU selective auto_reset after every monitored transition',
        'terminal_worlds_handled':True,
        'ever_safety_failure_worlds':int(np.count_nonzero(d['failed'])),
        'ever_overflow_worlds':int(np.count_nonzero(d['overflow'])),
        'nonfinite_positions_all_steps':int(np.count_nonzero(d['nonfinite'])),
        'nonfinite_reward_all_steps':int(np.count_nonzero(d['nonfinite'])),
        'max_reward_sum_err':float(d['err'].max()),
        'terminal_events':int(d['done'].sum()),
        'memory_samples_mib':mem,
        'peak_vram_mib_actual':max([x for x in mem if x is not None],default=None),
        'elapsed_sec':round(elapsed,3),
        'world_steps_per_sec':round(args.envs*args.steps/elapsed,2)
    }
    required=('ever_safety_failure_worlds','ever_overflow_worlds','nonfinite_positions_all_steps','nonfinite_reward_all_steps')
    good=all(out[k]==0 for k in required) and out['max_reward_sum_err']<1e-4 and out['peak_vram_mib_actual'] is not None
    out['status']='PASS_STICKY_RESET_MODE' if good else 'FAIL_STICKY'
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    return 0 if good else 2

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenario',type=Path,default=ROOT/'research/zhang_xue/configs/scenario_v91_bayes_1km.yaml')
    ap.add_argument('--envs',type=int,default=4096)
    ap.add_argument('--steps',type=int,default=900)
    ap.add_argument('--seed',type=int,default=44)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--motion',choices=('stationary','gentle_patrol'),default='gentle_patrol')
    ap.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(ap.parse_args()))
