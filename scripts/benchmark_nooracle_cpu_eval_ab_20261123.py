"""Paired, same-science CPU Eval benchmark of original CT095 vs Eval8192 candidate.

The CPU cannot afford 2 x 8192 physical worlds.  This bounded study scales the
checkpoint-packing *mechanism*: 12 checkpoints, three fixed seeds per policy,
full NoOracle PHY/MAC, and a fixed 20-world pool (6*3 real, 2 filler).
Reference is the pristine pre-change branch; no policy is trained here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REFERENCE = Path('/home/aduy/Documents/NCKH/uav_network_v9_ct095_h256_test1000_20261123')
CANDIDATE = Path('/home/aduy/Documents/NCKH/uav_network_eval8192_perf_candidate_20261123')


def measured_worker(mode, *, checkpoints, episodes, steps):
    import torch
    from uav_network.configuration import load_config
    from uav_network.experiments.deferred_gpu_evaluation import GPUActorOnlyPolicy
    from uav_network.experiments.deferred_gpu_batched import evaluate_group
    from uav_network.experiments.gpu_eval_layout import plan_validation
    from uav_network.environment.gpu.vector import WarpVectorEnv

    torch.set_num_threads(1)
    config=load_config('configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml')
    config.runtime.device='cpu'
    config.runtime.num_envs=2
    config.world.max_steps=steps
    config.tracking.enabled=False
    config.validate()
    assert (config.sensor.confirmation_threshold,config.matd3.hidden_sizes)==(0.95,[256,256])
    payloads=[]
    for i in range(checkpoints):
        torch.manual_seed(93001+i)
        actor=GPUActorOnlyPolicy(config,'cpu')
        payloads.append({'config':config.to_dict(),'purpose':'evaluation_only',
                         'actors':deepcopy(actor.actors.state_dict())})
    del actor
    if mode=='reference':
        pack=2
        capacity=pack*episodes
    else:
        pack=6
        capacity=pack*episodes+2
    actor_gpu_records={}
    timing=[]
    entire_eval_start=time.perf_counter()
    for gpu in (0,1):
        if mode=='reference':
            layout=plan_validation(checkpoints,episodes,capacity,gpu_index=gpu,gpu_count=2)
        else:
            layout=plan_validation(checkpoints,episodes,capacity,gpu_index=gpu,gpu_count=2,
                                   assignment='contiguous_batches')
        pool=None
        try:
            if mode!='reference':
                pc=deepcopy(config)
                pc.runtime.num_envs=capacity
                pool=WarpVectorEnv(pc,device='cpu')
            for batch in layout:
                indices=batch.checkpoint_indices
                start=time.perf_counter()
                results=evaluate_group(config,[payloads[i] for i in indices],device='cpu',
                                      seed=1000000044,episodes=episodes,vector=pool)
                elapsed=time.perf_counter()-start
                timing.append({'gpu':gpu,'indices':list(indices),'seconds':elapsed})
                for i,records in zip(indices,results):
                    if i in actor_gpu_records:raise ValueError('Repeated checkpoint')
                    actor_gpu_records[i]=records
        finally:
            if pool is not None:pool.close()
    if set(actor_gpu_records)!=set(range(checkpoints)):
        raise AssertionError('Missing checkpoint')
    entire_eval_seconds=time.perf_counter()-entire_eval_start
    data=json.dumps(actor_gpu_records,sort_keys=True,allow_nan=False,separators=(',',':'))
    canonical=hashlib.sha256(data.encode()).hexdigest()
    return {'mode':mode,'checkpoints':checkpoints,'episodes_each':episodes,'steps':steps,
            'total_real_episode_count':sum(len(x) for x in actor_gpu_records.values()),
            'physical_pool_worlds':capacity if pool is not None else 0,
            'score_sha256':canonical,'score_bytes':len(data),'scores':actor_gpu_records,
            'timing':timing,'eval_seconds':entire_eval_seconds, 'batch_seconds_only':sum(x['seconds'] for x in timing),
            'per_gpu_seconds':{str(i):sum(x['seconds'] for x in timing if x['gpu']==i)
                               for i in (0,1)}}


def invoke(mode, args):
    root=REFERENCE if mode=='reference' else CANDIDATE
    env=os.environ.copy();env['PYTHONPATH']=str(root)
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    cmd=[sys.executable,str(CANDIDATE/'scripts/benchmark_nooracle_cpu_eval_ab_20261123.py'),
         '--worker',mode,'--checkpoints',str(args.checkpoints),
         '--episodes',str(args.episodes),'--steps',str(args.steps)]
    proc=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=120)
    if proc.returncode:
        raise RuntimeError(mode+' worker failed: '+proc.stdout[-650:]+proc.stderr[-1500:])
    lines=[x for x in proc.stdout.splitlines() if x.startswith('CPU_EVAL_WORKER_JSON ')]
    if len(lines)!=1:raise RuntimeError('Missing worker report '+mode+' '+proc.stdout[-1000:])
    return json.loads(lines[0].split(' ',1)[1])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',choices=('reference','candidate'))
    ap.add_argument('--checkpoints',type=int,default=12)
    ap.add_argument('--episodes',type=int,default=3)
    ap.add_argument('--steps',type=int,default=24)
    ap.add_argument('--pairs',type=int,default=3)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.checkpoints < 1 or args.episodes < 1 or args.steps < 1:
        raise ValueError('Invalid benchmark dimensions')
    if args.worker:
        result=measured_worker(args.worker,checkpoints=args.checkpoints,
                               episodes=args.episodes,steps=args.steps)
        print('CPU_EVAL_WORKER_JSON '+json.dumps(result,allow_nan=False),flush=True)
        return
    if not args.output:raise ValueError('Missing --output')
    records=[]
    for i in range(args.pairs):
        modes=('reference','candidate') if i%2==0 else ('candidate','reference')
        for mode in modes:
            row=invoke(mode,args)
            records.append(row)
            print('CPU_EVAL_AB',i,mode,'seconds',round(row['eval_seconds'],3),
                  'score_sha256',row['score_sha256'],flush=True)
    all_hashes={x['score_sha256'] for x in records}
    identical=len(all_hashes)==1
    stats={m:{'median_eval_seconds':statistics.median(x['eval_seconds'] for x in records
                        if x['mode']==m),
              'all_eval_seconds':[x['eval_seconds'] for x in records if x['mode']==m]}
            for m in ('reference','candidate')}
    speedup=100*(stats['reference']['median_eval_seconds']/
                 stats['candidate']['median_eval_seconds']-1)
    report={'kind':'CPU-only full-science NoOracle V9 validation paired AB - not train or CUDA',
            'old_source':str(REFERENCE),'candidate_source':str(CANDIDATE),
            'science_profile':'CT095 H256x256 shared actor BN, NoOracle network on',
            'checkpoints':args.checkpoints,'episodes_each':args.episodes,
            'world_max_steps':args.steps,'trials':args.pairs,
            'score_parity_exact':identical,'score_sha256':next(iter(all_hashes)) if identical else None,
            'speedup_percent':speedup,
            'candidate_faster':speedup>0 and identical,'statistics':stats,
            'trials_without_large_scores':[{k:v for k,v in r.items() if k!='scores'} for r in records]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('CPU_EVAL_AB_REPORT '+json.dumps({k:report[k] for k in
           ('score_parity_exact','speedup_percent','candidate_faster','statistics')},
           allow_nan=False),flush=True)
    if not identical:
        raise AssertionError('Candidate changed mission scores: do not approve')

if __name__=='__main__':main()
