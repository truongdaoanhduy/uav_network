"""Safe GPU/CPU Eval8192 allocation smoke; *NEVER* invokes MATD3 training.

The policies are deterministic random, never historical or trained policies.
This probes physical capacity, Warp full NoOracle scene, vmap inference,
100-seed/policy packing and per-episode metric integrity.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from copy import deepcopy

import torch

from uav_network.configuration import load_config
from uav_network.experiments.deferred_gpu_batched import evaluate_group
from uav_network.experiments.deferred_gpu_evaluation import GPUActorOnlyPolicy
from uav_network.environment.gpu.vector import WarpVectorEnv


def smoke(*, device, gpu_index, policy_count, episodes, capacity, max_steps,
          first_policy):
    if capacity < policy_count*episodes:
        raise ValueError('Physical pool too small for complete policies')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU smoke without CUDA driver')
    torch.set_num_threads(1)
    c=load_config('configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml')
    assert (c.runtime.algorithm,c.sensor.confirmation_threshold,c.matd3.hidden_sizes)==(
        'matd3',0.95,[256,256])
    assert c.sensor.report_cell_hypotheses and c.network.enabled
    assert (c.world.num_uavs,c.world.num_targets,c.world.num_obstacles)==(6,35,25)
    c.runtime.device=device
    c.world.max_steps=max_steps  # CPU/GPU smoke ONLY; never modifies the production YAML.
    c.runtime.num_envs=capacity
    c.validate()
    payloads=[]
    for i in range(first_policy,first_policy+policy_count):
        torch.manual_seed(887766+i)
        actor=GPUActorOnlyPolicy(c,'cpu')
        payloads.append({'config':c.to_dict(),
                         'purpose':'evaluation_only',
                         'actors':deepcopy(actor.actors.state_dict())})
    del actor
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    vec=WarpVectorEnv(c,device='cuda:0' if device=='cuda' else 'cpu')
    allocation_seconds=time.perf_counter()-t0
    try:
        t_eval=time.perf_counter()
        grouped=evaluate_group(c,payloads,device='cuda:0' if device=='cuda' else 'cpu',
                               seed=1000000044,episodes=episodes,vector=vec)
        if device=='cuda': torch.cuda.synchronize()
        eval_seconds=time.perf_counter()-t_eval
        assert len(grouped)==policy_count
        assert all(len(r)==episodes and [x['seed'] for x in r]==list(
            range(1000000044,1000000044+episodes)) for r in grouped)
        samples=[x for y in grouped for x in y]
        checksum=hashlib.sha256(json.dumps(grouped,sort_keys=True,allow_nan=False).encode()).hexdigest()
        report={
            'kind':'physical_warp_evaluation_no_train',
            'device':device,'gpu_index':gpu_index,
            'real_checkpoints':policy_count,'episodes_each':episodes,
            'real_episodes':len(samples),'physical_worlds':capacity,
            'unused_worlds':capacity-policy_count*episodes,
            'max_steps_smoke_only':max_steps,
            'science':'NoOracle CT095 H256x256 PHY/MAC enabled',
            'seed_first':1000000044,'seed_last':1000000044+episodes-1,
            'source_policy_indices':[first_policy,first_policy+policy_count-1],
            'warp_state_bytes':vec.allocated_bytes,
            'allocate_seconds':allocation_seconds,'eval_seconds':eval_seconds,
            'overall_seconds':time.perf_counter()-t0,
            'cuda_allocated_bytes':torch.cuda.memory_allocated() if device=='cuda' else None,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'score_sha256':checksum,
            'mean_return':sum(x['return'] for x in samples)/len(samples),
            'mean_search_rate_percent':sum(x['search_rate_percent'] for x in samples)/len(samples),
            'mean_delivery_rate_percent':sum(x['delivery_rate_percent'] for x in samples)/len(samples),
        }
        return report
    finally:
        vec.close()
        del vec
        gc.collect()
        if device=='cuda': torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    ap.add_argument('--gpu-index',type=int,default=0)
    ap.add_argument('--policies',type=int,default=81)
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--world-pool',type=int,default=8192)
    ap.add_argument('--max-steps',type=int,default=12)
    ap.add_argument('--checkpoint-offset',type=int,default=0)
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    if not 0<args.policies<=81 or not 0<args.episodes<=100 or not 0<args.max_steps<=1500:
        raise ValueError('Invalid bounded Eval8192 smoke dimensions')
    if args.world_pool<args.policies*args.episodes or args.world_pool>8192:
        raise ValueError('Invalid physical pool size')
    result=smoke(device=args.device,gpu_index=args.gpu_index,policy_count=args.policies,
                 episodes=args.episodes,capacity=args.world_pool,
                 max_steps=args.max_steps,first_policy=args.checkpoint_offset)
    line=json.dumps(result,sort_keys=True,allow_nan=False)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(line+'\n')
    print('UAV_V9_EVAL8192_GPU_SMOKE_JSON '+line,flush=True)

if __name__=='__main__':main()
