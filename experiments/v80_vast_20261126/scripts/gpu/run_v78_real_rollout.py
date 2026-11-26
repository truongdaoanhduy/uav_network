#!/usr/bin/env python3
"""V78 experimental REAL independent Warp worlds -> MATD3/MASAC learner.

IMPORTANT: this is NOT research-certified or CPU-equivalent until the strict
parity gates pass. Requires `--allow-experimental-physics` to run. Never
silently substitute the V77 repeated-4-CPU-transition capacity fixture.

One vector transition contains E independently sampled target maps and E
independent radio/mobility/belief/reward states, not a repeated static tuple.
Learner reads (E,6,518) policy observations, gets global critic state by
flattening only observation data, and updates from real transitions.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/zhang_xue/src')]


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--algorithm',choices=['matd3','masac'],required=True)
    a.add_argument('--envs',type=int,default=4)
    a.add_argument('--steps',type=int,default=3)
    a.add_argument('--updates-per-step',type=int,default=1)
    a.add_argument('--batch-size',type=int,default=16)
    a.add_argument('--hidden',type=int,default=64)
    a.add_argument('--world-device',default='cpu')
    a.add_argument('--learner-device',default='cpu')
    a.add_argument('--seed',type=int,default=44)
    a.add_argument('--model',default='multi_obstacle',choices=['binary_nlos','multi_obstacle','rf_opaque'])
    a.add_argument('--allow-experimental-physics',action='store_true')
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'RUNNING','envs':args.envs,'steps':args.steps,'algorithm':args.algorithm,
            'batch_size':args.batch_size,'world_device':args.world_device,'learner_device':args.learner_device,
            'independent_world_runtime':True,'uses_repeated_cpu_fixtures':False,
            'verified_exact_cpu_model_parity':False,'paper_ready':False,
            'gpu_training_converged':False,'updates':0,'world_steps':0,
            'notes':['V78 dynamics and observation semantics are still experimental',
                     'No provenance/physics parity assertion',
                     'CPU host-sync handles terminal mask until GPU-autoreset kernel is validated']}
    start=time.time()
    try:
        if not args.allow_experimental_physics:
            result['status']='BLOCKED_EXPERIMENTAL_PHYSICS_REQUIRES_ACK'
            raise RuntimeError('add --allow-experimental-physics after reviewing V78 limitations')
        import numpy as np
        import torch
        import warp as wp
        from zhang_xue.gpu.mission_v78 import GPUWorldV78Params,WarpParallelMissionV78
        from uav_network.algorithms.configuration.loader import load_learner_config
        from uav_network.algorithms.matd3.learner import MATD3
        from uav_network.algorithms.masac.learner import MASAC
        from uav_network.algorithms.core.replay import TensorReplayBuffer
        def must_cuda(name):
            if name.startswith('cuda:'):
                if not torch.cuda.is_available() or int(name.split(':')[1])>=torch.cuda.device_count():
                    raise RuntimeError(f'CUDA unavailable: {name}')
        must_cuda(args.world_device);must_cuda(args.learner_device)
        cfg=load_learner_config(ROOT/'configs/learners/e4096_b8192.yaml')
        cfg.runtime.algorithm=args.algorithm;cfg.runtime.num_envs=args.envs
        hp=getattr(cfg,args.algorithm)
        hp.batch_size=args.batch_size
        hp.replay_capacity=max(args.batch_size*4,args.envs*8)
        hp.hidden_sizes=[args.hidden,args.hidden]
        cfg.validate()
        torch.manual_seed(args.seed)
        world=WarpParallelMissionV78(GPUWorldV78Params(envs=args.envs,device=args.world_device,seed=args.seed,
                                                         obstacle_model=args.model))
        initial=world.guard()
        if initial['active_truth_per_world_min']!=15 or initial['active_truth_per_world_max']!=15:
            raise RuntimeError('worlds must each have exactly 15 targets')
        if args.envs>1 and initial['distinct_target_layouts']<2:
            raise RuntimeError('worlds are not independently sampled')
        result['initial_world_invariants']=initial
        obs_dim=518; state_dim=6*obs_dim
        learner=(MATD3 if args.algorithm=='matd3' else MASAC)(cfg,observation_dim=obs_dim,state_dim=state_dim,
                                                              device=args.learner_device)
        replay=TensorReplayBuffer(hp.replay_capacity,(6,obs_dim),state_dim,
                                  device=args.learner_device,continuous_dim=2,seed=args.seed)
        result['replay_allocated_bytes']=replay.allocated_bytes
        def encoded_obs():
            raw=wp.to_torch(world.buffers['obs']).to(args.learner_device).clone()
            masks=wp.to_torch(world.buffers['mask']).to(args.learner_device).to(torch.float32).clone()
            # Preserve original GCS distance at index 9, repurpose to actor-alive
            # as expected by the retained learner. Last 3 are actual action mask.
            dist=raw[...,9:10].clone()
            raw[...,9]=(raw[...,4]>0).to(torch.float32)
            return torch.cat([raw,dist,masks],-1)
        statuses=[]
        for i in range(args.steps):
            old=encoded_obs(); state=old.reshape(args.envs,-1)
            action=learner.act_batch(old,deterministic=False,return_tensors=True)
            act_cont=action.continuous.to(args.world_device).contiguous()
            act_dest=action.destination.to(device=args.world_device,dtype=torch.int32).contiguous()
            world.step(wp.from_torch(act_cont,dtype=wp.float32),wp.from_torch(act_dest,dtype=wp.int32))
            nextobs=encoded_obs(); nextstate=nextobs.reshape(args.envs,-1)
            rew=wp.to_torch(world.buffers['reward']).to(args.learner_device,dtype=torch.float32).clone()
            ended=wp.to_torch(world.buffers['done']).to(args.learner_device,dtype=torch.float32).clone()
            replay.add_batch(old,state,action,rew,nextobs,nextstate,ended)
            step_result={'vector_step':i+1,'true_transition_count':(i+1)*args.envs,
                         'mean_reward':float(rew.mean().item()),'episodes_ended':int(ended.sum().item())}
            if len(replay)>=args.batch_size:
                step_result['updates']=[]
                for j in range(args.updates_per_step):
                    metrics=learner.update(replay.sample(args.batch_size))
                    out={k:float(v) for k,v in metrics.items()}
                    if not all(math.isfinite(v) for v in out.values()):
                        raise FloatingPointError('non-finite learner metrics')
                    step_result['updates'].append(out)
                    result['updates']+=1
            # NOTE: this synchronized mask transition is explicitly counted as a
            # CPU orchestration step. Environment dynamics remain Warp-native.
            d=world.buffers['done'].numpy().astype(np.uint8)
            if np.any(d):world.reset(d)
            statuses.append(step_result)
            result['world_steps']=i+1
            if (i+1)%max(1,min(args.steps,10))==0:world.guard()
            # Durable append-only progress: unexpected termination preserves earlier proof.
            args.output.with_suffix('.progress.jsonl').open('a').write(json.dumps(step_result,allow_nan=False)+'\n')
        if result['updates']==0:raise RuntimeError('no learner updates; increase --steps or reduce batch size')
        result['final_world_invariants']=world.guard()
        result['status']='EXPERIMENTAL_INDEPENDENT_WARP_ROLLOUT_AND_LEARNER_PASS_NOT_PARITY'
        result['final_record']=statuses[-1]
    except Exception as e:
        if result['status']=='RUNNING':result['status']='FAIL'
        result['error_type']=type(e).__name__;result['error']=str(e)
        result['traceback']=traceback.format_exc()
    finally:
        result['elapsed_s']=round(time.time()-start,3)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result['status'].startswith('EXPERIMENTAL_INDEPENDENT_WARP_ROLLOUT') else 3 if result['status'].startswith('BLOCKED') else 1
if __name__=='__main__':sys.exit(main())
