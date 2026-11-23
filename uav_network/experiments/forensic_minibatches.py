"""Read-only forensic records from ACTUAL sampled replay minibatches.

Never re-samples replay, draws action noise, backpropagates, or steps optimizers.
Q1/Q2 are model estimates, NOT empirically calibrated state-action returns.
Small actual joint transitions are retained for later offline investigation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ..algorithms.action_utils import joint_action, valid_destinations, canonical_continuous


class ReplayMinibatchForensics:
    def __init__(self, output, *, enabled=True, max_rows=16,
                 raw_full_first_updates=16, raw_rows=1):
        self.enabled=bool(enabled)
        self.max_rows=max(1,int(max_rows))
        self.raw_full_first_updates=max(0,int(raw_full_first_updates))
        self.raw_rows=max(0,int(raw_rows))
        self.path=Path(output)/'forensic-minibatches.jsonl'
        self.records=0
        if self.enabled:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.path.touch(exist_ok=True)

    @staticmethod
    def _stats(tensor):
        v=tensor.detach().float()
        if not v.numel():return {'mean':None,'min':None,'max':None}
        return {'mean':float(v.mean()),'min':float(v.min()),'max':float(v.max())}

    @torch.no_grad()
    def capture(self, learner, batch, *, global_step, update_index,
                critic_updates, actor_updates):
        if not self.enabled:return False
        count=min(self.max_rows,len(batch['reward']))
        device=learner.device
        mini={k:(v[:count].to(device=device,dtype=torch.long if k=='destination' else torch.float32)
                 if isinstance(v,torch.Tensor) else torch.as_tensor(v[:count],device=device,dtype=(torch.long if k=='destination' else torch.float32)))
              for k,v in batch.items()}
        obs=mini['observation'];state=mini['state'];routes=mini['destination']
        n=obs.shape[1];num_route=learner.destinations
        replay_action=joint_action(mini['continuous'],routes,num_route,
                                   learn_power=learner.learn_power)
        q1,q2=learner._critic_values(learner.critics,state,replay_action)
        deterministic,prob=learner._policy_outputs(obs)
        actor_routes=prob.argmax(-1)
        actor_continuous=canonical_continuous(deterministic,obs,actor_routes,
                                             learn_power=learner.learn_power)
        actor_joint=joint_action(actor_continuous,actor_routes,num_route,
                                 learn_power=learner.learn_power)
        qa1,qa2=learner._critic_values(learner.critics,state,actor_joint)
        if learner.share_actor_parameters:
            flat=obs.reshape(count*n,-1)
            ids=torch.arange(n,device=flat.device).expand(count,n).reshape(-1)
            actor=learner.actors[0]
            raw=actor.continuous(F.relu(actor.encoder(learner._actor_input(flat,ids))))
            raw=actor.normalize_pretanh(raw).reshape(count,n,-1)
        else:
            raw=torch.stack([actor.normalize_pretanh(actor.continuous(F.relu(actor.encoder(obs[:,i]))))
                             for i,actor in enumerate(learner.actors)],1)
        motion_raw=raw[...,:learner.config.motion_dim]
        derivative=1-motion_raw.tanh().square()
        valid=valid_destinations(obs,num_route)
        actual_valid=valid.gather(-1,routes[...,None]).squeeze(-1)
        motion_action=mini['continuous'][...,:learner.config.motion_dim]
        metrics={
            'reward/mean':float(mini['reward'].mean()),
            'reward/min':float(mini['reward'].min()),
            'reward/max':float(mini['reward'].max()),
            'reward/negative_fraction':float((mini['reward']<0).float().mean()),
            'terminal/fraction':float(mini['terminated'].float().mean()),
            'critic/q1_replay_mean':float(q1.mean()),
            'critic/q2_replay_mean':float(q2.mean()),
            'critic/q1_replay_abs_max':float(q1.abs().max()),
            'critic/q2_replay_abs_max':float(q2.abs().max()),
            'critic/twin_q_abs_gap_mean':float((q1-q2).abs().mean()),
            'critic/twin_q_abs_gap_max':float((q1-q2).abs().max()),
            'critic/q1_actor_deterministic_mean':float(qa1.mean()),
            'critic/q2_actor_deterministic_mean':float(qa2.mean()),
            'critic/q1_actor_minus_replay_mean':float((qa1-q1).mean()),
            'critic/q1_optimistic_when_actor_vs_replay_fraction':float((qa1>q1).float().mean()),
            'actor_motion/pretanh_abs_mean':float(motion_raw.abs().mean()),
            'actor_motion/pretanh_abs_max':float(motion_raw.abs().max()),
            'actor_motion/tanh_derivative_mean':float(derivative.mean()),
            'actor_motion/tanh_derivative_lt_1e3_fraction':float((derivative<1e-3).float().mean()),
            'actor_motion/saturated_gt_0_99_fraction':float((motion_raw.tanh().abs()>.99).float().mean()),
            'action/replay_saturated_gt_0_95_fraction':float((motion_action.abs()>.95).float().mean()),
            'action/selected_invalid_fraction':float((~actual_valid).float().mean()),
            'action/replay_silent_fraction':float((routes==0).float().mean()),
            'action/replay_gcs_fraction':float((routes==n).float().mean()),
            'action/replay_receiver_eligible_fraction':float((valid.sum(-1)>1).float().mean()),
            'action/deterministic_silent_fraction':float((actor_routes==0).float().mean()),
            'action/actor_minus_replay_xyz_abs_mean':float((actor_continuous[...,:learner.config.motion_dim]-motion_action).abs().mean()),
            'state/critic_global_abs_mean':float(state.abs().mean()),
            'state/critic_global_abs_max':float(state.abs().max()),
            'observation/actor_local_abs_mean':float(obs.abs().mean()),
            'observation/actor_local_abs_max':float(obs.abs().max()),
        }
        for i,name in enumerate('xyz'[:learner.config.motion_dim]):
            metrics[f'actor_motion/pretanh_{name}_signed_mean']=float(motion_raw[...,i].mean())
            metrics[f'actor_motion/tanh_derivative_{name}_mean']=float(derivative[...,i].mean())
            metrics[f'action/replay_{name}_signed_mean']=float(motion_action[...,i].mean())
            metrics[f'action/replay_{name}_saturated_fraction']=float((motion_action[...,i].abs()>.95).float().mean())
        raw_sample={'row_count':0,'hash_sha256':None,'data':{}}
        if self.records<self.raw_full_first_updates and self.raw_rows:
            r=min(self.raw_rows,count)
            fields=('observation','state','continuous','destination','reward',
                    'next_observation','next_state','terminated')
            data={k:mini[k][:r].detach().cpu().tolist() for k in fields}
            digest=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            raw_sample={'row_count':r,'hash_sha256':digest,'data':data}
        record={
            'schema':1,'global_step':int(global_step),'optimizer_update_index':int(update_index),
            'critic_updates':int(critic_updates),'actor_updates':int(actor_updates),
            'algorithm':learner.algorithm,
            'source':'actual_replay_minibatch_after_update',
            'scope':'first_N_nonrandom_rows_of_sampled_replay; Q_estimates_not_return_calibration',
            'observed_rows':count,'joint_agents':n,
            'metrics':metrics,'raw_sample':raw_sample,
        }
        with self.path.open('a') as fd:
            fd.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
        self.records+=1
        return True
