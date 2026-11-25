"""Bounded, real-gradient CPU smoke of the *packet* V55 env using repo VanillaMATD3/MASAC.

This intentionally refuses full GPU training until full MAC/PHY CPU-Warp parity.
Launch after running stage_in_repo_v55.sh on a clean project working branch:
  PYTHONPATH=src:. python src/train_v55_packet_cpu.py --algorithm matd3 --steps 10
"""
from __future__ import annotations
import argparse,json,math
import numpy as np

from uav_v54_7node_env import EnvConfigV54,ZhangXue7NodePacketEnv
from v55_state_derived_replay import StateDerivedReplayBuffer


def run_packet_cpu_gradient_smoke(*,algorithm:str='matd3',seed:int=44,
                                   steps:int=10,batch_size:int=8,
                                   hidden_size:int=64)->dict:
    import torch
    from uav_network.algorithms.vanilla_baselines import VanillaMATD3,VanillaMASAC
    if algorithm not in ('matd3','masac'): raise ValueError('algorithm must be matd3/masac')
    if not (1<=steps<=100 and 2<=batch_size<=64 and 16<=hidden_size<=512):
        raise ValueError('bounded CPU smoke only: steps <=100; batch 2..64; hidden 16..512')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    cfg=EnvConfigV54()
    env=ZhangXue7NodePacketEnv(cfg,seed=seed)
    obs_dim=env.obs_dim;state_dim=env.state_dim
    learner_kwargs=dict(obs_dim=obs_dim,state_dim=state_dim,num_agents=6,
                        cont_dim=2,disc_dim=7,hidden_sizes=(hidden_size,hidden_size),
                        device=torch.device('cpu'))
    agent=(VanillaMATD3(**learner_kwargs) if algorithm=='matd3'
           else VanillaMASAC(**learner_kwargs,target_entropy_disc_ratio=0.6))
    replay=StateDerivedReplayBuffer(capacity=max(128,steps+8),num_agents=6,
             obs_dim=obs_dim,state_dim=state_dim,cont_dim=2,disc_dim=7,
             device=torch.device('cpu'))
    initial_actor=[v.detach().clone() for v in agent.actor.parameters()]
    obs,state,mask=env.reset(seed=seed)
    rew=0.;n_update=0;last_info={};last_metrics={}
    for t in range(steps):
        with torch.no_grad():
            o=torch.from_numpy(obs).unsqueeze(0)
            st=torch.from_numpy(state).unsqueeze(0)
            mk=torch.from_numpy(mask).unsqueeze(0)
            cont,disc=agent.act(o,mk,explore=True)
        a_v=cont[0].numpy()*cfg.v_max_mps  # actor is tanh-normalized, env accepts m/s!
        new_obs,new_state,new_mask,r,term,trunc,info=env.step(a_v,disc[0].argmax(dim=-1).numpy())
        # Physical termination and technical horizon truncation are distinct.
        replay.add_batch(obs=o,state=st,act_cont=cont,act_disc_onehot=disc,
            rewards=torch.tensor([float(r)]),next_obs=torch.from_numpy(new_obs).unsqueeze(0),
            next_state=torch.from_numpy(new_state).unsqueeze(0),
            dones=torch.tensor([float(term)]),mask=mk,
            next_mask=torch.from_numpy(new_mask).unsqueeze(0))
        if replay.size>=batch_size:
            last_metrics=agent.update(replay.sample(batch_size),return_metrics=True)
            n_update+=1
        rew+=float(r);last_info=info
        obs,state,mask=new_obs,new_state,new_mask
        if term or trunc: break
    actor_delta=math.sqrt(sum(float(((p.detach()-q)**2).sum()) for p,q in zip(agent.actor.parameters(),initial_actor)))
    if n_update>0 and not math.isfinite(actor_delta):
        raise FloatingPointError('actor update produced NaN/Inf')
    return {'algorithm':algorithm,'seed':seed,'requested_steps':steps,
            'actual_steps':env.t,'updates':n_update,'actor_delta_l2':actor_delta,
            'total_return':rew,'obs_dim':obs_dim,'state_dim':state_dim,
            'replay_size':replay.size,'replay_bytes':replay.vram_bytes(),
            'last_training_metrics':{k:float(v) for k,v in last_metrics.items()
                                     if isinstance(v,(int,float))},
            'last_env':last_info,'device':'cpu','gpu_parity_claimed':False}


def main():
    a=argparse.ArgumentParser();a.add_argument('--algorithm',choices=['masac','matd3'],default='matd3')
    a.add_argument('--seed',type=int,default=44);a.add_argument('--steps',type=int,default=10)
    a.add_argument('--batch-size',type=int,default=8)
    args=a.parse_args()
    print(json.dumps(run_packet_cpu_gradient_smoke(algorithm=args.algorithm,seed=args.seed,
          steps=args.steps,batch_size=args.batch_size),indent=2))
if __name__=='__main__':main()
