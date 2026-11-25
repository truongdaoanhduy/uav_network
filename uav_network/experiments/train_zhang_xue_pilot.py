"""Small local CPU pilot exercising Zhang–Xue env + actual vanilla MASAC/MATD3.

Does not launch Kaggle, W&B, Hugging Face, CUDA, SimPy or Warp. A successful
pilot only confirms the end-to-end actor→env→replay→gradient plumbing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from uav_network.algorithms.vanilla_baselines import (
    GPUHybridReplayBuffer,
    VanillaMASAC,
    VanillaMATD3,
)
from uav_network.environment.zhang_xue_coop import (
    ZhangXueCooperativeEnv,
    ZhangXueConfig,
)


def run_cpu_pilot(
    algorithm: str = "matd3",
    seed: int = 44,
    episodes: int = 2,
    horizon_steps: int = 16,
    learning_starts: int = 8,
    batch_size: int = 8,
    hidden_sizes: Tuple[int,...] = (64,64),
    eval_episodes: int = 2,
    network_backend: str = "reference",
) -> dict:
    """Run a bounded CPU experiment and return metrics, without remote side effects.

    The standalone Zhang–Xue pilot state dimensions differ from the pre-existing
    800m Warp benchmark: never try to load those checkpoints as compatible.
    """
    algorithm=algorithm.lower()
    if algorithm not in ("matd3","masac"):
        raise ValueError("algorithm must be matd3 or masac")
    for name, value in (("episodes",episodes),("horizon_steps",horizon_steps),
                        ("learning_starts",learning_starts),("batch_size",batch_size),
                        ("eval_episodes",eval_episodes)):
        if int(value)<1:
            raise ValueError(f"{name} must be >=1")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    env=ZhangXueCooperativeEnv(
        ZhangXueConfig(horizon_steps=int(horizon_steps)),
        seed=seed,network_backend=network_backend,
    )
    learner_kw=dict(
        obs_dim=env.obs_dim,state_dim=env.state_dim,num_agents=env.num_agents,
        cont_dim=env.cont_dim,disc_dim=env.disc_dim,
        hidden_sizes=tuple(hidden_sizes),device=torch.device("cpu"),
    )
    # v3.2 scientific specification: target H_discrete = 0.6 * ln(K_valid).
    entropy_ratio = 0.6 if algorithm == "masac" else None
    agent=(VanillaMATD3(**learner_kw) if algorithm=="matd3"
           else VanillaMASAC(**learner_kw,target_entropy_disc_ratio=entropy_ratio))
    buffer=GPUHybridReplayBuffer(
        capacity=max(128,episodes*horizon_steps + 8),
        num_agents=env.num_agents,obs_dim=env.obs_dim,state_dim=env.state_dim,
        cont_dim=env.cont_dim,disc_dim=env.disc_dim,
        device=torch.device("cpu"),
    )
    actor_start=[x.detach().clone() for x in agent.actor.parameters()]
    all_returns=[]
    all_pdr=[]
    updates=0
    steps=0
    last_metrics={}
    for episode in range(episodes):
        obs,state,mask=env.reset(seed=seed + episode*7919)
        cumulative=0.0
        for t in range(horizon_steps):
            with torch.no_grad():
                o=torch.from_numpy(obs).unsqueeze(0)
                s=torch.from_numpy(state).unsqueeze(0)
                m=torch.from_numpy(mask).unsqueeze(0)
                cont,disc=agent.act(o,m,explore=True)
            next_obs,next_state,next_mask,reward,done,info=env.step(
                cont[0].numpy(),disc[0].argmax(dim=-1).numpy(),
            )
            with torch.no_grad():
                buffer.add_batch(
                    obs=o,state=s,act_cont=cont,act_disc_onehot=disc,
                    rewards=torch.tensor([float(reward)]),
                    next_obs=torch.from_numpy(next_obs).unsqueeze(0),
                    next_state=torch.from_numpy(next_state).unsqueeze(0),
                    dones=torch.tensor([float(done)]),
                    mask=m,next_mask=torch.from_numpy(next_mask).unsqueeze(0),
                )
            steps+=1
            cumulative+=float(reward)
            if buffer.size>=max(learning_starts,batch_size):
                last_metrics=agent.update(buffer.sample(batch_size),return_metrics=True)
                updates+=1
            obs,state,mask=next_obs,next_state,next_mask
            if done:
                break
        all_returns.append(cumulative)
        all_pdr.append(float(info["data_pdr"]))

    actor_shift=math.sqrt(sum(
        float(((new.detach()-old)**2).sum())
        for old,new in zip(actor_start,agent.actor.parameters())
    ))

    eval_returns=[]
    eval_pdr=[]
    for i in range(eval_episodes):
        obs,state,mask=env.reset(seed=1_000_000 + seed + i)
        cumulative=0.0
        for t in range(horizon_steps):
            with torch.no_grad():
                o=torch.from_numpy(obs).unsqueeze(0)
                m=torch.from_numpy(mask).unsqueeze(0)
                cont,disc=agent.act(o,m,explore=False)
            obs,state,mask,reward,done,info=env.step(
                cont[0].numpy(),disc[0].argmax(dim=-1).numpy(),
            )
            cumulative+=float(reward)
            if done:break
        eval_returns.append(cumulative)
        eval_pdr.append(float(info["data_pdr"]))
    for name,v in last_metrics.items():
        if isinstance(v,(float,int)) and not math.isfinite(float(v)):
            raise FloatingPointError(f"Non-finite learner metric: {name}")
    return {
        "backend":(
            "cpu_uavnetsim_v1_simpy_csma"
            if network_backend=="classic_simpy_v1"
            else "cpu_reference_not_uavnetsim"
        ),
        "algorithm":algorithm,
        "target_entropy_disc_ratio":entropy_ratio,
        "seed":int(seed),
        "scenario":"zhang_xue_1000m_20x20_6uav_10target",
        "obs_dim":env.obs_dim,
        "state_dim":env.state_dim,
        "disc_dim":env.disc_dim,
        "episodes":int(episodes),
        "total_steps":steps,
        "updates":updates,
        "replay_size":buffer.size,
        "actor_parameter_delta_norm":actor_shift,
        "mean_train_return":float(np.mean(all_returns)),
        "mean_train_pdr":float(np.mean(all_pdr)),
        "mean_eval_return":float(np.mean(eval_returns)),
        "mean_eval_pdr":float(np.mean(eval_pdr)),
        "learner_metrics":{k:float(v) for k,v in last_metrics.items()
                           if isinstance(v,(float,int))},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm",choices=("masac","matd3"),default="matd3")
    parser.add_argument("--network-backend",choices=("reference","classic_simpy_v1"),
                        default="reference")
    parser.add_argument("--episodes",type=int,default=2)
    parser.add_argument("--horizon-steps",type=int,default=16)
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--learning-starts",type=int,default=8)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--hidden-size",type=int,default=64)
    parser.add_argument("--eval-episodes",type=int,default=2)
    parser.add_argument("--output-json",type=Path)
    args=parser.parse_args()
    result=run_cpu_pilot(
        algorithm=args.algorithm,seed=args.seed,episodes=args.episodes,
        horizon_steps=args.horizon_steps,learning_starts=args.learning_starts,
        batch_size=args.batch_size,hidden_sizes=(args.hidden_size,args.hidden_size),
        eval_episodes=args.eval_episodes,
        network_backend=args.network_backend,
    )
    report=json.dumps(result,indent=2,sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(report+"\n")
    print(report)


if __name__=="__main__":
    main()
