"""Four actual environment transitions -> off-policy update, without GPU/online tracking.

This is a CPU integration gate, NOT a training benchmark or convergence claim.
Actor sees only observations; the centralized critic receives joint local data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from uav_network.algorithms.configuration import LearnerConfig
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.masac import MASAC
from zhang_xue.environment.configuration import make_v57_config
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.learners.actor_adapter import (
    ACTOR_POLICY_DIM, decode_policy_action, encode_actor_input, make_critic_state
)


def integration_gate(algorithm: str, *, seed=44, transitions=4):
    if algorithm not in ("matd3","masac") or not (1<=transitions<=12):
        raise ValueError("Expected matd3/masac and 1..12 short transitions")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    cfg=LearnerConfig()
    cfg.runtime.algorithm=algorithm
    cfg.matd3.hidden_sizes=[32,32]
    cfg.masac.hidden_sizes=[32,32]
    cfg.validate()
    model={"matd3":MATD3,"masac":MASAC}[algorithm](
        cfg,ACTOR_POLICY_DIM,6*ACTOR_POLICY_DIM,device="cpu"
    )
    env=ZhangXue7NodePacketEnv(make_v57_config(),seed=seed)
    raw,_,mask=env.reset(seed=seed)
    episodes=0
    rows=[]
    while len(rows)<transitions:
        encoded=encode_actor_input(raw,mask)
        state=make_critic_state(encoded)
        act=model.act(encoded,deterministic=True)
        velocity,destination=decode_policy_action(
            act.continuous,act.destination,mask,v_max_mps=env.cfg.v_max_mps
        )
        next_raw,_,next_mask,reward,terminated,truncated,info=env.step(
            velocity,destination
        )
        next_encoded=encode_actor_input(next_raw,next_mask)
        rows.append({
            "observation":encoded,
            "next_observation":next_encoded,
            "state":state,
            "next_state":make_critic_state(next_encoded),
            "continuous":np.asarray(act.continuous,dtype=np.float32),
            "destination":np.asarray(destination,dtype=np.int64),
            "reward":np.float32(reward),
            "terminated":np.float32(terminated),
        })
        raw,mask=next_raw,next_mask
        if terminated or truncated:
            episodes+=1
            raw,_,mask=env.reset(seed=seed+episodes)
    batch={key:np.stack([row[key] for row in rows]) for key in rows[0]}
    if batch["observation"].shape!=(transitions,6,ACTOR_POLICY_DIM):
        raise AssertionError("Unexpected actor observation shape")
    actor_before=next(model.actors.parameters()).detach().clone()
    metrics=model.update(batch)
    if model.updates<1 or not np.isfinite(batch["reward"]).all():
        raise AssertionError("Learner failed to perform an update")
    return {
        "algorithm":algorithm,
        "seed":seed,
        "transitions":len(rows),
        "episodes_reset":episodes,
        "actor_obs_dim":ACTOR_POLICY_DIM,
        "joint_critic_dim":6*ACTOR_POLICY_DIM,
        "gradient_update_count":model.updates,
        "actor_weight_unchanged_expected_if_delayed":bool(
            torch.equal(actor_before,next(model.actors.parameters()).detach())
        ),
        "mean_environment_reward":float(batch["reward"].mean()),
        "finite_report":all(
            np.isfinite(float(v)) for v in metrics.values()
            if isinstance(v,(int,float,np.number))
        )
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=44)
    p.add_argument("--transitions",type=int,default=4)
    p.add_argument("--json-out",type=Path,default=None)
    args=p.parse_args()
    data={a:integration_gate(a,seed=args.seed,transitions=args.transitions)
          for a in ("matd3","masac")}
    text=json.dumps(data,indent=2,sort_keys=True)
    if args.json_out is not None:
        destination=args.json_out.resolve()
        root=Path(__file__).resolve().parents[3]
        allowed=(root/"outputs").resolve()
        if not destination.is_relative_to(allowed):
            p.error("Output must be in gitignored project outputs/")
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(text+"\n",encoding="utf-8")
    print(text)


if __name__=="__main__":
    main()
