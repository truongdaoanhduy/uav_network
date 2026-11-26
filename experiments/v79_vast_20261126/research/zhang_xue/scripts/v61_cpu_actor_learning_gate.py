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

from uav_network.algorithms.configuration import LearnerConfig, load_learner_config
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.masac import MASAC
from zhang_xue.environment.configuration import EnvConfigV54, make_v57_config
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.learners.actor_adapter import (
    ACTOR_POLICY_DIM, encode_actor_input, make_critic_state, step_from_policy_action,
    encode_actor_input_scalable, make_critic_state_scalable,
)


def integration_gate(
    algorithm: str, *, seed=44, transitions=4,
    learner_config: LearnerConfig | None = None,
    scenario_config: EnvConfigV54 | None = None,
):
    if algorithm not in ("matd3","masac") or not (1<=transitions<=12):
        raise ValueError("Expected matd3/masac and 1..12 short transitions")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    import copy
    cfg=copy.deepcopy(learner_config) if learner_config is not None else LearnerConfig()
    cfg.runtime.algorithm=algorithm
    if learner_config is None:
        cfg.matd3.hidden_sizes=[32,32]
        cfg.masac.hidden_sizes=[32,32]
    cfg.validate()
    env=ZhangXue7NodePacketEnv(scenario_config or make_v57_config(),seed=seed)
    if bool(env.cfg.enable_tx_power_control) != (cfg.network.power_control == "learned"):
        raise ValueError("Learner TX power_control conflicts with scenario enable_tx_power_control")
    if cfg.world.num_uavs != env.actor_n or cfg.network.destination_mode != env.cfg.comm_action_mode:
        raise ValueError("Learner UAV count / communication action space conflicts with scenario")
    raw,_,mask=env.reset(seed=seed)
    use_scalable = env.actor_n != 6 or env.disc_dim != 7
    encode = encode_actor_input_scalable if use_scalable else encode_actor_input
    make_state = make_critic_state_scalable if use_scalable else make_critic_state
    actor_obs_dim = int(encode(raw,mask).shape[-1])
    model={"matd3":MATD3,"masac":MASAC}[algorithm](
        cfg, actor_obs_dim, env.actor_n*actor_obs_dim, device="cpu"
    )
    episodes=0
    rows=[]
    tx_power_samples=[]
    while len(rows)<transitions:
        encoded=encode(raw,mask)
        state=make_state(encoded)
        act=model.act(encoded,deterministic=True)
        next_raw,_,next_mask,reward,terminated,truncated,info = step_from_policy_action(
            env,act,mask,power_control=cfg.network.power_control,
        )
        tx_power_samples.append(float(np.mean(env.net.tx_power_per_radio[:env.actor_n])))
        next_encoded=encode(next_raw,next_mask)
        rows.append({
            "observation":encoded,
            "next_observation":next_encoded,
            "state":state,
            "next_state":make_state(next_encoded),
            "continuous":np.asarray(act.continuous,dtype=np.float32),
            "destination":np.asarray(act.destination,dtype=np.int64),
            "reward":np.float32(reward),
            "terminated":np.float32(terminated),
        })
        raw,mask=next_raw,next_mask
        if terminated or truncated:
            episodes+=1
            raw,_,mask=env.reset(seed=seed+episodes)
    batch={key:np.stack([row[key] for row in rows]) for key in rows[0]}
    if batch["observation"].shape!=(transitions,env.actor_n,actor_obs_dim):
        raise AssertionError("Unexpected actor observation shape")
    actor_before=next(model.actors.parameters()).detach().clone()
    metrics=model.update(batch)
    if model.updates<1 or not np.isfinite(batch["reward"]).all():
        raise AssertionError("Learner failed to perform an update")
    return {
        "algorithm":algorithm,
        "power_control":cfg.network.power_control,
        "mean_actor_tx_power_w":float(np.mean(tx_power_samples)),
        "seed":seed,
        "transitions":len(rows),
        "episodes_reset":episodes,
        "actor_obs_dim":actor_obs_dim,
        "joint_critic_dim":env.actor_n*actor_obs_dim,
        "continuous_dim":cfg.continuous_dim,
        "destinations":model.destinations,
        "num_uavs":env.actor_n,
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
    p.add_argument("--learner-config",type=Path)
    p.add_argument("--scenario-config",type=Path)
    p.add_argument("--learner-set",action="append",default=[],metavar="SECTION.FIELD=VALUE")
    p.add_argument("--scenario-set",action="append",default=[],metavar="FIELD=VALUE")
    p.add_argument("--algorithm",choices=("matd3","masac","both"),default="both")
    args=p.parse_args()
    use_config = bool(args.learner_config or args.learner_set)
    learner_cfg = (load_learner_config(args.learner_config,overrides=args.learner_set)
                   if use_config else None)
    scenario_cfg = (load_scenario(args.scenario_config,tuple(args.scenario_set))
                    if args.scenario_config or args.scenario_set else None)
    algorithms = ("matd3","masac") if args.algorithm=="both" else (args.algorithm,)
    data={a:integration_gate(a,seed=args.seed,transitions=args.transitions,
                             learner_config=learner_cfg,scenario_config=scenario_cfg)
          for a in algorithms}
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
