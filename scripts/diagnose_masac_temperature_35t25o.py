"""Paired MASAC temperature update diagnostic, same 35T25O observations/batches.

This is a small CPU *mechanistic* reproducibility test, NOT full 20k-episode
training or evidence for final delivery. Changes only temperature_loss.
Exact MASAC.update() executes for both conditions, with identical random
initial weights and same replay batch indices/random-generator seeds.

No Kaggle, WANDB, HuggingFace API, network, or GPU required.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.algorithms.masac import MASAC


def make_config(path, hidden):
    c=load_config(path)
    c.runtime.algorithm="masac"
    c.pql.enabled=False
    c.runtime.num_envs=1
    c.runtime.device="cpu"
    c.runtime.replay_device="cpu"
    c.runtime.critic_chunk_size=0
    c.runtime.torch_threads=1
    c.masac.share_actor_parameters=True
    c.masac.share_critic_parameters=True
    c.masac.actor_layer_norm=True
    c.masac.hidden_sizes=[hidden,hidden]
    c.masac.mask_aware_discrete_entropy_target=True
    c.masac.temperature_loss="log_alpha"
    c.masac.initial_alpha=1.0
    c.masac.target_continuous_entropy=-4.0
    c.masac.target_discrete_entropy_ratio=0.2
    c.masac.grad_clip_mode="none"
    return c.validate()


def make_real_warp_replay(c, steps):
    vec=WarpVectorEnv(c,device="cpu")
    rows=[]
    try:
        obs,_=vec.reset(seed=c.runtime.seed)
        for j in range(steps):
            act=vec.random_actions(obs["observation"])
            next_obs,reward,terminated,truncated,_=vec.step(
                {"continuous":act.continuous,"destination":act.destination})
            rows.append({
                "observation":obs["observation"][0].detach().clone(),
                "state":obs["state"][0].detach().clone(),
                "continuous":act.continuous[0].detach().clone(),
                "destination":act.destination[0].detach().clone(),
                "reward":reward[0].detach().clone(),
                "next_observation":next_obs["observation"][0].detach().clone(),
                "next_state":next_obs["state"][0].detach().clone(),
                "terminated":terminated[0].detach().clone(),
            })
            if bool((terminated|truncated).any()):
                obs,_=vec.reset(seed=c.runtime.seed+j+1)
            else:
                obs=next_obs
        dataset={k:torch.stack([r[k] for r in rows],0) for k in rows[0]}
        eligible=(dataset["observation"][..., -c.world.num_uavs-1:] > .5).sum(-1)>1
        replay={"steps":steps,"eligible_agent_state_fraction":float(eligible.float().mean().item()),
                "max_legal_destinations":int((dataset["observation"][...,-c.world.num_uavs-1:]>0.5).sum(-1).max().item()),
                "reward_mean":float(dataset["reward"].float().mean().item()),
                "terminal_fraction":float(dataset["terminated"].float().mean().item())}
        return dataset,replay
    finally:
        vec.close()


@torch.no_grad()
def action_snapshot(model,heldout):
    action=model.act_batch(heldout,deterministic=True,return_tensors=True)
    continuous=action.continuous.float()
    return {"motion_mean_abs":float(continuous[...,:3].abs().mean().item()),
            "motion_saturation_fraction":float((continuous[...,:3].abs()>0.95).float().mean().item()),
            "motion_component_std":float(continuous[...,:3].std().item()),
            "rf_nonzero_action_fraction":float((action.destination!=0).float().mean().item())},continuous.clone()


def run(cfg, steps, updates, batch_size, quiet=False):
    torch.set_num_threads(1)
    torch.manual_seed(cfg.runtime.seed)
    dataset,ds_meta=make_real_warp_replay(cfg,steps)
    models={}
    cfg.masac.temperature_loss="log_alpha"
    model_log=MASAC(cfg,observation_dim(cfg),state_dim(cfg),device="cpu")
    cfg.masac.temperature_loss="alpha"
    model_alpha=MASAC(cfg,observation_dim(cfg),state_dim(cfg),device="cpu")
    model_alpha.load_state_dict(model_log.state_dict(),strict=True)
    initial_weights=[p.detach().clone() for p in model_log.parameters()]
    assert all(torch.equal(x,y) for x,y in zip(initial_weights,model_alpha.parameters()))
    models={"log_alpha":model_log,"alpha":model_alpha}
    heldout=dataset["observation"][-min(64,steps):].clone()
    initial_action={mode:action_snapshot(m,heldout)[1] for mode,m in models.items()}
    assert torch.equal(initial_action["log_alpha"],initial_action["alpha"])
    gen=torch.Generator().manual_seed(900044)
    history=[]
    begin=time.monotonic()
    keep={0,1,2,3,4,5,10,20,30,50,75,100,150,200,300,500,updates}
    for it in range(1,updates+1):
        idx=torch.randint(0,steps,(batch_size,),generator=gen)
        batch={name:data[idx] for name,data in dataset.items()}
        metrics_by_mode={}
        for mode,model in models.items():
            torch.manual_seed(441000+it) # same policy sample noise per optimizer iteration
            metrics=model.update(batch,collect_metrics=it in keep)
            if it in keep:
                snap,act=action_snapshot(model,heldout)
                snap["mean_action_l2_change_from_initial"]=float(
                    (act-initial_action[mode]).square().mean().sqrt().item())
                metrics_by_mode[mode]={**snap, **{key:round(float(metrics[key]),8) for key in (
                    "alpha_continuous","alpha_discrete","continuous_entropy",
                    "discrete_entropy","actor_loss","critic_loss",
                    "actor_grad_norm","critic_grad_norm","target_mean")}}
        if it in keep:
            record={"update":it,"metrics":metrics_by_mode}
            history.append(record)
            if not quiet:
                print("UPDATE",it,"log_alpha",metrics_by_mode["log_alpha"],
                      "alpha",metrics_by_mode["alpha"],flush=True)
    final_mse=float((action_snapshot(model_log,heldout)[1]
                   -action_snapshot(model_alpha,heldout)[1]).square().mean().sqrt().item())
    return {
       "scope":"paired replay-only short CPU mechanistic stress-test; NOT full 35T25O train/eval",
       "same_35t25o_environment":True,
       "same_code":True,
       "exact_one_variable_changed":"masac.temperature_loss",
       "other_overrides":{"num_envs":1,"batch_size":batch_size,"hidden_sizes":cfg.masac.hidden_sizes,
                          "updates":updates,"warped_random_steps":steps,
                          "shared_actor_critic":True,"actor_layer_norm":True,
                          "mask_aware_discrete_entropy_target":True},
       "limits":"policy gradient uses common random numbers per step, but actors can diverge under different alpha; offline replay does not establish deployment returns or GPU/20k training effects",
       "real_warp_dataset":ds_meta,"wall_seconds":time.monotonic()-begin,
       "last_mean_action_difference_rms":final_mse,
       "history":history}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/comparison-independent-35t25o-h512-xyz-sensing.yaml")
    parser.add_argument("--steps",type=int,default=240)
    parser.add_argument("--updates",type=int,default=150)
    parser.add_argument("--batch",type=int,default=64)
    parser.add_argument("--hidden",type=int,default=64)
    parser.add_argument("--output",type=Path,
                        default=Path("docs/methods/evidence/masac_temp_35t25o_paired_cpu_2026-11-08.json"))
    args=parser.parse_args()
    if min(args.steps,args.updates,args.batch,args.hidden)<1:
        parser.error("all numeric settings must be positive")
    result=run(make_config(args.config,args.hidden),args.steps,args.updates,args.batch)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("SAVED",args.output,flush=True)


if __name__=="__main__":
    main()
