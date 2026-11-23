"""Read-only factorial SAC policy mode check on validated best Actor checkpoints.

Separates the two SAC stochastic choices: continuous Gaussian and categorical destination.
Uses identical five validation layout seeds and WarpMissionEnv for each mode; this is a
diagnostic of *saved best Actors*, not a claim about intermediate training checkpoints.
No W&B/Kaggle/HF writes, no learner updates, no randomly initialized Critic evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from uav_network.algorithms.action_utils import canonical_continuous, valid_destinations
from uav_network.environment.env import Action
from uav_network.environment.gpu.vector import WarpMissionEnv
from uav_network.experiments.evaluation_snapshots import load_evaluation_snapshot


MODES = (
    ("mean_argmax", True, True),
    ("sample_gaussian_argmax", False, True),
    ("mean_sample_categorical", True, False),
    ("sample_both", False, False),
)


@torch.no_grad()
def policy_action(model, observation, continuous_deterministic, categorical_deterministic):
    obs = torch.as_tensor(observation, dtype=torch.float32)[None]
    continuous, _, probs, _ = model._sample_agents(obs, deterministic=continuous_deterministic)
    destination = (probs.argmax(-1) if categorical_deterministic
                   else Categorical(probs=probs, validate_args=False).sample())
    action = canonical_continuous(continuous, obs, destination, learn_power=model.learn_power)
    return Action(action[0].numpy(), destination[0].numpy().astype(np.int64)), probs[0].numpy()


def load_verified_actor(run, root):
    ckpt = root / f"uav-network/{run['id']}/best/checkpoint.pt"
    expected = run["best_checkpoint_storage"]["sha256"]
    actual = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Actor SHA mismatch for {run['id']}")
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    if saved.get("algorithm") != "masac" or saved.get("purpose") != "evaluation_only":
        raise ValueError("Not a trusted MASAC-only evaluation snapshot")
    return load_evaluation_snapshot(saved), actual


def evaluate_one(model, mode, seed):
    label, c_det, d_det = mode
    torch.manual_seed(seed + 123456)
    env = WarpMissionEnv(model.config)
    total_return, counts = 0.0, {"actions": 0, "eligible": 0,
                                 "eligible_send": 0, "active": 0,
                                 "nonsilent": 0}
    try:
        observation, _ = env.reset(seed=seed)
        for step in range(model.config.world.max_steps):
            obs_t = torch.as_tensor(observation, dtype=torch.float32)
            mask = valid_destinations(obs_t, model.destinations).numpy()
            eligible = mask.sum(-1) > 1
            action, _ = policy_action(model, observation, c_det, d_det)
            destination = action.destination
            counts["actions"] += len(destination)
            counts["eligible"] += int(eligible.sum())
            counts["eligible_send"] += int(((destination != 0) & eligible).sum())
            counts["active"] += int((observation[:, 9] > 0.5).sum())
            counts["nonsilent"] += int((destination != 0).sum())
            observation, reward, done, truncated, info = env.step(action)
            total_return += reward
            if done or truncated:
                break
        return {
            "mode": label, "seed": seed, "steps": step+1,
            "return": float(total_return),
            "search": float(info["search_rate_percent"]),
            "delivery": float(info["delivery_rate_percent"]),
            "targets_found": int(info["targets_found"]),
            "targets_delivered": int(info["targets_delivered"]),
            "eligible_fraction": counts["eligible"] / max(1, counts["actions"]),
            "send_given_eligible": counts["eligible_send"] / max(1, counts["eligible"]),
            "counts": counts,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["c70aa6f4","7f5fd73f"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1000000044,1000000049)))
    parser.add_argument("--checkpoint-root",type=Path,
                        default=Path("outputs/wandb-ten-20261108/diagnostics/checkpoints"))
    parser.add_argument("--evidence",type=Path,
                        default=Path("docs/methods/evidence/wandb_latest_ten_2026-11-08.json"))
    parser.add_argument("--out",type=Path,
                        default=Path("outputs/wandb-ten-20261108/diagnostics/masac_four_policy_modes_2026-11-08.json"))
    args=parser.parse_args()
    torch.set_num_threads(1)
    runs=json.loads(args.evidence.read_text())["runs"]
    results=[]
    for name in args.runs:
        run=next(x for x in runs if x["id"]==name and x["algorithm"]=="masac")
        model,digest=load_verified_actor(run,args.checkpoint_root)
        model.actors.eval()
        for mode in MODES:
            rr=[]
            for seed in args.seeds:
                row=evaluate_one(model,mode,seed)
                rr.append(row)
                print("CASE",name,"MODE",mode[0],"SEED",seed,
                      "SEARCH",round(row['search'],3),"DELIVERY",round(row['delivery'],3),
                      "ELIGIBLE",round(row['eligible_fraction'],4),flush=True)
            mean={k:float(np.mean([z[k] for z in rr])) for k in
                  ("search","delivery","return","eligible_fraction","send_given_eligible")}
            print("MEAN",name,mode[0],mean,flush=True)
            results.append({"run":name,"case":run["case"],"saved_actor_sha256":digest,
                            "mode":mode[0],"mean":mean,"episodes":rr})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({
        "protocol":"same-checkpoint 2x2 factorial: mean-vs-sampled Gaussian, argmax-vs-sampled Categorical",
        "environment":"warp_packet CPU, same validation seeds, frozen saved best Actor, no training",
        "limitations":"Best snapshots only; not comparable directly with intermediate checkpoint validation or training metrics",
        "runs":results},indent=2)+"\n")
    print("OUTPUT",args.out)


if __name__=="__main__":
    main()
