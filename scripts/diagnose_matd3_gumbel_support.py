"""Read-only trained-MATD3 actor support check on the exact Warp environment.

This is NOT a trained Q-value comparison: immutable best HF snapshots carry
only Actor parameters. An untrained/random Critic must never be substituted.

Output reports conditional differences only when >1 destination is valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from uav_network.experiments.evaluation_snapshots import load_evaluation_snapshot
from uav_network.environment.gpu.vector import WarpMissionEnv
from uav_network.algorithms.matd3 import MATD3


def one_sample(learner, obs, seed):
    """Return observed soft/hard action-support statistics at a real state."""
    with torch.no_grad():
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        _, probs = learner._policy_outputs(obs)
        torch.manual_seed(seed)
        soft = MATD3._relaxed_destination(probs)
        hard = F.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
        mask_count = (obs[..., -learner.destinations:] > .5).sum(-1)
        max_prob = probs.max(-1).values
        l1 = (soft - hard).abs().sum(-1)
        l2 = ((soft - hard)**2).sum(-1).sqrt()
        selected_mass = torch.gather(soft, -1, hard.argmax(-1)[..., None]).squeeze(-1)
        entropy = -(soft * torch.log(soft.clamp_min(1e-30))).sum(-1)
    return {
        "mask_count": mask_count[0].cpu().numpy(),
        "policy_max_prob": max_prob[0].cpu().numpy(),
        "soft_hard_l1": l1[0].cpu().numpy(),
        "soft_hard_l2": l2[0].cpu().numpy(),
        "soft_selected_mass": selected_mass[0].cpu().numpy(),
        "soft_entropy": entropy[0].cpu().numpy(),
    }


def audited_checkpoint(snapshot, case, base):
    r = next(x for x in snapshot["runs"] if x["algorithm"] == "matd3" and x["case"] == case)
    path = (base/"checkpoints"/
            f"uav-network/{r['id']}/best/checkpoint.pt")
    if not path.exists():
        raise FileNotFoundError(f"Read-only local checkpoint required: {path}")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != r["best_checkpoint_storage"]["sha256"]:
        raise ValueError("Immutable W&B checkpoint digest mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("purpose") != "evaluation_only":
        raise ValueError("Unexpected checkpoint type")
    if any("critic" in key.lower() for key in payload["actors"]):
        raise AssertionError("Unexpected critic in actor snapshot")
    return r, path, payload, checksum


def summarize_metrics(metrics):
    keys = ["policy_max_prob", "soft_hard_l1", "soft_hard_l2",
            "soft_selected_mass", "soft_entropy"]
    mask_count = np.concatenate([m["mask_count"] for m in metrics])
    eligible = mask_count > 1
    result = {
        "total_agent_states": int(len(mask_count)),
        "eligible_agent_states": int(eligible.sum()),
        "eligible_fraction": float(eligible.mean()),
        "forced_silent_fraction": float((mask_count == 1).mean()),
        "unique_valid_destination_counts": {
            str(int(x)): int((mask_count == x).sum()) for x in np.unique(mask_count)
        }
    }
    for key in keys:
        values = np.concatenate([m[key] for m in metrics])
        result[key] = {
            "all_mean": float(np.mean(values)),
            "eligible_mean": float(np.mean(values[eligible])) if eligible.any() else None,
            "eligible_median": float(np.median(values[eligible])) if eligible.any() else None,
            "eligible_p90": float(np.quantile(values[eligible], 0.9)) if eligible.any() else None,
            "eligible_max": float(np.max(values[eligible])) if eligible.any() else None,
        }
    return result


def run_case(snapshot, case, folder, seeds, max_steps):
    run, checkpoint, saved, checksum = audited_checkpoint(snapshot, case, folder)
    learner = load_evaluation_snapshot(saved)
    learner.actors.eval()
    torch.set_num_threads(1)
    records = []
    steps_by_seed = []
    for seed in seeds:
        env = WarpMissionEnv(learner.config)
        try:
            obs, _ = env.reset(seed=seed)
            t = 0
            for t in range(max_steps):
                records.append(one_sample(learner, obs, seed=int(seed) + t + 1))
                action = learner.act(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            steps_by_seed.append({"seed":int(seed),"steps":t+1,
                                  "environment_ended": bool(terminated or truncated)})
        finally:
            env.close()
    return {
        "run_id": run["id"], "case": case,
        "source_commit": run["source_git_commit"],
        "checkpoint_sha256_verified": checksum,
        "checkpoint_has_trained_critic": False,
        "trained_q_gap_measurable": False,
        "diagnostic_only": "observed Actor soft-Gumbel vs hard onehot on real deterministic on-policy Warp trajectories",
        "seeds": steps_by_seed,
        "stats": summarize_metrics(records),
        "caution": ("No trained critic in evaluation-only checkpoint. "
                    "The measured action-vector difference cannot be translated "
                    "into Q-error or evidence that changing MATD3 improves reward.")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default="docs/methods/evidence/wandb_latest_ten_2026-11-08.json")
    parser.add_argument("--folder", default="outputs/wandb-ten-20261108/diagnostics")
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000000044, 1000000045])
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    snapshot = json.loads(Path(args.snapshot).read_text())
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = [run_case(snapshot, case, folder, args.seeds, args.max_steps)
            for case in args.cases]
    report = {
        "sample_scope": {
            "backend": "warp_packet", "policy": "best saved evaluation Actor",
            "policy_rollout": "deterministic", "categorical_sampling": "Gumbel tau=1.0",
            "use_random_critic": False, "critic_available": False},
        "results":data,
    }
    out = folder/"trained_matd3_actor_support_diagnostic.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print("DIAGNOSTIC",out)
    for r in data:
        s=r["stats"]
        print("RUN",r["run_id"],"CASE",r["case"],
              "AGENT_STATES",s["total_agent_states"],"ELIGIBLE",s["eligible_agent_states"],
              "ELIGIBLE_FRACTION",round(s["eligible_fraction"],4),
              "ELIGIBLE_L1",s["soft_hard_l1"]["eligible_mean"],
              "FORCED_SILENT",round(s["forced_silent_fraction"],4),
              "SEEDS",r["seeds"],flush=True)


if __name__ == "__main__":
    main()
