#!/usr/bin/env python3
"""Scenario-only mission audit with first-terminal metrics and paired counts.

The controller sees local observations only. Truth/ledgers are read separately
by the evaluator. Complete execution is not a proof of full mission success.
No claim that one inference backend must outperform another is an acceptance gate.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]
from v93_probe_metrics import TerminalSnapshots, compare_counts


def source_digest():
    h = hashlib.sha256()
    for folder in ("research/zhang_xue/src", "scripts/gpu"):
        for p in sorted((ROOT/folder).rglob("*.py")):
            h.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def run_mission(scenario, *, envs, steps, seed, device, trace_path=None, controller="frontier"):
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
    from v92_non_oracle_mission_probe import non_oracle_frontier_actions_np

    p = GPUWorldV80Params.from_yaml(scenario, envs=envs, device=device, seed=seed)
    if not 1 <= steps <= p.horizon:
        raise ValueError("steps must be within the configured horizon")
    w = WarpParallelMissionV80(p)
    snapshots = TerminalSnapshots(envs)
    truth = w.buffers["truth"].numpy().copy()
    ret = np.zeros(envs, dtype=np.float64)
    intervention = np.zeros(envs, dtype=np.float64)
    reward_error = 0.
    max_nonfinite = 0
    actions = wp.zeros((envs,6,2), dtype=wp.float32, device=device)
    dests = wp.zeros((envs,6), dtype=wp.int32, device=device)
    start = time.monotonic()

    def read_snapshot():
        count_pos = w.network.data["ledger_pos"].numpy().astype(np.int64)
        count_neg = w.network.data["ledger_neg"].numpy().astype(np.int64)
        own_pos = count_pos[:,np.arange(6),np.arange(6),:]
        own_neg = count_neg[:,np.arange(6),np.arange(6),:]
        return {
            "team_pos": own_pos.sum(axis=1), "team_neg": own_neg.sum(axis=1),
            "gcs_pos": count_pos[:,6,:,:].sum(axis=1),
            "gcs_neg": count_neg[:,6,:,:].sum(axis=1),
            **{key: w.buffers[key].numpy().copy() for key in
               ("gcs_tp","gcs_fp","gcs_fp_ever","termination_reason","episode_step","failed","battery",
                "positions","velocities","candidates","v_safe","safety_reason")},
            "overflow": w.network.data["overflow"].numpy().copy(),
            "episode_return": ret.copy(), "total_intervention": intervention.copy(),
            "previous_positions": previous_positions.copy(),
            "previous_velocities": previous_velocities.copy(),
            "raw_continuous": cont.copy(), "raw_discrete": dest.copy(),
        }

    for step in range(1, steps+1):
        active = ~snapshots.seen
        if not active.any():
            break
        obs, mask = w.buffers["obs"].numpy(), w.buffers["mask"].numpy()
        previous_positions = w.buffers["positions"].numpy().copy()
        previous_velocities = w.buffers["velocities"].numpy().copy()
        cont, dest = non_oracle_frontier_actions_np(obs, mask, v_max=p.v_max_mps)
        cont[~active], dest[~active] = 0., 0
        wp.copy(actions, wp.array(cont, dtype=wp.float32, device=device))
        wp.copy(dests, wp.array(dest, dtype=wp.int32, device=device))
        w.step(actions, dests)
        rewards = w.buffers["reward"].numpy()
        comps = w.buffers["reward_components"].numpy()
        ret[active] += rewards[active]
        intervention[active] += w.buffers["intervention"].numpy()[active]
        reward_error = max(reward_error, float(np.max(np.abs(comps[active].sum(axis=1)-rewards[active]))))
        max_nonfinite += int(np.count_nonzero(~np.isfinite(comps[active])))
        ended = w.buffers["done"].numpy().astype(bool) & active
        if ended.any():
            snapshots.capture(ended, read_snapshot())
    # A deliberately short audit retains a labelled partial snapshot as well.
    partial = ~snapshots.seen.copy()
    snapshots.capture(partial, read_snapshot())
    wp.synchronize_device(device)
    elapsed = time.monotonic()-start
    s = snapshots.values
    legal = np.zeros(400, dtype=bool)
    legal[w.legal.numpy().astype(int)] = True
    calibration = {}
    for where in ("team","gcs"):
        c = compare_counts(s[where+"_pos"][:,legal], s[where+"_neg"][:,legal], truth[:,legal],
                           prior=p.bayes_prior_occupancy,pd=p.pd,pf=p.pf,mf=p.ds_mf,me=p.ds_me)
        c.pop("bayes_posterior")
        calibration[where] = c
    seen = s["team_pos"] + s["team_neg"] > 0
    tp = s["gcs_tp"].sum(axis=1)
    reasons = s["termination_reason"]
    valid = int(np.count_nonzero(s["failed"])) == 0 and int(np.count_nonzero(s["overflow"])) == 0 and max_nonfinite == 0 and reward_error < 1e-4
    result = {
        "status": "PROBE_VALID" if valid else "PROBE_INVALID",
        "scenario_frozen": False, "controller": controller,
        "controller_uses_local_obs_only": True,
        "config": asdict(p), "scenario": str(scenario),
        "scenario_sha256": hashlib.sha256(Path(scenario).read_bytes()).hexdigest(),
        "source_digest": source_digest(), "steps_requested": steps,
        "elapsed_sec": round(elapsed,3), "partial_worlds": int(partial.sum()),
        "first_terminal_steps": s["episode_step"].tolist(),
        "full15_completed_worlds": int(np.count_nonzero(reasons & 4)),
        "mean_gcs_tp": float(tp.mean()), "mean_gcs_delivery_pct": float(tp.mean()/15*100),
        "mean_team_coverage_all400_pct": float(seen.mean()*100),
        "mean_team_coverage_legal_pct": float(seen[:,legal].mean()*100),
        "mean_fp_active": float(s["gcs_fp"].sum(axis=1).mean()),
        "mean_fp_ever": float(s["gcs_fp_ever"].sum(axis=1).mean()),
        "safety_failed_worlds": int(np.count_nonzero(s["failed"])),
        "overflow_worlds": int(np.count_nonzero(s["overflow"])),
        "mean_return": float(s["episode_return"].mean()),
        "mean_total_intervention": float(s["total_intervention"].mean()),
        "reward_sum_error": reward_error, "nonfinite_reward_terms": max_nonfinite,
        "paired_same_count_calibration": calibration,
        "calibration_scope": "visited legal cells, clustered by world; descriptive scores, not iid significance tests",
        "belief_comparison_scope": "paired estimator scores use identical terminal counts; separate mission runs may follow different trajectories/rewards",
    }
    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, truth=truth, legal=legal, partial=partial, **s)
        result["trace_file"] = trace_path.name
        result["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    return result


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    a.add_argument("--scenario",type=Path,default=ROOT/"research/zhang_xue/configs/scenario_v91_bayes_1km.yaml")
    a.add_argument("--envs",type=int,default=16)
    a.add_argument("--steps",type=int,default=900)
    seeds = a.add_mutually_exclusive_group()
    seeds.add_argument("--seed",type=int)
    seeds.add_argument("--seeds",default=None)
    a.add_argument("--device",default="cuda:0")
    a.add_argument("--output","--out",type=Path,required=True)
    args = a.parse_args(argv)
    chosen = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed if args.seed is not None else 44]
    runs = [run_mission(args.scenario,envs=args.envs,steps=args.steps,seed=seed,device=args.device,
                        trace_path=args.output.with_name(args.output.stem+f"_seed{seed}_trace.npz")) for seed in chosen]
    out = {"status":"PROBE_VALID" if all(r["status"]=="PROBE_VALID" for r in runs) else "PROBE_INVALID",
           "scenario_frozen":False,"runs":runs}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps(out,indent=2,allow_nan=False))
    return 0 if out["status"]=="PROBE_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
