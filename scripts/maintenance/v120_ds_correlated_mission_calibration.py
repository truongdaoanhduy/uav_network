"""V121 correction of the V120 CPU DS sensitivity experiment.

V120 passed unit vectors to a CPU velocity-in-m/s API (1m/s, not15m/s).
Historical V120 outputs are retained, not reused as corrected evidence.
Requested horizon is not necessarily observed horizon: report early termination.

Measures full-mission GCS delivery, false positives (ever/active), belief drops,
confirmation latency (AoI), and Brier score across:
  1. Correlated Markov 2-state sensing (rho in {0.0, 0.5, 0.7}) x fusion mode (sum vs zhang_max)
  2. Confirmation threshold sweep (confirm_th in {0.90, 0.95, 0.99}) at rho=0.0 and rho=0.7
  3. Evidence mass calibration (paper_fixed vs sensor_likelihood_ratio capped/uncapped)

Does NOT modify any core environment, Warp kernel, YAML, or algorithm file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from zhang_xue.belief.dempster_shafer import fuse_raw_provenance_counts
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv


def _load_cfg(yaml_path: str, **overrides: Any) -> EnvConfigV54:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.update(overrides)
    return EnvConfigV54(**data)


def _sector_patrol_xy(env: ZhangXue7NodePacketEnv, step_idx: int) -> np.ndarray:
    """Deterministic non-oracle sector-sweep patrol that visits all 6 strips repeatedly."""
    n = env.cfg.num_uavs
    pos = env.positions[:n, :2]
    # Sweep back-and-forth along Y while shifting X sub-lanes to cover the 1km grid
    strip_w = 880.0 / n
    lane_phase = (step_idx // 120) % 2
    x_centers = 60.0 + (np.arange(n) + 0.25 + 0.5 * lane_phase) * strip_w
    y_phase = (step_idx // 60) % 2
    y_goal = np.where((np.arange(n) + y_phase) % 2 == 0, 900.0, 100.0)
    goals = np.stack([x_centers, y_goal], axis=1)
    diff = goals - pos
    norm = np.linalg.norm(diff, axis=1, keepdims=True)
    cmd = np.where(norm > 1e-6, diff / np.maximum(norm, 1.0), 0.0)
    # CPU env.step expects m/s; Warp/actors use normalized XY instead.
    return (np.clip(cmd, -1.0, 1.0) * env.cfg.v_max_mps).astype(np.float32)


def _attach_markov_sensor_hook(env: ZhangXue7NodePacketEnv, rho: float, seed: int) -> None:
    """Wrap env.net.record_local_observation with stationary 2-state Markov correlation."""
    if rho <= 0.0:
        return
    orig_fn = env.net.record_local_observation
    prev_det: Dict[Tuple[int, int], bool] = {}
    corr_rng = np.random.default_rng(seed + 100_003)
    pd = float(env.cfg.pd)
    pf = float(env.cfg.pf)

    def _correlated_record(u: int, cid: int, det: bool, t_step_s: float) -> None:
        key = (int(u), int(cid))
        p = pd if bool(env.true_targets[cid]) else pf
        if key not in prev_det:
            new_det = bool(det)
        else:
            if prev_det[key]:
                prob_1 = p + rho * (1.0 - p)
            else:
                prob_1 = (1.0 - rho) * p
            new_det = bool(corr_rng.random() < prob_1)
        prev_det[key] = new_det
        orig_fn(u, cid, new_det, t_step_s)

    env.net.record_local_observation = _correlated_record  # type: ignore[method-assign]


def _compute_gcs_brier(env: ZhangXue7NodePacketEnv) -> Tuple[float | None, int]:
    """Legacy name: final observed-cell squared mass error; undefined if unseen."""
    gcs_id = env.net.gcs_id
    sq_errs: List[float] = []
    for cid in range(env.num_cells):
        pos_v = env.net.ledger_pos[gcs_id, : env.actor_n, cid]
        neg_v = env.net.ledger_neg[gcs_id, : env.actor_n, cid]
        ts_v = env.net.ledger_gen_ts_s[gcs_id, : env.actor_n, cid]
        if not (np.any(pos_v) or np.any(neg_v)):
            continue
        m_t, _, _ = fuse_raw_provenance_counts(
            pos_v, neg_v, ts_v, mf=env.evidence_mf, me=env.evidence_me, mode=env.cfg.ds_fusion_mode
        )
        y = 1.0 if bool(env.true_targets[cid]) else 0.0
        sq_errs.append((float(m_t) - y) ** 2)
    return (float(np.mean(sq_errs)) if sq_errs else None, len(sq_errs))


def _run_single(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_cfg(task["yaml_path"], **task["overrides"])
    env = ZhangXue7NodePacketEnv(cfg)
    env.reset(seed=int(task["seed"]))
    _attach_markov_sensor_hook(env, float(task["rho"]), int(task["seed"]))
    info: Dict[str, Any] = {}
    steps = int(task["steps"])
    term = trunc = False
    steps_run = 0
    trace_rows = []
    for t in range(steps):
        act_xy = _sector_patrol_xy(env, t)
        act_comm = np.ones((cfg.num_uavs,), dtype=np.int32)
        _, _, _, reward, term, trunc, info = env.step(act_xy, act_comm)
        steps_run = t + 1
        if task.get('trace_dir'):
            trace_rows.append(dict(step=steps_run, positions_xy=env.positions[:6,:2].tolist(),
                                   battery_j=env.battery_j.tolist(), reward=float(reward),
                                   metrics={k:info.get(k) for k in ['gcs_tp_confirmed','gcs_tp_active_confirmed',
                                   'gcs_fp_ever_confirmed','gcs_fp_latched','delivered_cell_coverage_pct',
                                   'safety_violation','termination_reason']},
                                   data_enqueued=int(env.net.stats.get('data_enqueued',0))))
        if term or trunc:
            break
    brier, obs_cells = _compute_gcs_brier(env)
    trace_path = None
    if task.get('trace_dir'):
        trace_path = str(Path(task['trace_dir']) / f"{task['group']}_{task['arm']}_rho{task['rho']}_s{task['seed']}.jsonl")
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text(''.join(json.dumps(r)+'\n' for r in trace_rows),encoding='utf-8')
    return {
        "trace_path": trace_path,
        "group": task["group"],
        "arm": task["arm"],
        "seed": int(task["seed"]),
        "scenario_sha256": hashlib.sha256(Path(task['yaml_path']).read_bytes()).hexdigest(),
        "scenario_overrides": task['overrides'],
        "rho": float(task["rho"]),
        "requested_steps": steps,
        "steps_run": steps_run,
        "action_units": "metres_per_second",
        "terminated": bool(term),
        "truncated": bool(trunc),
        "termination_reason": info.get("termination_reason"),
        "safety_violation": bool(info.get("safety_violation", False)),
        "gcs_tp_ever": int(info.get("gcs_tp_confirmed", 0)),
        "gcs_tp_active": int(info.get("gcs_tp_active_confirmed", 0)),
        "gcs_tp_belief_drops": int(info.get("gcs_tp_belief_drops", 0)),
        "gcs_fp_ever": int(info.get("gcs_fp_ever_confirmed", 0)),
        "gcs_fp_active": int(info.get("gcs_fp_latched", 0)),
        "delivered_cov_pct": float(info.get("delivered_cell_coverage_pct", 0.0)),
        "mean_first_confirm_s": float(info.get("mean_first_tp_confirm_latency_s", -1.0)),
        "gcs_mT_squared_error": round(brier, 6) if brier is not None else None,
        "gcs_observed_cells": obs_cells,
        "evidence_mf": round(float(env.evidence_mf), 6),
        "evidence_me": round(float(env.evidence_me), 6),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault((r["group"], r["arm"], r["rho"]), []).append(r)
    summary = []
    for (group, arm, rho), grp in sorted(by_key.items()):
        lat = [x["mean_first_confirm_s"] for x in grp if x["mean_first_confirm_s"] >= 0.0]
        errors = [x['gcs_mT_squared_error'] for x in grp if x['gcs_mT_squared_error'] is not None]
        summary.append(
            {
                "group": group,
                "arm": arm,
                "rho": rho,
                "n_seeds": len(grp),
                "mean_gcs_tp_ever": round(float(np.mean([x["gcs_tp_ever"] for x in grp])), 3),
                "mean_gcs_tp_active": round(float(np.mean([x["gcs_tp_active"] for x in grp])), 3),
                "mean_gcs_tp_belief_drops": round(float(np.mean([x["gcs_tp_belief_drops"] for x in grp])), 3),
                "mean_gcs_fp_ever": round(float(np.mean([x["gcs_fp_ever"] for x in grp])), 3),
                "mean_gcs_fp_active": round(float(np.mean([x["gcs_fp_active"] for x in grp])), 3),
                "mean_delivered_cov_pct": round(float(np.mean([x["delivered_cov_pct"] for x in grp])), 2),
                "mean_first_confirm_s": round(float(np.mean(lat)), 3) if lat else -1.0,
                "mean_gcs_mT_squared_error": round(float(np.mean(errors)), 6) if errors else None,
                "min_actual_steps": min(x["steps_run"] for x in grp),
                "early_terminated_runs": sum(x["terminated"] and x["steps_run"] < x["requested_steps"] for x in grp),
                "evidence_mf": grp[0]["evidence_mf"],
                "evidence_me": grp[0]["evidence_me"],
            }
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-yaml", default="research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46, 47, 48])
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--output", default="outputs/v121_scenario_audit/ds_sensitivity_corrected.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    source_files = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root/'research/zhang_xue/src').rglob('*.py')}
    source_files[str(Path(__file__).resolve().relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    tasks: List[Dict[str, Any]] = []
    # Group 1 (C04/C41/C66): Correlated Markov sensor rho x fusion mode (sum vs zhang_max)
    for rho in (0.0, 0.5, 0.7):
        for fmode in ("sum", "zhang_max"):
            for s in args.seeds:
                tasks.append(
                    {
                        "group": "c04_c41_c66_corr_fusion",
                        "arm": f"fusion_{fmode}",
                        "yaml_path": args.base_yaml,
                        "rho": rho,
                        "seed": s,
                        "steps": args.steps,
                        "overrides": {"ds_fusion_mode": fmode},
                    }
                )
    # Group 2 (C05/C35): confirm_th in {0.90, 0.95, 0.99} at rho=0.0 and rho=0.7
    for rho in (0.0, 0.7):
        for th in (0.90, 0.95, 0.99):
            for s in args.seeds:
                tasks.append(
                    {
                        "group": "c05_c35_threshold_sweep",
                        "arm": f"confirm_th_{th:.2f}",
                        "yaml_path": args.base_yaml,
                        "rho": rho,
                        "seed": s,
                        "steps": args.steps,
                        "overrides": {"confirm_th": th},
                    }
                )
    # Group 3 (C23): evidence_calibration paper_fixed vs sensor_likelihood_ratio (cap 0.80 vs 0.99)
    cal_arms = [
        ("paper_fixed_0.80", {"evidence_calibration": "paper_fixed"}),
        ("sensor_lr_cap_0.80", {"evidence_calibration": "sensor_likelihood_ratio", "bba_reliability_cap": 0.80}),
        ("sensor_lr_cap_0.99", {"evidence_calibration": "sensor_likelihood_ratio", "bba_reliability_cap": 0.99}),
    ]
    for arm_name, ov in cal_arms:
        for s in args.seeds:
            tasks.append(
                {
                    "group": "c23_mass_calibration",
                    "arm": arm_name,
                    "yaml_path": args.base_yaml,
                    "rho": 0.0,
                    "seed": s,
                    "steps": args.steps,
                    "overrides": ov,
                }
            )

    for task in tasks:
        task['trace_dir'] = args.trace_dir
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        raw_rows = list(ex.map(_run_single, tasks))

    report = {
        "revision": "V121_CPU_ACTION_UNITS_CORRECTED",
        "source_files_sha256": source_files,
        "source_sha256": hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest(),
        "trace_kind": "per-step mission state, not per-packet MAC event trace",
        "communication_controller": "always GCS_SEND request; legal routing/dirty selection remains in environment",
        "action_units": "metres_per_second",
        "mass_error_note": "Squared error of DS target mass; mT is a lower belief, not a calibrated predictive probability/Brier score.",
        "claim_limit": "Finite-seed synthetic Markov sensitivity, not sensor calibration or physical rho measurement. No auto-reset.",
        "script": "v120_ds_correlated_mission_calibration.py",
        "base_yaml": args.base_yaml,
        "seeds": args.seeds,
        "steps": args.steps,
        "summary": _aggregate(raw_rows),
        "raw_runs": raw_rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved V120 DS calibration ({len(raw_rows)} runs) -> {out_path}")


if __name__ == "__main__":
    main()
