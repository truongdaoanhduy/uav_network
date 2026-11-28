"""V121 unit-corrected CPU RF/energy sensitivity (V120 outputs are historical).

Measures:
  - C08 / C12: Zeng model curve + idealized straight-line budget sanity estimate.
    This estimate ignores constraints and does NOT prove mission feasibility.
  - C12: enable_accel_energy False vs True sensitivity
  - C13: rf_per_building_loss_db in {0, 6, 12, 20} dB on pure-GCS Base and pure-GCS RF-noise
  - C28: network_substeps 1 vs 2 discretization sensitivity on pure-GCS Base and RF-noise
  - C24: All comm=0 (NO_APP_SEND) control-plane airtime (HELLO/ACK) vs DATA=0 verification

Does NOT modify any core environment, Warp kernel, YAML, or algorithm file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.mobility.flight_safety_energy import (
    ZengBatteryConfig,
    uav_step_energy_j,
    zeng_propulsion_power_w,
)


def _load_cfg(yaml_path: str, **overrides: Any) -> EnvConfigV54:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.update(overrides)
    return EnvConfigV54(**data)


def _sector_patrol_xy(env: ZhangXue7NodePacketEnv, step_idx: int) -> np.ndarray:
    n = env.cfg.num_uavs
    pos = env.positions[:n, :2]
    strip_w = 880.0 / n
    lane_phase = (step_idx // 120) % 2
    x_centers = 60.0 + (np.arange(n) + 0.25 + 0.5 * lane_phase) * strip_w
    y_phase = (step_idx // 60) % 2
    y_goal = np.where((np.arange(n) + y_phase) % 2 == 0, 900.0, 100.0)
    goals = np.stack([x_centers, y_goal], axis=1)
    diff = goals - pos
    norm = np.linalg.norm(diff, axis=1, keepdims=True)
    cmd = np.where(norm > 1e-6, diff / np.maximum(norm, 1.0), 0.0)
    # The CPU environment consumes velocity in m/s, not normalized controls.
    return (np.clip(cmd, -1.0, 1.0) * env.cfg.v_max_mps).astype(np.float32)


def _run_episode(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_cfg(task["yaml_path"], **task["overrides"])
    env = ZhangXue7NodePacketEnv(cfg)
    env.reset(seed=int(task["seed"]))
    steps = int(task["steps"])
    comm_val = int(task.get("comm_val", 1))
    pdr_hist: List[float] = []
    total_airtime_us = 0.0
    info: Dict[str, Any] = {}
    term = trunc = False
    steps_run = 0
    trace_rows = []
    for t in range(steps):
        act_xy = _sector_patrol_xy(env, t)
        act_comm = np.full((cfg.num_uavs,), comm_val, dtype=np.int32)
        _, _, _, reward, term, trunc, info = env.step(act_xy, act_comm)
        steps_run = t + 1
        pdr = float(info.get("pdr_gcs", -1.0))
        if pdr >= 0.0:
            pdr_hist.append(pdr)
        total_airtime_us += float(np.sum(env.net.step_tx_airtime_us))
        if task.get('trace_dir'):
            trace_rows.append(dict(step=steps_run, positions_xy=env.positions[:6,:2].tolist(),
                                   battery_j=env.battery_j.tolist(), reward=float(reward),
                                   airtime_us=float(np.sum(env.net.step_tx_airtime_us)),
                                   metrics={k:info.get(k) for k in ['gcs_tp_confirmed','gcs_fp_ever_confirmed',
                                   'delivered_cell_coverage_pct','pdr_gcs','safety_violation','termination_reason']},
                                   data_enqueued=int(env.net.stats.get('data_enqueued',0))))
        if term or trunc:
            break
    trace_path = None
    if task.get('trace_dir'):
        trace_path = str(Path(task['trace_dir']) / f"{task['group']}_{task['arm']}_{task['param']}_s{task['seed']}.jsonl")
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trace_path).write_text(''.join(json.dumps(r)+'\n' for r in trace_rows),encoding='utf-8')
    return {
        "trace_path": trace_path,
        "group": task["group"],
        "arm": task["arm"],
        "param": task["param"],
        "seed": int(task["seed"]),
        "scenario_sha256": hashlib.sha256(Path(task['yaml_path']).read_bytes()).hexdigest(),
        "scenario_overrides": task['overrides'],
        "requested_steps": steps,
        "steps_run": steps_run,
        "action_units": "metres_per_second",
        "terminated": bool(term),
        "truncated": bool(trunc),
        "termination_reason": info.get("termination_reason"),
        "gcs_tp_ever": int(info.get("gcs_tp_confirmed", 0)),
        "delivered_cov_pct": float(info.get("delivered_cell_coverage_pct", 0.0)),
        "mean_pdr_gcs": round(float(np.mean(pdr_hist)), 4) if pdr_hist else -1.0,
        "mean_first_confirm_s": round(float(info.get("mean_first_tp_confirm_latency_s", -1.0)), 4),
        "min_battery_pct": round(float(info.get("min_battery_pct", 0.0)), 2),
        "data_enqueued": int(env.net.stats.get("data_enqueued", 0)),
        "data_delivered_gcs": int(env.net.stats.get("data_delivered_gcs", 0)),
        "events_processed": int(env.net.stats.get("events_processed", 0)),
        "total_tx_airtime_ms": round(total_airtime_us / 1000.0, 3),
        "safety_violation": bool(info.get("safety_violation", False)),
    }


def compute_c08_c12_energy_upper_bound(
    yaml_path: str, seeds: List[int], cruise_mps: float = 10.0, dwell_s_per_target: float = 3.0
) -> Dict[str, Any]:
    """Legacy function name; this is a heuristic budget estimate, not a bound.

    Straight legs, fixed cruise and deterministic3s dwell ignore obstacles,
    acceleration, separation, random sensing and packet delivery. A low cost
    here cannot establish existence of a feasible physical mission.
    """
    cfg = _load_cfg(yaml_path)
    bcfg = ZengBatteryConfig(e_max_j=cfg.e_max_j)
    p_cruise_w = uav_step_energy_j(cruise_mps, 0.0, 1.0, bcfg)
    p_hover_w = uav_step_energy_j(0.0, 0.0, 1.0, bcfg)
    p_vmax_w = uav_step_energy_j(cfg.v_max_mps, 0.0, 1.0, bcfg)
    seed_bounds = []
    for s in seeds:
        env = ZhangXue7NodePacketEnv(cfg)
        env.reset(seed=s)
        uav_xy = env.positions[: cfg.num_uavs, :2].copy()
        tgt_idx = np.flatnonzero(env.true_targets)
        tgt_xy = np.stack(
            [(tgt_idx % env.grid_w + 0.5) * env.cell_size_m, (tgt_idx // env.grid_w + 0.5) * env.cell_size_m],
            axis=1,
        )
        rem = set(range(len(tgt_xy)))
        cur_xy = uav_xy.copy()
        dist_per_uav = np.zeros(cfg.num_uavs, dtype=np.float64)
        cnt_per_uav = np.zeros(cfg.num_uavs, dtype=np.int32)
        while rem:
            best = None
            for u in range(cfg.num_uavs):
                for j in rem:
                    d = float(np.linalg.norm(tgt_xy[j] - cur_xy[u]))
                    score = dist_per_uav[u] + d
                    if best is None or score < best[0]:
                        best = (score, u, j, d)
            _, u_b, j_b, d_b = best  # type: ignore[misc]
            dist_per_uav[u_b] += d_b
            cur_xy[u_b] = tgt_xy[j_b]
            cnt_per_uav[u_b] += 1
            rem.remove(j_b)
        flight_t = dist_per_uav / cruise_mps
        dwell_t = cnt_per_uav * dwell_s_per_target
        makespan_s = float(np.max(flight_t + dwell_t))
        max_uav_energy_j = float(np.max(flight_t * p_cruise_w + dwell_t * p_hover_w))
        seed_bounds.append(
            {
                "seed": s,
                "num_targets": int(len(tgt_xy)),
                "max_uav_distance_m": round(float(np.max(dist_per_uav)), 2),
                "total_fleet_distance_m": round(float(np.sum(dist_per_uav)), 2),
                "idealized_makespan_s": round(makespan_s, 2),
                "idealized_time_below_horizon": bool(makespan_s <= cfg.safety_horizon_s),
                "max_uav_energy_j": round(max_uav_energy_j, 1),
                "max_uav_energy_pct_of_180kJ": round(100.0 * max_uav_energy_j / cfg.e_max_j, 2),
                "idealized_energy_below_battery": bool(max_uav_energy_j <= cfg.e_max_j),
            }
        )
    return {
        "mission_feasibility_proven": False,
        "estimate_kind": "greedy_straight_line_budget_sanity_check_not_a_formal_bound",
        "ignored_constraints": ["obstacle_avoidance", "acceleration", "uav_separation", "stochastic_sensing", "network_delivery"],
        "zeng_curve_summary": {
            "e_max_j": cfg.e_max_j,
            "horizon_s": cfg.safety_horizon_s,
            "p_hover_total_w": round(p_hover_w, 2),
            "p_cruise_10mps_total_w": round(p_cruise_w, 2),
            "p_vmax_15mps_total_w": round(p_vmax_w, 2),
            "propulsion_only_10mps_w": round(zeng_propulsion_power_w(10.0, bcfg), 2),
            "endurance_10mps_s": round(cfg.e_max_j / p_cruise_w, 1),
            "endurance_15mps_s": round(cfg.e_max_j / p_vmax_w, 1),
            "endurance_hover_s": round(cfg.e_max_j / p_hover_w, 1),
        },
        "per_seed_15target_idealized_estimate": seed_bounds,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_k: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        by_k.setdefault((r["group"], r["arm"], str(r["param"])), []).append(r)
    out = []
    for (g, a, p), grp in sorted(by_k.items()):
        pdrs = [x["mean_pdr_gcs"] for x in grp if x["mean_pdr_gcs"] >= 0.0]
        out.append(
            {
                "group": g,
                "arm": a,
                "param": p,
                "n_seeds": len(grp),
                "mean_gcs_tp_ever": round(float(np.mean([x["gcs_tp_ever"] for x in grp])), 3),
                "mean_cov_pct": round(float(np.mean([x["delivered_cov_pct"] for x in grp])), 2),
                "mean_pdr_gcs": round(float(np.mean(pdrs)), 4) if pdrs else -1.0,
                "mean_min_battery_pct": round(float(np.mean([x["min_battery_pct"] for x in grp])), 2),
                "mean_data_enqueued": round(float(np.mean([x["data_enqueued"] for x in grp])), 1),
                "mean_tx_airtime_ms": round(float(np.mean([x["total_tx_airtime_ms"] for x in grp])), 2),
                "any_safety_violation": any(x["safety_violation"] for x in grp),
                "min_actual_steps": min(x["steps_run"] for x in grp),
                "early_terminated_runs": sum(x["terminated"] and x["steps_run"] < x["requested_steps"] for x in grp),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-yaml", default="research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml")
    ap.add_argument("--rf-yaml", default="research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46, 47, 48])
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--output", default="outputs/v121_scenario_audit/rf_energy_corrected.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    source_files = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (root/'research/zhang_xue/src').rglob('*.py')}
    source_files[str(Path(__file__).resolve().relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    energy_bound = compute_c08_c12_energy_upper_bound(args.base_yaml, [44, 45, 46, 47, 48])
    tasks: List[Dict[str, Any]] = []
    # C13: Building loss sweep on pure-GCS Base and pure-GCS RF-noise (no C20 confounding)
    for arm, yml, base_ov in [
        ("base_pure_gcs", args.base_yaml, {}),
        ("rf_noise_pure_gcs", args.rf_yaml, {"enable_peer_dissemination": False, "enable_data_mule_forwarding": False}),
    ]:
        for loss_db in (0.0, 6.0, 12.0, 20.0):
            for s in args.seeds:
                tasks.append(
                    {
                        "group": "c13_building_loss_db",
                        "arm": arm,
                        "param": loss_db,
                        "yaml_path": yml,
                        "seed": s,
                        "steps": args.steps,
                        "overrides": {**base_ov, "rf_per_building_loss_db": loss_db},
                    }
                )
        # C28: network_substeps 1 vs 2
        for substeps in (1, 2):
            for s in args.seeds:
                tasks.append(
                    {
                        "group": "c28_network_substeps",
                        "arm": arm,
                        "param": substeps,
                        "yaml_path": yml,
                        "seed": s,
                        "steps": args.steps,
                        "overrides": {**base_ov, "network_substeps": substeps},
                    }
                )
    # C12: enable_accel_energy False vs True
    for accel in (False, True):
        for s in args.seeds:
            tasks.append(
                {
                    "group": "c12_accel_energy",
                    "arm": "base_pure_gcs",
                    "param": accel,
                    "yaml_path": args.base_yaml,
                    "seed": s,
                    "steps": args.steps,
                    "overrides": {"enable_accel_energy": accel},
                }
            )
    # C24: comm=0 (NO_APP_SEND) control-plane airtime check
    for s in args.seeds:
        tasks.append(
            {
                "group": "c24_control_plane_comm0",
                "arm": "base_pure_gcs_comm0",
                "param": "NO_APP_SEND",
                "yaml_path": args.base_yaml,
                "seed": s,
                "steps": args.steps,
                "comm_val": 0,
                "overrides": {},
            }
        )

    for task in tasks:
        task['trace_dir'] = args.trace_dir
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        raw_rows = list(ex.map(_run_episode, tasks))

    report = {
        "revision": "V121_CPU_ACTION_UNITS_CORRECTED",
        "source_files_sha256": source_files,
        "source_sha256": hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest(),
        "trace_kind": "per-step mission state, not per-packet MAC event trace",
        "communication_controller": "fixed GCS_SEND or fixed NO_APP_SEND; environmental routing/dirty selection",
        "action_units": "metres_per_second",
        "claim_limit": "Finite-seed synthetic sensitivity. Idealized route estimate is not a physical/network feasibility proof. No auto-reset.",
        "script": "v120_cpu_rf_substeps_energy_calibration.py",
        "seeds": args.seeds,
        "steps": args.steps,
        "c08_c12_energy_reachability": energy_bound,
        "summary": _aggregate(raw_rows),
        "raw_runs": raw_rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved V120 RF/Energy/Substeps calibration ({len(raw_rows)} runs) -> {out_path}")


if __name__ == "__main__":
    main()
