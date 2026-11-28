"""V122 Multi-Seed CPU Scientific Calibration & Sensitivity Suite (5 seeds x 300 steps).

Produces real, reproducible CPU evidence for the remaining V121 open scientific items:
  A. C04/C23/C41/C66: DS temporal-correlation mitigation under Markov rho in {0.0, 0.5, 0.7}
     (canonical_raw_iid vs reliability_cap_0.65 vs revisit_cooldown_9s vs ar1_weighted_rho07)
  B. C13/C32: Al-Hourani / 3GPP building RF attenuation sweep {12, 20, 23} dB x {Base, RF-stress}
  C. C12/C83: Zeng 2019 steady-level vs corrected Appendix A acceleration energy
  D. C33/C65: Compact DS frame (70+8k B) vs UavNetSim 1082 B airtime utilization check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from zhang_xue.belief.ds_correlation_mitigation import DSCorrelationMitigator
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv

ROOT = Path(__file__).resolve().parents[2]
BASE_YAML = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"
RF_YAML = ROOT / "research/zhang_xue/configs/scenario_v111_ds_rf_stress_no_peer_1km.yaml"


def _load_cfg(yaml_path: Path, **overrides: Any) -> EnvConfigV54:
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
    return (np.clip(cmd, -1.0, 1.0) * env.cfg.v_max_mps).astype(np.float64)


def _run_ds_mitigation_arm(
    seed: int, steps: int, rho: float, arm: str
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if arm == "reliability_cap_0.65":
        overrides = {"ds_mf": 0.65, "ds_me": 0.65}
    cfg = _load_cfg(BASE_YAML, **overrides)
    env = ZhangXue7NodePacketEnv(cfg)
    env.reset(seed=seed)

    mit = None
    if arm == "revisit_cooldown_9s":
        mit = DSCorrelationMitigator(num_uavs=cfg.num_uavs, num_cells=env.num_cells, mode="revisit_cooldown", cooldown_s=9)
    elif arm == "ar1_weighted_rho07":
        mit = DSCorrelationMitigator(num_uavs=cfg.num_uavs, num_cells=env.num_cells, mode="ar1_weighted", rho_hat=0.7)

    orig_fn = env.net.record_local_observation
    prev_det: Dict[tuple[int, int], bool] = {}
    corr_rng = np.random.default_rng(seed + 100_003)
    pd, pf = float(cfg.pd), float(cfg.pf)

    # Track GCS-delivered mitigated confirmation in parallel for ar1_weighted / cooldown
    mit_tp_ever = np.zeros(env.num_cells, dtype=np.uint8)
    mit_fp_ever = np.zeros(env.num_cells, dtype=np.uint8)

    def _hook(u: int, cid: int, det: bool, t_step_s: float) -> None:
        key = (int(u), int(cid))
        p = pd if bool(env.true_targets[cid]) else pf
        if rho <= 0.0 or key not in prev_det:
            new_det = bool(det)
        else:
            prob_1 = (p + rho * (1.0 - p)) if prev_det[key] else ((1.0 - rho) * p)
            new_det = bool(corr_rng.random() < prob_1)
        prev_det[key] = new_det

        if arm == "revisit_cooldown_9s" and mit is not None:
            w = mit.record(u, cid, new_det, int(round(t_step_s)))
            if w > 0.0:
                orig_fn(u, cid, new_det, t_step_s)
        elif arm == "ar1_weighted_rho07" and mit is not None:
            mit.record(u, cid, new_det, int(round(t_step_s)))
            orig_fn(u, cid, new_det, t_step_s)
        else:
            orig_fn(u, cid, new_det, t_step_s)

    env.net.record_local_observation = _hook  # type: ignore[method-assign]
    info: Dict[str, Any] = {}
    for t in range(steps):
        act_xy = _sector_patrol_xy(env, t)
        act_comm = np.ones((cfg.num_uavs,), dtype=np.int32)
        _, _, _, _, term, trunc, info = env.step(act_xy, act_comm)
        if arm == "ar1_weighted_rho07" and mit is not None:
            # Evaluate mitigated DS on cells that have been delivered to GCS
            for cid in np.flatnonzero(env.net.gcs_latest_u_s >= 0):
                if mit.is_confirmed(int(cid), mf=cfg.ds_mf, me=cfg.ds_me, confirm_th=cfg.confirm_th, uncertainty_ceiling=cfg.ds_uncertainty_ceiling):
                    if bool(env.true_targets[cid]):
                        mit_tp_ever[cid] = 1
                    else:
                        mit_fp_ever[cid] = 1
        if term or trunc:
            break

    if arm == "ar1_weighted_rho07":
        tp_ever = int(mit_tp_ever.sum())
        fp_ever = int(mit_fp_ever.sum())
    else:
        tp_ever = int(info.get("gcs_tp_confirmed", 0))
        fp_ever = int(info.get("gcs_fp_ever_confirmed", 0))

    return {
        "arm": arm,
        "rho": rho,
        "seed": seed,
        "gcs_tp_ever": tp_ever,
        "gcs_fp_ever": fp_ever,
        "delivered_cov_pct": float(info.get("delivered_cell_coverage_pct", 0.0)),
        "mean_first_confirm_s": float(info.get("mean_first_tp_confirm_latency_s", -1.0)),
    }


def _run_rf_energy_arm(
    yaml_path: Path, label: str, seed: int, steps: int, overrides: Dict[str, Any]
) -> Dict[str, Any]:
    cfg = _load_cfg(yaml_path, **overrides)
    env = ZhangXue7NodePacketEnv(cfg)
    env.reset(seed=seed)
    info: Dict[str, Any] = {}
    total_airtime_us = 0
    for t in range(steps):
        act_xy = _sector_patrol_xy(env, t)
        act_comm = np.ones((cfg.num_uavs,), dtype=np.int32)
        _, _, _, _, term, trunc, info = env.step(act_xy, act_comm)
        total_airtime_us += int(env.net.step_tx_airtime_us[: cfg.num_uavs].sum())
        if term or trunc:
            break
    data_deliv = int(info.get("data_delivered_gcs", 0))
    return {
        "label": label,
        "seed": seed,
        "gcs_tp_ever": int(info.get("gcs_tp_confirmed", 0)),
        "delivered_cov_pct": float(info.get("delivered_cell_coverage_pct", 0.0)),
        "pdr_gcs": float(info.get("pdr_gcs", -1.0)),
        "min_battery_pct": float(info.get("min_battery_pct", 0.0)),
        "data_delivered_gcs": data_deliv,
        "total_uav_tx_airtime_ms": round(total_airtime_us / 1000.0, 2),
        "equiv_uavnetsim_1082b_data_airtime_ms": round(data_deliv * 4.328, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46, 47, 48])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--output", default="outputs/v122_scenario_audit/v122_scientific_calibration.json")
    args = ap.parse_args()

    import concurrent.futures, os
    workers = max(1, min(8, (os.cpu_count() or 2)))
    ds_jobs = [(s, args.steps, rho, arm) for rho in (0.0, 0.5, 0.7) for arm in ("canonical_raw_iid", "reliability_cap_0.65", "revisit_cooldown_9s", "ar1_weighted_rho07") for s in args.seeds]
    rf_jobs = [(ypath, f"{prof_name}_loss_{int(loss_db)}dB", s, args.steps, {"rf_per_building_loss_db": loss_db}) for prof_name, ypath in (("base", BASE_YAML), ("rf_stress", RF_YAML)) for loss_db in (12.0, 20.0, 23.0) for s in args.seeds]
    en_jobs = [(BASE_YAML, f"zeng_accel_{str(accel).lower()}", s, args.steps, {"enable_accel_energy": accel}) for accel in (False, True) for s in args.seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        ds_rows = list(ex.map(lambda a: _run_ds_mitigation_arm(*a), ds_jobs)) if workers == 1 else [f.result() for f in [ex.submit(_run_ds_mitigation_arm, *a) for a in ds_jobs]]
        rf_rows = [f.result() for f in [ex.submit(_run_rf_energy_arm, *a) for a in rf_jobs]]
        energy_rows = [f.result() for f in [ex.submit(_run_rf_energy_arm, *a) for a in en_jobs]]

    def _agg(rows: List[Dict[str, Any]], key_fields: List[str], metric_fields: List[str]) -> List[Dict[str, Any]]:
        buckets: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in rows:
            k = tuple(r[f] for f in key_fields)
            buckets.setdefault(k, []).append(r)
        out = []
        for k, grp in sorted(buckets.items()):
            rec = {f: v for f, v in zip(key_fields, k)}
            rec["n_seeds"] = len(grp)
            for mf in metric_fields:
                vals = [float(x[mf]) for x in grp if x[mf] is not None and float(x[mf]) >= 0.0]
                rec[f"mean_{mf}"] = round(float(np.mean(vals)), 3) if vals else -1.0
            out.append(rec)
        return out

    payload = {
        "seeds": args.seeds,
        "steps": args.steps,
        "base_yaml_sha256": hashlib.sha256(BASE_YAML.read_bytes()).hexdigest(),
        "exp_a_ds_correlation_mitigation": _agg(
            ds_rows, ["rho", "arm"], ["gcs_tp_ever", "gcs_fp_ever", "delivered_cov_pct"]
        ),
        "exp_b_rf_building_loss_sweep": _agg(
            rf_rows, ["label"], ["gcs_tp_ever", "delivered_cov_pct", "pdr_gcs", "total_uav_tx_airtime_ms", "equiv_uavnetsim_1082b_data_airtime_ms"]
        ),
        "exp_c_zeng_2019_corrected_accel_energy": _agg(
            energy_rows, ["label"], ["min_battery_pct", "gcs_tp_ever", "delivered_cov_pct"]
        ),
        "raw_counts": {"ds_runs": len(ds_rows), "rf_runs": len(rf_rows), "energy_runs": len(energy_rows)},
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("exp_a_ds_correlation_mitigation", "exp_b_rf_building_loss_sweep", "exp_c_zeng_2019_corrected_accel_energy")}, indent=2))


if __name__ == "__main__":
    main()
