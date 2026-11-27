#!/usr/bin/env python3
"""V98 Scenario-Only Non-Oracle Ablation Suite (S11 DS<->Bayes, PEER relay, Energy).

Runs strictly with the deterministic non-oracle frontier controller (ZERO RL
training or algorithm changes) across 48 worlds (3 seeds x 16 envs x 900 steps):
1. S11 reward-matched estimator ablation: DS vs Bayes with shared_coverage_reward=True.
2. PEER dissemination trajectory ablation under rf_opaque: enable_peer=True vs False.
3. Energy headroom audit: min/mean remaining battery across 900-step trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src"), str(ROOT / "scripts/gpu")]

from v92_non_oracle_mission_probe import non_oracle_frontier_actions_np
from v93_probe_metrics import TerminalSnapshots


def _run_case(scenario: Path, overrides: tuple[str, ...], *, envs: int, steps: int, seeds: list[int], device: str) -> dict:
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

    t0 = time.monotonic()
    deliv, cov_l, fp_ever, bat_min, bat_mean, ret_all = [], [], [], [], [], []
    s_gcs, s_peer, s_e2e, s_retx, s_drop, s_air = [], [], [], [], [], []
    n_action_peer = n_action_gcs = n_action_silent = 0
    world_seed = [int(seed) for seed in seeds for _ in range(envs)]
    world_env_id = [int(e) for _ in seeds for e in range(envs)]
    safety_fails = overflows = full15 = 0

    for seed in seeds:
        p = GPUWorldV80Params.from_yaml(scenario, envs=envs, device=device, seed=seed, scenario_overrides=overrides)
        w = WarpParallelMissionV80(p)
        snaps = TerminalSnapshots(envs)
        ret = np.zeros(envs, dtype=np.float64)
        airtime_cumulative_us = np.zeros(envs, dtype=np.float64)
        act_wp = wp.zeros((envs, 6, 2), dtype=wp.float32, device=device)
        dst_wp = wp.zeros((envs, 6), dtype=wp.int32, device=device)

        for _ in range(1, steps + 1):
            active = ~snaps.seen
            if not active.any():
                break
            cont, dest = non_oracle_frontier_actions_np(
                w.buffers["obs"].numpy(), w.buffers["mask"].numpy(), v_max=p.v_max_mps
            )
            n_action_peer += int(np.count_nonzero(dest[active] == 2))
            n_action_gcs += int(np.count_nonzero(dest[active] == 1))
            n_action_silent += int(np.count_nonzero(dest[active] == 0))
            cont[~active], dest[~active] = 0.0, 0
            wp.copy(act_wp, wp.array(cont, dtype=wp.float32, device=device))
            wp.copy(dst_wp, wp.array(dest, dtype=wp.int32, device=device))
            w.step(act_wp, dst_wp)
            # NetworkRuntimeV67.step() clears tx_airtime EACH transition before the MAC kernel.
            # The terminal snapshot of tx_airtime is therefore only one-step usage, not episode total.
            airtime_step_us = w.network.data["tx_airtime"].numpy().astype(np.float64).sum(axis=1)
            if not np.isfinite(airtime_step_us).all() or np.any(airtime_step_us < 0):
                raise RuntimeError("nonfinite or negative per-step network TX airtime")
            airtime_cumulative_us[active] += airtime_step_us[active]
            ret[active] += w.buffers["reward"].numpy()[active]
            ended = w.buffers["done"].numpy().astype(bool) & active
            if ended.any():
                cpos = w.network.data["ledger_pos"].numpy().astype(np.int64)
                cneg = w.network.data["ledger_neg"].numpy().astype(np.int64)
                snaps.capture(ended, {
                    "seen": (cpos[:, np.arange(6), np.arange(6), :].sum(axis=1) + cneg[:, np.arange(6), np.arange(6), :].sum(axis=1)) > 0,
                    "gcs_tp": w.buffers["gcs_tp"].numpy().sum(axis=1),
                    "fp_ever": w.buffers["gcs_fp_ever"].numpy().sum(axis=1),
                    "bat": w.buffers["battery"].numpy().copy(),
                    "reason": w.buffers["termination_reason"].numpy().copy(),
                    "failed": w.buffers["failed"].numpy().copy(),
                    "overflow": w.network.data["overflow"].numpy().copy(),
                    "ret": ret.copy(),
                    "stats_gcs": w.network.data["stats_gcs"].numpy().astype(np.int64).copy(),
                    "stats_peer": w.network.data["stats_peer"].numpy().astype(np.int64).copy(),
                    "stats_e2e": w.network.data["stats_e2e"].numpy().astype(np.int64).copy(),
                    "stats_retry": w.network.data["stats_retry"].numpy().astype(np.int64).copy(),
                    "stats_drop": w.network.data["stats_drop"].numpy().astype(np.int64).copy(),
                    "tx_airtime_s": airtime_cumulative_us / 1.0e6,
                })
        if (~snaps.seen).any():
            cpos = w.network.data["ledger_pos"].numpy().astype(np.int64)
            cneg = w.network.data["ledger_neg"].numpy().astype(np.int64)
            snaps.capture(~snaps.seen.copy(), {
                "seen": (cpos[:, np.arange(6), np.arange(6), :].sum(axis=1) + cneg[:, np.arange(6), np.arange(6), :].sum(axis=1)) > 0,
                "gcs_tp": w.buffers["gcs_tp"].numpy().sum(axis=1),
                "fp_ever": w.buffers["gcs_fp_ever"].numpy().sum(axis=1),
                "bat": w.buffers["battery"].numpy().copy(),
                "reason": w.buffers["termination_reason"].numpy().copy(),
                "failed": w.buffers["failed"].numpy().copy(),
                "overflow": w.network.data["overflow"].numpy().copy(),
                "ret": ret.copy(),
                "stats_gcs": w.network.data["stats_gcs"].numpy().astype(np.int64).copy(),
                "stats_peer": w.network.data["stats_peer"].numpy().astype(np.int64).copy(),
                "stats_e2e": w.network.data["stats_e2e"].numpy().astype(np.int64).copy(),
                "stats_retry": w.network.data["stats_retry"].numpy().astype(np.int64).copy(),
                "stats_drop": w.network.data["stats_drop"].numpy().astype(np.int64).copy(),
                "tx_airtime_s": airtime_cumulative_us / 1.0e6,
            })
        wp.synchronize_device(device)
        s = snaps.values
        legal = w.legal.numpy().astype(int)
        deliv.extend((s["gcs_tp"] / 15.0 * 100.0).tolist())
        cov_l.extend((s["seen"][:, legal].mean(axis=1) * 100.0).tolist())
        fp_ever.extend(s["fp_ever"].tolist())
        bat_min.extend(s["bat"].min(axis=1).tolist())
        bat_mean.extend(s["bat"].mean(axis=1).tolist())
        ret_all.extend(s["ret"].tolist())
        s_gcs.extend(s["stats_gcs"].tolist())
        s_peer.extend(s["stats_peer"].tolist())
        s_e2e.extend(s["stats_e2e"].tolist())
        s_retx.extend(s["stats_retry"].tolist())
        s_drop.extend(s["stats_drop"].tolist())
        s_air.extend(s["tx_airtime_s"].tolist())
        safety_fails += int(np.count_nonzero(s["failed"]))
        overflows += int(np.count_nonzero(s["overflow"]))
        full15 += int(np.count_nonzero(s["reason"] & 4))

    return {
        "scenario": scenario.name,
        "overrides": list(overrides),
        "worlds": envs * len(seeds),
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "mean_gcs_delivery_pct": round(float(np.mean(deliv)), 3),
        "mean_legal_coverage_pct": round(float(np.mean(cov_l)), 3),
        "mean_fp_ever": round(float(np.mean(fp_ever)), 3),
        "mean_return": round(float(np.mean(ret_all)), 2),
        "min_battery_remaining_j": round(float(np.min(bat_min)), 1),
        "mean_battery_remaining_j": round(float(np.mean(bat_mean)), 1),
        "full15_worlds": full15,
        "safety_failed_worlds": safety_fails,
        "overflow_worlds": overflows,
        "actions_selected": {"SILENT": n_action_silent, "GCS": n_action_gcs, "PEER": n_action_peer},
        "per_world": {
            "seed": world_seed, "env_id": world_env_id,
            "gcs_delivery_pct": deliv, "legal_coverage_pct": cov_l,
            "fp_ever": fp_ever, "mean_battery_j": bat_mean,
            "return": ret_all,
        },
        "mac_telemetry_mean": {
            "stats_gcs": round(float(np.mean(s_gcs)), 2),
            "stats_peer": round(float(np.mean(s_peer)), 2),
            "stats_e2e": round(float(np.mean(s_e2e)), 2),
            "stats_retry": round(float(np.mean(s_retx)), 2),
            "stats_drop": round(float(np.mean(s_drop)), 2),
            "tx_airtime_s_total": round(float(np.mean(s_air)), 4),
            "tx_airtime_measurement_method": "sum post-network-step airtime over each active 1-second transition; GPU register reset each step",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seeds", default="44,2027,10044")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    cfg = ROOT / "research/zhang_xue/configs"
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cases = {
        "s11_bayes_shared_cov": (cfg / "scenario_v95_bayes_shared_coverage_1km.yaml", ()),
        "s11_ds_shared_cov": (cfg / "scenario_v95_ds_shared_coverage_1km.yaml", ()),
        "peer_on_rf_opaque": (cfg / "scenario_v91_bayes_1km.yaml", ("rf_obstacle_model=rf_opaque", "rf_per_building_loss_db=0.0", "enable_peer_dissemination=true")),
        "peer_off_rf_opaque": (cfg / "scenario_v91_bayes_1km.yaml", ("rf_obstacle_model=rf_opaque", "rf_per_building_loss_db=0.0", "enable_peer_dissemination=false")),
    }
    res = {k: _run_case(sc, ov, envs=args.envs, steps=args.steps, seeds=seeds, device=args.device) for k, (sc, ov) in cases.items()}
    ok = all(r["safety_failed_worlds"] == 0 and r["overflow_worlds"] == 0 for r in res.values())
    on, off = res["peer_on_rf_opaque"]["per_world"], res["peer_off_rf_opaque"]["per_world"]
    assert on["seed"] == off["seed"] and on["env_id"] == off["env_id"], "world alignment failed"
    paired = {}
    rng = np.random.default_rng(20261127)
    for field in ("gcs_delivery_pct", "legal_coverage_pct", "fp_ever", "mean_battery_j", "return"):
        diff = np.asarray(off[field], dtype=np.float64) - np.asarray(on[field], dtype=np.float64)
        # Exploratory paired-bootstrap interval; do not treat 16 worlds in each seed as 16 independent seeds.
        sample_seed = [int(x) for x in seeds]
        blockmeans = np.asarray([np.mean(diff[np.asarray(on["seed"]) == sd]) for sd in sample_seed])
        samples = rng.choice(blockmeans, size=(3000,len(blockmeans)), replace=True).mean(axis=1)
        paired[field] = {"mean_off_minus_on": float(np.mean(diff)),
                         "bootstrap_seed_block_ci95": [float(np.quantile(samples,0.025)),float(np.quantile(samples,0.975))],
                         "paired_world_deltas": diff.tolist()}
    out = {
        "status": "PASS" if ok else "FAIL",
        "gate": "V98_SCENARIO_ONLY_ABLATION_SUITE",
        "device": args.device,
        "seeds": seeds,
        "envs_per_seed": args.envs,
        "result_status_scope": "zero safety/overflow at end of each world only, NOT proof PEER is useful",
        "paired_world_analysis": True,
        "paired_seed_block_count": len(seeds),
        "peer_action_necessity_confirmed": False,
        "coverage_term_matched_only_acknowledged": True,
        "battery_energy_accounting_verified": False,
        "peer_on_off_paired": paired,
        "results": res,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
