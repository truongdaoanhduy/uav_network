#!/usr/bin/env python3
"""V96 S13 empirical OOD RF sensitivity sweep on Vast cuda:0 (non-oracle frontier controller)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src"), str(ROOT / "scripts/gpu")]

from v92_non_oracle_mission_probe import non_oracle_frontier_actions_np
from v93_probe_metrics import TerminalSnapshots

REGIMES = (
    ("multi_00db", ("rf_obstacle_model=multi_obstacle", "rf_per_building_loss_db=0.0")),
    ("multi_06db", ("rf_obstacle_model=multi_obstacle", "rf_per_building_loss_db=6.0")),
    ("multi_12db", ("rf_obstacle_model=multi_obstacle", "rf_per_building_loss_db=12.0")),
    ("multi_20db", ("rf_obstacle_model=multi_obstacle", "rf_per_building_loss_db=20.0")),
    ("rf_opaque",  ("rf_obstacle_model=rf_opaque",      "rf_per_building_loss_db=0.0")),
)


def _iqm_stat(x: np.ndarray) -> float:
    arr = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if arr.size == 0:
        raise ValueError("IQM requires nonempty data")
    n = arr.size
    left, right = 0.25 * n, 0.75 * n
    weights = np.clip(np.minimum(np.arange(n, dtype=float) + 1.0, right)
                      - np.maximum(np.arange(n, dtype=float), left), 0.0, 1.0)
    return float(np.dot(arr, weights) / (right - left))


def _iqm_and_ci95(x: np.ndarray, *, n_boot: int = 2000, seed: int = 0,
                  seed_clusters: int = 3) -> dict:
    """Cluster-bootstrap the ACTUAL IQM statistic, not the mean.

    Each mission seed produces 16 world instances, which may be dependent.
    We bootstrap the seed blocks and preserve all worlds within each block.
    Just three independent seeds is insufficient for tight inferential claims;
    the interval is exploratory, not a hypothesis test.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    if not np.isfinite(arr).all() or arr.size == 0:
        raise ValueError("Nonfinite/empty scores are an invalid gate")
    if seed_clusters < 1 or arr.size % seed_clusters:
        raise ValueError("Seed cluster count must divide world count")
    blocks = arr.reshape(seed_clusters, -1)
    rng = np.random.default_rng(seed)
    iqm_samples = np.empty(n_boot, dtype=np.float64)
    mean_samples = np.empty(n_boot, dtype=np.float64)
    for it in range(n_boot):
        sample = blocks[rng.integers(0, seed_clusters, size=seed_clusters)].ravel()
        iqm_samples[it] = _iqm_stat(sample)
        mean_samples[it] = float(sample.mean())
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "iqm": _iqm_stat(arr),
        "ci95": [float(v) for v in np.percentile(iqm_samples, [2.5, 97.5])],
        "mean_ci95": [float(v) for v in np.percentile(mean_samples, [2.5, 97.5])],
        "bootstrap_statistic": "IQM",
        "bootstrap_unit": "mission_seed_cluster",
        "independent_seed_clusters": seed_clusters,
        "caution": "Only 3 independent seed clusters; descriptive, do not assert significance",
    }



def _eval_regime(scenario: Path, overrides: tuple[str, ...], *, envs: int, steps: int, seeds: list[int], device: str) -> dict:
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

    delivery_pct, cov_legal_pct, returns, ev_ratio = [], [], [], []
    safety_fails = overflow_worlds = full15 = 0
    t0 = time.monotonic()

    for seed in seeds:
        p = GPUWorldV80Params.from_yaml(scenario, envs=envs, device=device, seed=seed, scenario_overrides=overrides)
        w = WarpParallelMissionV80(p)
        snaps = TerminalSnapshots(envs)
        ret = np.zeros(envs, dtype=np.float64)
        act_wp = wp.zeros((envs, 6, 2), dtype=wp.float32, device=device)
        dst_wp = wp.zeros((envs, 6), dtype=wp.int32, device=device)

        for _ in range(1, steps + 1):
            active = ~snaps.seen
            if not active.any():
                break
            cont, dest = non_oracle_frontier_actions_np(
                w.buffers["obs"].numpy(), w.buffers["mask"].numpy(), v_max=p.v_max_mps
            )
            cont[~active], dest[~active] = 0.0, 0
            wp.copy(act_wp, wp.array(cont, dtype=wp.float32, device=device))
            wp.copy(dst_wp, wp.array(dest, dtype=wp.int32, device=device))
            w.step(act_wp, dst_wp)
            ret[active] += w.buffers["reward"].numpy()[active]
            ended = w.buffers["done"].numpy().astype(bool) & active
            if ended.any():
                cpos = w.network.data["ledger_pos"].numpy().astype(np.int64)
                cneg = w.network.data["ledger_neg"].numpy().astype(np.int64)
                snaps.capture(ended, {
                    "team_ev": (cpos[:, np.arange(6), np.arange(6), :] + cneg[:, np.arange(6), np.arange(6), :]).sum(axis=(1, 2)),
                    "gcs_ev": (cpos[:, 6, :, :] + cneg[:, 6, :, :]).sum(axis=(1, 2)),
                    "seen": (cpos[:, np.arange(6), np.arange(6), :].sum(axis=1) + cneg[:, np.arange(6), np.arange(6), :].sum(axis=1)) > 0,
                    "gcs_tp": w.buffers["gcs_tp"].numpy().sum(axis=1),
                    "reason": w.buffers["termination_reason"].numpy().copy(),
                    "failed": w.buffers["failed"].numpy().copy(),
                    "overflow": w.network.data["overflow"].numpy().copy(),
                    "ret": ret.copy(),
                })
        if (~snaps.seen).any():
            cpos = w.network.data["ledger_pos"].numpy().astype(np.int64)
            cneg = w.network.data["ledger_neg"].numpy().astype(np.int64)
            snaps.capture(~snaps.seen.copy(), {
                "team_ev": (cpos[:, np.arange(6), np.arange(6), :] + cneg[:, np.arange(6), np.arange(6), :]).sum(axis=(1, 2)),
                "gcs_ev": (cpos[:, 6, :, :] + cneg[:, 6, :, :]).sum(axis=(1, 2)),
                "seen": (cpos[:, np.arange(6), np.arange(6), :].sum(axis=1) + cneg[:, np.arange(6), np.arange(6), :].sum(axis=1)) > 0,
                "gcs_tp": w.buffers["gcs_tp"].numpy().sum(axis=1),
                "reason": w.buffers["termination_reason"].numpy().copy(),
                "failed": w.buffers["failed"].numpy().copy(),
                "overflow": w.network.data["overflow"].numpy().copy(),
                "ret": ret.copy(),
            })
        wp.synchronize_device(device)
        s = snaps.values
        legal = w.legal.numpy().astype(int)
        delivery_pct.extend((s["gcs_tp"] / 15.0 * 100.0).tolist())
        cov_legal_pct.extend((s["seen"][:, legal].mean(axis=1) * 100.0).tolist())
        returns.extend(s["ret"].tolist())
        ev_ratio.extend((s["gcs_ev"] / np.maximum(1, s["team_ev"])).tolist())
        safety_fails += int(np.count_nonzero(s["failed"]))
        overflow_worlds += int(np.count_nonzero(s["overflow"]))
        full15 += int(np.count_nonzero(s["reason"] & 4))

    return {
        "overrides": list(overrides),
        "total_worlds": envs * len(seeds),
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "gcs_delivery_pct": _iqm_and_ci95(np.array(delivery_pct), seed_clusters=len(seeds)),
        "team_coverage_legal_pct": _iqm_and_ci95(np.array(cov_legal_pct), seed_clusters=len(seeds)),
        "gcs_to_team_evidence_ratio": _iqm_and_ci95(np.array(ev_ratio), seed_clusters=len(seeds)),
        "episode_return": _iqm_and_ci95(np.array(returns), seed_clusters=len(seeds)),
        "full15_completed_worlds": full15,
        "safety_failed_worlds": safety_fails,
        "overflow_worlds": overflow_worlds,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seeds", default="44,2027,10044")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",")]
    results = {name: _eval_regime(args.scenario, ov, envs=args.envs, steps=args.steps, seeds=seeds, device=args.device)
               for name, ov in REGIMES}
    deliv_curve = [results[name]["gcs_delivery_pct"]["mean"] for name, _ in REGIMES]
    ev_curve = [results[name]["gcs_to_team_evidence_ratio"]["mean"] for name, _ in REGIMES]
    out = {
        "gate": "S13_OOD_RF_SENSITIVITY",
        "rf_claim_scope": "Nominal parametric urban-shadowing surrogate; NOT field-calibrated",
        "scenario": str(args.scenario),
        "scenario_sha256": hashlib.sha256(args.scenario.read_bytes()).hexdigest(),
        "device": args.device,
        "seeds": seeds,
        "envs_per_seed": args.envs,
        "steps": args.steps,
        "monotonic_delivery_drop_00_to_opaque": deliv_curve[0] >= deliv_curve[-1],
        "monotonic_evidence_ratio_drop_00_to_opaque": ev_curve[0] >= ev_curve[-1],
        "regimes": results,
        "status": "PROBE_VALID" if all(v["safety_failed_worlds"] == 0 and v["overflow_worlds"] == 0 for v in results.values()) else "PROBE_INVALID",
        "statistical_caveat": "Seed-cluster IQM intervals; 3 independent seeds is not enough for strong Q1 claims",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0 if out["status"] == "PROBE_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
