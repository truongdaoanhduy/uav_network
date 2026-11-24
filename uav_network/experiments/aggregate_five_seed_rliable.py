"""5-Seed Paper Aggregator (rliable / marl-eval Seed-Level Stratified Bootstrap IQM + 95% CI).

Merges per-seed `marl_eval_results.json` across seeds `[44, 45, 46, 47, 48]` for Vanilla MATD3
and Vanilla MASAC without inflating sample sizes across episodes, and computes:
- Seed-level IQM + 95% Bootstrap CI for Episode Return, Delivery Ratio %, Multi-Hop %, Mean AoI
- Probability of Improvement P(MATD3 > MASAC) with 95% Bootstrap CI (Mann-Whitney U statistic)
- Paired 5-seed Sim-to-Net Gap Delta (Full UavNetSim CSMA/CA minus Warp Slotted M=4) with 95% CI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from uav_network.experiments.telemetry_wandb_hf import iqm_stratified_bootstrap_ci


def probability_of_improvement(
    scores_a: np.ndarray, scores_b: np.ndarray, n_resamples: int = 2000, seed: int = 42
) -> Tuple[float, float, float]:
    """Compute P(A > B) and 95% bootstrap CI across seed-level means (rliable standard)."""
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()

    def _poi(x: np.ndarray, y: np.ndarray) -> float:
        diff = x[:, None] - y[None, :]
        return float(np.mean((diff > 0.0) + 0.5 * (diff == 0.0)))

    point = _poi(a, b)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        boots[i] = _poi(a[ia], b[ib])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def aggregate_runs(run_dirs: List[Path], output_path: Path) -> Dict[str, Any]:
    merged_marl_eval: Dict[str, Any] = {}
    seed_means: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    for rdir in run_dirs:
        me_file = rdir / "marl_eval_results.json"
        if not me_file.exists():
            continue
        data = json.loads(me_file.read_text(encoding="utf-8"))
        for env_key, tasks in data.items():
            for task_key, algos in tasks.items():
                for algo_key, seeds in algos.items():
                    for seed_key, payload in seeds.items():
                        merged_marl_eval.setdefault(env_key, {}).setdefault(task_key, {}).setdefault(algo_key, {})[seed_key] = payload
                        abs_m = payload.get("absolute_metrics")
                        if not abs_m:
                            continue
                        for m_name, ep_vals in abs_m.items():
                            if ep_vals:
                                seed_means.setdefault(env_key, {}).setdefault(algo_key, {}).setdefault(m_name, []).append(
                                    float(np.mean(ep_vals))
                                )

    summary: Dict[str, Any] = {"seed_level_metrics": {}, "comparisons": {}, "sim_to_net_gap_5seed": {}}

    for env_key, algos in seed_means.items():
        summary["seed_level_metrics"][env_key] = {}
        for algo_key, metrics in algos.items():
            summary["seed_level_metrics"][env_key][algo_key] = {}
            for m_name, vals in metrics.items():
                arr = np.asarray(vals, dtype=np.float64)
                iqm, lo, hi = iqm_stratified_bootstrap_ci(arr.reshape(1, -1))
                summary["seed_level_metrics"][env_key][algo_key][m_name] = {
                    "n_seeds": int(len(arr)),
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "iqm": iqm,
                    "ci_low": lo,
                    "ci_high": hi,
                }

        if "vanilla_matd3" in algos and "vanilla_masac" in algos:
            summary["comparisons"][env_key] = {}
            for m_name in ("episode_return", "delivery_ratio_pct", "multihop_ratio_pct", "mean_aoi_s"):
                if m_name in algos["vanilla_matd3"] and m_name in algos["vanilla_masac"]:
                    p, lo, hi = probability_of_improvement(
                        np.asarray(algos["vanilla_matd3"][m_name]),
                        np.asarray(algos["vanilla_masac"][m_name]),
                    )
                    summary["comparisons"][env_key][f"P_MATD3_gt_MASAC_{m_name}"] = {
                        "point": p, "ci_low": lo, "ci_high": hi,
                    }

    # Paired Sim-to-Net Gap across seeds
    for slot_k, csma_k in (("warp_slotted_fp32", "uavnetsim_full_csma"), ("numpy_slotted_reference", "csma_event_proxy")):
        if slot_k in seed_means and csma_k in seed_means:
            for algo_key in set(seed_means[slot_k]).intersection(seed_means[csma_k]):
                summary["sim_to_net_gap_5seed"][f"{algo_key}_{slot_k}_vs_{csma_k}"] = {}
                for m_name in ("episode_return", "delivery_ratio_pct", "mean_aoi_s"):
                    a = np.asarray(seed_means[slot_k][algo_key].get(m_name, []), dtype=np.float64)
                    b = np.asarray(seed_means[csma_k][algo_key].get(m_name, []), dtype=np.float64)
                    n = min(len(a), len(b))
                    if n > 0:
                        delta = b[:n] - a[:n]
                        iqm, lo, hi = iqm_stratified_bootstrap_ci(delta.reshape(1, -1))
                        summary["sim_to_net_gap_5seed"][f"{algo_key}_{slot_k}_vs_{csma_k}"][m_name] = {
                            "n_paired_seeds": int(n),
                            "delta_mean": float(delta.mean()),
                            "delta_iqm": iqm,
                            "ci_low": lo,
                            "ci_high": hi,
                        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_path.parent / "merged_marl_eval_results.json").write_text(
        json.dumps(merged_marl_eval, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Seed rliable/marl-eval Aggregator")
    parser.add_argument("--runs", nargs="+", required=True, help="List of per-seed run directories")
    parser.add_argument("--out", required=True, help="Output summary JSON path")
    args = parser.parse_args()
    res = aggregate_runs([Path(r) for r in args.runs], Path(args.out))
    print("5-SEED AGGREGATION COMPLETE:", json.dumps(res["comparisons"], indent=2))


if __name__ == "__main__":
    main()
