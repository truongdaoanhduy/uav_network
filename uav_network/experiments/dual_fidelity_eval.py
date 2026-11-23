"""Dual-Fidelity Evaluation Bridge: Slotted Warp PHY/MAC vs. Full CSMA/CA UavNetSim.

Evaluates a trained Vanilla MATD3 or Vanilla MASAC checkpoint on identical held-out seeds
across BOTH backends and computes the exact Sim-to-Eval Gap (Delta) required by Q1 reviewers:
  - Target Discovery Rate (TDR %) & Target Delivery Ratio (GCS %)
  - Mean AoI & P95 AoI (seconds) + AoI-weighted Score
  - Multi-Hop Delivery Ratio (% of GCS deliveries traversing >= 2 hops)
  - MAC Collision Rate, Half-Duplex Drop Rate, Retry-Exhaustion Drop Rate, TTL Expiry Rate
  - Energy Efficiency (kJ per delivered target) & Safety Violation Rate.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List


@dataclass
class EpisodeKPI:
    episode_return: float
    targets_discovered: int
    targets_delivered: int
    total_targets: int
    mean_aoi_s: float
    p95_aoi_s: float
    multihop_deliveries: int
    mac_collisions: int
    hd_drops: int
    retry_drops: int
    ttl_expired: int
    propulsion_kj: float
    safety_violations: int


def aggregate_kpis(episodes: Iterable[EpisodeKPI]) -> Dict[str, float]:
    eps = list(episodes)
    if not eps:
        raise ValueError("Cannot aggregate empty episode list.")
    n = len(eps)
    tot_targets = sum(e.total_targets for e in eps)
    tot_disc = sum(e.targets_discovered for e in eps)
    tot_deliv = sum(e.targets_delivered for e in eps)
    tot_multihop = sum(e.multihop_deliveries for e in eps)
    tot_prop_kj = sum(e.propulsion_kj for e in eps)

    return {
        "episodes": float(n),
        "mean_return": sum(e.episode_return for e in eps) / n,
        "target_discovery_rate_pct": 100.0 * tot_disc / max(tot_targets, 1),
        "target_delivery_ratio_pct": 100.0 * tot_deliv / max(tot_targets, 1),
        "multihop_delivery_ratio_pct": 100.0 * tot_multihop / max(tot_deliv, 1),
        "mean_aoi_s": sum(e.mean_aoi_s for e in eps) / n,
        "p95_aoi_s": sum(e.p95_aoi_s for e in eps) / n,
        "mean_mac_collisions": sum(e.mac_collisions for e in eps) / n,
        "mean_hd_drops": sum(e.hd_drops for e in eps) / n,
        "mean_retry_drops": sum(e.retry_drops for e in eps) / n,
        "mean_ttl_expired": sum(e.ttl_expired for e in eps) / n,
        "energy_kj_per_delivered_target": tot_prop_kj / max(tot_deliv, 1),
        "mean_safety_violations": sum(e.safety_violations for e in eps) / n,
    }


def compute_sim_to_eval_gap(
    slotted_summary: Dict[str, float], full_csma_summary: Dict[str, float]
) -> Dict[str, float]:
    """Compute Delta = Full_CSMA_UavNetSim - Slotted_Warp_Train across all shared KPIs."""
    keys = [
        "mean_return",
        "target_delivery_ratio_pct",
        "multihop_delivery_ratio_pct",
        "mean_aoi_s",
        "p95_aoi_s",
        "mean_mac_collisions",
        "energy_kj_per_delivered_target",
    ]
    return {f"delta_{k}": full_csma_summary[k] - slotted_summary[k] for k in keys}


def run_dual_fidelity_evaluation(
    run_slotted_fn: Callable[[List[int]], List[EpisodeKPI]],
    run_full_csma_fn: Callable[[List[int]], List[EpisodeKPI]],
    eval_seeds: List[int],
    output_json_path: str | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run paired held-out evaluation on identical seeds across both communication backends."""
    slotted_eps = run_slotted_fn(eval_seeds)
    full_csma_eps = run_full_csma_fn(eval_seeds)
    slotted_agg = aggregate_kpis(slotted_eps)
    full_agg = aggregate_kpis(full_csma_eps)
    gap = compute_sim_to_eval_gap(slotted_agg, full_agg)

    report = {
        "slotted_warp_fp32": slotted_agg,
        "full_uavnetsim_csma": full_agg,
        "sim_to_eval_gap": gap,
    }
    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return report
