"""Dual-Fidelity Evaluation KPIs & Scientifically Sound Q1 Aggregation (Streamfix v3).

Key Q1 paper metric fixes:
- `energy_kj_per_delivered_target`: returns `None` (JSON `null` / N/A) when `targets_delivered == 0`
  (never fabricates `3.02 kJ/target` for a 0-delivery run); always reports unconditional `mean_propulsion_kj`.
- `mean_aoi_s` & `p95_aoi_s`: conditional strictly on delivered true targets (`None` when 0 deliveries),
  computed via exact pooled packet samples (`aoi_samples_s`) when available; separate explicit
  `censored_mean_aoi_s` for TTL-censored reporting.
- `multihop_delivery_ratio_pct`, `confirmation_precision_pct`, `gcs_report_precision_pct`: return `None`
  when their denominator is 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class EpisodeKPI:
    episode_return: float
    targets_discovered: int
    targets_delivered: int
    total_targets: int
    mean_aoi_s: Optional[float]
    p95_aoi_s: Optional[float]
    multihop_deliveries: int
    mac_collisions: int
    hd_drops: int
    retry_drops: int
    ttl_expired: int
    propulsion_kj: float
    safety_violations: int
    false_reports_delivered: int = 0
    false_confirmed_cells: int = 0
    total_reported_cells: int = 0
    observed_cells: int = 0
    total_cells: int = 400
    radio_energy_j: float = 0.0
    censored_mean_aoi_s: float = 120.0
    aoi_samples_s: Optional[List[float]] = None


def _safe_ratio(num: float, den: float, scale: float = 1.0) -> Optional[float]:
    return float(scale * num / den) if den > 0 else None


def aggregate_kpis(episodes: Iterable[EpisodeKPI]) -> Dict[str, Optional[float]]:
    """Aggregate episode KPIs with strict zero-denominator & conditional-AoI hygiene."""
    eps = list(episodes)
    if not eps:
        raise ValueError("Cannot aggregate empty episode list.")

    n = len(eps)
    tot_targets = sum(e.total_targets for e in eps)
    tot_disc = sum(e.targets_discovered for e in eps)
    tot_deliv = sum(e.targets_delivered for e in eps)
    tot_multihop = sum(e.multihop_deliveries for e in eps)
    tot_prop_kj = sum(e.propulsion_kj for e in eps)
    total_false_gcs = sum(e.false_reports_delivered for e in eps)
    total_false_confirm = sum(e.false_confirmed_cells for e in eps)
    total_reported = sum(e.total_reported_cells for e in eps)
    total_observed = sum(e.observed_cells for e in eps)
    total_cells = sum(e.total_cells for e in eps)

    valid_aoi_eps = [e for e in eps if e.targets_delivered > 0 and e.mean_aoi_s is not None]
    pooled_aois = [float(a) for e in eps if e.aoi_samples_s for a in e.aoi_samples_s]
    if pooled_aois:
        mean_aoi: Optional[float] = float(sum(pooled_aois) / len(pooled_aois))
        s_aois = sorted(pooled_aois)
        p95_idx = min(len(s_aois) - 1, max(0, int(math.ceil(0.95 * len(s_aois))) - 1))
        p95_aoi: Optional[float] = float(s_aois[p95_idx])
    elif valid_aoi_eps:
        w_sum = sum(e.targets_delivered for e in valid_aoi_eps)
        mean_aoi = float(sum(float(e.mean_aoi_s) * e.targets_delivered for e in valid_aoi_eps) / w_sum)
        p95_vals = [float(e.p95_aoi_s) * e.targets_delivered for e in valid_aoi_eps if e.p95_aoi_s is not None]
        p95_aoi = float(sum(p95_vals) / w_sum) if p95_vals else None
    else:
        mean_aoi, p95_aoi = None, None

    return {
        "episodes": float(n),
        "delivered_episodes": float(len(valid_aoi_eps)),
        "total_targets_delivered": float(tot_deliv),
        "mean_return": sum(e.episode_return for e in eps) / n,
        "target_discovery_rate_pct": _safe_ratio(tot_disc, tot_targets, 100.0),
        "target_delivery_ratio_pct": _safe_ratio(tot_deliv, tot_targets, 100.0),
        "map_coverage_pct": _safe_ratio(total_observed, total_cells, 100.0),
        "confirmation_precision_pct": _safe_ratio(tot_disc, total_reported, 100.0),
        "gcs_report_precision_pct": _safe_ratio(tot_deliv, tot_deliv + total_false_gcs, 100.0),
        "false_confirmed_cells": float(total_false_confirm),
        "false_reports_delivered": float(total_false_gcs),
        "mission_success_rate_pct": 100.0 * sum(e.targets_delivered >= e.total_targets for e in eps) / n,
        "multihop_delivery_ratio_pct": _safe_ratio(tot_multihop, tot_deliv, 100.0),
        "mean_aoi_s": mean_aoi,
        "p95_aoi_s": p95_aoi,
        "censored_mean_aoi_s": sum(e.censored_mean_aoi_s for e in eps) / n,
        "mean_mac_collisions": sum(e.mac_collisions for e in eps) / n,
        "mean_hd_drops": sum(e.hd_drops for e in eps) / n,
        "mean_retry_drops": sum(e.retry_drops for e in eps) / n,
        "mean_ttl_expired": sum(e.ttl_expired for e in eps) / n,
        "mean_propulsion_kj": tot_prop_kj / n,
        "energy_kj_per_delivered_target": _safe_ratio(tot_prop_kj, tot_deliv, 1.0),
        "mean_radio_energy_j": sum(e.radio_energy_j for e in eps) / n,
        "mean_safety_violations": sum(e.safety_violations for e in eps) / n,
    }


def compute_sim_to_eval_gap(
    slotted_summary: Dict[str, Optional[float]], full_csma_summary: Dict[str, Optional[float]]
) -> Dict[str, Optional[float]]:
    keys = [
        "mean_return",
        "target_delivery_ratio_pct",
        "multihop_delivery_ratio_pct",
        "mean_aoi_s",
        "p95_aoi_s",
        "mean_mac_collisions",
        "mean_propulsion_kj",
        "energy_kj_per_delivered_target",
    ]
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        a, b = slotted_summary.get(k), full_csma_summary.get(k)
        out[f"delta_{k}"] = (
            float(b - a) if (a is not None and b is not None and math.isfinite(a) and math.isfinite(b)) else None
        )
    return out


def run_dual_fidelity_evaluation(
    slotted_eval_fn,
    full_csma_eval_fn,
    eval_seeds: List[int],
) -> Dict[str, Dict[str, Optional[float]]]:
    slotted_eps = slotted_eval_fn(eval_seeds) if callable(slotted_eval_fn) else slotted_eval_fn
    csma_eps = full_csma_eval_fn(eval_seeds) if callable(full_csma_eval_fn) else full_csma_eval_fn
    slotted_sum = aggregate_kpis(slotted_eps)
    csma_sum = aggregate_kpis(csma_eps)
    gap = compute_sim_to_eval_gap(slotted_sum, csma_sum)
    return {
        "slotted_warp_fp32": slotted_sum,
        "full_uavnetsim_csma": csma_sum,
        "sim_to_eval_gap": gap,
    }
