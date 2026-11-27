"""Add-only episode-level statistics for GPU Validation and held-out Test.

NEVER replace pooled estimators (energy per delivered, packet PDR, false
confirmations/measurement, target-conditional delay) by an unweighted mean.
Standard deviation is sample SD (ddof=1) across *independent scenarios* of
one frozen checkpoint, not variance across independent training runs.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
from statistics import fmean, stdev


POOLED = frozenset({
    "energy_per_delivered_target_j",
    "false_confirmation_per_1000_measurements",
    "false_confirmation_rate_percent",
    "packet_pdr_percent",
    "first_discovery_to_gcs_delay_delivered_mean_s",
})

# Every existing public mission aggregate receives a well-defined per-episode
# observation. For pooled metrics, this per-episode mean and SD are *additional*
# diagnostics; the unchanged pooled value is the paper's primary estimate.
def per_episode_metric(row, name):
    c = row.get("comparison_metrics") or {}
    sensing = row.get("sensing_counts") or {}
    packets = row.get("packet_counts") or {}
    found, delivered = row.get("targets_found"), row.get("targets_delivered")
    if name == "discovery_rate_percent":
        return 100 * c["target_discovery_fraction"]
    if name == "gcs_delivery_rate_percent":
        return 100 * c["gcs_delivery_fraction"]
    if name == "discovery_time_capped_mean_s":
        return c["target_discovery_time_capped_mean_s"]
    if name == "gcs_delivery_time_capped_mean_s":
        return c["target_delivery_time_capped_mean_s"]
    if name == "energy_per_delivered_target_j":
        return c["uav_energy_j"]/delivered if delivered else None
    if name == "mission_success_rate_percent":
        return 100 * int(row["success"])
    if name == "mean_total_uav_energy_j":
        return c["uav_energy_j"]
    if name == "mean_targets_found":
        return found
    if name == "mean_targets_delivered":
        return delivered
    if name == "false_confirmation_per_1000_measurements":
        count = sensing.get("measurements")
        failures = sensing.get("false_confirmation")
        return 1000*failures/count if count and failures is not None else None
    if name == "false_confirmation_rate_percent":
        count = sensing.get("measurements")
        failures = sensing.get("false_confirmation")
        return 100*failures/count if count and failures is not None else None
    if name == "packet_pdr_percent":
        sent, received = packets.get("sent"), packets.get("received")
        return 100*received/sent if sent and received is not None else None
    if name == "return":
        return row["return"]
    if name == "episode_length":
        return row["episode_length"]
    if name == "search_rate_percent":
        return row["search_rate_percent"]
    if name == "delivery_rate_percent":
        return row["delivery_rate_percent"]
    if name in c:
        return c[name]
    if name == "delivery_given_discovery_percent":
        return 100*delivered/found if found else None
    if name == "search_delivery_gap_percentage_points":
        return row["search_rate_percent"]-row["delivery_rate_percent"]
    raise KeyError("Missing per-episode value definition for "+name)


def episode_statistics(records, mission_metrics):
    """Mean +/- sample SD per metric across fixed-seed episodes.

    Missing denominator -> null, excluded only from that metric, counted in
    n_missing. Do not silently turn missing metrics into zero.
    """
    if not records:
        raise ValueError("Cannot summarize empty Validation/Test records")
    metric_keys = dict.fromkeys([
        *mission_metrics,
        "return", "episode_length", "search_rate_percent", "delivery_rate_percent",
        "delivery_given_discovery_percent", "search_delivery_gap_percentage_points",
    ])
    for row in records:
        for key in (row.get("comparison_metrics") or {}):
            metric_keys.setdefault("comparison/"+key, None)
    stats = {}
    for metric in metric_keys:
        values = []
        for row in records:
            value = (row.get("comparison_metrics", {}).get(metric[11:])
                     if metric.startswith("comparison/") else per_episode_metric(row, metric))
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Non-numeric episode metric: "+metric)
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Non-finite episode metric: "+metric)
            values.append(value)
        n = len(values)
        mean = float(fmean(values)) if n else None
        sd = float(stdev(values)) if n >= 2 else None
        original = mission_metrics.get(metric)
        pooled = metric in POOLED
        stats[metric] = {
            "n": n, "n_total": len(records), "n_missing": len(records)-n,
            "mean": mean, "std": sd,
            "mean_pm_std": (f"{mean:.6g} ± {sd:.6g}" if sd is not None
                            else (f"{mean:.6g} ± N/A" if mean is not None else None)),
            "reported_value": original,
            "reported_aggregation": ("pooled_ratio_or_target_weighted_mean"
                                      if pooled else "episode_mean"),
            "std_definition": "sample_std_across_scenarios_ddof1",
        }
        # A pooled ratio can exist even when zero episodes have an individual
        # denominator, and its SD must NOT be mixed with the pooled estimate.
        if pooled:
            stats[metric]["pooled_value"] = original
    return stats


def export_episode_statistics_csv(statistics, destination, *, checkpoint=None):
    """Long-format CSV; easy to compare and filter without dropping old fields."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["checkpoint_global_step", "metric", "reported_aggregation",
              "reported_value", "episode_mean", "episode_sd", "mean_pm_std",
              "valid_episodes", "missing_episodes", "total_episodes"]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for metric, stat in statistics.items():
            writer.writerow({
                "checkpoint_global_step": checkpoint, "metric": metric,
                "reported_aggregation": stat["reported_aggregation"],
                "reported_value": stat["reported_value"],
                "episode_mean": stat["mean"], "episode_sd": stat["std"],
                "mean_pm_std": stat["mean_pm_std"],
                "valid_episodes": stat["n"], "missing_episodes": stat["n_missing"],
                "total_episodes": stat["n_total"],
            })
    return destination

def export_per_episode_metrics_csv(records, statistics, destination):
    """Preserve individual Test scenario metrics for paired analysis and debugging."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed", "episode", "termination_reason", "episode_length",
        *(key for key in statistics if key != "episode_length"),
    ]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {
                "seed": rec["seed"], "episode": rec["episode"],
                "termination_reason": rec.get("termination_reason"),
                "episode_length": rec["episode_length"],
            }
            for metric in statistics:
                if metric == "episode_length":
                    row[metric] = rec["episode_length"]
                elif metric.startswith("comparison/"):
                    row[metric] = rec.get("comparison_metrics", {}).get(metric[11:])
                else:
                    row[metric] = per_episode_metric(rec, metric)
            writer.writerow(row)
    return destination
