"""Paper metrics from real Warp terminal information, shared by every GPU evaluator.

AUC is the normalized time-integral of target completion fraction over the
mission horizon. Targets missing at the horizon are right-censored. Energy
per delivered target is pooled across episodes, not averaged per episode.
"""
import math
from statistics import fmean
from .comparison_metrics import core_mission_metrics


def _times(values, n, horizon, label):
    if not isinstance(values, (list, tuple)) or len(values) != n:
        raise ValueError("Missing per-target " + label + " timestamps")
    seen, capped = [], []
    for v in values:
        if v is None:
            seen.append(False)
            capped.append(horizon)
        else:
            t = float(v)
            if not math.isfinite(t) or not 0 <= t <= horizon + 1.e-5:
                raise ValueError("Invalid " + label + " timestamp")
            seen.append(True)
            capped.append(min(t, horizon))
    return seen, capped


def gpu_episode_record(info, config, *, seed, episode, episode_return):
    n = int(config.world.num_targets)
    horizon = float(config.world.max_steps * config.world.dt_s)
    found, delivered = int(info["targets_found"]), int(info["targets_delivered"])
    if n <= 0 or horizon <= 0 or not 0 <= delivered <= found <= n:
        raise ValueError("Invalid target counts or horizon")
    report = info.get("report_diagnostics")
    if not isinstance(report, dict):
        raise ValueError("Warp terminal record lacks timestamps")
    seen, discovery = _times(report.get("first_discovery_s"), n, horizon, "discovery")
    arrived, delivery = _times(report.get("first_delivery_s"), n, horizon, "GCS delivery")
    if sum(seen) != found or sum(arrived) != delivered:
        raise ValueError("Observed timestamps disagree with terminal counts")
    if any(arrived[i] and (not seen[i] or delivery[i] < discovery[i]-1e-5) for i in range(n)):
        raise ValueError("GCS delivery before discovery")
    success, steps = bool(info["success"]), int(info["step"])
    if success != (delivered == n) or not 0 <= steps <= config.world.max_steps:
        raise ValueError("Invalid termination state")
    network = info.get("network") or {}
    energy = float(info["energy"]["propulsion_energy_j"]) + float(network.get("uav_communication_energy_j", 0))
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("Invalid observed energy")
    sensing = info.get("sensing") or {}
    measured, false = sensing.get("measurements"), sensing.get("false_confirmation")
    if measured is not None and false is not None:
        measured, false = int(measured), int(false)
        if not 0 <= false <= measured:
            raise ValueError("Invalid false-confirmation counters")
    else:
        measured = false = None
    coverage = info.get("coverage_percent")
    if coverage is not None:
        coverage = float(coverage)
        if not math.isfinite(coverage) or not 0 <= coverage <= 100:
            raise ValueError("Invalid map coverage")
    coverage_auc=info.get("coverage_auc")
    if coverage_auc is not None:
        coverage_auc=float(coverage_auc)
        if not math.isfinite(coverage_auc) or not 0 <= coverage_auc <= 1:
            raise ValueError("Invalid observed coverage AUC")
    packet_sent, packet_delivered = ((network.get("generated_packets"),network.get("received_packets"))
                                     if network.get("enabled") else (None,None))
    if packet_sent is not None and packet_delivered is not None:
        packet_sent, packet_delivered = int(packet_sent), int(packet_delivered)
        if not 0 <= packet_delivered <= packet_sent:
            raise ValueError("Invalid packet counters")
    else:
        packet_sent = packet_delivered = None
    d_time, g_time = fmean(discovery), fmean(delivery)
    comparison = {
        "target_discovery_fraction": found/n, "gcs_delivery_fraction": delivered/n,
        "mission_success": float(success),
        "target_discovery_time_capped_mean_s": d_time,
        "target_delivery_time_capped_mean_s": g_time,
        "discovery_auc": max(0., min(1., 1. - d_time/horizon)),
        "delivery_auc": max(0., min(1., 1. - g_time/horizon)),
        "mission_completion_time_capped_s": steps*config.world.dt_s if success else horizon,
        "uav_energy_j": energy,
        "uav_j_per_gcs_delivery": energy/delivered if delivered else None,
        "map_coverage_percent": coverage,
        "coverage_auc": coverage_auc,
        "false_confirmation_per_1000_measurements": (1000.*false/measured if measured else None),
        "false_confirmation_rate_percent": (100.*false/measured if measured else None),
        "first_discovery_to_gcs_delay_delivered_mean_s": report.get("first_discovery_to_delivery_mean_s"),
    }
    return {"seed": int(seed), "episode": int(episode), "return": float(episode_return),
            "success": success, "search_rate_percent": float(info["search_rate_percent"]),
            "delivery_rate_percent": float(info["delivery_rate_percent"]),
            "episode_length": steps, "termination_reason": info.get("termination_reason"),
            "targets_found": found, "targets_delivered": delivered,
            "comparison_metrics": comparison,
            "sensing_counts": {"measurements": measured, "false_confirmation": false},
            "packet_counts": {"sent": packet_sent, "received": packet_delivered},
            "report_diagnostics": {"first_discovery_s": report["first_discovery_s"],
                                   "first_delivery_s": report["first_delivery_s"]}}


def aggregate_gpu_mission_metrics(records):
    if not records:
        raise ValueError("No observed validation episodes")
    core = core_mission_metrics(records)
    comparison = [r["comparison_metrics"] for r in records]
    def average(field):
        values = [r.get(field) for r in comparison]
        return float(fmean(values)) if all(v is not None for v in values) else None
    sensing = [r["sensing_counts"] for r in records]
    packets = [r["packet_counts"] for r in records]
    has_sensing = all(r["measurements"] is not None and r["false_confirmation"] is not None for r in sensing)
    has_packets = all(r["sent"] is not None and r["received"] is not None for r in packets)
    measures = sum(r["measurements"] for r in sensing) if has_sensing else 0
    sent = sum(r["sent"] for r in packets) if has_packets else 0
    # Pool real target-level GCS delays. Averaging per-episode delays would
    # overweight episodes that delivered only one target.
    delivered_delays = []
    for rec in records:
        report = rec["report_diagnostics"]
        for discovery, delivery in zip(report["first_discovery_s"],
                                       report["first_delivery_s"]):
            if discovery is not None and delivery is not None:
                delivered_delays.append(float(delivery) - float(discovery))
    mission = {**core,
        "discovery_auc": average("discovery_auc"),
        "delivery_auc": average("delivery_auc"),
        "mission_success_rate_percent": 100. * fmean(int(r["success"]) for r in records),
        "mission_completion_time_capped_s": average("mission_completion_time_capped_s"),
        "map_coverage_percent": average("map_coverage_percent"),
        "coverage_auc": average("coverage_auc"),
        "false_confirmation_per_1000_measurements": (1000.*sum(r["false_confirmation"] for r in sensing)/measures if measures else None),
        "false_confirmation_rate_percent": (100.*sum(r["false_confirmation"] for r in sensing)/measures if measures else None),
        "packet_pdr_percent": (100.*sum(r["received"] for r in packets)/sent if sent else None),
        "first_discovery_to_gcs_delay_delivered_mean_s": (
            float(fmean(delivered_delays)) if delivered_delays else None),
        "mean_total_uav_energy_j": average("uav_energy_j"),
        "mean_targets_found": fmean(r["targets_found"] for r in records),
        "mean_targets_delivered": fmean(r["targets_delivered"] for r in records),
    }
    return {"core_metrics": core, "mission_metrics": mission}


def gpu_eval_wandb_payload(result):
    source = result["source"]
    values = {"global_step": int(source["global_step"]),
              "gpu_eval/checkpoint_global_step": int(source["global_step"]),
              "gpu_eval/checkpoint_episode": int(source["episodes"]),
              "gpu_eval/policy_version": int(source["policy_version"]),
              "gpu_eval/mean_return": float(result["mean_return"]),
              "gpu_eval/search_rate_percent": float(result["mean_search_rate_percent"]),
              "gpu_eval/delivery_rate_percent": float(result["mean_delivery_rate_percent"]),
              "gpu_eval/mission_success_rate_percent": 100.*float(result["success_rate"])}
    # Do not reconstruct variance from mission success or another proxy.
    # This is the observed std of held-out episodic returns when available.
    # Held-out return variance comes from evaluated episodes, not a proxy.
    # Older GPU evaluators name this `mean_return_std` in the diagnostic
    # payload; newer callers may supply `return_std` at the top level.
    return_std = result.get("return_std")
    if return_std is None:
        return_std = result.get("episode_statistics", {}).get("return", {}).get("std")
    if isinstance(return_std, (int, float)) and math.isfinite(return_std):
        values["gpu_eval/mean_return_std"] = float(return_std)
        values["gpu_eval/return_std"] = float(return_std)
    for key, val in result.get("mission_metrics", {}).items():
        if val is not None:
            if not isinstance(val, (float,int)) or not math.isfinite(val):
                raise ValueError("Nonfinite GPU Evaluation metric: "+key)
            values["gpu_eval/"+key] = float(val)
    return values


PERIODIC_EVAL_PREFIX = "gpu_eval_periodic/"
PERIODIC_EVAL_AXIS = PERIODIC_EVAL_PREFIX + "checkpoint_global_step"
_PERIODIC_METRICS = (
    # Real held-out episodic return/dispersion at this checkpoint.
    "mean_return", "return_std",
    "discovery_rate_percent", "gcs_delivery_rate_percent", "discovery_auc",
    "delivery_auc", "map_coverage_percent", "coverage_auc", "energy_per_delivered_target_j",
    "gcs_delivery_time_capped_mean_s", "mission_success_rate_percent",
)


def periodic_eval_wandb_payload(transition, metrics):
    """Log observed milestone values on their real checkpoint transition axis."""
    # A checkpoint evaluated later is plotted at its TRUE collection step.
    values = {"global_step": int(transition), PERIODIC_EVAL_AXIS: int(transition)}
    for key in _PERIODIC_METRICS:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[PERIODIC_EVAL_PREFIX + key] = value
    return values


def periodic_gpu_eval_wandb_payload(payload):
    metrics = {key.removeprefix("gpu_eval/"): value for key, value in payload.items()}
    metrics.setdefault("discovery_rate_percent", metrics.get("search_rate_percent"))
    metrics.setdefault("gcs_delivery_rate_percent", metrics.get("delivery_rate_percent"))
    return periodic_eval_wandb_payload(metrics["checkpoint_global_step"], metrics)
