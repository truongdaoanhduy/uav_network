"""Offline cross-layer evidence audit (symptoms and competing explanations).

Reads local JSONL or immutable incremental diagnostic .tar.gz parts. Does not
mutate source files, environment, checkpoints or any running W&B/Kaggle job.
Can be run after a crash if the last completed diagnostic part survived.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import tarfile

MAX_LINE_BYTES = 2_000_000
MAX_RECORDS_PER_SOURCE = 250_000
SOURCE_NAMES = (
    "training-health-snapshots.jsonl", "mission-cause-snapshots.jsonl",
    "forensic-transitions.jsonl", "run-provenance.jsonl",
    "learner-layer-snapshots.jsonl",
    "action-cbf-samples.jsonl", "episodes.jsonl",
    "cohorts.jsonl", "diagnostics.jsonl", "evaluations/events.jsonl",
)
# Per-root-cause classification requires joint context, not a lone threshold.
# Source/field names below match runtime metrics exactly.
PROBES = (
    ("Actor tanh saturation", "learner",
     "actor_motion_tanh_derivative_lt_1e3_fraction", .5, "gt",
     "Compare Q-action vs pre-tanh gradient on the same XYZ axis."),
    ("Actor encoder gradient collapse", "learner",
     "actor_first_encoder_grad_postclip", 1e-9, "lt",
     "Compare first/last encoder gradients before/after clipping and LR."),
    ("Critic action gradient near zero (X)", "learner",
     "gradprobe/critic_facing_dloss_da_x_abs_mean", 1e-8, "lt",
     "Check TD/Q calibration; absence of a gradient is not proof of incorrect Q."),
    ("Tanh blocks upstream signal (X)", "learner",
     "gradprobe/pretanh_dloss_dz_x_abs_mean", 1e-8, "lt",
     "Require nonzero critic-facing gradient AND near-zero tanh derivative."),
    ("High safety intervention", "action_cbf",
     "action_probe/cbf_projected_fraction", .7, "gt",
     "Cross-check >1m/s correction; tiny projection is not a stall."),
    ("Material CBF correction", "action_cbf",
     "action_probe/cbf_correction_gt_1mps_fraction", .3, "gt",
     "Compare raw unsafe commands, post-safety blocked and delivery."),
    ("Boundary outward commands", "action_cbf",
     "action_probe/command_outward_at_boundary_fraction", .3, "gt",
     "Inspect X/Y/Z saturation and position-specific motion commands."),
    ("UAV stalls despite command", "action_cbf",
     "action_probe/stalled_with_strong_command_fraction", .2, "gt",
     "Inspect safety fallback, energy, collisions and current speed."),
    ("High confirmed-to-delivery gap", "mission",
     "mission_probe/found_undelivered_target_fraction", .3, "gt",
     "Find whether report is in FIFO or in-flight, then inspect network."),
    ("Discovered targets absent from FIFO", "mission",
     "mission_probe/pending_not_in_fifo_fraction", .5, "gt",
     "Could be in-flight; inspect packet/session and report recovery counters."),
    ("Confirmed targets absent from FIFO and active packet queue", "mission",
     "mission_probe/pending_without_fifo_or_active_packet_fraction", .5, "gt",
     "Inspect report recovery and expired/dropped counters; active sessions are counted separately."),
    ("Active packet retries", "mission",
     "mission_probe/active_packet_retry_mean", 2, "gt",
     "Inspect MAC collisions, CCA busy and PHY power margin."),
    ("Optimistic relay geometry disconnected", "mission",
     "mission_probe/pending_with_report_but_no_optimistic_path_fraction", .5, "gt",
     "No geometric multi-hop chain; cannot be fixed by PHY alone."),
    ("Reports approaching TTL", "mission",
     "mission_probe/pending_report_ttl_gt_75pct_fraction", .5, "gt",
     "Check packet retry, queue, buffer, routing and expiry counters."),
    ("Reports fill UAV FIFO", "mission",
     "mission_probe/buffer_full_fraction", .5, "gt",
     "Check repeated confirmations, drops and custody bottlenecks."),
    ("Low battery", "mission",
     "mission_probe/battery_below_10pct_fraction", .2, "gt",
     "Separate energy depletion from motion stall and transmission failure."),
    ("Reward decomposition discrepancy", "mission",
     "mission_probe/reward_accounting_abs_discrepancy_mean", 1e-3, "gt",
     "Audit physics reward components; component sum must match step reward."),
    ("Shaping dominates observed rewards", "mission",
     "mission_probe/reward_shaping_share_of_abs_components_mean", .9, "gt",
     "Compare successful delivery and reward gradient rather than assuming shaping is bad."),
    ("Exploration overlap", "mission",
     "mission_probe/visited_sample_uav_overlap_fraction", .8, "gt",
     "Sample covers first 16 worlds only; inspect spatial trajectories separately."),
    ("False confirmations", "mission",
     "mission_probe/sensing/interval_false_confirmation_per_1000_measurements", 5, "gt",
     "Check altitude, Bayes posterior, Pf and verified-empty belief reset."),
    ("Replay evictions without learning", "replay",
     "replay/evicted_unsampled_fraction", .05, "gt",
     "Inspect replay fill, CPU/GPU split, age and learner throughput."),
    ("GPU1 low headroom", "resource",
     "memory/cuda1_driver_free_now_gib", 2, "lt",
     "Check peak allocator, driver min-free, co-resident workloads and OOM events."),
    ("GPU0 low headroom", "resource",
     "memory/cuda0_driver_free_now_gib", 2, "lt",
     "Check Warp allocation and high-water driver usage."),
    ("Host RAM low headroom", "resource",
     "memory/cgroup_headroom_now_gib", 3, "lt",
     "Check RSS peak, cgroup limit, shared memory and kernel OS kill records."),
    ("PQL collector uses stale policy", "system",
     "pql/collector_policy_lag", 2, "gt",
     "Inspect mailbox synchronization and policy publication lag."),
    ("PQL critic uses stale policy", "system",
     "pql/critic_policy_lag", 2, "gt",
     "Inspect async critic policy refresh and queue length."),
    ("PQL actor uses stale critic", "system",
     "pql/policy_critic_lag", 8, "gt",
     "Inspect queued policy updates, worker backpressure and actor timing."),
    ("PQL collector queue backs up", "system",
     "pql/queued_collections", 1, "gt",
     "Check policy learner throughput, memory peaks, worker wait time."),
    ("MASAC logstd collapses to lower clamp", "learner",
     "masac_probe/std_clamped_low_fraction", .5, "gt",
     "Check log_std/mean gradients, alpha entropy and route masks."),
    ("MASAC logstd saturates upper clamp", "learner",
     "masac_probe/std_clamped_high_fraction", .5, "gt",
     "Inspect entropy alpha schedule, target entropy and replay coverage."),
)


def finite(value):
    return type(value) in (int, float) and math.isfinite(float(value))


def read_jsonl_bytes(data, limit=MAX_RECORDS_PER_SOURCE):
    records = []
    bad = 0
    for line in data.splitlines():
        if len(records) >= limit:
            break
        if not line.strip():
            continue
        if len(line) > MAX_LINE_BYTES:
            bad += 1
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                bad += 1
        except (UnicodeDecodeError, ValueError, TypeError):
            bad += 1
    return records, bad


def load_records(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    records = {}
    provenance = {}
    for name in SOURCE_NAMES:
        path = folder / name
        if path.is_file():
            rows, invalid = read_jsonl_bytes(path.read_bytes())
            records[name] = rows
            provenance[name] = {"mode": "local_jsonl", "rows": len(rows),
                                "invalid_rows": invalid}
    # Local complete JSONL supersedes incremental archived chunks; never count
    # both. Archived members can repeat across parts only if uploader failed;
    # this normal archive implementation uses nonoverlapping byte offsets.
    archive_files = sorted((folder / "diagnostic-archives").glob("*.tar.gz"))
    archived = {k: [] for k in SOURCE_NAMES if k not in records}
    invalid_counts = Counter()
    for archive_path in archive_files:
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar:
                    if member.name not in archived or not member.isfile():
                        continue
                    if member.size > 64_000_000 or member.size < 0:
                        invalid_counts[member.name] += 1
                        continue
                    stream = tar.extractfile(member)
                    if stream is None:
                        continue
                    rows, bad = read_jsonl_bytes(stream.read(), MAX_RECORDS_PER_SOURCE - len(archived[member.name]))
                    archived[member.name].extend(rows)
                    invalid_counts[member.name] += bad
        except (OSError, tarfile.TarError):
            invalid_counts["corrupt_archive"] += 1
    for name, rows in archived.items():
        if rows:
            records[name] = rows
            provenance[name] = {"mode": "incremental_tar_gz", "rows": len(rows),
                                "invalid_rows": invalid_counts[name]}
    return records, provenance, {name: count for name, count in invalid_counts.items() if count > 0}


def _last_three_parts(values):
    if not values:
        return None
    segments = []
    for lo, hi in ((0, max(1, len(values)//3)),
                   (len(values)//3, max(1, 2*len(values)//3)),
                   (2*len(values)//3, len(values))):
        selected = values[lo:hi]
        segments.append(statistics.median(selected) if selected else None)
    return segments


def _metric_series(snapshots, source, metric):
    result = []
    for snapshot in snapshots:
        sub = snapshot.get(source, {})
        if not isinstance(sub, dict):
            continue
        value = sub.get(metric)
        step = snapshot.get("global_step")
        if finite(value) and type(step) is int:
            result.append((step, float(value)))
    return sorted(result)


def _evaluate_probe(snapshots, item):
    title, source, key, threshold, op, next_test = item
    series = _metric_series(snapshots, source, key)
    if not series:
        return {"name": title, "status": "not_measured", "metric": key,
                "source": source, "next_check": next_test}
    observed = [p[1] for p in series]
    hits = [(step, value) for step, value in series
            if (value > threshold if op == "gt" else value < threshold)]
    return {
        "name": title, "status": "symptom_seen" if hits else "not_triggered",
        "metric": key, "source": source, "threshold": threshold, "condition": op,
        "n_samples": len(series), "hit_count": len(hits),
        "first_trigger": dict(global_step=hits[0][0], value=hits[0][1]) if hits else None,
        "median": statistics.median(observed), "latest": observed[-1],
        "early_middle_late_median": _last_three_parts(observed),
        "next_check": next_test,
    }


# Co-occurring evidence in a single recorded collection window. This is
# stronger *localization* than isolated thresholds, not causal proof.
MECHANISM_PATTERNS = (
    ("critic_to_tanh_gradient_bottleneck", "Critic still drives X, but tanh blocks X",
     (("learner", "gradprobe/critic_facing_dloss_da_x_abs_mean", "gt", 1e-6),
      ("learner", "gradprobe/pretanh_dloss_dz_x_abs_mean", "lt", 1e-8),
      ("learner", "gradprobe/actor_tanh_derivative_x_mean_fullbatch", "lt", 1e-3)),
     "Inspect pre-tanh X logits and whether Critic pushes toward saturated sign."),
    ("critic_q_action_flat", "Critic X-action signal is nearly flat with nonsaturated action",
     (("learner", "gradprobe/critic_facing_dloss_da_x_abs_mean", "lt", 1e-8),
      ("learner", "gradprobe/actor_tanh_derivative_x_mean_fullbatch", "gt", .05)),
     "Check Q calibration, rewards, TD error, next-action targets and Q1/Q2 disagreement."),
    ("relu_encoder_gradient_bottleneck", "Actor head receives gradient but encoder does not",
     (("learner", "gradprobe/pretanh_dloss_dz_x_abs_mean", "gt", 1e-6),
      ("learner", "gradprobe/encoder_first_grad_preclip_norm", "lt", 1e-9),
      ("learner", "gradprobe/encoder_final_relu_blocked_fraction_fullbatch", "gt", .8)),
     "Compare layer activation distributions and first/last encoder gradients."),
    ("clipping_limits_actor", "Actor first encoder clipped strongly",
     (("learner", "gradprobe/encoder_first_grad_preclip_norm", "gt", 1e-3),
      ("learner", "gradprobe/encoder_first_grad_postclip_norm", "lt", 1e-4),
      ("learner", "gradprobe/monitored_parameter_clipped_fraction", "gt", .5)),
     "Check pre/post clipping and config grad_clip_mode on the same update."),
    ("rf_action_gate", "RF Q signal exists, but no RF gradient is delivered",
     (("learner", "gradprobe/critic_facing_dloss_da_rf_abs_mean", "gt", 1e-6),
      ("learner", "gradprobe/pretanh_dloss_dz_rf_abs_mean", "lt", 1e-9),
      ("action_cbf", "action_probe/mask_silent_only_fraction", "gt", .8)),
     "Align Actor update replay mask versus *collector* mask; sample populations differ."),
    ("cbf_unsafe_stall", "Material CBF intervention, outward commands and UAV stalls",
     (("action_cbf", "action_probe/cbf_correction_gt_1mps_fraction", "gt", .3),
      ("action_cbf", "action_probe/command_outward_at_boundary_fraction", "gt", .2),
      ("action_cbf", "action_probe/stalled_with_strong_command_fraction", "gt", .1)),
     "Inspect scene-specific motion and evaluate nominal policy vs shielded execution safely."),
    ("confirmed_without_report", "Many confirmed targets lack a current FIFO report",
     (("mission", "mission_probe/pending_target_observation_count", "gt", 10),
      ("mission", "mission_probe/pending_not_in_fifo_fraction", "gt", .5)),
     "Check in-flight custody and report creation/drop/recovery before blaming lost reports."),
    ("geometric_relay_disconnected", "Reports exist but no optimistic GCS route",
     (("mission", "mission_probe/pending_target_observation_count", "gt", 10),
      ("mission", "mission_probe/pending_with_report_but_no_optimistic_path_fraction", "gt", .5)),
     "Inspect UAV spacing/mobility/range/altitude; geometry excludes PHY and buffer constraints."),
    ("report_ttl_buffer_bottleneck", "Report TTL nears expiry while buffers are full",
     (("mission", "mission_probe/pending_report_observation_count", "gt", 10),
      ("mission", "mission_probe/pending_report_ttl_gt_75pct_fraction", "gt", .5),
      ("mission", "mission_probe/buffer_full_fraction", "gt", .5)),
     "Inspect retries, custody and buffered target aging in episode reports."),
    ("masac_stochastic_variance_clamped", "MASAC low log-std clamp and entropy collapse",
     (("learner", "masac_probe/std_clamped_low_fraction", "gt", .5),
      ("learner", "continuous_entropy", "lt", .2)),
     "Compare alpha_continuous, action-mask eligibility and gradients through log_std."),
    ("pql_policy_lag_with_backpressure", "PQL collector lag coincides with queue backpressure",
     (("system", "pql/collector_policy_lag", "gt", 2),
      ("system", "pql/queued_collections", "gt", 1)),
     "Separate asynchronous staleness from policy stochasticity and replica seed variation."),
)


def _evaluate_patterns(snapshots):
    result = []
    for code, label, conditions, followup in MECHANISM_PATTERNS:
        checked = 0
        positive = []
        first = None
        last = None
        for snapshot in snapshots:
            values = []
            for source, metric, op, threshold in conditions:
                raw = snapshot.get(source, {})
                value = raw.get(metric) if isinstance(raw, dict) else None
                if not finite(value):
                    break
                values.append((metric, float(value), op, threshold))
            if len(values) < len(conditions):
                continue
            checked += 1
            if all(value > threshold if op == "gt" else value < threshold
                   for _metric, value, op, threshold in values):
                step = snapshot.get("global_step")
                if type(step) is int:
                    positive.append(step)
                    if first is None:
                        first = {"global_step": step,
                                 "evidence": {m:v for m,v,_o,_t in values}}
                    last = step
        result.append({
            "id": code, "description": label,
            "status": "not_evaluable" if not checked else "cooccurrence_seen" if positive else "not_seen",
            "fully_observed_windows": checked, "matching_windows": len(positive),
            "first_match": first, "last_match_global_step": last,
            "necessary_metrics": [f"{source}/{metric}" for source,metric,_,_ in conditions],
            "next_discriminating_test": followup,
            "interpretation": "Co-occurrence, not causality; matching signals must be compared to episode and held-out evaluation outcomes.",
        })
    return result


def _episode_analysis(rows):
    if not rows:
        return {"status": "not_measured", "reason": "No episodes.jsonl records"}
    valid = [r for r in rows if finite(r.get("search_rate_percent")) and finite(r.get("delivery_rate_percent"))]
    if not valid:
        return {"status": "invalid", "reason": "Missing finite mission outcomes"}
    result = {"status": "measured", "episodes": len(valid),
              "mean_search_percent": statistics.mean(float(r["search_rate_percent"]) for r in valid),
              "mean_delivery_percent": statistics.mean(float(r["delivery_rate_percent"]) for r in valid),
              "mean_search_delivery_gap_percent": statistics.mean(
                  float(r["search_rate_percent"]) - float(r["delivery_rate_percent"]) for r in valid),
              "success_rate": (statistics.mean(float(r["success"]) for r in valid)
                               if all(finite(r.get("success")) for r in valid) else None),
              "termination_reasons": dict(Counter(r.get("termination_reason", "unknown") for r in valid)),
              "total_report_expired": sum(int(r.get("expired_reports", 0)) for r in valid),
              "total_report_dropped": sum(int(r.get("dropped_reports", 0)) for r in valid),
              "total_report_regenerated": sum(int(r.get("regenerated_reports", 0)) for r in valid),
              "episode_first_global_step": min(int(r.get("global_step", 0)) for r in valid),
              "episode_last_global_step": max(int(r.get("global_step", 0)) for r in valid)}
    result["early_middle_late_delivery_percent"] = _last_three_parts(
        [float(r["delivery_rate_percent"]) for r in sorted(valid, key=lambda x: x.get("global_step", 0))])
    raw_network = [r.get("network", {}) for r in valid]
    generated = sum(int(x.get("generated_packets", 0)) for x in raw_network if isinstance(x, dict))
    received = sum(int(x.get("received_packets", 0)) for x in raw_network if isinstance(x, dict))
    result["network_generated_packets"] = generated
    result["network_received_packets"] = received
    result["network_pooled_packet_pdr"] = received / generated if generated else None
    diagnostics = [x.get("diagnostics", {}) for x in raw_network if isinstance(x, dict)]
    agg = Counter()
    for raw in diagnostics:
        if not isinstance(raw, dict):
            continue
        for key in ("data_attempts", "data_retries", "cca_busy", "cca_checks",
                    "data_below_required_power_attempts", "data_decode_count",
                    "data_phy_successes", "custody_peer_transfers",
                    "relay_toward_gcs_transfers", "relay_away_from_gcs_transfers"):
            if finite(raw.get(key)):
                agg[key] += int(raw[key])
    result["network_diagnostic_counts"] = dict(agg)
    for label, numerator, denominator in (
        ("retry_per_data_attempt", "data_retries", "data_attempts"),
        ("cca_busy_fraction", "cca_busy", "cca_checks"),
        ("data_below_required_power_fraction", "data_below_required_power_attempts", "data_attempts"),
        ("data_phy_success_fraction", "data_phy_successes", "data_decode_count"),
        ("relay_toward_gcs_fraction", "relay_toward_gcs_transfers", "custody_peer_transfers"),
    ):
        d = agg[denominator]
        result["network_" + label] = agg[numerator] / d if d else None
    measurements = 0
    false_confirmations = 0
    for r in valid:
        sense = r.get("sensing", {})
        if isinstance(sense, dict):
            measurements += int(sense.get("measurements", 0))
            false_confirmations += int(sense.get("false_confirmation", 0))
    result["sensing_measurements"] = measurements
    result["sensing_false_confirmations"] = false_confirmations
    result["sensing_false_confirmation_per_1000_measurements"] = (
        1000 * false_confirmations / measurements if measurements else None)
    return result


def _configuration_analysis(folder):
    """Freeze important scientific controls, and flag *missing* seed evidence."""
    path = Path(folder) / "resolved_config.yaml"
    if not path.is_file():
        return {"status": "not_measured", "reason": "resolved_config.yaml missing"}
    try:
        import yaml
        config = yaml.safe_load(path.read_text())
    except (ImportError, OSError, ValueError, TypeError):
        return {"status": "invalid", "reason": "unreadable resolved configuration"}
    if not isinstance(config, dict):
        return {"status": "invalid", "reason": "config is not a mapping"}
    groups = {key: config.get(key, {}) for key in
              ("world", "sensor", "network", "energy", "reward", "masac",
               "matd3", "runtime", "tracking", "pql")}
    get = lambda group, key: groups[group].get(key) if isinstance(groups[group], dict) else None
    algorithm = get("runtime", "algorithm")
    train_seed = get("runtime", "seed")
    validation_seed = get("tracking", "periodic_evaluation_seed")
    eval_n = get("tracking", "periodic_evaluation_episodes")
    batch = get(algorithm, "batch_size") if algorithm in ("masac", "matd3") else None
    updates = get(algorithm, "updates_per_step") if algorithm in ("masac", "matd3") else None
    envs = get("runtime", "num_envs")
    data_reuse = (updates * batch / envs if all(type(x) is int and x > 0
                                               for x in (updates,batch,envs)) else None)
    selected = {
        "algorithm": algorithm,
        "pql_enabled": get("pql", "enabled"),
        "pql_deterministic_requested": get("runtime", "deterministic"),
        "training_seed": train_seed,
        "validation_first_seed": validation_seed,
        "validation_episode_count": eval_n,
        "eval_seed": get("runtime", "eval_seed"),
        "evaluation_episodes_configured": get("runtime", "eval_episodes"),
        "batch": batch, "updates_per_collection": updates,
        "num_envs": envs, "sample_reuse_proxy_updates_times_batch_per_env": data_reuse,
        "replay_capacity": get(algorithm, "replay_capacity") if algorithm in ("masac", "matd3") else None,
        "replay_cpu_fraction": get("runtime", "replay_cpu_fraction"),
        "shared_actor": get(algorithm, "share_actor_parameters") if algorithm in ("masac", "matd3") else None,
        "shared_critic": get(algorithm, "share_critic_parameters") if algorithm in ("masac", "matd3") else None,
        "actor_layernorm": get(algorithm, "actor_layer_norm") if algorithm in ("masac", "matd3") else None,
        "output_normalization": get("matd3", "output_normalization") if algorithm=="matd3" else None,
        "grad_clip_mode": get(algorithm, "grad_clip_mode") if algorithm in ("masac", "matd3") else None,
        "grad_clip_norm": get(algorithm, "grad_clip_norm") if algorithm in ("masac", "matd3") else None,
        "motion_safety": get("world", "motion_safety"),
        "verified_empty_belief": get("sensor", "verified_empty_belief"),
        "belief_patch_cells": get("sensor", "belief_patch_cells"),
        "belief_coarse_cells": get("sensor", "belief_coarse_cells"),
        "confirmation_threshold": get("sensor", "confirmation_threshold"),
        "sensor_fine_altitude_m": get("sensor", "fine_altitude_m"),
        "targets": get("world", "num_targets"),
        "uavs": get("world", "num_uavs"),
        "map_size_m": get("world", "map_size_m"),
        "max_steps": get("world", "max_steps"),
        "report_buffer_bytes": get("network", "buffer_bytes"),
        "report_payload_bytes": get("network", "report_bytes"),
        "report_ttl_s": get("network", "report_ttl_s"),
        "power_control": get("network", "power_control"),
        "receiver_admission_mask": get("network", "receiver_admission_mask"),
        "reward_unsafe_action_coefficient": get("reward", "unsafe_action"),
        "reward_shield_correction_coefficient": get("reward", "shield_correction"),
        "reward_shield_intervention_coefficient": get("reward", "shield_intervention"),
        "reward_information_gain_coefficient": get("reward", "information_gain"),
        "reward_investigation_coefficient": get("reward", "investigation_shaping"),
        "compile_models": get("runtime", "compile_models"),
        "amp_mode": get("runtime", "amp_mode"),
        "critic_chunk_size": get("runtime", "critic_chunk_size"),
        "vectorize_critics": get("runtime", "vectorize_critics"),
    }
    notes = []
    if type(eval_n) is int and eval_n < 30:
        notes.append("Periodic validation uses <30 episodes; evaluate selected checkpoints on separate held-out seeds.")
    if get("pql", "enabled"):
        notes.append("PQL policy/critic are asynchronous; collector and learner snapshots may represent different update times.")
    if selected["verified_empty_belief"] not in (None, 0):
        notes.append("False-confirmation verified-empty belief reset is configured as nonzero; do not call it missing.")
    if selected["reward_unsafe_action_coefficient"] not in (None, 0):
        notes.append("Unsafe-action reward penalty is already configured; do not assume it is absent.")
    if selected["reward_shield_intervention_coefficient"] not in (None, 0):
        notes.append("CBF-intervention penalty is already configured; measure its efficacy before modifying reward.")
    return {"status": "measured", "fields": selected, "notes": notes}


def _performance_analysis(folder):
    path = Path(folder) / "summary.json"
    if not path.is_file():
        return {"status": "not_measured"}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {"status": "invalid"}
    keys = ("elapsed_s", "startup_s", "joint_transitions_per_s",
            "collection_wait_s", "learner_s", "replay_sample_host_s",
            "optimizer_update_host_s", "stop_reason", "episodes",
            "global_step", "updates", "sample_updates_per_transition")
    evidence = {k: data[k] for k in keys if k in data}
    for nested in ("resource_memory", "replay", "reproducibility", "periodic_evaluation"):
        if isinstance(data.get(nested), dict):
            evidence[nested] = data[nested]
    return {"status": "measured", "values": evidence}


def _evaluation_analysis(folder):
    summaries = []
    for path in sorted((Path(folder)/"evaluations").glob("*/metrics.json")):
        try:
            data = json.loads(path.read_text())
            if not all(finite(data.get(key)) for key in
                       ("mean_delivery_rate_percent", "mean_search_rate_percent", "success_rate")):
                continue
            summaries.append({
                "source": str(path.relative_to(folder)),
                "episodes": data.get("episodes"),
                "mean_delivery_percent": data["mean_delivery_rate_percent"],
                "mean_search_percent": data["mean_search_rate_percent"],
                "success_rate": data["success_rate"],
                "scenario_hash": data.get("protocol", {}).get("scenario_hash"),
                "seed_overlap": data.get("protocol", {}).get("training_seed_overlap"),
                "validation_seed_overlap": data.get("protocol", {}).get("validation_seed_overlap"),
                "step": data.get("protocol", {}).get("training_global_step"),
            })
        except (OSError, ValueError, TypeError):
            continue
    return {
        "status": "measured" if summaries else "not_measured",
        "evaluations": summaries, "evaluation_count": len(summaries),
        "small_validation_count": sum(1 for r in summaries
                                      if type(r["episodes"]) is int and r["episodes"] < 30),
    }




def _forensic_analysis(records):
    """Never claim world-complete causal trace from sparse sampled records."""
    if not records:
        return {"status":"not_measured","records":0,
                "scope":"no selected-world transitions available"}
    errors = []
    worlds = set()
    exact_steps = 0
    for record in records:
        worlds.add(record.get("world_index"))
        account = record.get("reward_accounting", {})
        if account.get("is_exactly_next_step"):
            exact_steps += 1
        value = account.get("step_abs_discrepancy")
        if finite(value) and value > .001:
            errors.append({"global_step":record.get("global_step_after"),
                           "world":record.get("world_index"),"abs_diff":value})
    return {"status":"sampled","records":len(records),
            "world_ids":sorted(w for w in worlds if type(w)==int),
            "exact_adjacent_transitions":exact_steps,
            "reward_discrepancies_gt_1e3":errors[:20],
            "reward_discrepancy_count":len(errors),
            "sample_record_keys":sorted(records[0]),
            "scope":"Only worlds/steps explicitly sampled, not all transitions. "
                    "Raw observation and critic state in compressed forensic-transitions.jsonl."}


def _layerwise_gradient_analysis(rows):
    """Summarize sparse real learner updates, never infer missing as zero."""
    if not rows:
        return {"status":"not_measured","records":0,
                "caveat":"No update-level layer gradient records in this archive."}
    zero_or_missing={}
    actor_evaluated=critic_evaluated=0
    for item in rows:
        values=item.get("metrics",{})
        if not isinstance(values,dict):
            continue
        for name,value in values.items():
            if not isinstance(name,str) or not name.startswith("layerprobe/"):
                continue
            if name.endswith(("/postclip_grad_norm",
                              "/postclip_norm_min_across_modules")):
                scope=name.split("/",2)[1]
                if scope=="actor" and item.get("policy_updated_in_metric")!=1:
                    # MATD3 may carry forward previous Actor metrics when no
                    # Actor update ran. This is NOT gradient disappearance.
                    continue
                if name.endswith("/postclip_grad_norm"):
                    if scope=="actor":
                        actor_evaluated+=1
                    else:
                        critic_evaluated+=1
                if finite(value) and value<=1e-12:
                    zero_or_missing.setdefault(name,[]).append({
                        "global_step":item.get("global_step"),
                        "critic_updates":item.get("critic_updates"),
                        "actor_updates":item.get("actor_updates"),
                        "grad_norm":value,
                        "meaning":"min_across_modules may indicate a single agent's silent gradient"
                                   if name.endswith("min_across_modules") else "mean gradient norm",
                    })
            if name.endswith("/missing_gradient_fraction") and finite(value) and value>0:
                if name.startswith("layerprobe/actor/") and item.get("policy_updated_in_metric")!=1:
                    continue
                zero_or_missing.setdefault(name,[]).append({
                    "global_step":item.get("global_step"),
                    "fraction_missing":value,
                })
    return {
        "status":"measured","records":len(rows),
        "actor_layer_gradient_points":actor_evaluated,
        "critic_layer_gradient_points":critic_evaluated,
        "near_zero_or_missing_layer_events":zero_or_missing,
        "near_zero_or_missing_layer_event_count":sum(len(x) for x in zero_or_missing.values()),
        "caveat":"Gradient zeros are symptoms. Delayed actor updates are excluded "
                 "from absent-Actor classification. Full per-layer raw metrics reside "
                 "in learner-layer-snapshots.jsonl inside HF tar.gz; "
                 "sparse cadence does not observe every optimizer step.",
    }


def analyze(folder):
    folder = Path(folder)
    sources, provenance, archive_errors = load_records(folder)
    snapshots = sources.get("training-health-snapshots.jsonl", [])
    episodes = sources.get("episodes.jsonl", [])
    findings = [_evaluate_probe(snapshots, row) for row in PROBES]
    result = {
        "schema": 1, "label": "observational evidence triage, not root-cause proof",
        "source_directory": str(folder),
        "provenance": provenance, "archive_errors": archive_errors,
        "snapshots_observed": len(snapshots), "episodes_observed": len(episodes),
        "findings": findings, "mechanism_patterns": _evaluate_patterns(snapshots),
        "forensic": _forensic_analysis(sources.get("forensic-transitions.jsonl", [])),
        "layerwise_gradients": _layerwise_gradient_analysis(
            sources.get("learner-layer-snapshots.jsonl", [])),
        "episodes": _episode_analysis(episodes),
        "evaluation": _evaluation_analysis(folder),
        "configuration": _configuration_analysis(folder),
        "performance": _performance_analysis(folder),
    }
    # Missing instrumentation is a first-class result, never recast as zero.
    result["coverage"] = {
        "measured_probes": sum(r["status"] != "not_measured" for r in findings),
        "symptoms_seen": sum(r["status"] == "symptom_seen" for r in findings),
        "missing_probes": [r["name"] for r in findings if r["status"] == "not_measured"],
    }
    return result


def render_markdown(data):
    e = data["episodes"]
    lines = [
        "# MATD3/UAV cross-layer evidence report",
        "",
        "**Interpretation:** observed symptoms only; never a confirmed causal root cause from a threshold.",
        "",
        f"- Source: `{data['source_directory']}`",
        f"- Health samples: {data['snapshots_observed']}; episodes: {data['episodes_observed']}",
        f"- Probes measured: {data['coverage']['measured_probes']}/{len(data['findings'])}",
        "",
        "## Forensic transition evidence",
        "",
        f"- Captured transitions: {data.get('forensic',{}).get('records',0)}; "
        f"scope: {data.get('forensic',{}).get('status','not_measured')}",
        f"- Sampled reward inconsistencies (>0.001): "
        f"{data.get('forensic',{}).get('reward_discrepancy_count',0)}",
        "",
        "## Per-layer Actor/Q1/Q2 gradient evidence",
        "",
        f"- Gradient update snapshots: {data.get('layerwise_gradients',{}).get('records',0)}; "
        f"status: {data.get('layerwise_gradients',{}).get('status','not_measured')}",
        f"- Zero/missing layer symptoms: "
        f"{data.get('layerwise_gradients',{}).get('near_zero_or_missing_layer_event_count',0)}",
        "- Zeros during a delayed MATD3 Actor step are not treated as vanishing; "
        "consult sampled activation/backward gradients and clipping.",
        "",
        "## Episode evidence",
        "",
    ]
    if e.get("status") == "measured":
        lines.extend([
            f"- Mean search: {e['mean_search_percent']:.2f}%; mean delivery: {e['mean_delivery_percent']:.2f}%; gap: {e['mean_search_delivery_gap_percent']:.2f} percentage points",
            f"- Success: {e['success_rate'] if e['success_rate'] is not None else 'unavailable'}; reports expired/dropped/regenerated: {e['total_report_expired']}/{e['total_report_dropped']}/{e['total_report_regenerated']}",
            f"- Packet PDR: {e['network_pooled_packet_pdr'] if e['network_pooled_packet_pdr'] is not None else 'unavailable'}; false confirmations per 1000 measurements: {e['sensing_false_confirmation_per_1000_measurements'] if e['sensing_false_confirmation_per_1000_measurements'] is not None else 'unavailable'}",
        ])
    else:
        lines.append("- Episode evidence unavailable.")
    lines.extend(["", "## Mechanism-oriented probes", "",
                  "| Hypothesis/symptom | Status | Samples | First trigger step | Late median |",
                  "|---|---|---:|---:|---:|"])
    for item in data["findings"]:
        first = item.get("first_trigger")
        late = (item.get("early_middle_late_median") or [None,None,None])[-1]
        late_text = f"{late:.4g}" if late is not None else "—"
        lines.append(f"| {item['name']} | {item['status']} | {item.get('n_samples', 0)} | {first['global_step'] if first else '—'} | {late_text} |")
    lines.extend(["", "## Mechanism co-occurrence (not causal proof)", "",
                  "| Pattern | Result | Observable windows | Matched windows | First step |",
                  "|---|---|---:|---:|---:|"])
    for pattern in data.get("mechanism_patterns", []):
        first = pattern.get("first_match")
        lines.append(f"| {pattern['description']} | {pattern['status']} | {pattern['fully_observed_windows']} | {pattern['matching_windows']} | {first['global_step'] if first else '—'} |")
    config = data.get("configuration", {})
    lines.extend(["", "## Static scientific controls", ""])
    if config.get("status") == "measured":
        for key, val in config["fields"].items():
            lines.append(f"- `{key}`: `{val}`")
        for note in config.get("notes", []):
            lines.append(f"- **Caution:** {note}")
    else:
        lines.append("- Configuration unavailable; do not infer missing config options.")
    lines.extend(["", "## Next discriminating checks", ""])
    for item in data["findings"]:
        if item["status"] in {"symptom_seen", "not_measured"}:
            lines.append(f"- **{item['name']} ({item['status']}):** {item['next_check']}")
    lines.extend(["", "## Evidence limitations", "",
                  "- Numeric thresholds are triage heuristics; no single metric proves root cause.",
                  "- Delayed/async evaluation and sampled W&B steps may not align perfectly.",
                  "- Snapshot ratios are conditional on the observed population, not independent episode-level estimates.",
                  "- Geometric route potential excludes obstacle/NLOS, MAC collisions, admission and RF power.",
                  "- A 5-episode repeatedly reused validation bank cannot establish final held-out generalization.",
                  "- Missing metrics remain unavailable; the report never fills them with inferred zeros."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local run output folder")
    parser.add_argument("--output", required=True, help="New report folder; must be empty")
    args = parser.parse_args(argv)
    dest = Path(args.output)
    if dest.exists() and any(dest.iterdir()):
        parser.error("Output folder must be empty to avoid overwriting earlier evidence")
    result = analyze(args.input)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "causal_evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (dest / "causal_evidence.md").write_text(render_markdown(result))
    print(json.dumps({"output": str(dest), "coverage": result["coverage"],
                      "episodes_observed": result["episodes_observed"]}))


if __name__ == "__main__":
    main()
