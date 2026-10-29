"""Failure-aware mission outcomes from observed trajectories, without changing the env."""
import hashlib
import json
import numpy as np

from ..environment.geometry import segment_distance
from ..environment.observations import OBSERVATION_VERSION


# Units and preferred direction are explicit; no aggregate 'best algorithm' score.
METRICS = {
    "gcs_delivery_fraction": ("fraction", "higher"),
    "target_discovery_fraction": ("fraction", "higher"),
    "delivery_given_discovery_fraction": ("fraction", "higher"),
    "mission_success": ("fraction", "higher"),
    "delivery_auc": ("fraction", "higher"),
    "discovery_auc": ("fraction", "higher"),
    "target_delivery_time_capped_mean_s": ("s", "lower"),
    "mission_completion_time_capped_s": ("s", "lower"),
    "first_discovery_to_gcs_delay_delivered_mean_s": ("s", "lower"),
    "report_delay_delivered_mean_s": ("s", "lower"),
    "uav_energy_j": ("J", "lower"),
    "uav_j_per_gcs_delivery": ("J/delivered target", "lower"),
    "uav_radio_energy_j": ("J", "lower"),
    "uav_radio_j_per_gcs_delivery": ("J/delivered target", "lower"),
    "path_length_m": ("m", "lower"),
    "safe_episode": ("fraction", "higher"),
    "safety_violation_steps": ("observed steps including reset", "lower"),
    "swept_peer_violation_pairs": ("pair segments including reset", "lower"),
    "swept_obstacle_violation_segments": ("UAV-obstacle segments including reset", "lower"),
    "boundary_violation_samples": ("UAV samples including reset", "lower"),
    "shield_interventions_per_uav_step": ("interventions/UAV-step", "lower"),
    "blocked_moves_per_uav_step": ("blocks/UAV-step", "lower"),
    "packet_pdr_fraction": ("fraction of per-hop generated packets", "higher"),
    "data_below_required_power_fraction": ("fraction of DATA attempts", "lower"),
    "relay_toward_gcs_fraction": ("fraction of peer custody transfers", "higher"),
    "tx_power_margin_db_mean": ("dB", "descriptive"),
    "relay_gcs_distance_improvement_mean_m": ("m/peer custody transfer", "higher"),
    "shield_correction_mean_per_step": ("normalized velocity correction/step", "lower"),
    "return": ("configured reward units", "higher"),
}


def evaluation_protocol(config, seeds, progress):
    """Hash only shared scientific conditions, including the backend's scene generator."""
    values = config.to_dict()
    scenario = {k: values[k] for k in ("world", "sensor", "network", "energy", "reward")}
    # New V3 radio fields must not relabel historical V1/V2 physics when they
    # are at their exact legacy defaults. Non-default V3 values remain hashed.
    network = scenario["network"]
    if network.get("power_control") == "learned":
        network.pop("power_control", None)
        network.pop("link_budget_margin_db", None)
    if not scenario["sensor"].get("coverage_grid_cells", 0):
        scenario["sensor"].pop("coverage_grid_cells", None)
    if (network.get("pa_efficiency") == 1.0
            and network.get("tx_circuit_power_w") == 0.0
            and network.get("rx_circuit_power_w") == 0.0):
        for key in ("pa_efficiency", "tx_circuit_power_w", "rx_circuit_power_w"):
            network.pop(key, None)
    scenario["environment_backend"] = config.runtime.environment_backend
    scenario["observation_version"] = OBSERVATION_VERSION
    digest = hashlib.sha256(json.dumps(scenario, sort_keys=True, allow_nan=False).encode()).hexdigest()
    next_seed = progress.get("next_seed")
    history_seeds = [r["seed"] for r in progress.get("records", []) if type(r.get("seed")) is int]
    earliest = min([config.runtime.seed, *history_seeds])
    interval = [earliest, next_seed - 1] if type(next_seed) is int else None
    overlap = any(interval[0] <= s <= interval[1] for s in seeds) if interval else None
    validation_intervals = []
    if (config.tracking.enabled and config.tracking.evaluation_interval > 0
            and progress.get("episodes", 0) >= config.tracking.evaluation_interval):
        validation_intervals.append([config.runtime.eval_seed,
                                     config.runtime.eval_seed + config.runtime.eval_episodes - 1])
    if (config.tracking.evaluation_interval_steps > 0
            and progress.get("global_step", 0) >= config.tracking.evaluation_interval_steps):
        validation_intervals.append([config.tracking.periodic_evaluation_seed,
                                     config.tracking.periodic_evaluation_seed +
                                     config.tracking.periodic_evaluation_episodes - 1])
    validation_interval = validation_intervals[0] if validation_intervals else None
    validation_overlap = any(low <= seed <= high for low, high in validation_intervals for seed in seeds)
    # Earlier checkpoints may have used a different validation bank; old checkpoints lack that history.
    if config.runtime.warm_start_checkpoint != "none" and not validation_overlap:
        validation_overlap = None
    return {"version": 3, "scenario_hash": digest, "scenario": scenario, "seeds": list(seeds),
            "deterministic_policy": True, "training_seed_interval_conservative": interval,
            "training_seed_overlap": overlap, "training_seed": config.runtime.seed,
            "validation_seed_interval_current": validation_interval,
            "validation_seed_intervals_current": validation_intervals,
            "validation_seed_overlap": validation_overlap,
            "exposure_provenance": "resumed_history_incomplete" if config.runtime.warm_start_checkpoint != "none"
                                   else "saved_fresh_training_configuration",
            "training_global_step": progress.get("global_step"),
            "training_episodes": progress.get("episodes"),
            "timing_resolution_s": config.world.dt_s,
            "timing_basis": "first observation at simulation step boundary; not exact packet timestamp",
            "safety_tolerance_m": 1e-3,
            "scope": "evaluation scenarios conditional on this trained checkpoint; not training-seed robustness"}


class MissionMetrics:
    def __init__(self, config, positions, obstacles, found, delivered):
        self.config = config
        self.previous = np.asarray(positions, dtype=float).copy()
        self.obstacles = np.asarray(obstacles, dtype=float).copy()
        self.discovered = np.full(config.world.num_targets, np.nan)
        self.delivered = np.full(config.world.num_targets, np.nan)
        self.step = 0
        self.path_length = 0.
        self.obstacle_violations = self.peer_violations = self.boundary_violations = self.unsafe_steps = 0
        self._mark(found, delivered, 0.)
        self._safety(self.previous, self.previous)

    def _mark(self, found, delivered, time_s):
        found, delivered = np.asarray(found, dtype=bool), np.asarray(delivered, dtype=bool)
        if (found.shape != self.discovered.shape or delivered.shape != self.delivered.shape
                or np.any(delivered & ~found)
                or np.any(~found & np.isfinite(self.discovered))
                or np.any(~delivered & np.isfinite(self.delivered))):
            raise ValueError("Target observations must have consistent shapes and monotonic discovery/delivery")
        self.discovered[found & np.isnan(self.discovered)] = time_s
        self.delivered[delivered & np.isnan(self.delivered)] = time_s

    def _safety(self, start, end):
        w = self.config.world
        if end.shape != (w.num_uavs, 2) or not np.isfinite(end).all():
            raise ValueError("Mission positions must be finite [num_uavs,2] coordinates")
        tolerance = 1e-3
        obstacle = sum(segment_distance(a, b, o[:2]) < o[2] + w.obstacle_clearance_m - tolerance
                       for a, b in zip(start, end) for o in self.obstacles)
        peer = sum(segment_distance(start[i] - start[j], end[i] - end[j], [0, 0]) <
                   w.safety_distance_m - tolerance
                   for i in range(w.num_uavs) for j in range(i))
        boundary = int(np.any((end < -tolerance) | (end > w.map_size_m + tolerance), axis=1).sum())
        self.obstacle_violations += int(obstacle)
        self.peer_violations += int(peer)
        self.boundary_violations += boundary
        self.unsafe_steps += int(bool(obstacle or peer or boundary))

    def observe(self, positions, found, delivered, step):
        if type(step) is not int or not self.step < step <= self.config.world.max_steps:
            raise ValueError("Mission steps must increase within the configured horizon")
        positions = np.asarray(positions, dtype=float)
        self._safety(self.previous, positions)
        self.path_length += float(np.linalg.norm(positions - self.previous, axis=1).sum())
        self._mark(found, delivered, step * self.config.world.dt_s)
        self.previous, self.step = positions.copy(), step

    def finish(self, info, episode_return):
        w = self.config.world
        horizon = w.max_steps * w.dt_s
        found, delivered = np.isfinite(self.discovered), np.isfinite(self.delivered)
        f, d = int(found.sum()), int(delivered.sum())
        success = bool(delivered.all())
        if success != bool(info["success"]):
            raise ValueError("Mission success disagrees with observed GCS deliveries")
        net = info["network"]
        radio_energy = float(net.get("uav_communication_energy_j", 0.))
        energy = float(info["energy"]["propulsion_energy_j"]) + radio_energy
        capped_delivery = np.where(delivered, self.delivered, horizon)
        capped_discovery = np.where(found, self.discovered, horizon)
        generated = net.get("generated_packets", 0)
        received = net.get("received_packets", 0)
        diagnostics = net.get("diagnostics") or {}
        attempts = diagnostics.get("data_attempts", 0)
        peer_transfers = diagnostics.get("custody_peer_transfers", 0)
        power_margin = diagnostics.get("data_power_margin_db") or {}
        relay_progress = diagnostics.get("relay_gcs_distance_improvement_m") or {}
        motion = info.get("motion_counts", {})
        return {
            "targets_found": f, "targets_delivered": d, "target_discovery_fraction": f / w.num_targets,
            "gcs_delivery_fraction": d / w.num_targets,
            "delivery_given_discovery_fraction": d / f if f else None,
            "mission_success": float(success),
            "mission_completion_time_capped_s": self.step * w.dt_s if success else horizon,
            "target_delivery_time_capped_mean_s": float(capped_delivery.mean()),
            "delivery_auc": float(1. - capped_delivery.mean() / horizon),
            "discovery_auc": float(1. - capped_discovery.mean() / horizon),
            "first_discovery_to_gcs_delay_delivered_mean_s":
                float((self.delivered[delivered] - self.discovered[delivered]).mean()) if d else None,
            "report_delay_delivered_mean_s": net.get("report_e2e_delay_s") if d and net.get("enabled") else None,
            "uav_energy_j": energy, "uav_j_per_gcs_delivery": energy / d if d else None,
            "uav_radio_energy_j": radio_energy,
            "uav_radio_j_per_gcs_delivery": radio_energy / d if d else None,
            "packet_pdr_fraction": received / generated if generated and net.get("enabled") else None,
            "data_below_required_power_fraction": (diagnostics.get("data_below_required_power_attempts", 0) / attempts
                                                    if attempts else None),
            "relay_toward_gcs_fraction": (diagnostics.get("relay_toward_gcs_transfers", 0) / peer_transfers
                                           if peer_transfers else None),
            "tx_power_margin_db_mean": power_margin.get("mean"),
            "relay_gcs_distance_improvement_mean_m": relay_progress.get("mean"),
            "shield_correction_mean_per_step": motion.get("shield_correction", 0.) / max(1, self.step),
            "path_length_m": self.path_length, "safe_episode": float(self.unsafe_steps == 0),
            "safety_violation_steps": self.unsafe_steps,
            "swept_peer_violation_pairs": self.peer_violations,
            "swept_obstacle_violation_segments": self.obstacle_violations,
            "boundary_violation_samples": self.boundary_violations,
            "shield_interventions_per_uav_step":
                info.get("motion_counts", {}).get("shield_interventions", 0) / max(1, self.step * w.num_uavs),
            "blocked_moves_per_uav_step": info.get("blocked_moves", 0) / max(1, self.step * w.num_uavs),
            "return": float(episode_return),
            "first_discovery_observed_s": [float(t) if np.isfinite(t) else None for t in self.discovered],
            "first_gcs_delivery_observed_s": [float(t) if np.isfinite(t) else None for t in self.delivered],
        }


def summarize_metrics(records, bootstrap_samples=2000, seed=44):
    rng = np.random.default_rng(seed)
    result = {}
    for name, (unit, direction) in METRICS.items():
        values = np.array([r["comparison_metrics"][name] for r in records
                           if r["comparison_metrics"].get(name) is not None], dtype=float)
        n = len(values)
        interval = None
        if n > 1:
            means = values[rng.integers(n, size=(bootstrap_samples, n))].mean(axis=1)
            interval = list(map(float, np.quantile(means, [.025, .975])))
        result[name] = {"mean": float(values.mean()) if n else None, "ci95": interval,
                        "episodes": n, "unit": unit, "preferred": direction}
    energy = sum(r["comparison_metrics"]["uav_energy_j"] for r in records)
    delivered = sum(r["comparison_metrics"]["targets_delivered"] for r in records)
    result["pooled_uav_j_per_gcs_delivery"] = {"value": energy / delivered if delivered else None,
                                             "total_energy_j": energy, "total_targets_delivered": delivered}
    radio = sum(r["comparison_metrics"].get("uav_radio_energy_j", 0.) for r in records)
    result["pooled_uav_radio_j_per_gcs_delivery"] = {
        "value": radio / delivered if delivered else None,
        "total_radio_energy_j": radio, "total_targets_delivered": delivered}
    return result
