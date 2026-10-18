"""Privileged Scenario-V3 feasibility controller.

This controller intentionally reads ground-truth target/report state.  It is an
engineering oracle used to reject structurally impossible scenarios before MARL
training; it is not a baseline policy and must never be used for learned-policy
scores.
"""
from __future__ import annotations

import numpy as np

from ..communication.link_budget import nominal_required_power_w, power_fraction
from ..environment.env import Action, SwarmSearchEnv


def power_action_for_required(required_w, network_config, margin_db=3.0):
    """Map required radiated power plus a dB margin back to the [-1, 1] action."""
    factor = 10.0 ** (float(margin_db) / 10.0)
    selected = min(float(network_config.tx_power_max_w),
                   max(float(network_config.tx_power_min_w), float(required_w) * factor))
    return 2.0 * power_fraction(selected, network_config) - 1.0


def assign_search_targets(positions, targets, discovered, active):
    """Greedily make deterministic unique nearest-target assignments.

    Returns one target id or ``None`` per UAV.  Only globally undiscovered targets
    are eligible.  Global nearest-pair assignment avoids sending multiple UAVs to
    the same target while enough work remains.
    """
    positions = np.asarray(positions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    discovered = np.asarray(discovered, dtype=bool)
    active = np.asarray(active, dtype=bool)
    result = [None] * len(positions)
    uavs = set(np.flatnonzero(active).tolist())
    remaining = set(np.flatnonzero(~discovered).tolist())
    while uavs and remaining:
        best = min((float(np.linalg.norm(positions[i] - targets[t])), i, t)
                   for i in uavs for t in remaining)
        _, i, t = best
        result[i] = t
        uavs.remove(i)
        remaining.remove(t)
    return result


def _peer_action_index(source, peer, num_uavs):
    peers = [i for i in range(num_uavs) if i != source]
    return peers.index(peer) + 1


def _required(env, source, destination):
    n = env.config.world.num_uavs
    destination_xy = env.gcs if destination == n else env.positions[destination]
    return nominal_required_power_w(
        env.positions[source], destination_xy, env.obstacles, env.config.network,
        env.config.sensor.altitude_m, destination == n)


def _communication_choice(env, source, report):
    """Return (destination-action-index, required-power) or None.

    Prefer direct GCS when it is presently admissible.  Otherwise choose a peer
    that advances custody toward GCS, maximizing progress then preferring lower
    nominal required power.
    """
    n = env.config.world.num_uavs
    network = env.config.network
    if env.network._eligible(source, n, env.positions, report):
        return n, _required(env, source, n)

    source_gcs = float(np.linalg.norm(env.positions[source] - env.gcs))
    candidates = []
    for peer in range(n):
        if peer == source or not env.network._eligible(source, peer, env.positions, report):
            continue
        progress = source_gcs - float(np.linalg.norm(env.positions[peer] - env.gcs))
        if progress <= 1e-9:
            continue
        required = _required(env, source, peer)
        if required <= network.tx_power_max_w:
            candidates.append((-progress, required, peer))
    if not candidates:
        return None
    _, required, peer = min(candidates)
    return _peer_action_index(source, peer, n), required


def _motion_action(env, assignments, return_to_gcs):
    n = env.config.world.num_uavs
    acceleration = np.zeros((n, 2), dtype=np.float64)
    w = env.config.world
    for i in range(n):
        if not env.active[i]:
            continue
        target_id = assignments[i]
        if return_to_gcs[i]:
            # Delivery urgency wins when a report carrier has no usable direct or
            # progress-peer route.  Closing GCS distance creates a future route.
            goal = env.gcs
        elif target_id is not None:
            goal = env.targets[target_id]
        else:
            # Idle UAVs return toward the GCS instead of burning energy on an
            # arbitrary patrol once all current undiscovered targets are assigned.
            goal = env.gcs
        delta = np.asarray(goal, dtype=float) - env.positions[i]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            desired_velocity = np.zeros(2)
        else:
            # Slow down close to the goal; otherwise fly at the speed limit.
            desired_speed = min(w.max_speed_mps, distance / max(w.dt_s, 1e-12))
            desired_velocity = delta / distance * desired_speed
        acceleration[i] = np.clip(
            (desired_velocity - env.velocities[i]) / (w.max_accel_mps2 * w.dt_s), -1.0, 1.0)
    return acceleration


def oracle_action(env, margin_db=3.0):
    """Construct one privileged search/relay/power action for a native environment."""
    n = env.config.world.num_uavs
    communication = [None] * n
    return_to_gcs = np.zeros(n, dtype=bool)
    for source in range(n):
        if not env.active[source] or not env.reports.buffers[source]:
            continue
        # Oldest report first, then target id for deterministic ties.
        reports = sorted(env.reports.buffers[source].values(),
                         key=lambda r: (r.created_s, r.target_id, r.source_id))
        for report in reports:
            choice = _communication_choice(env, source, report)
            if choice is not None:
                communication[source] = choice
                break
        return_to_gcs[source] = communication[source] is None

    discovered = env.reports.known.any(axis=0)
    assignments = assign_search_targets(env.positions, env.targets, discovered, env.active)
    continuous = np.zeros((n, 3), dtype=np.float32)
    continuous[:, :2] = _motion_action(env, assignments, return_to_gcs).astype(np.float32)
    destination = np.zeros(n, dtype=np.int64)

    for source, choice in enumerate(communication):
        if choice is None:
            continue
        destination[source], required = choice
        continuous[source, 2] = np.float32(
            power_action_for_required(required, env.config.network, margin_db))
    return Action(continuous, destination)


def _finish_result(env, seed):
    info = env.info()
    network = info.get("network", {})
    return {
        "seed": int(seed),
        "success": bool(info["success"]),
        "episode_length": int(info["step"]),
        "search_rate_percent": float(info["search_rate_percent"]),
        "delivery_rate_percent": float(info["delivery_rate_percent"]),
        "battery_fraction": list(info["energy"]["battery_fraction"]),
        "propulsion_energy_j": float(info["energy"]["propulsion_energy_j"]),
        "uav_radio_energy_j": float(network.get("uav_communication_energy_j", 0.0)),
        "uav_tx_energy_j": float(network.get("uav_tx_energy_j", 0.0)),
        "uav_rx_energy_j": float(network.get("uav_rx_energy_j", 0.0)),
        "packet_pdr_percent": float(network.get("packet_pdr_percent", 0.0)),
        "phy_success_percent": float(network.get("phy_success_percent", 0.0)),
        "report_average_hops": float(network.get("report_average_hops", 0.0)),
        "expired_reports": int(info.get("expired_reports", 0)),
        "dropped_reports": int(info.get("dropped_reports", 0)),
        "network_diagnostics": network.get("diagnostics", {}),
        "report_diagnostics": info.get("report_diagnostics", {}),
    }


def run_explicit_feasibility(config, seed, positions, targets, obstacles, margin_db=3.0):
    """Run one explicit native scenario with the privileged oracle."""
    env = SwarmSearchEnv(config)
    try:
        env.reset(seed=seed, options={"positions": np.asarray(positions, dtype=float),
                                      "targets": np.asarray(targets, dtype=float),
                                      "obstacles": np.asarray(obstacles, dtype=float).reshape(-1, 3)})
        while True:
            _, _, terminated, truncated, _ = env.step(oracle_action(env, margin_db))
            if terminated or truncated:
                return _finish_result(env, seed)
    finally:
        env.close()


def run_feasibility(config, seeds, margin_db=3.0):
    """Run deterministic generated scenarios and return raw records plus aggregates."""
    records = []
    for seed in [int(value) for value in seeds]:
        env = SwarmSearchEnv(config)
        try:
            env.reset(seed=seed)
            while True:
                _, _, terminated, truncated, _ = env.step(oracle_action(env, margin_db))
                if terminated or truncated:
                    records.append(_finish_result(env, seed))
                    break
        finally:
            env.close()
    if not records:
        raise ValueError("seeds must contain at least one seed")
    return {
        "controller": "privileged_feasibility_oracle",
        "not_a_policy_baseline": True,
        "episodes": len(records),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "mean_search_rate_percent": float(np.mean([r["search_rate_percent"] for r in records])),
        "mean_delivery_rate_percent": float(np.mean([r["delivery_rate_percent"] for r in records])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
        "mean_final_battery_fraction": float(np.mean([np.mean(r["battery_fraction"]) for r in records])),
        "records": records,
    }
