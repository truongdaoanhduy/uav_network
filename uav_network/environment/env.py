"""Decentralized observations and team rewards for a 2D search mission."""
from dataclasses import dataclass
import numpy as np

from ..communication.reports import ReportStore
from ..communication.link_budget import report_reference_tx_energy_j
from .energy import funded_motion
from .observations import centralized_state, local_observations, observation_dim, state_dim
from .sensing import visible_target_ids
from .world import generate_scenario


@dataclass
class Action:
    continuous: np.ndarray  # [N,3]: acceleration X/Y and normalized TX power
    destination: np.ndarray  # [N]: silent, ordered peers, GCS

    def validate(self, num_uavs):
        a, d = np.asarray(self.continuous), np.asarray(self.destination)
        if (a.shape != (num_uavs, 3) or not np.issubdtype(a.dtype, np.number)
                or not np.isfinite(a).all() or np.any(np.abs(a) > 1)):
            raise ValueError(f"Continuous actions must be finite [-1,1] values of shape ({num_uavs},3)")
        if (d.shape != (num_uavs,) or not np.issubdtype(d.dtype, np.integer)
                or np.any(d < 0) or np.any(d > num_uavs)):
            raise ValueError("Destination indices must be integers from 0 through num_uavs")


class SwarmSearchEnv:
    def __init__(self, config):
        self.config = config.validate()
        self.network = None
        self._communication_trace = False
        self.steps = 0
        self._ended = True

    def enable_communication_trace(self):
        """Capture successful whole-report hops until drained or the next reset."""
        self._communication_trace = True
        if self.network is not None and self.network.communication_events is None:
            self.network.communication_events = []

    def drain_communication_events(self):
        if self.network is None or self.network.communication_events is None:
            return []
        events, self.network.communication_events = self.network.communication_events, []
        return events

    @property
    def time_s(self):
        return self.steps * self.config.world.dt_s

    @property
    def observation_dim(self):
        return observation_dim(self.config)

    @property
    def state_dim(self):
        return state_dim(self.config)

    @property
    def active(self):
        return self.battery_j > 0 if self.config.energy.enabled else np.ones(len(self.battery_j), dtype=bool)

    @property
    def battery_fraction(self):
        return self.battery_j / self.config.energy.battery_capacity_j

    def reset(self, seed=None, options=None):
        self.close()
        self.seed = self.config.runtime.seed if seed is None else int(seed)
        scene = generate_scenario(self.config, np.random.default_rng(self.seed), options)
        self.positions, self.targets, self.obstacles, self.gcs = (
            scene.positions, scene.targets, scene.obstacles, scene.gcs)
        self.generation_fallback = scene.generation_fallback
        self.generation_attempts = scene.generation_attempts
        self.velocities = np.zeros_like(self.positions)
        self.battery_j = np.full(self.config.world.num_uavs, self.config.energy.battery_capacity_j, dtype=float)
        self.propulsion_energy_j = np.zeros_like(self.battery_j)
        self.reports = ReportStore(self.config)
        self.steps = self.blocked_moves = self.boundary_clips = 0
        self._ended = False
        if self.config.network.enabled:
            from ..communication.backend import UavNetSimNetwork
            self.network = UavNetSimNetwork(self.config, self.reports, scene, self.seed, self.battery_j)
            if self._communication_trace:
                self.network.communication_events = []
        self.motion_counts = {"obstacle_blocks": 0, "peer_blocks": 0, "commanded_stalls": 0}
        self.motion_counts.update({f"shield_{name}": 0 for name in (
            "interventions", "obstacle_interventions", "peer_interventions",
            "boundary_interventions", "infeasible")})
        self.motion_counts["shield_correction"] = 0.0
        self.reward_components = {}
        self.action_counts = {"silent": 0, "peer": 0, "gcs": 0, "saturated": 0}
        self._sense()
        return local_observations(self), self.info()

    def _sense(self):
        for i, position in enumerate(self.positions):
            if not self.active[i]:
                continue
            for target_id in visible_target_ids(position, self.targets, self.obstacles, self.config.sensor):
                self.reports.discover(i, target_id, self.time_s)

    def step(self, action):
        if self._ended:
            raise RuntimeError("Call reset before stepping a finished environment")
        action.validate(self.config.world.num_uavs)
        before_found = self.reports.known.any(axis=0).sum()
        before_delivery = self.reports.delivered.sum()
        before_expired, before_dropped = self.reports.expired, self.reports.dropped
        motion_details = {}
        self.positions, self.velocities, blocked, clipped, flight_energy = funded_motion(
            self.positions, self.velocities, np.asarray(action.continuous)[:, :2],
            self.obstacles, self.battery_j, self.config, diagnostics=motion_details)
        for key, value in motion_details.items():
            self.motion_counts[key] += value
        self.propulsion_energy_j += flight_energy
        self.blocked_moves += blocked
        self.boundary_clips += clipped
        self.reports.expire(self.time_s)
        self._sense()
        comm_energy = 0.0
        if self.network:
            comm_energy = self.network.step(action, self.positions,
                                           self.time_s + self.config.world.dt_s)
        self.steps += 1
        self.velocities[~self.active] = 0
        self.reports.expire(self.time_s)
        r = self.config.reward
        communication_efficiency = (comm_energy / report_reference_tx_energy_j(self.config.network)
                                    if comm_energy else 0.0)
        shield_correction = float(motion_details.get("shield_correction", 0.0))
        reward = (r.discovery * (self.reports.known.any(axis=0).sum() - before_found)
                  + r.delivery * (self.reports.delivered.sum() - before_delivery)
                  - r.blocked_motion * blocked - r.step_cost
                  - r.expired_report * (self.reports.expired - before_expired)
                  - r.dropped_report * (self.reports.dropped - before_dropped)
                  - r.communication_energy * comm_energy
                  - r.propulsion_energy_per_kj * flight_energy.sum() / 1000
                  - r.communication_efficiency * communication_efficiency
                  - r.shield_correction * shield_correction)
        components = {"discovery": r.discovery * (self.reports.known.any(axis=0).sum() - before_found),
                      "delivery": r.delivery * (self.reports.delivered.sum() - before_delivery),
                      "blocked_motion": -r.blocked_motion * blocked, "step_cost": -r.step_cost,
                      "expired_report": -r.expired_report * (self.reports.expired - before_expired),
                      "dropped_report": -r.dropped_report * (self.reports.dropped - before_dropped),
                      "communication_energy": -r.communication_energy * comm_energy,
                      "propulsion_energy": -r.propulsion_energy_per_kj * flight_energy.sum() / 1000,
                      "communication_efficiency": -r.communication_efficiency * communication_efficiency,
                      "shield_correction": -r.shield_correction * shield_correction}
        for key, value in components.items():
            self.reward_components[key] = self.reward_components.get(key, 0.0) + float(value)
        dest = np.asarray(action.destination)
        self.action_counts["silent"] += int((dest == 0).sum())
        self.action_counts["gcs"] += int((dest == self.config.world.num_uavs).sum())
        self.action_counts["peer"] += int(((dest > 0) & (dest < self.config.world.num_uavs)).sum())
        self.action_counts["saturated"] += int((np.abs(np.asarray(action.continuous)[:, :2]) >= .99).sum())
        # max_steps is the mission deadline, observed by the actor and critic.
        # It is a finite-horizon terminal state, not an external collection cutoff.
        terminated = bool(self.reports.delivered.all() or not self.active.any()
                          or self.steps >= self.config.world.max_steps)
        truncated = False
        self._ended = terminated or truncated
        return local_observations(self), float(reward), terminated, truncated, self.info()

    def state(self):
        return centralized_state(self)

    def info(self):
        found, delivered = int(self.reports.known.any(axis=0).sum()), int(self.reports.delivered.sum())
        count = self.config.world.num_targets
        rf = self.network.energy_j[:-1] if self.network else np.zeros_like(self.propulsion_energy_j)
        tx_rf = self.network.tx_energy_j[:-1] if self.network else np.zeros_like(self.propulsion_energy_j)
        rx_rf = self.network.rx_energy_j[:-1] if self.network else np.zeros_like(self.propulsion_energy_j)
        return {
            "generation_fallback": self.generation_fallback,
            "generation_attempts": self.generation_attempts,
            "report_diagnostics": self.reports.diagnostics(self.time_s),"motion_counts": dict(self.motion_counts), "reward_components": dict(self.reward_components), "action_counts": dict(self.action_counts),
                "buffer_reports": sum(len(b) for b in self.reports.buffers),
                "buffer_bytes": sum(self.reports.bytes_used(i) for i in range(self.config.world.num_uavs)),
                "mean_speed_mps": float(np.linalg.norm(self.velocities, axis=1).mean()),
                "step": self.steps, "targets_found": found, "targets_delivered": delivered,
                "search_rate_percent": 100 * found / count,
                "delivery_rate_percent": 100 * delivered / count,
                "success": bool(self.reports.delivered.all()), "blocked_moves": self.blocked_moves,
                "boundary_clips": self.boundary_clips, "expired_reports": self.reports.expired,
                "dropped_reports": self.reports.dropped, "regenerated_reports": self.reports.regenerated,
                "termination_reason": ("success" if self.reports.delivered.all() else
                                       "battery_exhausted" if not self.active.any() else
                                       "time_limit" if self.steps >= self.config.world.max_steps else None),
                "energy": {"per_uav_propulsion_energy_j": self.propulsion_energy_j.tolist(),
                           "per_uav_rf_energy_j": rf.tolist(),
                           "per_uav_tx_radio_energy_j": tx_rf.tolist(),
                           "per_uav_rx_radio_energy_j": rx_rf.tolist(),
                           "per_uav_total_energy_j": (self.propulsion_energy_j + rf).tolist(),
                           "battery_j": self.battery_j.tolist(),
                           "battery_fraction": self.battery_fraction.tolist(),
                           "active_uavs": int(self.active.sum()),
                           "propulsion_energy_j": float(self.propulsion_energy_j.sum())},
                "network": self.network.metrics() if self.network else {"enabled": False}}

    def close(self):
        if self.network:
            self.network.close()
            self.network = None
