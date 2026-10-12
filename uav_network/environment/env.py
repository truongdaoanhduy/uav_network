"""Decentralized observations and team rewards for a 2D search mission."""
from dataclasses import dataclass
import numpy as np

from ..communication.reports import ReportStore
from .motion import move_uavs
from .observations import centralized_state, local_observations
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
        self.steps = 0
        self._ended = True

    @property
    def time_s(self):
        return self.steps * self.config.world.dt_s

    @property
    def observation_dim(self):
        return local_observations(self).shape[1]

    @property
    def state_dim(self):
        return len(self.state())

    def reset(self, seed=None, options=None):
        self.close()
        self.seed = self.config.runtime.seed if seed is None else int(seed)
        scene = generate_scenario(self.config, np.random.default_rng(self.seed), options)
        self.positions, self.targets, self.obstacles, self.gcs = (
            scene.positions, scene.targets, scene.obstacles, scene.gcs)
        self.velocities = np.zeros_like(self.positions)
        self.reports = ReportStore(self.config)
        self.steps = self.blocked_moves = self.boundary_clips = 0
        self._ended = False
        if self.config.network.enabled:
            from ..communication.backend import UavNetSimNetwork
            self.network = UavNetSimNetwork(self.config, self.reports, scene, self.seed)
        self._sense()
        return local_observations(self), self.info()

    def _sense(self):
        for i, position in enumerate(self.positions):
            for target_id in visible_target_ids(position, self.targets, self.obstacles, self.config.sensor):
                self.reports.discover(i, target_id, self.time_s)

    def step(self, action):
        if self._ended:
            raise RuntimeError("Call reset before stepping a finished environment")
        action.validate(self.config.world.num_uavs)
        before_found = self.reports.known.any(axis=0).sum()
        before_delivery = self.reports.delivered.sum()
        before_expired, before_dropped = self.reports.expired, self.reports.dropped
        self.positions, self.velocities, blocked, clipped = move_uavs(
            self.positions, self.velocities, np.asarray(action.continuous)[:, :2],
            self.obstacles, self.config)
        self.steps += 1
        self.blocked_moves += blocked
        self.boundary_clips += clipped
        self.reports.expire(self.time_s)
        self._sense()
        comm_energy = 0.0
        if self.network:
            comm_energy = self.network.step(action, self.positions, self.time_s)
        r = self.config.reward
        reward = (r.discovery * (self.reports.known.any(axis=0).sum() - before_found)
                  + r.delivery * (self.reports.delivered.sum() - before_delivery)
                  - r.blocked_motion * blocked - r.step_cost
                  - r.expired_report * (self.reports.expired - before_expired)
                  - r.dropped_report * (self.reports.dropped - before_dropped)
                  - r.communication_energy * comm_energy)
        terminated = bool(self.reports.delivered.all())
        truncated = self.steps >= self.config.world.max_steps and not terminated
        self._ended = terminated or truncated
        return local_observations(self), float(reward), terminated, truncated, self.info()

    def state(self):
        return centralized_state(self)

    def info(self):
        found, delivered = int(self.reports.known.any(axis=0).sum()), int(self.reports.delivered.sum())
        count = self.config.world.num_targets
        return {"step": self.steps, "targets_found": found, "targets_delivered": delivered,
                "search_rate_percent": 100 * found / count,
                "delivery_rate_percent": 100 * delivered / count,
                "success": bool(self.reports.delivered.all()), "blocked_moves": self.blocked_moves,
                "boundary_clips": self.boundary_clips, "expired_reports": self.reports.expired,
                "dropped_reports": self.reports.dropped,
                "network": self.network.metrics() if self.network else {"enabled": False}}

    def close(self):
        if self.network:
            self.network.close()
            self.network = None
