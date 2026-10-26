"""Seeded synthetic scenarios; no external target dataset is required."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Scenario:
    positions: np.ndarray
    targets: np.ndarray
    obstacles: np.ndarray
    gcs: np.ndarray


def _outside(point, obstacles, clearance=0):
    return all(np.linalg.norm(point - o[:2]) >= o[2] + clearance + 1e-9 for o in obstacles)


def generate_scenario(config, rng, options=None):
    w = config.world
    gcs = np.asarray(w.gcs_position_m or [w.map_size_m / 2, 0], dtype=float)
    if options is not None:
        missing = {"positions", "targets", "obstacles"} - options.keys()
        if missing:
            raise ValueError(f"Explicit scenario is missing {sorted(missing)}")
        scenario = Scenario(np.asarray(options["positions"], dtype=float),
                            np.asarray(options["targets"], dtype=float),
                            np.asarray(options["obstacles"], dtype=float).reshape(-1, 3), gcs)
        validate_scenario(config, scenario)
        return scenario
    obstacles = []
    if w.num_obstacles and 2 * (w.obstacle_radius_max_m + w.obstacle_clearance_m + 1e-6) >= w.map_size_m:
        raise ValueError("Obstacles cannot fit inside this map")
    # Disjoint clearance disks inside the map keep the free flight region connected.
    for _ in range(w.num_obstacles):
        radius = rng.uniform(w.obstacle_radius_min_m, w.obstacle_radius_max_m)
        for _ in range(w.placement_attempts):
            padding = radius + w.obstacle_clearance_m + 1e-6
            center = rng.uniform(padding, w.map_size_m - padding, 2)
            if (np.linalg.norm(center - gcs) > w.launch_radius_m + padding
                    and _outside(center, obstacles, radius + 2 * w.obstacle_clearance_m + 1e-6)):
                obstacles.append(np.r_[center, radius])
                break
        else:
            raise ValueError("Cannot place obstacles: reduce their count/radius or increase the map")
    positions = []
    for _ in range(w.num_uavs):
        for _ in range(w.placement_attempts):
            point = gcs + rng.uniform(-w.launch_radius_m, w.launch_radius_m, 2)
            if (np.all((point >= 0) & (point <= w.map_size_m))
                    and np.linalg.norm(point - gcs) <= w.launch_radius_m
                    and _outside(point, obstacles, w.obstacle_clearance_m)
                    and all(np.linalg.norm(point - p) >= w.launch_min_spacing_m for p in positions)):
                positions.append(point)
                break
        else:
            raise ValueError("Cannot place UAVs with the configured launch spacing")
    targets = []
    target_clearance = max(0, w.obstacle_clearance_m - config.sensor.footprint_radius_m) + 1e-6
    for _ in range(w.num_targets):
        for _ in range(w.placement_attempts):
            point = rng.uniform(0, w.map_size_m, 2)
            if np.linalg.norm(point - gcs) >= w.target_exclusion_radius_m and _outside(point, obstacles, target_clearance):
                targets.append(point)
                break
        else:
            raise ValueError("Cannot place targets outside exclusion zones")
    scenario = Scenario(np.array(positions), np.array(targets),
                        np.array(obstacles).reshape(-1, 3), gcs)
    validate_scenario(config, scenario)
    return scenario


def validate_scenario(config, scene):
    w = config.world
    for name, shape in [("positions", (w.num_uavs, 2)), ("targets", (w.num_targets, 2)),
                        ("obstacles", (w.num_obstacles, 3))]:
        a = getattr(scene, name)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"{name} must be a finite array of shape {shape}")
        if np.any(a[:, :2] < 0) or np.any(a[:, :2] > w.map_size_m):
            raise ValueError(f"{name} lies outside the map")
    if np.any(scene.obstacles[:, 2] <= 0):
        raise ValueError("Obstacle radii must be positive")
    for i, p in enumerate(scene.positions):
        if not _outside(p, scene.obstacles, w.obstacle_clearance_m):
            raise ValueError("UAV starts inside a forbidden obstacle clearance")
        if any(np.linalg.norm(p - q) < w.safety_distance_m for q in scene.positions[:i]):
            raise ValueError("Initial UAVs violate the safety distance")
    target_clearance = max(0, w.obstacle_clearance_m - config.sensor.footprint_radius_m)
    if any(not _outside(p, scene.obstacles, target_clearance) for p in scene.targets):
        raise ValueError("Target lies inside an obstacle or outside camera reach of its clearance")
    if not _outside(scene.gcs, scene.obstacles):
        raise ValueError("GCS lies inside an obstacle")
