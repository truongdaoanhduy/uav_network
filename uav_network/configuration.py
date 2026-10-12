"""Validated configuration; use YAML or --set group.name=value."""
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
import math

import yaml


@dataclass
class WorldConfig:
    map_size_m: float = 1000.0
    num_uavs: int = 6
    num_targets: int = 10
    num_obstacles: int = 8
    dt_s: float = 1.0
    max_steps: int = 800
    max_speed_mps: float = 20.0
    max_accel_mps2: float = 2.0
    safety_distance_m: float = 15.0
    obstacle_clearance_m: float = 10.0
    obstacle_radius_min_m: float = 30.0
    obstacle_radius_max_m: float = 70.0
    launch_radius_m: float = 150.0
    launch_min_spacing_m: float = 30.0
    target_exclusion_radius_m: float = 200.0
    gcs_position_m: list | None = None
    placement_attempts: int = 5000


@dataclass
class SensorConfig:
    altitude_m: float = 100.0
    full_fov_deg: float = 90.0
    obstacle_occlusion: bool = True
    visible_target_slots: int = 4
    visible_obstacle_slots: int = 3

    @property
    def footprint_radius_m(self):
        return self.altitude_m * math.tan(math.radians(self.full_fov_deg / 2))


@dataclass
class NetworkConfig:
    enabled: bool = True
    upstream_path: str = "third_party/UavNetSim"
    peer_range_m: float = 500.0
    gcs_range_m: float = 500.0
    tx_power_min_w: float = 0.1
    tx_power_max_w: float = 0.4
    report_bytes: int = 1024
    buffer_bytes: int = 3072
    report_ttl_s: float = 300.0
    payload_bytes: int = 1024
    packet_lifetime_s: float = 10.0
    max_queue_packets: int = 64
    queue_poll_us: float = 1000.0
    max_retries: int = 5
    obstacle_polygon_sides: int = 16
    bit_rate_bps: float = 2_000_000.0
    carrier_frequency_hz: float = 2_400_000_000.0
    bandwidth_hz: float = 22_000_000.0
    sinr_threshold_db: float = 6.0
    cca_threshold_dbm: float = -82.0
    noise_density_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    nlos_model: str = "urban"


@dataclass
class RewardConfig:
    discovery: float = 10.0
    delivery: float = 50.0
    blocked_motion: float = 0.2
    expired_report: float = 10.0
    dropped_report: float = 10.0
    step_cost: float = 0.01
    communication_energy: float = 0.01


@dataclass
class MASACConfig:
    hidden_sizes: list = field(default_factory=lambda: [256, 256])
    replay_capacity: int = 50_000
    batch_size: int = 256
    learning_starts: int = 2000
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 0.0003
    initial_alpha: float = 0.2
    target_continuous_entropy: float = -3.0
    target_discrete_entropy_ratio: float = 0.9
    grad_clip_norm: float = 10.0


@dataclass
class RuntimeConfig:
    seed: int = 44
    eval_seed: int = 10044
    episodes: int = 1000
    eval_episodes: int = 3
    device: str = "auto"
    torch_threads: int = 1
    output_dir: str = "outputs/run"
    checkpoint_interval: int = 100


@dataclass
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    masac: MASACConfig = field(default_factory=MASACConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self):
        defaults = Config()
        for group in fields(self):
            obj, default = getattr(self, group.name), getattr(defaults, group.name)
            for item in fields(obj):
                value, example = getattr(obj, item.name), getattr(default, item.name)
                name = f"{group.name}.{item.name}"
                if isinstance(example, bool):
                    valid = isinstance(value, bool)
                elif isinstance(example, int):
                    valid = type(value) is int and value >= 0
                elif isinstance(example, float):
                    valid = type(value) in (float, int) and math.isfinite(value)
                elif isinstance(example, str):
                    valid = isinstance(value, str) and bool(value)
                else:
                    valid = value is None or isinstance(value, (list, tuple))
                if not valid:
                    raise ValueError(f"Invalid {name}: {value!r}")
        w, s, n, a, r = self.world, self.sensor, self.network, self.masac, self.runtime
        positive = [w.map_size_m, w.num_uavs, w.num_targets, w.dt_s, w.max_steps,
                    w.max_speed_mps, w.max_accel_mps2, w.safety_distance_m,
                    w.launch_radius_m, w.launch_min_spacing_m, w.placement_attempts,
                    w.obstacle_radius_min_m, w.obstacle_radius_max_m,
                    s.altitude_m, s.visible_target_slots, s.visible_obstacle_slots,
                    n.peer_range_m, n.gcs_range_m, n.tx_power_min_w, n.tx_power_max_w,
                    n.report_bytes, n.buffer_bytes, n.report_ttl_s, n.payload_bytes,
                    n.packet_lifetime_s, n.max_queue_packets, n.queue_poll_us, n.max_retries,
                    n.bit_rate_bps, n.carrier_frequency_hz, n.bandwidth_hz,
                    a.replay_capacity, a.batch_size, a.updates_per_step,
                    a.learning_rate, a.initial_alpha, a.grad_clip_norm,
                    r.episodes, r.eval_episodes, r.torch_threads, r.checkpoint_interval]
        if any(value <= 0 for value in positive):
            raise ValueError("Counts, scales, time intervals and capacities must be positive")
        if (w.obstacle_clearance_m < 0 or w.target_exclusion_radius_m < 0
                or w.obstacle_radius_min_m > w.obstacle_radius_max_m):
            raise ValueError("Invalid world clearances or obstacle radius range")
        if w.launch_min_spacing_m < w.safety_distance_m:
            raise ValueError("Launch spacing must be at least the safety distance")
        if not 0 < s.full_fov_deg < 180:
            raise ValueError("FOV must be strictly between 0 and 180 degrees")
        if n.tx_power_max_w < n.tx_power_min_w or n.buffer_bytes < n.report_bytes:
            raise ValueError("Invalid TX power range or buffer smaller than one report")
        if n.obstacle_polygon_sides < 8 or n.nlos_model not in {"urban", "free_space"}:
            raise ValueError("Invalid obstacle polygon or A2A NLOS model")
        if (not 0 <= a.gamma <= 1 or not 0 < a.tau <= 1
                or not 0 <= a.target_discrete_entropy_ratio <= 1
                or a.batch_size > a.replay_capacity):
            raise ValueError("Invalid MASAC discount, averaging, entropy or replay capacity")
        if not a.hidden_sizes or any(type(x) is not int or x <= 0 for x in a.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if r.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if w.gcs_position_m is not None:
            if (len(w.gcs_position_m) != 2 or
                    any(type(x) not in (int, float) or not math.isfinite(x)
                        or x < 0 or x > w.map_size_m for x in w.gcs_position_m)):
                raise ValueError("GCS must be a finite XY point inside the map")
        return self

    def to_dict(self):
        return asdict(self)


def _apply(obj, values, prefix=""):
    if not isinstance(values, dict):
        raise ValueError(f"{prefix or 'Config'} must be a mapping")
    for name, value in values.items():
        if name not in {f.name for f in fields(obj)}:
            raise ValueError(f"Unknown configuration key: {prefix}{name}")
        if is_dataclass(getattr(obj, name)):
            _apply(getattr(obj, name), value, f"{prefix}{name}.")
        else:
            setattr(obj, name, value)


def load_config(path=None, overrides=()):
    config = Config()
    if path is not None:
        _apply(config, yaml.safe_load(Path(path).read_text()) or {})
    for override in overrides:
        key, sep, value = override.partition("=")
        parts = key.split(".")
        if not sep or len(parts) != 2:
            raise ValueError("Overrides use group.name=value")
        _apply(config, {parts[0]: {parts[1]: yaml.safe_load(value)}})
    return config.validate()
