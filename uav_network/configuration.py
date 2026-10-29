"""Validated configuration; use YAML or --set group.name=value."""
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
import math
from copy import deepcopy
import re

import yaml


@dataclass
class WorldConfig:
    generation_version: int = 3
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
    motion_safety: str = "dynamic_window"
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
    coverage_grid_cells: int = 0

    @property
    def footprint_radius_m(self):
        return self.altitude_m * math.tan(math.radians(self.full_fov_deg / 2))


@dataclass
class NetworkConfig:
    enabled: bool = True
    receiver_admission_mask: bool = True
    rediscovery_recovery: bool = True
    ack_uses_data_power: bool = True
    upstream_path: str = "third_party/UavNetSim"
    peer_range_m: float = 500.0
    gcs_range_m: float = 500.0
    tx_power_min_w: float = 0.1
    tx_power_max_w: float = 0.4
    tx_power_mapping: str = "linear"
    power_control: str = "learned"
    link_budget_margin_db: float = 3.0
    # Radio hardware defaults preserve the historical RF-output-only energy model.
    pa_efficiency: float = 1.0
    tx_circuit_power_w: float = 0.0
    rx_circuit_power_w: float = 0.0
    power_limited_admission: bool = False
    link_state_observation: bool = False
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
    propagation_delay_enabled: bool = False  # Opt in for new studies; preserve saved legacy runs.
    obstacle_extra_loss_db: float = 0.0


@dataclass
class EnergyConfig:
    enabled: bool = True
    battery_capacity_j: float = 277_200.0
    idle_power_w: float = 20.0
    hover_power_w: float = 120.0
    speed_squared_coefficient: float = 0.5
    acceleration_squared_coefficient: float = 2.0


@dataclass
class RewardConfig:
    discovery: float = 10.0
    delivery: float = 50.0
    blocked_motion: float = 0.2
    expired_report: float = 10.0
    dropped_report: float = 10.0
    step_cost: float = 0.01
    communication_energy: float = 0.01
    propulsion_energy_per_kj: float = 0.01
    communication_efficiency: float = 0.0
    shield_correction: float = 0.0


@dataclass
class OffPolicyConfig:
    hidden_sizes: list = field(default_factory=lambda: [256, 256])
    share_actor_parameters: bool = False
    share_critic_parameters: bool = False
    replay_capacity: int = 50_000
    batch_size: int = 256
    learning_starts: int = 2000
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 0.0003
    grad_clip_norm: float = 10.0


@dataclass
class MASACConfig(OffPolicyConfig):
    initial_alpha: float = 0.2
    target_continuous_entropy: float = -3.0
    target_discrete_entropy_ratio: float = 0.9


@dataclass
class MATD3Config(OffPolicyConfig):
    policy_delay: int = 2
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise_std: float = 0.1
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    epsilon_decay_steps: int = 100_000


@dataclass
class RuntimeConfig:
    algorithm: str = "masac"
    seed: int = 44
    eval_seed: int = 10044
    episodes: int = 1000
    eval_episodes: int = 1
    device: str = "auto"
    torch_threads: int = 1
    output_dir: str = "outputs/run"
    checkpoint_interval: int = 100
    num_envs: int = 1
    environment_backend: str = "native"
    shared_memory: bool = True
    collector_overlap: bool = True
    replay_device: str = "auto"
    replay_cpu_fraction: float = 0.0
    deterministic: bool = True
    compile_models: bool = False
    amp_mode: str = "off"
    coalesce_idle_polls: bool = True
    fused_optimizers: bool = False
    vectorize_critics: bool = False
    critic_chunk_size: int = 0
    log_interval_steps: int = 100
    warm_start_checkpoint: str = "none"
    max_train_seconds: float = 0.0


@dataclass
class PQLConfig:
    enabled: bool = False
    mode: str = "async"
    critic_sample_ratio: float = 8.0  # minibatch updates per eligible vector collection
    warmup_collection_steps: int = 32
    max_pending_collections: int = 2
    max_pending_policy_updates: int = 2
    collector_device: str = "auto"
    learner_device: str = "auto"
    policy_device: str = "auto"
    worker_timeout_s: float = 120.0
    checkpoint_replay: bool = True


@dataclass
class TrackingConfig:
    enabled: bool = False
    credentials_file: str = "key.txt"
    entity: str = "auto"
    project: str = "auto"
    run_name: str = "auto"
    group: str = "uav-network-2d"
    tags: list = field(default_factory=lambda: ["uav-network", "2d", "fov", "uavnetsim"])
    log_checkpoint: bool = True
    checkpoint_upload_interval_seconds: float = 300.0
    resumable_checkpoint_interval_seconds: float = 0.0
    resumable_checkpoint_max_uploads: int = 4
    local_checkpoints_to_keep: int = 2
    checkpoint_store: str = "huggingface"
    checkpoint_repo: str = "auto"
    checkpoint_repo_private: bool = False
    system_metrics: bool = True
    artifact_timeout_seconds: int = 120
    evaluation_interval: int = 50
    evaluation_interval_steps: int = 0  # Absolute joint transitions; zero disables.
    periodic_evaluation_episodes: int = 5
    periodic_evaluation_seed: int = 1000000044
    evaluation_timeout_seconds: int = 300
    diagnostic_window: int = 20
    delivery_gap_threshold_percent: float = 40.0
    return_drop_threshold: float = 20.0
    blocked_rate_threshold: float = 0.3


@dataclass
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    masac: MASACConfig = field(default_factory=MASACConfig)
    matd3: MATD3Config = field(default_factory=MATD3Config)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pql: PQLConfig = field(default_factory=PQLConfig)

    @property
    def algorithm_config(self):
        return getattr(self, self.runtime.algorithm)

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
        if w.generation_version != 3:
            raise ValueError("Unsupported scene generation version; use its pinned historical source")
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
        if w.motion_safety not in {"dynamic_window", "reject"}:
            raise ValueError("motion_safety must be dynamic_window or reject")
        if w.launch_min_spacing_m < w.safety_distance_m:
            raise ValueError("Launch spacing must be at least the safety distance")
        if not 0 < s.full_fov_deg < 180:
            raise ValueError("FOV must be strictly between 0 and 180 degrees")
        if (n.tx_power_min_w <= 0 or n.tx_power_max_w <= 0
                or n.tx_power_max_w < n.tx_power_min_w or n.buffer_bytes < n.report_bytes):
            raise ValueError("Invalid TX power range or buffer smaller than one report")
        if (not math.isfinite(n.pa_efficiency) or not 0 < n.pa_efficiency <= 1
                or not math.isfinite(n.tx_circuit_power_w) or n.tx_circuit_power_w < 0
                or not math.isfinite(n.rx_circuit_power_w) or n.rx_circuit_power_w < 0):
            raise ValueError("Radio PA efficiency must be in (0,1] and circuit powers finite/nonnegative")
        if s.coverage_grid_cells > 32:
            raise ValueError("sensor.coverage_grid_cells must be in [0,32]")
        if n.power_control not in {"learned", "link_budget"}:
            raise ValueError("network.power_control must be learned or link_budget")
        if not 0 <= n.link_budget_margin_db <= 30:
            raise ValueError("network.link_budget_margin_db must be in [0,30]")
        if n.tx_power_mapping not in {"linear", "log"}:
            raise ValueError("network.tx_power_mapping must be linear or log")
        if type(n.power_limited_admission) is not bool or type(n.link_state_observation) is not bool:
            raise ValueError("network V2 feature flags must be booleans")
        if n.obstacle_polygon_sides < 8:
            raise ValueError("Invalid obstacle polygon")
        if n.nlos_model not in {"urban", "suburban"}:
            raise ValueError("network.nlos_model must be one of: urban, suburban")
        if type(n.propagation_delay_enabled) is not bool:
            raise ValueError("network.propagation_delay_enabled must be a boolean")
        if (not 0 <= a.gamma <= 1 or not 0 < a.tau <= 1
                or not 0 <= a.target_discrete_entropy_ratio <= 1
                or a.batch_size > a.replay_capacity):
            raise ValueError("Invalid MASAC discount, averaging, entropy or replay capacity")
        if not a.hidden_sizes or any(type(x) is not int or x <= 0 for x in a.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if r.device not in {"auto", "cpu", "cuda"} or r.replay_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device and replay_device must be auto, cpu or cuda")
        if (type(r.replay_cpu_fraction) not in (int, float)
                or not 0 <= r.replay_cpu_fraction < 1
                or (r.replay_cpu_fraction > 0 and r.replay_device != "cpu")):
            raise ValueError("runtime.replay_cpu_fraction must be in [0, 1) and hybrid replay requires replay_device=cpu")
        p = self.pql
        if (p.mode not in {"async", "synchronous"} or p.critic_sample_ratio <= 0
                or min(p.max_pending_collections, p.max_pending_policy_updates) < 1
                or p.worker_timeout_s <= 0):
            raise ValueError("Invalid PQL mode, ratios, queue bounds or worker timeout")
        for value in (p.collector_device, p.learner_device, p.policy_device):
            if value not in {"auto", "cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", value):
                raise ValueError("PQL devices must be auto, cpu, cuda or cuda:N")
        if r.environment_backend not in {"native", "warp_packet"}:
            raise ValueError("environment_backend must be native or warp_packet")
        if r.algorithm not in {"masac", "matd3"}:
            raise ValueError("algorithm must be masac or matd3")
        if r.num_envs < 1 or r.log_interval_steps < 1:
            raise ValueError("num_envs and log_interval_steps must be positive")
        if r.critic_chunk_size < 0:
            raise ValueError("runtime.critic_chunk_size must be nonnegative; zero disables chunking")
        if r.amp_mode not in {"off", "fp16"}:
            raise ValueError("runtime.amp_mode must be off or fp16")
        if r.max_train_seconds < 0:
            raise ValueError("max_train_seconds must be nonnegative; zero disables the soft deadline")
        for name, minimum in (("evaluation_interval", 0), ("evaluation_interval_steps", 0),
                              ("periodic_evaluation_episodes", 1), ("periodic_evaluation_seed", 0),
                              ("evaluation_timeout_seconds", 1)):
            value = getattr(self.tracking, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid evaluation setting {name}: expected integer >= {minimum}")
        if self.tracking.diagnostic_window < 2:
            raise ValueError("diagnostic_window must be at least two episodes")
        for value in (self.tracking.delivery_gap_threshold_percent, self.tracking.return_drop_threshold,
                      self.tracking.blocked_rate_threshold):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Diagnostic thresholds must be finite and nonnegative")
        if (type(self.tracking.local_checkpoints_to_keep) is not int
                or self.tracking.local_checkpoints_to_keep < 1):
            raise ValueError("local_checkpoints_to_keep must be a positive integer")
        if self.tracking.checkpoint_store not in {"huggingface", "wandb"}:
            raise ValueError("checkpoint_store must be huggingface or legacy wandb")
        if (self.tracking.checkpoint_repo != "auto"
                and not re.fullmatch(r"[\w.-]+/[\w.-]+", self.tracking.checkpoint_repo)):
            raise ValueError("checkpoint_repo must be auto or owner/repository")
        t = self.tracking
        if (not math.isfinite(t.resumable_checkpoint_interval_seconds)
                or t.resumable_checkpoint_interval_seconds < 0
                or type(t.resumable_checkpoint_max_uploads) is not int
                or t.resumable_checkpoint_max_uploads < 1):
            raise ValueError("Invalid resumable_checkpoint interval or upload limit")
        if (t.resumable_checkpoint_interval_seconds > 0 and t.enabled and t.log_checkpoint
                and t.checkpoint_store == "huggingface"
                and self.pql.enabled and not self.pql.checkpoint_replay):
            raise ValueError("resumable_checkpoint requires full PQL replay")
        if self.tracking.artifact_timeout_seconds < 1:
            raise ValueError("artifact_timeout_seconds must be positive")
        if (not math.isfinite(self.tracking.checkpoint_upload_interval_seconds)
                or self.tracking.checkpoint_upload_interval_seconds < 0):
            raise ValueError("checkpoint_upload_interval_seconds must be finite and nonnegative")
        if self.algorithm_config.replay_capacity < r.num_envs:
            raise ValueError("Replay capacity must accommodate num_envs transitions")
        e = self.energy
        if e.battery_capacity_j <= 0 or any(v < 0 for v in (
                e.idle_power_w, e.hover_power_w, e.speed_squared_coefficient,
                e.acceleration_squared_coefficient, n.obstacle_extra_loss_db,
                self.reward.propulsion_energy_per_kj, self.reward.communication_efficiency,
                self.reward.shield_correction)):
            raise ValueError("Battery capacity must be positive and energy/reward/loss coefficients nonnegative")
        for learner in (self.masac, self.matd3):
            if (not 0 <= learner.gamma <= 1 or not 0 < learner.tau <= 1
                    or any(v <= 0 for v in (learner.replay_capacity, learner.batch_size,
                                           learner.updates_per_step, learner.learning_rate,
                                           learner.grad_clip_norm))
                    or learner.batch_size > learner.replay_capacity
                    or not learner.hidden_sizes
                    or any(type(v) is not int or v < 1 for v in learner.hidden_sizes)):
                raise ValueError("Invalid off-policy learner configuration")
        t = self.matd3
        if (t.policy_delay < 1 or t.epsilon_decay_steps < 1
                or not 0 <= t.final_epsilon <= t.initial_epsilon <= 1
                or min(t.target_noise_std, t.target_noise_clip, t.exploration_noise_std) < 0):
            raise ValueError("Invalid MATD3 delay, noise or epsilon schedule")
        if w.gcs_position_m is not None:
            if (len(w.gcs_position_m) != 2 or
                    any(type(x) not in (int, float) or not math.isfinite(x)
                        or x < 0 or x > w.map_size_m for x in w.gcs_position_m)):
                raise ValueError("GCS must be a finite XY point inside the map")
        if not isinstance(self.tracking.tags, (list, tuple)) or any(not isinstance(t, str) or not t for t in self.tracking.tags):
            raise ValueError("tracking.tags must be a list of nonempty strings")
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


def with_legacy_feature_defaults(values):
    """Fill only newly introduced optional settings, never required science fields."""
    if not isinstance(values, dict):
        raise ValueError("Checkpoint configuration must be a mapping")
    result = deepcopy(values)
    for group, defaults in (
        ("sensor", {"coverage_grid_cells": 0}),
        ("network", {"power_control": "learned", "link_budget_margin_db": 3.0}),
    ):
        if group in result:
            if not isinstance(result[group], dict):
                raise ValueError(f"Checkpoint {group} must be a mapping")
            for key, value in defaults.items():
                result[group].setdefault(key, value)
    return result
