"""Independent learner hyperparameters; does not encode an old UAV scenario."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
import math

@dataclass
class OffPolicyConfig:
    hidden_sizes: list = field(default_factory=lambda: [256, 256])
    share_actor_parameters: bool = False
    share_critic_parameters: bool = False
    actor_layer_norm: bool = False
    actor_batch_norm: bool = False
    critic_layer_norm: bool = False
    critic_batch_norm: bool = False
    activation: str = "relu"
    sync_critic_loss_scale: bool = False
    critic_huber_delta: float = 0.0
    replay_capacity: int = 50_000
    batch_size: int = 256
    learning_starts: int = 2000
    updates_per_step: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 0.0003
    grad_clip_norm: float = 10.0
    grad_clip_mode: str = "global"


@dataclass
class MASACConfig(OffPolicyConfig):
    shared_critic_mode: str = "dense"
    temperature_loss: str = "log_alpha"  # "alpha" selects the original SAC dual loss.
    initial_alpha: float = 0.2
    target_continuous_entropy: float = -2.0
    target_discrete_entropy_ratio: float = 0.9
    target_discrete_entropy_final_ratio: float = 0.9
    target_discrete_entropy_anneal_updates: int = 0
    # Historical entropy follows mask size; the reference uses a fixed action count.
    mask_aware_discrete_entropy_target: bool = True


@dataclass
class MATD3Config(OffPolicyConfig):
    # Original author p_train uses 1e-3; zero preserves historical UAV runs.
    actor_output_regularization: float = 0.0
    pre_tanh_reg: float = 0.0
    # Only an explicit ablation changes PyTorch default Linear initialization.
    initialization: str = "default"
    policy_delay: int = 2
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise_std: float = 0.1
    output_normalization: bool = False
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    epsilon_decay_steps: int = 100_000

@dataclass
class LearnerWorld:
    num_uavs: int = 6

@dataclass
class LearnerRadio:
    power_control: str = "fixed"
    destination_mode: str = "per_peer"  # "compact_3" = SILENT, GCS, SHARE_PEERS

@dataclass
class LearnerRuntime:
    algorithm: str = "matd3"
    num_envs: int = 1
    fused_optimizers: bool = False
    vectorize_critics: bool = False
    critic_chunk_size: int = 0
    amp_mode: str = "off"

@dataclass
class LearnerConfig:
    world: LearnerWorld = field(default_factory=LearnerWorld)
    network: LearnerRadio = field(default_factory=LearnerRadio)
    runtime: LearnerRuntime = field(default_factory=LearnerRuntime)
    matd3: MATD3Config = field(default_factory=MATD3Config)
    masac: MASACConfig = field(default_factory=MASACConfig)
    motion_dim: int = 2
    continuous_dim: int = 3
    scenario_id: str = "zhang-xue-7radio-v60"

    def validate(self):
        if type(self.world.num_uavs) is not int or not 1 <= self.world.num_uavs <= 30:
            raise ValueError("num_uavs must be an integer in [1,30]")
        if self.motion_dim not in (2,):
            raise ValueError("This actor adapter uses XY controls, not old XYZ controls")
        if self.continuous_dim not in (self.motion_dim, self.motion_dim + 1):
            raise ValueError("continuous_dim must be 2 (motors only) or 3 (legacy/RF head)")
        if self.network.power_control == "learned" and self.continuous_dim != self.motion_dim + 1:
            raise ValueError("learned RF power requires the third continuous dimension")
        if self.network.destination_mode not in ("per_peer", "compact_3"):
            raise ValueError("destination_mode must be per_peer or compact_3")
        if self.network.power_control not in ("fixed", "learned"):
            raise ValueError("power_control must be fixed or learned")
        if self.runtime.algorithm not in ("matd3", "masac"):
            raise ValueError("Algorithm must be MATD3 or MASAC")
        if type(self.runtime.num_envs) is not int or self.runtime.num_envs < 1:
            raise ValueError("runtime.num_envs must be a positive integer")
        if self.runtime.amp_mode not in ("off", "fp16", "bf16"):
            raise ValueError("Unsupported AMP mode")
        if self.matd3.policy_delay < 1 or type(self.matd3.policy_delay) is not int:
            raise ValueError("MATD3 policy_delay must be a positive integer")
        for name in ("matd3", "masac"):
            c = getattr(self, name)
            if not c.hidden_sizes or any(type(n) is not int or n < 1 for n in c.hidden_sizes):
                raise ValueError(f"{name}.hidden_sizes must be positive integers")
            for field_name in ("batch_size", "replay_capacity", "learning_starts", "updates_per_step"):
                val = getattr(c, field_name)
                if type(val) is not int or val < 1:
                    raise ValueError(f"{name}.{field_name} must be a positive integer")
            for field_name in ("learning_rate", "gamma", "tau"):
                val = getattr(c, field_name)
                if not isinstance(val, (float, int)) or isinstance(val, bool) or not math.isfinite(val) or val <= 0:
                    raise ValueError(f"{name}.{field_name} must be finite and positive")
            gc = c.grad_clip_norm
            if not isinstance(gc, (float, int)) or isinstance(gc, bool) or not math.isfinite(gc) or gc < 0:
                raise ValueError(f"{name}.grad_clip_norm must be finite and nonnegative")
            if c.gamma > 1 or c.tau > 1:
                raise ValueError(f"{name} gamma/tau must not exceed one")
            if c.grad_clip_mode not in ("global", "per_parameter", "none"):
                raise ValueError(f"{name}.grad_clip_mode is unsupported")
            for key in ("critic_layer_norm", "critic_batch_norm", "sync_critic_loss_scale"):
                if type(getattr(c, key)) is not bool:
                    raise ValueError(f"{name}.{key} requires a boolean")
            if c.critic_layer_norm and c.critic_batch_norm:
                raise ValueError(f"{name}: select one critic normalization")
            if c.activation not in ("relu", "silu"):
                raise ValueError(f"{name}.activation must be 'relu' or 'silu'")
            if not isinstance(c.critic_huber_delta, (int, float)) or isinstance(c.critic_huber_delta, bool) or not math.isfinite(c.critic_huber_delta) or c.critic_huber_delta < 0:
                raise ValueError(f"{name}.critic_huber_delta must be finite and nonnegative")
        if self.masac.initial_alpha <= 0 or not math.isfinite(self.masac.initial_alpha):
            raise ValueError("MASAC alpha must be positive and finite")
        if self.matd3.initial_epsilon < 0 or self.matd3.final_epsilon < 0:
            raise ValueError("MATD3 epsilon must be nonnegative")
        if not isinstance(self.matd3.pre_tanh_reg, (int, float)) or isinstance(self.matd3.pre_tanh_reg, bool) or not math.isfinite(self.matd3.pre_tanh_reg) or self.matd3.pre_tanh_reg < 0:
            raise ValueError("MATD3 pre_tanh_reg must be finite and nonnegative")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        obj=cls()
        if not isinstance(payload,dict) or set(payload)!=set(obj.to_dict()):
            raise ValueError("Unexpected learner config schema; use archived source for old checkpoints")
        for name,value in payload.items():
            target=getattr(obj,name)
            if hasattr(target,"__dataclass_fields__"):
                # V72 checkpoint v4 predates this diagnostic scheduling field.
                # Migrate only that exact key set; scientific fields stay strict.
                if (name == "runtime" and isinstance(value, dict) and
                        set(value) == set(asdict(target)) - {"num_envs"}):
                    value = {**value, "num_envs": 1}
                if name in ("matd3", "masac") and isinstance(value, dict):
                    defaults = asdict(target)
                    v81_opt = {"critic_layer_norm", "activation", "sync_critic_loss_scale",
                               "critic_huber_delta", "pre_tanh_reg"}
                    missing = set(defaults) - set(value)
                    if missing and missing <= v81_opt:
                        value = {**{k: defaults[k] for k in missing}, **value}
                if set(value)!=set(asdict(target)):
                    raise ValueError("Unexpected learner field: "+name)
                setattr(obj,name,type(target)(**value))
            else:
                setattr(obj,name,value)
        return obj.validate()
