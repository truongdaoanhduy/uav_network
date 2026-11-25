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
    critic_batch_norm: bool = False
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
    target_continuous_entropy: float = -3.0
    target_discrete_entropy_ratio: float = 0.9
    target_discrete_entropy_final_ratio: float = 0.9
    target_discrete_entropy_anneal_updates: int = 0
    # Historical entropy follows mask size; the reference uses a fixed action count.
    mask_aware_discrete_entropy_target: bool = True


@dataclass
class MATD3Config(OffPolicyConfig):
    # Original author p_train uses 1e-3; zero preserves historical UAV runs.
    actor_output_regularization: float = 0.0
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

@dataclass
class LearnerRuntime:
    algorithm: str = "matd3"
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
        if self.world.num_uavs != 6:
            raise ValueError("Current seven-radio scenario requires exactly six learning actors")
        if self.motion_dim not in (2,):
            raise ValueError("This actor adapter uses XY controls, not old XYZ controls")
        if self.continuous_dim != self.motion_dim + 1:
            raise ValueError("Learner retains a fixed/zeroed RF-power head")
        if self.network.power_control != "fixed":
            raise ValueError("Learned TX power not implemented in the seven-radio scenario")
        if self.runtime.algorithm not in ("matd3","masac"):
            raise ValueError("Algorithm must be MATD3 or MASAC")
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
                if set(value)!=set(asdict(target)):
                    raise ValueError("Unexpected learner field: "+name)
                setattr(obj,name,type(target)(**value))
            else:
                setattr(obj,name,value)
        return obj.validate()
