from .shared import *

@dataclass
class EnvConfigV54:
    num_uavs: int = 6                       # v6.3: scalable N-UAV (e.g., 4, 6, 8, 10, 12)
    grid_w: int = 20                        # v6.3: scalable belief grid (20x20 or 40x40)
    comm_action_mode: str = 'per_peer'      # 'per_peer' (legacy N+1) | 'compact_3' (0=NO_SEND,1=GCS,2=PEER)
    map_size_m: float = 1000.0
    uav_alt_m: float = 50.0
    v_max_mps: float = 25.0
    a_max_mps2: float = 5.0
    fov_radius_m: float = 50.0
    uav_safe_distance_m: float = 50.0
    w_blocked_motion: float = 0.2
    dt_s: float = 1.0
    safety_horizon_s: int = 1500
    num_targets: int = 15
    pd: float = 0.85
    pf: float = 0.03
    ds_mf: float = 0.80
    ds_me: float = 0.80
    confirm_th: float = 0.90
    tx_power_w: float = 0.5
    # Balanced reward weights
    w_loc_team: float = 0.50
    w_p2p: float = 0.25
    r_tp: float = 25.0
    r_fp: float = 12.5
    fp_hold: float = 0.10
    w_aoi: float = 6.0
    t_ref_s: float = 60.0
    lambda_b: float = 5.0
    # Distinguish a source-faithful Zhang MAX-count baseline from a
    # provenance-safe novel packet-network extension.
    ds_fusion_mode: str = 'sum'  # sum | zhang_max
    ds_aoi_eta: float = 0.02  # 1/s; frozen discount at actual packet reception
    evidence_calibration: str = 'paper_fixed' # paper_fixed | sensor_likelihood_ratio
    ds_uncertainty_ceiling: float = 0.20
    # Team coverage reward/peer reward computed from local evidence/actual packet events.
    reward_peer_cap_per_step: int = 12
    # 5 x 10 x 10: target mass, empty mass, unknown mass, AoI, own location.
    obs_map_side: int = 10
    # v5.7 scientific & physics upgrades (backward-compatible flags)
    safety_mode: str = 'kinematic_shield'  # only physically constrained policy supported
    enable_3d_sensing_occlusion: bool = True  # C2 fix: block ground sensing through 70-80m buildings
    enable_accel_energy: bool = False       # v5.7 Zeng 2019 acceleration thrust-tilt + kinetic delta
    enable_fp_unlatch: bool = False         # v5.7 allow FP correction when negative evidence lowers m_T < confirm_th
    obs_pool_mode: str = 'mean'             # 'mean' (v5.5) | 'semantic' (v5.7: max m_T, min m_F, mean m_U, max AoI)
    peer_reward_mode: str = 'cell_count'    # 'cell_count' (v5.5) | 'info_gain' (v5.7 anti-reward-hacking)
    aoi_reward_mode: str = 'linear'         # 'linear' (v5.4) | 'exponential' (v5.7 non-vanishing gradient)
    e_max_j: float = 180_000.0              # 180 kJ (50 Wh v5.4) | 270 kJ (75 Wh v5.7 mission-matched)
    bba_reliability_cap: float | None = None  # v5.7 cap (e.g. 0.80) for sensor_likelihood_ratio
    safety_failure_penalty: float = 100.0  # v5.8: fail-closed terminal collision cost (not a safety proof)
    safety_failure_terminate: bool = True
    battery_termination_mode: str = 'any_depleted'  # legacy | 'all_depleted' team-resilient mission
    aoi_ewma_end_to_end: bool = False       # v6.2: UQ8.8 EWMA effective counts in 8B packet
    enable_tx_power_control: bool = False   # v6.2: continuous per-UAV TX power [0.05, 0.50] W
    enable_peer_dissemination: bool = True  # controlled research ablation, mask SHARE_PEERS when false
    enable_data_mule_forwarding: bool = False  # v6.5: store-carry-forward multi-origin relay via SHARE_PEERS
    coalesce_duplicate_queue: bool = False    # optional protocol variant; retain baseline FIFO by default
    scale_obstacles_with_map: bool = True      # v6.5: scale 5 buildings & 2 blackouts proportionally for 2-3 km maps
    rf_obstacle_model: str = 'binary_nlos'      # binary_nlos | multi_obstacle | rf_opaque
    rf_per_building_loss_db: float = 12.0       # parametric RF sweep, not an empirical material calibration
    tx_power_min_w: float = 0.05
    tx_power_max_w: float = 0.50
    network_substeps: int = 1               # 1=legacy 1s | 2=500ms midpoint mobility-PHY sync
    gcs_target_exclusion_m: float = 0.0   # v5.8 configurable ground-target/GCS separation
    target_min_separation_m: float = 0.0  # v5.8 min separation; 0 preserves v5.7 benchmark
    # CPU CSMA/CA tunables; Warp packet timings remain compile-time until parity gate.
    mac_difs_us: int = 50
    mac_sifs_us: int = 10
    mac_slot_us: int = 20
    mac_cw_min: int = 31
    mac_max_retries: int = 5
    mac_ack_timeout_us: int = 180
    mac_queue_capacity: int = 64
    mac_noise_power_w: float = 4.0e-11
    mac_snr_threshold_db: float = 6.0
    mac_cca_threshold_dbm: float = -82.0
    # v7.1 reward scaling, continuous AoI ablation & decentralized peer selection
    reward_scale_mode: str = 'raw'           # 'raw' | 'per_target_norm'
    continuous_aoi_penalty_w: float = 0.0    # >=0 per-step continuous mapwide AoI penalty weight
    compact3_decentralized_peer_select: bool = True  # False explicitly selects legacy oracle nearest-peer experiment

    def __post_init__(self):
        # Scientific knobs must fail closed; otherwise a typo silently picks
        # the obsolete motion/reward branch and corrupts comparison runs.
        modes={
            "safety_mode": ("kinematic_shield",),
            "obs_pool_mode": ("mean", "semantic"),
            "peer_reward_mode": ("cell_count", "info_gain"),
            "aoi_reward_mode": ("linear", "exponential"),
            "battery_termination_mode": ("any_depleted", "all_depleted"),
            "reward_scale_mode": ("raw", "per_target_norm"),
            "rf_obstacle_model": ("binary_nlos", "multi_obstacle", "rf_opaque"),
        }
        for key, legal in modes.items():
            if getattr(self,key) not in legal:
                raise ValueError(f"{key} must be one of {legal}")
        strictly_positive=(
            "map_size_m", "uav_alt_m", "v_max_mps", "a_max_mps2",
            "fov_radius_m", "uav_safe_distance_m", "dt_s", "tx_power_w",
            "e_max_j", "t_ref_s",
        )
        for key in strictly_positive:
            value=getattr(self,key)
            if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
                raise ValueError(f"{key} must be finite and positive")
        if type(self.safety_horizon_s) is not int or self.safety_horizon_s<1:
            raise ValueError("safety_horizon_s must be a positive integer")
        optional_mass=self.bba_reliability_cap
        if optional_mass is not None and (type(optional_mass) not in (int,float)
                or not math.isfinite(optional_mass) or not 0<optional_mass<1):
            raise ValueError("bba_reliability_cap must lie strictly inside (0, 1)")
        for key in (
            "w_loc_team", "w_p2p", "r_tp", "r_fp", "fp_hold", "w_aoi",
            "w_blocked_motion", "ds_mf", "ds_me", "ds_uncertainty_ceiling",
        ):
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in (
            "enable_3d_sensing_occlusion", "enable_accel_energy", "enable_fp_unlatch",
            "safety_failure_terminate", "enable_peer_dissemination",
            "enable_data_mule_forwarding", "coalesce_duplicate_queue",
            "scale_obstacles_with_map", "compact3_decentralized_peer_select",
        ):
            if type(getattr(self,key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        if type(self.network_substeps) is not int or self.network_substeps not in (1, 2):
            raise ValueError("network_substeps must be either 1 or 2")
        if (type(self.rf_per_building_loss_db) not in (int, float)
                or not math.isfinite(self.rf_per_building_loss_db)
                or self.rf_per_building_loss_db < 0):
            raise ValueError("rf_per_building_loss_db must be finite and non-negative")
        if type(self.enable_tx_power_control) is not bool:
            raise ValueError("enable_tx_power_control must be a boolean")
        for field_name in ("tx_power_min_w", "tx_power_max_w"):
            val = getattr(self, field_name)
            if type(val) not in (int, float) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.tx_power_max_w < self.tx_power_min_w:
            raise ValueError("Require tx_power_min_w <= tx_power_max_w")
        if self.enable_tx_power_control and not self.tx_power_min_w <= self.tx_power_w <= self.tx_power_max_w:
            raise ValueError("Default TX power outside learned control range")
        CsmaProtocolConfig(
            difs_us=self.mac_difs_us, sifs_us=self.mac_sifs_us,
            slot_us=self.mac_slot_us, cw_min=self.mac_cw_min,
            max_retries=self.mac_max_retries, ack_timeout_us=self.mac_ack_timeout_us,
        )
        if type(self.mac_queue_capacity) is not int or not (1 <= self.mac_queue_capacity <= 200):
            raise ValueError("mac_queue_capacity must be an integer in 1..200 (CPU only; Warp differs)")
        if not (math.isfinite(self.mac_noise_power_w) and self.mac_noise_power_w > 0):
            raise ValueError("mac_noise_power_w must be finite and positive")
        if not (math.isfinite(self.mac_snr_threshold_db) and
                math.isfinite(self.mac_cca_threshold_dbm)):
            raise ValueError("MAC PHY threshold must be finite")
        if isinstance(self.num_uavs, bool) or not isinstance(self.num_uavs, int) or not (1 <= self.num_uavs <= 30):
            raise ValueError('num_uavs must be an integer in [1, 30]')
        if isinstance(self.grid_w, bool) or not isinstance(self.grid_w, int) or self.grid_w < 10 or self.grid_w % self.obs_map_side != 0 or (self.grid_w * self.grid_w) > 4096:
            raise ValueError('grid_w must be a positive multiple of obs_map_side with grid_w^2 <= 4096')
        if self.comm_action_mode not in ('per_peer', 'compact_3'):
            raise ValueError('comm_action_mode must be per_peer or compact_3')
        if not (math.isfinite(self.continuous_aoi_penalty_w) and self.continuous_aoi_penalty_w >= 0.0):
            raise ValueError('continuous_aoi_penalty_w must be finite and non-negative')
        if not (math.isfinite(self.lambda_b) and self.lambda_b >= 0.0):
            raise ValueError('lambda_b must be finite and non-negative')
        if abs(self.map_size_m - self.grid_w * CELL_SIZE_M) > 1e-6:
            raise ValueError('map_size_m must match grid_w * CELL_SIZE_M (50m cells)')
        if isinstance(self.num_targets, bool) or not isinstance(self.num_targets, int) or self.num_targets<1 or self.num_targets>(self.grid_w * self.grid_w):
            raise ValueError('invalid number of targets')
        if self.ds_fusion_mode not in ('sum','zhang_max'):
            raise ValueError('ds_fusion_mode must be sum or zhang_max')
        if self.evidence_calibration not in ('paper_fixed','sensor_likelihood_ratio'):
            raise ValueError('unknown evidence_calibration')
        if not (0 < self.pf < self.pd < 1):
            raise ValueError('Require 0 < Pf < Pd < 1')
        if not (0<self.ds_mf<1 and 0<self.ds_me<1 and 0<self.confirm_th<=1
                and 0<=self.ds_uncertainty_ceiling<=1):
            raise ValueError('invalid DS masses or thresholds')
        if not(math.isfinite(self.ds_aoi_eta) and self.ds_aoi_eta>=0):
            raise ValueError('invalid AoI decay rate')
        # The mission clock, 8-byte generation timestamps, and AoI accumulation
        # currently count *one second per environment step*. Do not silently
        # accept a different dt_s as if seconds were still tracked correctly.
        if self.dt_s != 1.0:
            raise ValueError('Current packet/AoI clock requires dt_s=1.0 second')
        if self.aoi_ewma_end_to_end:
            # One source can observe the same cell once per step. With an N-step
            # mission, sum(exp(-eta*j)) is the maximum effective count per cell.
            # Above UQ8.8's range, transport clips evidence silently.
            from ..belief.dempster_shafer import UQ88_MAX_VAL
            eta = float(self.ds_aoi_eta)
            n = int(self.safety_horizon_s)
            max_effective_count = (float(n) if eta == 0.0 else
                                   -math.expm1(-eta * n) / -math.expm1(-eta))
            if max_effective_count > UQ88_MAX_VAL:
                raise ValueError('UQ8.8 effective-count saturation possible: '
                                 'increase ds_aoi_eta or shorten safety_horizon_s')
        if self.obs_map_side!=10:
            raise ValueError('currently supported map encoder is exactly 10x10')
        if not (math.isfinite(self.dt_s) and self.dt_s>0 and
                math.isfinite(self.t_ref_s) and self.t_ref_s>0):
            raise ValueError('invalid timestep or AoI reference')
        if isinstance(self.reward_peer_cap_per_step, bool) or not isinstance(self.reward_peer_cap_per_step, int) or not (0 <= self.reward_peer_cap_per_step <= self.grid_w * self.grid_w):
            raise ValueError('invalid peer information cap')
        if self.safety_failure_penalty < 0.0 or not math.isfinite(self.safety_failure_penalty):
            raise ValueError('safety_failure_penalty must be finite and nonnegative')
        if (not math.isfinite(self.gcs_target_exclusion_m) or self.gcs_target_exclusion_m < 0.0 or
                not math.isfinite(self.target_min_separation_m) or self.target_min_separation_m < 0.0):
            raise ValueError('Target/GCS exclusion radii must be finite and nonnegative')

def make_v57_config(**overrides) -> EnvConfigV54:
    """Recommended v5.7 Q1-ready configuration with all physics, sensing, AoI, and reward fixes enabled."""
    defaults = dict(
        safety_mode="kinematic_shield",
        enable_3d_sensing_occlusion=True,
        enable_accel_energy=True,
        enable_fp_unlatch=True,
        obs_pool_mode="semantic",
        peer_reward_mode="info_gain",
        aoi_reward_mode="exponential",
        e_max_j=270_000.0,  # 75 Wh matches 1500s horizon without forced early battery death
        w_p2p=0.10,
        safety_failure_penalty=100.0,
        safety_failure_terminate=True,
        bba_reliability_cap=0.80,
    )
    defaults.update(overrides)
    return EnvConfigV54(**defaults)


def make_v62_config(**overrides) -> EnvConfigV54:
    """v6.2 Q1-aligned configuration: end-to-end UQ8.8 AoI EWMA, Deng entropy reward,
    continuous TX power control, 2-substep midpoint PHY-mobility sync, and braking shield."""
    defaults = dict(
        enable_3d_sensing_occlusion=True,
        enable_accel_energy=True,
        enable_fp_unlatch=True,
        obs_pool_mode="semantic",
        peer_reward_mode="info_gain",
        aoi_reward_mode="exponential",
        e_max_j=270_000.0,
        w_p2p=0.10,
        bba_reliability_cap=0.80,
        aoi_ewma_end_to_end=True,
        enable_tx_power_control=True,
        network_substeps=2,
    )
    defaults.update(overrides)
    return EnvConfigV54(**defaults)


def make_v63_q1_config(**overrides) -> EnvConfigV54:
    """v6.3 Q1-Clean & Scalable configuration (UavNetSim-v1 PHY aligned, zero bloat).

    - Scalable `num_uavs` (default 6; change freely to 4, 8, 10, 12 for scalability experiments).
    - Compact 3-action discrete comm (`0=NO_SEND, 1=SEND_GCS, 2=SHARE_PEERS`) so policy is N-invariant.
    - Reference power is 0.1 W. Source UavNetSim uses CCA=-82 dBm and thermal noise with 22MHz BW + 7dB NF; this is not full simulator parity.
    - Pure Zeng 2019 Eq. 12 steady-level battery (`e_max_j=180_000 J`, `horizon=900 s`, `v_max=15 m/s`).
    - Pure Zhang 2024 `paper_fixed` DS (0.80/0.80) + end-to-end Xue 2026 UQ8.8 EWMA AoI + Deng Entropy reward.
    - No redundant continuous RF power head, no tiny `r_airtime` noise (`lambda_b=0.0`).
    """
    defaults = dict(
        num_uavs=6,
        grid_w=20,
        comm_action_mode="compact_3",
        v_max_mps=15.0,
        a_max_mps2=5.0,
        safety_horizon_s=900,
        tx_power_w=0.1,
        mac_cca_threshold_dbm=-82.0,
        mac_noise_power_w=1e-3 * 10.0 ** ((-174.0 + 10.0 * math.log10(22.0e6) + 7.0) / 10.0),
        enable_3d_sensing_occlusion=True,
        enable_accel_energy=False,
        enable_fp_unlatch=True,
        obs_pool_mode="semantic",
        peer_reward_mode="info_gain",
        aoi_reward_mode="exponential",
        evidence_calibration="paper_fixed",
        e_max_j=180_000.0,
        w_p2p=0.10,
        lambda_b=0.0,
        aoi_ewma_end_to_end=True,
        enable_tx_power_control=False,
    )
    defaults.update(overrides)
    return EnvConfigV54(**defaults)


def make_uavnetsim_reference_config(**overrides) -> EnvConfigV54:
    """Explicit PHY parameter reference, NOT packet-event-equivalent simulation.

    UavNetSim master: transmitting_power=.1 W, CCA=-82 dBm, 802.11b BW=22MHz,
    thermal noise PSD=-174 dBm/Hz, receiver noise figure=7 dB.
    https://github.com/Zihao-Felix-Zhou/UavNetSim/blob/master/utils/config.py
    """
    defaults = dict(
        tx_power_w=0.1,
        mac_cca_threshold_dbm=-82.0,
        mac_noise_power_w=1e-3 * 10.0 ** ((-174.0 + 10.0 * math.log10(22e6) + 7.0) / 10.0),
        mac_snr_threshold_db=6.0,
        mac_slot_us=20,
        mac_sifs_us=10,
        mac_difs_us=50,
        mac_cw_min=31,
        mac_max_retries=5,
        mac_ack_timeout_us=180,
        mac_queue_capacity=200,
    )
    defaults.update(overrides)
    return EnvConfigV54(**defaults)


def make_multihop_research_config(**overrides) -> EnvConfigV54:
    """Optional denser-interference setting; DOES NOT match upstream PHY noise.

    This is an intentionally different, more challenging research ablation.
    Never describe its -70 dBm CCA and 4e-11 W noise as UavNetSim defaults.
    """
    defaults = dict(mac_cca_threshold_dbm=-70.0, mac_noise_power_w=4.0e-11)
    defaults.update(overrides)
    return make_v63_q1_config(**defaults)


def make_v65_datamule_config(**overrides) -> EnvConfigV54:
    """v6.5 Q1 Research Baseline: True Data-Mule Multi-Origin Forwarding (w_p2p=0.0).

    - SHARE_PEERS forwards multi-origin provenance entries (origin != dst) so UAVs
      without a direct GCS route can eventually relay via contacts with peers.
    - Removes artificial peer-shaping reward (`w_p2p=0.0`); peers share purely for mission AoI/TP.
    - Selected UavNetSim PHY parameters; network/MAC-event parity NOT established.
    """
    defaults = dict(
        enable_data_mule_forwarding=True,
        scale_obstacles_with_map=True,
        w_p2p=0.0,
    )
    defaults.update(overrides)
    return make_v63_q1_config(**defaults)
