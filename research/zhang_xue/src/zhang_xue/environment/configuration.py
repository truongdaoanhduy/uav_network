from .shared import *

@dataclass
class EnvConfigV54:
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
    safety_mode: str = 'hard_veto'          # 'hard_veto' (v5.4 legacy) | 'kinematic_shield' (v5.7 default in EnvConfigV57)
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

    def __post_init__(self):
        CsmaProtocolConfig(
            difs_us=self.mac_difs_us, sifs_us=self.mac_sifs_us,
            slot_us=self.mac_slot_us, cw_min=self.mac_cw_min,
            max_retries=self.mac_max_retries, ack_timeout_us=self.mac_ack_timeout_us,
        )
        if type(self.mac_queue_capacity) is not int or not (1 <= self.mac_queue_capacity <= 64):
            raise ValueError("mac_queue_capacity must be an integer in 1..64")
        if not (math.isfinite(self.mac_noise_power_w) and self.mac_noise_power_w > 0):
            raise ValueError("mac_noise_power_w must be finite and positive")
        if not (math.isfinite(self.mac_snr_threshold_db) and
                math.isfinite(self.mac_cca_threshold_dbm)):
            raise ValueError("MAC PHY threshold must be finite")
        if self.map_size_m != GRID_W * CELL_SIZE_M:
            raise ValueError('map_size_m must match 20x20 physical sensor grid')
        if self.num_targets<1 or self.num_targets>NUM_CELLS:
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
        if self.obs_map_side!=10:
            raise ValueError('currently supported map encoder is exactly 10x10')
        if not (math.isfinite(self.dt_s) and self.dt_s>0 and
                math.isfinite(self.t_ref_s) and self.t_ref_s>0):
            raise ValueError('invalid timestep or AoI reference')
        if not (0 <= self.reward_peer_cap_per_step <= 400):
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
