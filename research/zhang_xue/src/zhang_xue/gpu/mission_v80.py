"""V80 GPU-vectorized Warp UAV Mission Runtime with Feasible Kinematic Safety Shield.

Extends `WarpParallelMissionV79` (preserving 100% V79 test compatibility) by swapping only
the motion/safety kernel to `_candidate_v80` + `_move_v80` (66-candidate acceleration-disk
projection + 3-sweep coordinate descent + CPU-parity physical collision detector).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from zhang_xue.gpu.mission_v79 import GPUWorldV79Params, WarpParallelMissionV79


@dataclass
class GPUWorldV80Params(GPUWorldV79Params):
    dt_s: float = 1.0
    strict_braking_terminate: bool = False  # False = CPU trajectory_safety_violations parity

    def validate(self) -> None:
        super().validate()
        # V79 fixed MAC duration, sensing sampling and physical shield assume dt=1s.
        # Reject alternative dt instead of silently simulating a different experiment.
        if not math.isfinite(self.dt_s) or abs(self.dt_s - 1.0) > 1.0e-12:
            raise ValueError("V80 Warp runtime requires dt_s=1.0; MAC and safety kernels use a 1s tick")


class WarpParallelMissionV80(WarpParallelMissionV79):
    """V80 Warp GPU runtime with feasible lateral evasion + 3-sweep coordinate descent shield."""

    def __init__(self, params: GPUWorldV80Params):
        super().__init__(params)

    def step(self, continuous, discrete):
        from zhang_xue.gpu.kernels.inject_v67 import compute_dirty_dest_v67, inject_v67, resolve_compact3_v67
        from zhang_xue.gpu.kernels.mission_v79_kernels import _reward_v79, _sense_visible_v79
        from zhang_xue.gpu.kernels.mission_v80_shield import _candidate_v80, _move_v80
        from zhang_xue.gpu.kernels.mission_v93_validation import validate_actions_v93

        w = self.buffers
        n = self.network
        p = self.params
        if tuple(continuous.shape) != (self.envs, 6, 2) or tuple(discrete.shape) != (self.envs, 6):
            raise ValueError("act shape should be (E,6,2) and (E,6)")
        self._launch(validate_actions_v93, self.envs, [
            continuous, discrete, w["failed"], w["safety_reason"],
        ])
        self._launch(compute_dirty_dest_v67, (self.envs, 6, 7), [n.state, n.dims])
        self._launch(resolve_compact3_v67, (self.envs, 6), [n.state, n.dims, discrete, w["pos_xy"], n._dest_buf])
        self._launch(inject_v67, (self.envs, 6), [n.state, n.dims, n._dest_buf])
        self._capture_hello_offers()
        n.step(
            duration_us=1_000_000, max_events=p.max_events, tx_power_w=p.tx_power_w,
            cca_threshold_w=self.cca_w, noise_w=p.noise_power_w, snr_threshold=self.snr_lin,
            check_host_guards=False,
        )
        self._launch(_candidate_v80, (self.envs, 6), [
            continuous, w["velocities"], w["candidates"],
            float(p.dt_s), float(p.v_max_mps), float(p.a_max_mps2),
        ])
        self._launch(_move_v80, self.envs, [
            w["positions"], w["velocities"], w["candidates"], w["v_safe"], w["pos_xy"],
            w["battery"], w["battery_comp"], self.build_wp, len(self.buildings), n.state, w["failed"],
            w["safety_reason"], w["unsafe_mask"], w["intervention"],
            float(p.dt_s), float(p.v_max_mps), float(p.a_max_mps2),
            float(p.uav_safe_distance_m), float(self.pa_dc_w),
            int(p.strict_braking_terminate),
        ])
        self._update_phy()
        self._launch(_sense_visible_v79, (self.envs, 6), [
            n.state, w["positions"], w["truth"], w["episode_step"], w["episode_id"],
            w["battery"], w["unsafe_mask"], w["failed"], self.build_wp, len(self.buildings),
            int(p.seed), float(p.pd), float(p.pf), float(p.ds_aoi_eta),
            int(p.static_raw_count), int(p.sensor_revisit_cooldown_steps),
        ])
        self._launch(_reward_v79, self.envs, [
            n.state, w["truth"], w["team_u_prev"], w["gcs_tp"], w["gcs_tp_active"], w["gcs_tp_drops"],
            w["gcs_fp"], w["gcs_fp_ever"],
            w["battery"], w["episode_step"], w["failed"], w["intervention"], w["reward"],
            w["reward_components"], w["terminated"], w["truncated"], w["done"], w["termination_reason"],
            self.logmf, self.logme, float(p.ds_aoi_eta), int(p.static_raw_count),
            float(p.confirm_th),
            float(p.ds_uncertainty_ceiling), float(p.r_tp), float(p.r_fp), float(p.fp_hold),
            int(p.enable_fp_unlatch), float(p.safety_failure_penalty), float(p.w_aoi),
            float(p.w_loc_team), float(p.w_blocked_motion), float(p.lambda_b),
            float(p.continuous_aoi_penalty_w), float(p.t_ref_s), int(p.horizon),
            int(p.shared_coverage_reward), {"ds_ignorance": 0, "deng_entropy": 1, "first_visit": 2}[p.exploration_reward_mode],
        ])
        self._observe()
        return w["obs"], w["mask"], w["reward"], w["terminated"], w["truncated"], w["done"]
