"""V79 GPU-vectorized Warp UAV Mission Runtime (YAML-synchronized, masked selective reset).

Resolves V78 P0/P1 defects while keeping `mission_v78.py` untouched for V78 test compatibility:
- `GPUWorldV79Params.from_yaml()` eliminates YAML <-> runtime mismatch (`12.0 dB` vs `20.0 dB`).
- Selective `reset(done)` / `auto_reset_gpu()` uses `step_v67_masked` so non-reset worlds never
  experience +0.6s radio clock drift.
- Separates `terminated` and `truncated` so learners never zero Bellman bootstrap at horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import numpy as np

from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
from zhang_xue.gpu.state_v67 import HAS_WARP, wp


@dataclass
class GPUWorldV79Params:
    envs: int = 4
    device: str = "cpu"
    seed: int = 44
    queue_capacity: int = 64
    obstacle_model: str = "multi_obstacle"
    per_building_loss_db: float = 12.0
    noise_power_w: float = 4.38957709293155e-13
    tx_power_w: float = 0.1
    cca_threshold_dbm: float = -82.0
    snr_threshold_db: float = 6.0
    horizon: int = 900
    max_events: int = 4096
    uav_alt_m: float = 50.0
    v_max_mps: float = 15.0
    a_max_mps2: float = 5.0
    uav_safe_distance_m: float = 50.0
    e_max_j: float = 180000.0
    pd: float = 0.85
    pf: float = 0.03
    ds_mf: float = 0.80
    ds_me: float = 0.80
    ds_aoi_eta: float = 0.02
    static_raw_count: bool = False
    obs_schema_v2: bool = False
    bayesian_count_mode: bool = False
    bayes_prior_occupancy: float = 15.0 / 392.0
    confirm_th: float = 0.90
    ds_uncertainty_ceiling: float = 0.20
    r_tp: float = 25.0
    r_fp: float = 12.5
    fp_hold: float = 0.10
    enable_fp_unlatch: bool = True
    safety_failure_penalty: float = 100.0
    w_aoi: float = 6.0
    w_loc_team: float = 0.50
    w_blocked_motion: float = 0.20
    lambda_b: float = 0.0
    continuous_aoi_penalty_w: float = 0.0
    t_ref_s: float = 60.0
    enable_peer: bool = True
    enable_data_mule: bool = True
    decentralized_peer_select: bool = True

    def validate(self) -> None:
        if not (1 <= self.envs <= 8192 and 1 <= self.horizon <= 32767):
            raise ValueError("envs must be in [1, 8192] and horizon in [1, 32767]")
        if self.obstacle_model not in ("binary_nlos", "multi_obstacle", "rf_opaque"):
            raise ValueError(f"Unsupported obstacle_model: {self.obstacle_model!r}")
        if not (1 <= self.queue_capacity <= 127):
            raise ValueError("queue_capacity must be in [1, 127]")
        if self.per_building_loss_db < 0.0 or not math.isfinite(self.per_building_loss_db):
            raise ValueError("per_building_loss_db must be finite non-negative")
        for name in ("noise_power_w", "tx_power_w", "uav_alt_m", "v_max_mps",
                     "a_max_mps2", "uav_safe_distance_m", "e_max_j", "t_ref_s"):
            x = getattr(self, name)
            if not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        for name in ("pd", "pf", "ds_mf", "ds_me", "confirm_th", "ds_uncertainty_ceiling"):
            x = getattr(self, name)
            if not isinstance(x, (float, int)) or not math.isfinite(x) or not 0.0 < x < 1.0:
                raise ValueError(f"{name} must be strictly in (0, 1)")
        if self.pf >= self.pd:
            raise ValueError("pf must be smaller than pd")
        if self.max_events < 1 or self.max_events > 1_000_000:
            raise ValueError("max_events must lie within [1, 1000000]")
        if not math.isfinite(self.ds_aoi_eta) or self.ds_aoi_eta < 0.0:
            raise ValueError("ds_aoi_eta must be finite and nonnegative")
        if type(self.bayesian_count_mode) is not bool or not math.isfinite(self.bayes_prior_occupancy) or not 0.0 < self.bayes_prior_occupancy < 1.0:
            raise ValueError("invalid Bayesian posterior configuration")
        if self.bayesian_count_mode and (not self.static_raw_count or self.ds_aoi_eta != 0.0):
            raise ValueError("Bayes mode requires persistent nondecayed raw counts")
        if type(self.obs_schema_v2) is not bool:
            raise ValueError("obs_schema_v2 must be boolean")
        if type(self.static_raw_count) is not bool:
            raise ValueError("static_raw_count must be boolean")
        if self.static_raw_count and self.ds_aoi_eta != 0.0:
            raise ValueError("raw uint16 counts cannot be decayed; AoI uses timestamps")

    @classmethod
    def from_scenario(cls, scenario, *, envs: int = 4, device: str = "cpu", seed: int = 44, **overrides) -> "GPUWorldV79Params":
        if scenario.num_uavs != 6 or scenario.grid_w != 20 or abs(scenario.map_size_m - 1000.0) > 1e-6:
            raise ValueError("WarpParallelMissionV79 currently supports 6 UAVs on 20x20 (1000m) map")
        if scenario.comm_action_mode != "compact_3":
            raise ValueError("WarpParallelMissionV79 requires comm_action_mode='compact_3'")
        if scenario.enable_tx_power_control:
            raise ValueError("WarpParallelMissionV79 does not support per-UAV learned TX power")
        if not scenario.aoi_ewma_end_to_end:
            raise ValueError("WarpParallelMissionV79 requires aoi_ewma_end_to_end=True")
        # Fail closed: the GPU kernels are not general-purpose interpreters of EnvConfigV54.
        # Quietly accepting an unsupported YAML knob silently changes the scientific experiment.
        required_exact = {
            "num_targets": 15, "dt_s": 1.0, "fov_radius_m": 50.0,
            "obs_map_side": 10, "obs_pool_mode": "semantic",
            "ds_fusion_mode": "sum", "evidence_calibration": "paper_fixed",
            "safety_mode": "kinematic_shield", "enable_3d_sensing_occlusion": True,
            "enable_accel_energy": False, "network_substeps": 1,
            "coalesce_duplicate_queue": False, "reward_scale_mode": "raw",
            "aoi_reward_mode": "exponential", "battery_termination_mode": "all_depleted",
            "safety_failure_terminate": True, "w_p2p": 0.0,
            "gcs_target_exclusion_m": 0.0, "target_min_separation_m": 0.0,
            "mac_difs_us": 50, "mac_sifs_us": 10, "mac_slot_us": 20,
            "mac_cw_min": 31, "mac_max_retries": 5, "mac_ack_timeout_us": 180,
        }
        for field, expected in required_exact.items():
            actual = getattr(scenario, field)
            if actual != expected:
                raise ValueError(
                    f"Unsupported GPU YAML {field}={actual!r}; this Warp build only models "
                    f"{field}={expected!r}. Do not label a non-parity run as supported."
                )
        eta = float(scenario.ds_aoi_eta)
        kwargs = dict(
            envs=envs, device=device, seed=seed,
            queue_capacity=int(scenario.mac_queue_capacity),
            obstacle_model=str(scenario.rf_obstacle_model),
            per_building_loss_db=float(scenario.rf_per_building_loss_db),
            noise_power_w=float(scenario.mac_noise_power_w),
            tx_power_w=float(scenario.tx_power_w),
            cca_threshold_dbm=float(scenario.mac_cca_threshold_dbm),
            snr_threshold_db=float(scenario.mac_snr_threshold_db),
            horizon=int(scenario.safety_horizon_s),
            uav_alt_m=float(scenario.uav_alt_m),
            v_max_mps=float(scenario.v_max_mps),
            a_max_mps2=float(scenario.a_max_mps2),
            uav_safe_distance_m=float(scenario.uav_safe_distance_m),
            e_max_j=float(scenario.e_max_j),
            pd=float(scenario.pd), pf=float(scenario.pf),
            ds_mf=float(scenario.ds_mf), ds_me=float(scenario.ds_me),
            ds_aoi_eta=eta,
            static_raw_count=(scenario.static_evidence_mode == "raw_u16_count"),
            obs_schema_v2=bool(scenario.obs_schema_v2),
            bayesian_count_mode=(scenario.belief_inference_mode == "bayesian_logodds"),
            bayes_prior_occupancy=float(scenario.bayes_prior_occupancy),
            confirm_th=float(scenario.confirm_th),
            ds_uncertainty_ceiling=float(scenario.ds_uncertainty_ceiling),
            r_tp=float(scenario.r_tp), r_fp=float(scenario.r_fp),
            fp_hold=float(scenario.fp_hold),
            enable_fp_unlatch=bool(scenario.enable_fp_unlatch),
            safety_failure_penalty=float(scenario.safety_failure_penalty),
            w_aoi=float(scenario.w_aoi), w_loc_team=float(scenario.w_loc_team),
            w_blocked_motion=float(scenario.w_blocked_motion),
            lambda_b=float(scenario.lambda_b),
            continuous_aoi_penalty_w=float(scenario.continuous_aoi_penalty_w),
            t_ref_s=float(scenario.t_ref_s),
            enable_peer=bool(scenario.enable_peer_dissemination),
            enable_data_mule=bool(scenario.enable_data_mule_forwarding),
            decentralized_peer_select=bool(scenario.compact3_decentralized_peer_select),
        )
        kwargs.update(overrides)
        params = cls(**kwargs)
        params.validate()
        return params

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, envs: int = 4, device: str = "cpu", seed: int = 44,
        scenario_overrides: tuple[str, ...] = (), **overrides,
    ) -> "GPUWorldV79Params":
        scenario = load_scenario(path, overrides=scenario_overrides)
        return cls.from_scenario(scenario, envs=envs, device=device, seed=seed, **overrides)


class WarpParallelMissionV79:
    """V79 GPU-vectorized independent UAV worlds with masked selective reset & YAML parity."""

    def __init__(self, params: GPUWorldV79Params):
        if not HAS_WARP:
            raise RuntimeError("Requires NVIDIA Warp")
        params.validate()
        self.params = params
        self.device = params.device
        self.envs = params.envs
        self.cca_w = 1e-3 * math.pow(10.0, params.cca_threshold_dbm / 10.0)
        self.snr_lin = math.pow(10.0, params.snr_threshold_db / 10.0)
        self.pa_dc_w = 0.5 + params.tx_power_w / 0.20
        self.network = WarpCompressedRadioNetworkV67(
            params.envs, num_uavs=6, grid_w=20, device=params.device, seed=params.seed,
            q_cap=params.queue_capacity, enable_peer=params.enable_peer,
            enable_data_mule=params.enable_data_mule,
            decentralized_peer_select=params.decentralized_peer_select,
            aoi_ewma=True, aoi_eta=params.ds_aoi_eta,
            enable_peer_telemetry=params.obs_schema_v2,
        )
        specs = {
            "positions": ((params.envs, 7, 3), wp.float32),
            "velocities": ((params.envs, 6, 2), wp.float32),
            "candidates": ((params.envs, 6, 2), wp.float32),
            "v_safe": ((params.envs, 6, 2), wp.float32),
            "pos_xy": ((params.envs, 6, 2), wp.float32),
            "battery": ((params.envs, 6), wp.float32),
            "truth": ((params.envs, 400), wp.uint8),
            "team_u_prev": ((params.envs, 400), wp.float32),
            "gcs_tp": ((params.envs, 400), wp.uint8),
            "gcs_fp": ((params.envs, 400), wp.uint8),
            "gcs_fp_ever": ((params.envs, 400), wp.uint8),
            "episode_step": ((params.envs,), wp.int32),
            "episode_id": ((params.envs,), wp.int32),
            "terminated": ((params.envs,), wp.uint8),
            "truncated": ((params.envs,), wp.uint8),
            "done": ((params.envs,), wp.uint8),
            "termination_reason": ((params.envs,), wp.uint8),  # valid after step, before auto_reset_gpu
            "failed": ((params.envs,), wp.uint8),
            "safety_reason": ((params.envs,), wp.uint8),
            "unsafe_mask": ((params.envs, 6), wp.uint8),
            "reward": ((params.envs,), wp.float32),
            "reward_components": ((params.envs, 8), wp.float32),
            "intervention": ((params.envs,), wp.float32),
            "obs": ((params.envs, 6, 514 + (35 if params.obs_schema_v2 else 0)), wp.float32),
            "mask": ((params.envs, 6, 3), wp.uint8),
        }
        self.buffers = {k: wp.zeros(sh, dtype=dt, device=params.device) for k, (sh, dt) in specs.items()}
        buildings, blackouts, blackout_r = scale_canonical_geometry(1000.0, 20)
        self.buildings, self.blackouts, self.blackout_r = buildings, blackouts, blackout_r
        self.build_wp = wp.array(np.asarray(buildings, dtype=np.float32), dtype=wp.float32, device=params.device)
        self.blackout_wp = wp.array(np.asarray(blackouts, dtype=np.int32), dtype=wp.int32, device=params.device)
        legal = [
            cid for cid in range(400)
            if all((((cid % 20 + 0.5) * 50.0 - b[0]) ** 2 + ((cid // 20 + 0.5) * 50.0 - b[1]) ** 2) > b[2] ** 2
                   for b in buildings)
        ]
        self.legal = wp.array(np.asarray(legal, dtype=np.int32), dtype=wp.int32, device=params.device)
        self.obstacle_model_id = ("binary_nlos", "multi_obstacle", "rf_opaque").index(params.obstacle_model)
        self.logmf = math.log1p(-params.ds_mf)
        self.logme = math.log1p(-params.ds_me)
        self.bayes_log_prior = math.log(params.bayes_prior_occupancy / (1.0 - params.bayes_prior_occupancy))
        self.bayes_llr_pos = math.log(params.pd / params.pf)
        self.bayes_llr_neg = math.log((1.0 - params.pd) / (1.0 - params.pf))
        self.reset()

    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def _reset_with_device_mask(self, done_wp):
        from zhang_xue.gpu.kernels.mission_v79_reset_obs import (
            _finalize_reset_v79, _reset_world_v79, _clear_masked_diagnostics_v79, step_v67_masked,
        )
        from zhang_xue.gpu.kernels.reset_v67 import reset_cells_v67, reset_world_v67
        w = self.buffers
        n = self.network
        self._launch(reset_world_v67, self.envs, [n.state, n.dims, done_wp, int(self.params.seed)])
        self._launch(reset_cells_v67, (self.envs, 400), [n.state, n.dims, done_wp])
        self._launch(_reset_world_v79, self.envs, [
            w["positions"], w["velocities"], w["battery"], w["truth"], w["team_u_prev"],
            w["gcs_tp"], w["gcs_fp"], w["gcs_fp_ever"], w["episode_step"], w["episode_id"],
            done_wp, w["terminated"], w["truncated"], w["failed"], w["unsafe_mask"],
            w["reward"], w["intervention"], w["pos_xy"], self.legal, int(len(self.legal)),
            15, float(self.params.e_max_j), float(self.params.uav_alt_m), int(self.params.seed),
        ])
        self._launch(_clear_masked_diagnostics_v79, self.envs, [
            done_wp, w["termination_reason"], w["safety_reason"], w["reward_components"],
        ])
        self._update_phy()
        self._capture_hello_offers()
        # Masked HELLO warmup: ONLY reset worlds advance 600ms; running worlds stay frozen!
        self._launch(step_v67_masked, self.envs, [
            n.state, n.dims, done_wp, 600_000, int(self.params.max_events),
            float(self.params.tx_power_w), float(self.cca_w),
            float(self.params.noise_power_w), float(self.snr_lin),
        ])
        self._launch(_finalize_reset_v79, self.envs, [n.state, done_wp])
        self._observe()
        return w["obs"], w["mask"]

    def reset(self, done: Optional[np.ndarray] = None):
        a = np.ones(self.envs, dtype=np.uint8) if done is None else np.asarray(done, dtype=np.uint8)
        if a.shape != (self.envs,):
            raise ValueError("done mask shape mismatch")
        wp.copy(self.buffers["done"], wp.array(a, dtype=wp.uint8, device=self.device))
        return self._reset_with_device_mask(self.buffers["done"])

    def auto_reset_gpu(self):
        """Zero host-sync selective reset using the on-device `done` buffer."""
        return self._reset_with_device_mask(self.buffers["done"])

    def _capture_hello_offers(self):
        if not self.params.obs_schema_v2:
            return
        from zhang_xue.gpu.kernels.mission_v89_hello_peer import capture_hello_offer_v89
        self._launch(capture_hello_offer_v89, (self.envs, 7), [
            self.network.state, self.buffers["positions"], self.buffers["velocities"],
        ])

    def _update_phy(self):
        from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67
        n = self.network
        self._launch(compute_channel_gain_v67, (self.envs, 7), [
            self.buffers["positions"], n.data["radio_alive"], self.build_wp, self.blackout_wp,
            n.data["gain"], 7, len(self.buildings), len(self.blackouts), self.blackout_r,
            20, 50.0, self.obstacle_model_id, float(self.params.per_building_loss_db),
        ])

    def _observe(self):
        from zhang_xue.gpu.kernels.inject_v67 import compute_dirty_dest_v67
        from zhang_xue.gpu.kernels.mission_v79_reset_obs import _observe_core_v79, _observe_map_v79
        n = self.network
        w = self.buffers
        p = self.params
        self._launch(compute_dirty_dest_v67, (self.envs, 6, 7), [n.state, n.dims])
        self._launch(_observe_core_v79, (self.envs, 6), [
            n.state, n.dims, w["positions"], w["velocities"], w["battery"],
            self.blackout_wp, len(self.blackouts), self.blackout_r,
            w["obs"], w["mask"], float(p.e_max_j), float(p.queue_capacity), float(p.v_max_mps),
        ])
        self._launch(_observe_map_v79, (self.envs, 6, 100), [
            n.state, w["positions"], w["episode_step"], w["obs"],
            self.logmf, self.logme, float(p.ds_aoi_eta), float(p.t_ref_s),
            int(p.static_raw_count), int(p.bayesian_count_mode),
            self.bayes_log_prior, self.bayes_llr_pos, self.bayes_llr_neg,
        ])
        if p.obs_schema_v2:
            from zhang_xue.gpu.kernels.mission_v89_hello_peer import observe_hello_peers_v89
            self._launch(observe_hello_peers_v89, (self.envs, 6), [
                n.state, w["positions"], w["velocities"], w["obs"], float(p.v_max_mps),
            ])

    def step(self, continuous, discrete):
        from zhang_xue.gpu.kernels.inject_v67 import resolve_compact3_v67, inject_v67
        from zhang_xue.gpu.kernels.mission_v79_kernels import _move_v79, _reward_v79, _sense_visible_v79
        from zhang_xue.gpu.mission_v78 import _candidate_v78
        w = self.buffers
        n = self.network
        p = self.params
        if tuple(continuous.shape) != (self.envs, 6, 2) or tuple(discrete.shape) != (self.envs, 6):
            raise ValueError("act shape should be (E,6,2) and (E,6)")
        self._launch(resolve_compact3_v67, (self.envs, 6), [n.state, n.dims, discrete, w["pos_xy"], n._dest_buf])
        self._launch(inject_v67, (self.envs, 6), [n.state, n.dims, n._dest_buf])
        self._capture_hello_offers()
        n.step(duration_us=1_000_000, max_events=p.max_events, tx_power_w=p.tx_power_w,
               cca_threshold_w=self.cca_w, noise_w=p.noise_power_w, snr_threshold=self.snr_lin,
               check_host_guards=False)
        self._launch(_candidate_v78, (self.envs, 6), [continuous, w["velocities"], w["candidates"]])
        self._launch(_move_v79, self.envs, [
            w["positions"], w["velocities"], w["candidates"], w["v_safe"], w["pos_xy"],
            w["battery"], self.build_wp, len(self.buildings), n.state, w["failed"], w["safety_reason"],
            w["unsafe_mask"], w["intervention"], float(p.v_max_mps), float(p.a_max_mps2),
            float(p.uav_safe_distance_m), float(self.pa_dc_w),
        ])
        self._update_phy()
        self._launch(_sense_visible_v79, (self.envs, 6), [
            n.state, w["positions"], w["truth"], w["episode_step"], w["episode_id"],
            w["battery"], w["unsafe_mask"], w["failed"], self.build_wp, len(self.buildings),
            int(p.seed), float(p.pd), float(p.pf), float(p.ds_aoi_eta),
            int(p.static_raw_count),
        ])
        self._launch(_reward_v79, self.envs, [
            n.state, w["truth"], w["team_u_prev"], w["gcs_tp"], w["gcs_fp"], w["gcs_fp_ever"],
            w["battery"], w["episode_step"], w["failed"], w["intervention"], w["reward"],
            w["reward_components"], w["terminated"], w["truncated"], w["done"], w["termination_reason"], self.logmf, self.logme,
            float(p.ds_aoi_eta), int(p.static_raw_count), int(p.bayesian_count_mode),
            self.bayes_log_prior, self.bayes_llr_pos, self.bayes_llr_neg,
            float(p.confirm_th), float(p.ds_uncertainty_ceiling),
            float(p.r_tp), float(p.r_fp), float(p.fp_hold), int(p.enable_fp_unlatch),
            float(p.safety_failure_penalty), float(p.w_aoi), float(p.w_loc_team),
            float(p.w_blocked_motion), float(p.lambda_b), float(p.continuous_aoi_penalty_w),
            float(p.t_ref_s), int(p.horizon),
        ])
        self._observe()
        return w["obs"], w["mask"], w["reward"], w["terminated"], w["truncated"], w["done"]

    def guard(self):
        errors = self.network.data["overflow"].numpy()
        failed = self.buffers["failed"].numpy()
        if np.any(errors):
            raise RuntimeError(f"Warp packet event/gen overflow in {int(np.count_nonzero(errors))} worlds")
        if np.any(failed):
            raise RuntimeError(f"unsafe flight or sampling error in {int(np.count_nonzero(failed))} worlds")
        if not np.isfinite(self.buffers["obs"].numpy()).all():
            raise FloatingPointError("nonfinite observations")
        truth = self.buffers["truth"].numpy()
        sums = truth.sum(axis=1)
        return {
            "environments": self.envs,
            "episode_steps": int(self.buffers["episode_step"].numpy().max()),
            "active_truth_per_world_min": int(sums.min()),
            "active_truth_per_world_max": int(sums.max()),
            "distinct_target_layouts": int(np.unique(truth, axis=0).shape[0]),
        }
