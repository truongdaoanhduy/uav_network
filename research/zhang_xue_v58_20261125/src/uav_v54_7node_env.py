"""v5.4 Complete 7-Node Packet-Level RL Environment (6 Actors + 1 Passive GCS).

Integrates:
  - Real SevenRadioCsmaCaNetwork (CSMA/CA, half-duplex, SINR, 3D NLoS, 2 blackouts, Q=64, E2E ACK)
  - Numerically stable canonical DS (FP32/FP64 int-diff log-space)
  - Per-UAV Zeng 2019 physical battery + acceleration clamping (a_max = 5 m/s^2)
  - Regime (c) termination: battery depletion or 15/15 GCS confirm -> terminated=True; H=1500 -> truncated=True
  - AoI metrics: confirm latency (microsecond timestamps), delivered-cell AoI and censored map-wide age
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from uav_v54_7radio_csmaca_sim import (
    ACTOR_N,
    CANONICAL_BLACKOUT_CENTERS_2,
    CANONICAL_BUILDINGS_5,
    CELL_SIZE_M,
    GCS_ID,
    GCS_POS_M,
    GRID_W,
    NUM_CELLS,
    RADIO_N,
    SevenRadioCsmaCaNetwork,
    is_in_blackout,
    segment_hits_cylinder_3d_fp32,
)
from uav_v54_ds_and_battery import (
    ZengBatteryConfig,
    clamp_velocity_with_accel,
    counts_to_canonical_ds_stable,
    fuse_provenance_counts,
    kinematic_safety_shield,
    trajectory_safety_violations,
    sensor_likelihood_to_bba,
    uav_step_energy_j,
)

DISC_DIM: int = 7  # 0=NO_SEND, 1..5=5 peer UAVs (bijective), 6=GCS_ID


def action_to_dest_radio_id(actor_id: int, action_idx: int) -> int:
    if not 0 <= action_idx < DISC_DIM or not 0 <= actor_id < ACTOR_N:
        raise ValueError(f"Invalid comm action {action_idx} for actor {actor_id}")
    if action_idx == 0:
        return -1
    if action_idx == 6:
        return GCS_ID
    peer = action_idx - 1
    return peer + 1 if peer >= actor_id else peer


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

    def __post_init__(self):
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


class ZhangXue7NodePacketEnv:
    """End-to-end 6-UAV + 1-GCS packet-level CSMA/CA environment."""

    def __init__(self, cfg: EnvConfigV54 | None = None, seed: int = 20261125) -> None:
        self.cfg = cfg or EnvConfigV54()
        self.obs_dim = 14 + 5 * self.cfg.obs_map_side**2
        self.state_dim = ACTOR_N * self.obs_dim
        if self.cfg.evidence_calibration == 'sensor_likelihood_ratio':
            self.evidence_mf, self.evidence_me = sensor_likelihood_to_bba(
                self.cfg.pd, self.cfg.pf, max_mass=self.cfg.bba_reliability_cap
            )
        else:
            self.evidence_mf, self.evidence_me = self.cfg.ds_mf, self.cfg.ds_me
        self.batt_cfg = ZengBatteryConfig(e_max_j=self.cfg.e_max_j)
        self.rng = np.random.default_rng(seed)
        self.net = SevenRadioCsmaCaNetwork(tx_power_w=self.cfg.tx_power_w, seed=seed)

        self.positions = np.zeros((RADIO_N, 3), dtype=np.float64)
        self.velocities = np.zeros((ACTOR_N, 2), dtype=np.float64)
        self.battery_j = np.full(ACTOR_N, self.batt_cfg.e_max_j, dtype=np.float64)
        self.true_targets = np.zeros(NUM_CELLS, dtype=bool)
        self.team_u_prev = np.ones(NUM_CELLS, dtype=np.float64)
        self.gcs_tp_latch = np.zeros(NUM_CELLS, dtype=np.int8)
        self.gcs_fp_latch = np.zeros(NUM_CELLS, dtype=np.int8)
        self.t = 0
        self.episode_id = 0
        self.epoch_us = 0

        # AoI accumulators (all 3 definitions)
        self.confirm_latencies_s: List[float] = []
        self.delivered_age_sum_s: float = 0.0
        self.delivered_age_samples: int = 0
        self.mapwide_age_sum_s: float = 0.0
        self.mapwide_age_samples: int = 0

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.net.rng = np.random.default_rng(seed + 101)
        self.episode_id += 1
        self.t = 0
        self.positions.fill(0.0)
        for u in range(ACTOR_N):
            self.positions[u] = [90.0 + 65.0 * u, 110.0 + 30.0 * (u % 2), self.cfg.uav_alt_m]
        self.positions[GCS_ID] = np.array(GCS_POS_M, dtype=np.float64)
        self.velocities.fill(0.0)
        self.battery_j.fill(self.batt_cfg.e_max_j)

        self.true_targets.fill(False)
        # Bounded placement: reject impossible configurations instead of hanging
        # forever in a rejection loop (num_targets can be <=400 yet infeasible).
        gcs_x, gcs_y = self.positions[GCS_ID, :2]
        legal_cells = []
        for cid in range(NUM_CELLS):
            cx = (cid % GRID_W + 0.5) * CELL_SIZE_M
            cy = (cid // GRID_W + 0.5) * CELL_SIZE_M
            if any(math.hypot(cx - bx, cy - by) <= br for bx, by, br, _ in CANONICAL_BUILDINGS_5):
                continue
            if math.hypot(cx - gcs_x, cy - gcs_y) < self.cfg.gcs_target_exclusion_m:
                continue
            legal_cells.append(cid)
        if len(legal_cells) < self.cfg.num_targets:
            raise ValueError(
                f"Not enough legal target cells: {len(legal_cells)} < {self.cfg.num_targets}"
            )
        if self.cfg.target_min_separation_m == 0.0:
            chosen = self.rng.choice(legal_cells, size=self.cfg.num_targets, replace=False)
            self.true_targets[chosen] = True
        else:
            selected_xy = []
            chosen = []
            for cid in self.rng.permutation(legal_cells):
                xy = np.array([((cid % GRID_W) + 0.5) * CELL_SIZE_M,
                               ((cid // GRID_W) + 0.5) * CELL_SIZE_M])
                if all(float(np.linalg.norm(xy - other)) >= self.cfg.target_min_separation_m
                       for other in selected_xy):
                    selected_xy.append(xy)
                    chosen.append(int(cid))
                    if len(chosen) == self.cfg.num_targets:
                        break
            if len(chosen) < self.cfg.num_targets:
                raise ValueError(
                    f"Only {len(chosen)} targets can be placed under min separation "
                    f"{self.cfg.target_min_separation_m} m with this randomized placement; "
                    "reduce count/separation or use a larger map"
                )
            self.true_targets[chosen] = True

        self.team_u_prev.fill(1.0)
        self.gcs_tp_latch.fill(0)
        self.gcs_fp_latch.fill(0)
        self.confirm_latencies_s.clear()
        self.delivered_age_sum_s = 0.0
        self.delivered_age_samples = 0
        self.mapwide_age_sum_s = 0.0
        self.mapwide_age_samples = 0

        self.net.reset(self.positions, episode_id=self.episode_id)
        # Warmup 0.6 s of HELLO/DV discovery before step 0 so initial 1-hop routes exist
        if not self.net.run_until_us(600_000):
            raise RuntimeError("Network HELLO warmup exceeded event budget")
        # Keep a strictly monotonic network clock; epoch is used only for mission AoI.
        self.epoch_us = self.net.sim_time_us
        self.net.mission_epoch_us = self.epoch_us
        self.net.step_tx_airtime_us.fill(0)
        self.net.step_new_peer_cells = 0
        self.net.step_peer_info_gain = 0.0
        for u in range(ACTOR_N):
            self.net.set_radio_alive(u, bool(self.battery_j[u] > 0.0))
        return self._build_obs_state_mask()

    def _fused_cell(self, receiver: int, cid: int) -> Tuple[float, float, float]:
        """Receiver-private evidence only. Never read the GCS map for a UAV actor."""
        net=self.net
        return fuse_provenance_counts(
            net.ledger_pos[receiver,:,cid],net.ledger_neg[receiver,:,cid],
            net.ledger_gen_ts_s[receiver,:,cid],net.ledger_rx_age_s[receiver,:,cid],
            mf=self.evidence_mf,me=self.evidence_me,
            eta=self.cfg.ds_aoi_eta if self.cfg.ds_fusion_mode=='sum' else 0.0,
            mode=self.cfg.ds_fusion_mode,
        )

    def _local_belief_features(self, u: int) -> np.ndarray:
        """5x10x10 compressed map (20x20 source grid, 2x2 pooling).

        Channels: mass target, mass empty, mass unknown, normalized source AoI,
        own UAV location. All belief evidence is present in receiver u's ledger.
        Unknown cells have mass (0,0,1), AoI 1; no ground truth enters actor.
        """
        src_p=self.net.ledger_pos[u]
        src_n=self.net.ledger_neg[u]
        valid=(src_p.astype(np.int32)+src_n.astype(np.int32))>0
        any_seen=np.any(valid,axis=0)
        masses=np.zeros((3,NUM_CELLS),np.float32)
        masses[2].fill(1.0)
        for cid in np.flatnonzero(any_seen):
            masses[:,cid] = self._fused_cell(u,int(cid))
        latest=np.where(valid,self.net.ledger_gen_ts_s[u],-1).max(axis=0)
        age=np.ones(NUM_CELLS,dtype=np.float32)
        age[any_seen]=np.minimum(1.0,np.maximum(0.0,
                              (self.t-latest[any_seen])/self.cfg.t_ref_s))
        loc=np.zeros(NUM_CELLS,np.float32)
        x=int(np.clip(self.positions[u,0]//CELL_SIZE_M,0,GRID_W-1))
        y=int(np.clip(self.positions[u,1]//CELL_SIZE_M,0,GRID_W-1))
        loc[y*GRID_W+x]=1.0
        channels=np.concatenate((masses,age[None,:],loc[None,:]),axis=0)
        side=self.cfg.obs_map_side
        grid=channels.reshape(5,side,2,side,2)
        if self.cfg.obs_pool_mode == 'semantic':
            pooled=np.empty((5,side,side),dtype=np.float32)
            pooled[0]=grid[0].max(axis=(1,3))   # Target mass: preserve peak (no 4x dilution)
            pooled[1]=grid[1].min(axis=(1,3))   # Empty mass: conservative min
            pooled[2]=grid[2].mean(axis=(1,3))  # Unknown mass: mean unexplored density
            pooled[3]=grid[3].max(axis=(1,3))   # AoI: prioritize stalest cell
            pooled[4]=grid[4].max(axis=(1,3))   # Own location: exact one-hot
        else:
            pooled=grid.mean(axis=(2,4))
            # The pooled location is upweighted to preserve a one-hot 10x10 actor cue.
            pooled[4]*=4.0
        return pooled.astype(np.float32).reshape(-1)

    def _build_obs_state_mask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Receiver-private 14 telemetry + 5 spatial maps. No target truth or global stats."""
        obs = np.zeros((ACTOR_N, self.obs_dim), dtype=np.float32)
        mask = np.zeros((ACTOR_N, DISC_DIM), dtype=np.uint8)
        mask[:, 0] = 1
        for u in range(ACTOR_N):
            own_radio = not is_in_blackout(self.positions[u], CANONICAL_BLACKOUT_CENTERS_2)
            own_alive = self.battery_j[u] > 0
            soc = max(0.0, self.battery_j[u] / self.batt_cfg.e_max_j)
            gcs_dx = (self.positions[GCS_ID, 0] - self.positions[u, 0]) / self.cfg.map_size_m
            gcs_dy = (self.positions[GCS_ID, 1] - self.positions[u, 1]) / self.cfg.map_size_m
            gcs_has_route = 1.0 if GCS_ID in self.net.routes[u] else 0.0
            dirty_gcs = self.net.count_all_dirty_entries_non_oracle(u, GCS_ID) / 400.0
            obs[u,:14] = [
                self.positions[u, 0] / self.cfg.map_size_m,
                self.positions[u, 1] / self.cfg.map_size_m,
                self.velocities[u, 0] / self.cfg.v_max_mps,
                self.velocities[u, 1] / self.cfg.v_max_mps,
                soc, float(own_radio and own_alive),
                len(self.net.radios[u].queue) / float(self.net.queue_cap),
                gcs_dx, gcs_dy, math.hypot(gcs_dx, gcs_dy),
                gcs_has_route, dirty_gcs,
                float(np.count_nonzero((self.net.ack_wm_pos[u] > 0) |
                       (self.net.ack_wm_neg[u] > 0))) / float(RADIO_N * NUM_CELLS),
                float(len(self.net.routes[u])) / float(RADIO_N),
            ]
            obs[u,14:] = self._local_belief_features(u)
            for a_idx in range(1, DISC_DIM):
                dst = action_to_dest_radio_id(u, a_idx)
                if own_radio and own_alive and len(self.net.get_dirty_entries_non_oracle(u, dst)) > 0:
                    mask[u, a_idx] = 1
        state=obs.reshape(-1).copy()
        return obs,state,mask

    def step(
        self, act_vel_cmd: np.ndarray, act_comm_disc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, bool, Dict[str, float]]:
        c = self.cfg
        self.net.step_tx_airtime_us.fill(0)
        self.net.step_new_peer_cells = 0
        self.net.step_peer_info_gain = 0.0
        for u in range(ACTOR_N):
            self.net.set_radio_alive(u, bool(self.battery_j[u] > 0.0))

        act_vel_cmd = np.asarray(act_vel_cmd, dtype=np.float64)
        act_comm_disc = np.asarray(act_comm_disc)
        if act_vel_cmd.shape != (ACTOR_N, 2) or not np.isfinite(act_vel_cmd).all():
            raise ValueError("act_vel_cmd must be finite shape (6, 2)")
        if act_comm_disc.shape != (ACTOR_N,):
            raise ValueError("act_comm_disc must have shape (6,)")
        if not np.issubdtype(act_comm_disc.dtype, np.integer):
            raise ValueError("act_comm_disc must contain integers")
        if np.any((act_comm_disc < 0) | (act_comm_disc >= DISC_DIM)):
            raise ValueError("act_comm_disc must contain actions in [0, 6]")

        # 1. Enqueue prior-step dirty evidence per actor comm action (strictly non-oracle)
        for u in range(ACTOR_N):
            dst = action_to_dest_radio_id(u, int(act_comm_disc[u]))
            if dst >= 0 and self.battery_j[u]>0:
                entries = self.net.get_dirty_entries_non_oracle(u, dst)
                if entries:
                    self.net.enqueue_app_data(u, dst, entries)

        # 2. Advance REAL 7-radio CSMA/CA discrete-event network for dt_s (1,000,000 us)
        target_us = self.net.sim_time_us + int(round(c.dt_s * 1_000_000))
        if not self.net.run_until_us(target_us):
            raise RuntimeError("CSMA/CA event cap hit: refusing to advance RL state with stale network time")

        # 3. Compute GCS TP/FP latch & AoI rewards from ACTUAL packets received at GCS
        r_gcs_tp = r_gcs_aoi = r_gcs_fp = 0.0
        t_step_s = self.t + 1
        for cid in range(NUM_CELLS):
            if not np.any(self.net.ledger_pos[GCS_ID,:,cid]) and not np.any(self.net.ledger_neg[GCS_ID,:,cid]):
                continue
            m_t, _, m_u = self._fused_cell(GCS_ID, cid)
            if m_t >= c.confirm_th and m_u <= c.ds_uncertainty_ceiling:
                if bool(self.true_targets[cid]) and self.gcs_tp_latch[cid] == 0:
                    self.gcs_tp_latch[cid] = 1
                    rx_t_s = (self.net.gcs_last_rx_us[cid] - self.epoch_us) / 1e6
                    lat_s = max(0.0, rx_t_s - float(self.net.gcs_latest_u_s[cid]))
                    self.confirm_latencies_s.append(lat_s)
                    r_gcs_tp += c.r_tp
                    if c.aoi_reward_mode == 'exponential':
                        r_gcs_aoi += c.w_aoi * math.exp(-lat_s / c.t_ref_s)
                    else:
                        r_gcs_aoi += c.w_aoi * max(0.0, 1.0 - lat_s / c.t_ref_s)
                elif (not bool(self.true_targets[cid])) and self.gcs_fp_latch[cid] == 0:
                    self.gcs_fp_latch[cid] = 1
                    r_gcs_fp -= c.r_fp
            elif c.enable_fp_unlatch and (not bool(self.true_targets[cid])) and self.gcs_fp_latch[cid] == 1:
                self.gcs_fp_latch[cid] = 0

        # Update all 3 AoI accumulators at end of step
        for cid in range(NUM_CELLS):
            u_s = int(self.net.gcs_latest_u_s[cid])
            if u_s >= 0:
                age = max(0.0, float(t_step_s - u_s))
                self.delivered_age_sum_s += age
                self.delivered_age_samples += 1
                self.mapwide_age_sum_s += age
            else:
                self.mapwide_age_sum_s += float(t_step_s)
            self.mapwide_age_samples += 1

        # 4. Motion update: v5.7 Kinematic Safety Shield (strict a_max) or legacy hard_veto
        old_xy_all = self.positions[:ACTOR_N, :2].copy()
        old_vel_all = self.velocities.copy()
        alive_mask = self.battery_j > 0.0
        shield_intervention = np.zeros(ACTOR_N, dtype=np.float64)
        safety_violations = {"boundary": [], "building": [], "pairwise": []}
        safety_violation = False

        if c.safety_mode == 'kinematic_shield':
            v_safe_all, shield_intervention = kinematic_safety_shield(
                old_xy_all, old_vel_all, act_vel_cmd, alive_mask,
                CANONICAL_BUILDINGS_5, c.map_size_m, c.dt_s,
                c.v_max_mps, c.a_max_mps2, c.uav_safe_distance_m,
            )
            rejected_motion_count = int(np.count_nonzero((shield_intervention > 1e-6) & alive_mask))
            safety_violations = trajectory_safety_violations(
                old_xy_all, v_safe_all, alive_mask, CANONICAL_BUILDINGS_5,
                map_size_m=c.map_size_m, dt_s=c.dt_s,
                uav_safe_distance_m=c.uav_safe_distance_m,
            )
            safety_violation = any(safety_violations.values())
            for u in range(ACTOR_N):
                if alive_mask[u]:
                    self.velocities[u] = v_safe_all[u]
                    self.positions[u, :2] = old_xy_all[u] + v_safe_all[u] * c.dt_s
                else:
                    self.velocities[u] = np.zeros(2, dtype=np.float64)
                actual_speed = float(np.linalg.norm(self.velocities[u]))
                own_airtime_s = float(self.net.step_tx_airtime_us[u]) / 1e6
                self.battery_j[u] = max(0.0, self.battery_j[u] - uav_step_energy_j(
                    actual_speed, own_airtime_s, c.dt_s, self.batt_cfg,
                    v_prev_xy=old_vel_all[u] if c.enable_accel_energy else None,
                    v_new_xy=self.velocities[u] if c.enable_accel_energy else None,
                ))
                if self.battery_j[u] <= 0.0:
                    self.net.set_radio_alive(u, False)
        else:
            disp = np.zeros((ACTOR_N, 2), dtype=np.float64)
            accepted = np.ones(ACTOR_N, dtype=bool)
            for u in range(ACTOR_N):
                if not alive_mask[u]:
                    accepted[u] = False
                    disp[u] = 0.0
                    continue
                v_new = clamp_velocity_with_accel(
                    self.velocities[u], act_vel_cmd[u], c.dt_s, c.v_max_mps, c.a_max_mps2
                )
                candidate = old_xy_all[u] + v_new * c.dt_s
                geometry_blocked = (
                    candidate[0] < 10.0 or candidate[0] > c.map_size_m - 10.0
                    or candidate[1] < 10.0 or candidate[1] > c.map_size_m - 10.0
                )
                for bx, by, br, _ in CANONICAL_BUILDINGS_5:
                    if geometry_blocked:
                        break
                    delta = candidate - old_xy_all[u]
                    t_closest = float(np.clip(np.dot(np.array([bx, by]) - old_xy_all[u], delta)
                                              / max(1e-12, float(np.dot(delta, delta))), 0.0, 1.0))
                    closest = old_xy_all[u] + t_closest * delta
                    if float(np.linalg.norm(closest - np.array([bx, by]))) <= br + 2.0:
                        geometry_blocked = True
                accepted[u] = not geometry_blocked
                disp[u] = v_new * c.dt_s

            for _ in range(ACTOR_N):
                changed = False
                actual_disp = disp * accepted[:, None]
                for i in range(ACTOR_N):
                    for j in range(i + 1, ACTOR_N):
                        rel0 = old_xy_all[i] - old_xy_all[j]
                        reld = actual_disp[i] - actual_disp[j]
                        tau = float(np.clip(-np.dot(rel0, reld) /
                                            max(1e-12, float(np.dot(reld, reld))), 0.0, 1.0))
                        min_sep = float(np.linalg.norm(rel0 + tau * reld))
                        if min_sep < c.uav_safe_distance_m - 1e-6:
                            if accepted[i] or accepted[j]:
                                changed = True
                                accepted[i] = False
                                accepted[j] = False
                if not changed:
                    break

            rejected_motion_count = int(np.count_nonzero((~accepted) & alive_mask))
            for u in range(ACTOR_N):
                if accepted[u]:
                    self.positions[u, :2] = old_xy_all[u] + disp[u]
                    self.velocities[u] = disp[u] / c.dt_s
                    actual_speed = float(np.linalg.norm(self.velocities[u]))
                else:
                    self.velocities[u] = np.zeros(2, dtype=np.float64)
                    actual_speed = 0.0
                own_airtime_s = float(self.net.step_tx_airtime_us[u]) / 1e6
                self.battery_j[u] = max(0.0, self.battery_j[u] - uav_step_energy_j(
                    actual_speed, own_airtime_s, c.dt_s, self.batt_cfg,
                    v_prev_xy=old_vel_all[u] if c.enable_accel_energy else None,
                    v_new_xy=self.velocities[u] if c.enable_accel_energy else None,
                ))
                if self.battery_j[u] <= 0.0:
                    self.net.set_radio_alive(u, False)

        self.net.update_positions(self.positions)

        # 5. Local sensing & team-union exploration reward
        for u in range(ACTOR_N):
            if self.battery_j[u] <= 0.0:
                continue
            ux, uy = self.positions[u, 0], self.positions[u, 1]
            cmin_x = max(0, int((ux - c.fov_radius_m) // CELL_SIZE_M))
            cmax_x = min(GRID_W - 1, int((ux + c.fov_radius_m) // CELL_SIZE_M))
            cmin_y = max(0, int((uy - c.fov_radius_m) // CELL_SIZE_M))
            cmax_y = min(GRID_W - 1, int((uy + c.fov_radius_m) // CELL_SIZE_M))
            for cy in range(cmin_y, cmax_y + 1):
                for cx in range(cmin_x, cmax_x + 1):
                    cell_x = (cx + 0.5) * CELL_SIZE_M
                    cell_y = (cy + 0.5) * CELL_SIZE_M
                    if math.hypot(cell_x - ux, cell_y - uy) <= c.fov_radius_m:
                        if c.enable_3d_sensing_occlusion:
                            # C2 fix: skip cells inside building footprint or occluded by 70-80m buildings
                            occluded = False
                            for bx, by, br, bh in CANONICAL_BUILDINGS_5:
                                if math.hypot(cell_x - bx, cell_y - by) <= br:
                                    occluded = True
                                    break
                                if segment_hits_cylinder_3d_fp32(
                                    self.positions[u], np.array([cell_x, cell_y, 0.0]), bx, by, br, bh
                                ):
                                    occluded = True
                                    break
                            if occluded:
                                continue
                        cid = cy * GRID_W + cx
                        det = bool(self.rng.random() < (c.pd if self.true_targets[cid] else c.pf))
                        self.net.record_local_observation(u, cid, det, t_step_s)

        r_loc_team = 0.0
        for cid in range(NUM_CELLS):
            if c.ds_fusion_mode == 'zhang_max':
                # B10 fix: respect zhang_max fusion mode in team exploration reward
                n_p = int(max(self.net.ledger_pos[u, u, cid] for u in range(ACTOR_N)))
                n_n = int(max(self.net.ledger_neg[u, u, cid] for u in range(ACTOR_N)))
            else:
                n_p = int(sum(self.net.ledger_pos[u, u, cid] for u in range(ACTOR_N)))
                n_n = int(sum(self.net.ledger_neg[u, u, cid] for u in range(ACTOR_N)))
            if n_p > 0 or n_n > 0:
                _, _, u_now = counts_to_canonical_ds_stable(n_p, n_n, self.evidence_mf, self.evidence_me)
                # Signed potential difference avoids rewarding noise-induced oscillations.
                r_loc_team += c.w_loc_team * (self.team_u_prev[cid] - u_now)
                self.team_u_prev[cid] = u_now

        # Active airtime & FP hold penalties
        uav_airtime_total_s = float(self.net.step_tx_airtime_us[:ACTOR_N].sum()) / 1e6
        r_airtime = -c.lambda_b * uav_airtime_total_s
        # Reward *new, actual, receiver-integrated* peer information, not transmissions.
        if c.peer_reward_mode == 'info_gain':
            r_peer_information = c.w_p2p * min(float(c.reward_peer_cap_per_step), self.net.step_peer_info_gain)
        else:
            r_peer_information = c.w_p2p * min(c.reward_peer_cap_per_step, self.net.step_new_peer_cells)
        r_fp_hold = -c.fp_hold * float(self.gcs_fp_latch.sum())

        if c.safety_mode == 'kinematic_shield':
            r_blocked_motion = -c.w_blocked_motion * float(shield_intervention.sum())
        else:
            r_blocked_motion = -c.w_blocked_motion * rejected_motion_count
        r_safety_failure = -c.safety_failure_penalty if safety_violation else 0.0
        reward = (r_loc_team + r_gcs_tp + r_gcs_aoi + r_gcs_fp + r_fp_hold
                  + r_airtime + r_blocked_motion + r_peer_information + r_safety_failure)
        self.t += 1

        terminated = bool(
            np.all(self.gcs_tp_latch[self.true_targets] == 1) or np.any(self.battery_j <= 0.0)
            or (safety_violation and c.safety_failure_terminate)
        )
        truncated = bool((not terminated) and (self.t >= c.safety_horizon_s))
        obs, state, mask = self._build_obs_state_mask()

        info = {
            "step": self.t,
            "terminated": terminated,
            "truncated": truncated,
            "gcs_tp_confirmed": int(self.gcs_tp_latch[self.true_targets].sum()),
            "gcs_fp_latched": int(self.gcs_fp_latch.sum()),
            "r_loc_team": round(r_loc_team, 4),
            "r_gcs_tp": round(r_gcs_tp, 4),
            "r_gcs_aoi": round(r_gcs_aoi, 4),
            "r_airtime": round(r_airtime, 5),
            "r_peer_information": round(r_peer_information,5),
            "peer_new_cells_rx": int(self.net.step_new_peer_cells),
            "ds_fusion_mode":c.ds_fusion_mode,
            "ds_aoi_eta":c.ds_aoi_eta,
            "evidence_calibration":c.evidence_calibration,
            "r_blocked_motion": round(r_blocked_motion, 4),
            "r_safety_failure": float(r_safety_failure),
            "safety_violation": bool(safety_violation),
            "safety_violations": safety_violations,
            "rejected_motion_count": rejected_motion_count,
            "mean_first_tp_confirm_latency_s": (
                round(float(np.mean(self.confirm_latencies_s)), 4) if self.confirm_latencies_s else -1.0
            ),
            "mean_gcs_aoi_delivered_cells_s": (
                round(self.delivered_age_sum_s / self.delivered_age_samples, 3)
                if self.delivered_age_samples > 0
                else -1.0
            ),
            "delivered_cell_coverage_pct": round(float((self.net.gcs_latest_u_s >= 0).mean() * 100.0), 2),
            # For never-delivered cells we use elapsed mission time as an explicit
            # initialization convention; this is NOT ground-truth AoI.
            "mean_gcs_aoi_mapwide_censored_s": round(self.mapwide_age_sum_s / max(1, self.mapwide_age_samples), 3),
            "min_battery_pct": round(float(self.battery_j.min() / self.batt_cfg.e_max_j * 100.0), 2),
            "data_delivered_gcs": self.net.stats["data_delivered_gcs"],
            "mac_ack_ok": self.net.stats["mac_ack_ok"],
            "e2e_ack_rx": self.net.stats["e2e_ack_rx"],
        }
        return obs, state, mask, float(reward), terminated, truncated, info
