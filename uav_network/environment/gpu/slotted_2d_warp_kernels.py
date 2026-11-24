"""Pure FP32 NVIDIA Warp Kernels completing the full GPU Environment Step (Items 3-12).

All kernels strictly use single-precision (wp.float32 / wp.int32 / wp.uint8), zero dynamic allocations,
zero host-device synchronization, and 1-thread-per-env static unrolling for 100% CUDA Graph capture
compatibility on Kaggle Tesla T4 (Turing SM 7.5).

Kernels:
  1. `motion_2d_safety_fp32`: 2D accel/vel integration, boundary clamp + penalty, cylindrical building
     projection + penalty, pairwise UAV-UAV < 15m safety check, node_pos update for PHY.
  2. `assemble_reward_and_done_fp32`: computes team reward, updates episode KPIs, sets `dones`,
     increments per-env step clock.
  3. `selective_auto_reset_fp32`: resets ONLY finished worlds (`dones[e] == 1`) in-place on GPU using
     pre-staged scene templates (100% CUDA Graph safe).
  4. `build_obs_state_mask_fp32`: constructs exact 96D NoOracle local `obs` (reading `local_ds` when
     `belief_sharing_mode == 1`), exact 256D Centralized Critic `global_state`, and exact 7D Adaptive
     `action_mask` directly in zero-copy GPU buffers (ZERO CPU round-trip!).
"""

from __future__ import annotations

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

from uav_network.communication.gpu.phy_sinr_fp32 import GCS_IDX, NUM_NODES, NUM_UAVS
from uav_network.environment.gpu.ds_aoi_belief_fp32 import CELL_SIZE_M, GRID_H, GRID_W, NUM_CELLS

OBS_DIM = 96
STATE_DIM = 256
DISC_ACTION_DIM = 7
MAX_QUEUE = 32
MAP_SIZE_M = 800.0
CRUISE_Z_M = 50.0
NOMINAL_RANGE_M = 480.0
SAFETY_DIST_M = 15.0
MAX_SPEED_MPS = 20.0
MAX_ACCEL_MPS2 = 4.0
REPORT_TTL_S = 120.0
BATTERY_CAPACITY_J = 150000.0


if wp is not None:
    wp.set_module_options({"enable_backward": False})

    @wp.kernel
    def motion_2d_safety_fp32(
        act_cont: wp.array3d(dtype=wp.float32),       # (E, 6, 2) in [-1, 1]
        uav_pos: wp.array2d(dtype=wp.vec3f),          # (E, 6)
        uav_vel: wp.array2d(dtype=wp.vec3f),          # (E, 6)
        uav_alive: wp.array2d(dtype=wp.int32),        # (E, 6)
        node_pos: wp.array2d(dtype=wp.vec3f),         # (E, 7) 0..5 UAVs, 6 GCS
        gcs_pos: wp.array(dtype=wp.vec3f),            # (E,)
        obstacles: wp.array2d(dtype=wp.vec4f),        # (E, max_obs) (cx, cy, r, h)
        num_obstacles: wp.int32,
        last_act_norm: wp.array2d(dtype=wp.float32),  # (E, 6)
        step_safety_viol: wp.array(dtype=wp.int32),   # (E,)
        dt_s: wp.float32,
    ):
        """2D project-and-penalize motion safety kernel (non-terminal -> prevents suicide exploit)."""
        e = wp.tid()
        viol = wp.int32(0)

        for u in range(NUM_UAVS):
            if uav_alive[e, u] == wp.int32(0):
                node_pos[e, u] = uav_pos[e, u]
                continue

            ax = wp.clamp(act_cont[e, u, 0], wp.float32(-1.0), wp.float32(1.0)) * wp.float32(MAX_ACCEL_MPS2)
            ay = wp.clamp(act_cont[e, u, 1], wp.float32(-1.0), wp.float32(1.0)) * wp.float32(MAX_ACCEL_MPS2)
            last_act_norm[e, u] = wp.sqrt(ax * ax + ay * ay) / wp.float32(MAX_ACCEL_MPS2 * 1.41421356)

            v = uav_vel[e, u]
            vx = v[0] + ax * dt_s
            vy = v[1] + ay * dt_s
            spd = wp.sqrt(vx * vx + vy * vy)
            if spd > wp.float32(MAX_SPEED_MPS):
                sc = wp.float32(MAX_SPEED_MPS) / spd
                vx = vx * sc
                vy = vy * sc

            p = uav_pos[e, u]
            nx = p[0] + vx * dt_s
            ny = p[1] + vy * dt_s

            # Boundary clamp + violation penalty
            if nx < wp.float32(5.0) or nx > wp.float32(MAP_SIZE_M - 5.0) or ny < wp.float32(5.0) or ny > wp.float32(MAP_SIZE_M - 5.0):
                viol = viol + wp.int32(1)
                nx = wp.clamp(nx, wp.float32(5.0), wp.float32(MAP_SIZE_M - 5.0))
                ny = wp.clamp(ny, wp.float32(5.0), wp.float32(MAP_SIZE_M - 5.0))
                vx = wp.float32(0.0)
                vy = wp.float32(0.0)

            # Building cylinder projection + violation penalty
            for o in range(num_obstacles):
                ob = obstacles[e, o]
                dx = nx - ob[0]
                dy = ny - ob[1]
                d_ob = wp.sqrt(dx * dx + dy * dy)
                if d_ob < ob[2] + wp.float32(4.0):
                    viol = viol + wp.int32(1)
                    push = (ob[2] + wp.float32(5.0)) / wp.max(d_ob, wp.float32(1e-4))
                    nx = ob[0] + dx * push
                    ny = ob[1] + dy * push
                    vx = wp.float32(0.0)
                    vy = wp.float32(0.0)

            new_p = wp.vec3f(nx, ny, wp.float32(CRUISE_Z_M))
            uav_pos[e, u] = new_p
            uav_vel[e, u] = wp.vec3f(vx, vy, wp.float32(0.0))
            node_pos[e, u] = new_p

        node_pos[e, wp.static(GCS_IDX)] = gcs_pos[e]

        # Pairwise UAV-UAV safety check (< 15 m)
        s2 = wp.float32(SAFETY_DIST_M * SAFETY_DIST_M)
        for i in range(NUM_UAVS):
            if uav_alive[e, i] == wp.int32(0):
                continue
            pi = uav_pos[e, i]
            for j in range(i + 1, wp.static(NUM_UAVS)):
                if uav_alive[e, j] == wp.int32(0):
                    continue
                pj = uav_pos[e, j]
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                if dx * dx + dy * dy < s2:
                    viol = viol + wp.int32(1)
                    # Project the candidate pose instead of recording only a penalty.
                    # Mirror the CPU reference; keep PHY node positions synchronized.
                    d = wp.sqrt(dx * dx + dy * dy)
                    if d < wp.float32(1.0e-7):
                        dx = wp.float32(-1.0)
                        dy = wp.float32(0.0)
                        d = wp.float32(1.0)
                    safe_d = wp.float32(SAFETY_DIST_M + 0.01)
                    px = wp.clamp(pi[0] - safe_d * dx / d, wp.float32(5.0), wp.float32(MAP_SIZE_M - 5.0))
                    py = wp.clamp(pi[1] - safe_d * dy / d, wp.float32(5.0), wp.float32(MAP_SIZE_M - 5.0))
                    p_safe = wp.vec3f(px, py, wp.float32(CRUISE_Z_M))
                    uav_pos[e, j] = p_safe
                    node_pos[e, j] = p_safe
                    uav_vel[e, j] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))

        step_safety_viol[e] = viol

    @wp.kernel
    def assemble_reward_and_done_fp32(
        step_delta_u: wp.array(dtype=wp.float32),
        step_aoi_score: wp.array(dtype=wp.float32),
        step_prop_kj: wp.array(dtype=wp.float32),
        step_safety_viol: wp.array(dtype=wp.int32),
        step_retry_drops: wp.array(dtype=wp.int32),
        step_ttl_expired: wp.array(dtype=wp.int32),
        step_new_gcs_deliv: wp.array(dtype=wp.int32),
        step_false_gcs_deliv: wp.array(dtype=wp.int32),
        step_multihop_deliv: wp.array(dtype=wp.int32),
        step_collisions: wp.array(dtype=wp.int32),
        step_hd_drops: wp.array(dtype=wp.int32),
        uav_alive: wp.array2d(dtype=wp.int32),
        env_step_count: wp.array(dtype=wp.int32),
        step_time_s: wp.array(dtype=wp.float32),
        # Outputs & accumulators
        rewards_out: wp.array(dtype=wp.float32),
        dones_out: wp.array(dtype=wp.float32),
        terminated_out: wp.array(dtype=wp.float32),
        ep_return: wp.array(dtype=wp.float32),
        ep_true_deliv: wp.array(dtype=wp.int32),
        ep_false_deliv: wp.array(dtype=wp.int32),
        ep_multihop: wp.array(dtype=wp.int32),
        ep_collisions: wp.array(dtype=wp.int32),
        ep_hd_drops: wp.array(dtype=wp.int32),
        ep_retry_drops: wp.array(dtype=wp.int32),
        ep_ttl_expired: wp.array(dtype=wp.int32),
        ep_prop_kj: wp.array(dtype=wp.float32),
        ep_safety_viol: wp.array(dtype=wp.int32),
        w1_ds: wp.float32,
        w2_aoi: wp.float32,
        w3_prop: wp.float32,
        w4_safety: wp.float32,
        w5_drop: wp.float32,
        max_steps: wp.int32,
    ):
        """Assemble team reward, update KPI accumulators, and set episode done flags."""
        e = wp.tid()
        drops = step_retry_drops[e] + step_ttl_expired[e]
        r = (
            w1_ds * step_delta_u[e]
            + w2_aoi * step_aoi_score[e]
            - w3_prop * step_prop_kj[e]
            - w4_safety * wp.float32(step_safety_viol[e])
            - w5_drop * wp.float32(drops)
        )
        rewards_out[e] = r
        ep_return[e] = ep_return[e] + r
        ep_true_deliv[e] = ep_true_deliv[e] + step_new_gcs_deliv[e]
        ep_false_deliv[e] = ep_false_deliv[e] + step_false_gcs_deliv[e]
        ep_multihop[e] = ep_multihop[e] + step_multihop_deliv[e]
        ep_collisions[e] = ep_collisions[e] + step_collisions[e]
        ep_hd_drops[e] = ep_hd_drops[e] + step_hd_drops[e]
        ep_retry_drops[e] = ep_retry_drops[e] + step_retry_drops[e]
        ep_ttl_expired[e] = ep_ttl_expired[e] + step_ttl_expired[e]
        ep_prop_kj[e] = ep_prop_kj[e] + step_prop_kj[e]
        ep_safety_viol[e] = ep_safety_viol[e] + step_safety_viol[e]

        sc = env_step_count[e] + wp.int32(1)
        env_step_count[e] = sc
        alive_sum = wp.int32(0)
        for u in range(NUM_UAVS):
            alive_sum = alive_sum + uav_alive[e, u]

        terminated_out[e] = wp.float32(1.0) if alive_sum == wp.int32(0) else wp.float32(0.0)
        if sc >= max_steps or alive_sum == wp.int32(0):
            dones_out[e] = wp.float32(1.0)
            step_time_s[e] = wp.float32(0.0)
        else:
            dones_out[e] = wp.float32(0.0)
            step_time_s[e] = wp.float32(sc)

    @wp.kernel
    def selective_auto_reset_fp32(
        dones: wp.array(dtype=wp.float32),
        init_uav_pos: wp.array2d(dtype=wp.vec3f),
        uav_pos: wp.array2d(dtype=wp.vec3f),
        uav_vel: wp.array2d(dtype=wp.vec3f),
        uav_battery_j: wp.array2d(dtype=wp.float32),
        uav_alive: wp.array2d(dtype=wp.int32),
        node_pos: wp.array2d(dtype=wp.vec3f),
        last_act_norm: wp.array2d(dtype=wp.float32),
        env_step_count: wp.array(dtype=wp.int32),
        ds_pos_cnt: wp.array2d(dtype=wp.int32),
        ds_neg_cnt: wp.array2d(dtype=wp.int32),
        ds_m_f: wp.array2d(dtype=wp.float32),
        ds_m_u: wp.array2d(dtype=wp.float32),
        cell_reported: wp.array2d(dtype=wp.int32),
        gcs_delivered: wp.array2d(dtype=wp.int32),
        gcs_true_aoi_s: wp.array2d(dtype=wp.float32),
        ep_radio_energy_j: wp.array(dtype=wp.float32),
        local_ds_pos: wp.array3d(dtype=wp.int32),
        local_ds_neg: wp.array3d(dtype=wp.int32),
        local_ds_m_f: wp.array3d(dtype=wp.float32),
        local_ds_m_u: wp.array3d(dtype=wp.float32),
        local_cell_reported: wp.array3d(dtype=wp.int32),
        q_count: wp.array2d(dtype=wp.int32),
        ep_return: wp.array(dtype=wp.float32),
        ep_true_deliv: wp.array(dtype=wp.int32),
        ep_false_deliv: wp.array(dtype=wp.int32),
        ep_multihop: wp.array(dtype=wp.int32),
        ep_collisions: wp.array(dtype=wp.int32),
        ep_hd_drops: wp.array(dtype=wp.int32),
        ep_retry_drops: wp.array(dtype=wp.int32),
        ep_ttl_expired: wp.array(dtype=wp.int32),
        ep_prop_kj: wp.array(dtype=wp.float32),
        ep_safety_viol: wp.array(dtype=wp.int32),
    ):
        """CUDA-Graph-safe selective in-place auto-reset for worlds with dones[e] > 0.5."""
        e = wp.tid()
        if dones[e] <= wp.float32(0.5):
            return

        env_step_count[e] = wp.int32(0)
        ep_return[e] = wp.float32(0.0)
        ep_true_deliv[e] = wp.int32(0)
        ep_false_deliv[e] = wp.int32(0)
        ep_multihop[e] = wp.int32(0)
        ep_collisions[e] = wp.int32(0)
        ep_hd_drops[e] = wp.int32(0)
        ep_retry_drops[e] = wp.int32(0)
        ep_ttl_expired[e] = wp.int32(0)
        ep_prop_kj[e] = wp.float32(0.0)
        ep_safety_viol[e] = wp.int32(0)
        ep_radio_energy_j[e] = wp.float32(0.0)

        for u in range(NUM_UAVS):
            uav_pos[e, u] = init_uav_pos[e, u]
            node_pos[e, u] = init_uav_pos[e, u]
            uav_vel[e, u] = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
            last_act_norm[e, u] = wp.float32(0.0)
            uav_battery_j[e, u] = wp.float32(BATTERY_CAPACITY_J)
            uav_alive[e, u] = wp.int32(1)
            q_count[e, u] = wp.int32(0)

        for c in range(NUM_CELLS):
            ds_pos_cnt[e, c] = wp.int32(0)
            ds_neg_cnt[e, c] = wp.int32(0)
            ds_m_f[e, c] = wp.float32(0.0)
            ds_m_u[e, c] = wp.float32(1.0)
            cell_reported[e, c] = wp.int32(0)
            gcs_delivered[e, c] = wp.int32(0)
            gcs_true_aoi_s[e, c] = wp.float32(-1.0)
            for u in range(NUM_UAVS):
                local_ds_pos[e, u, c] = wp.int32(0)
                local_ds_neg[e, u, c] = wp.int32(0)
                local_ds_m_f[e, u, c] = wp.float32(0.0)
                local_ds_m_u[e, u, c] = wp.float32(1.0)
                local_cell_reported[e, u, c] = wp.int32(0)

    @wp.kernel
    def build_obs_state_mask_fp32(
        uav_pos: wp.array2d(dtype=wp.vec3f),
        uav_vel: wp.array2d(dtype=wp.vec3f),
        uav_battery_j: wp.array2d(dtype=wp.float32),
        uav_alive: wp.array2d(dtype=wp.int32),
        gcs_pos: wp.array(dtype=wp.vec3f),
        obstacles: wp.array2d(dtype=wp.vec4f),
        num_obstacles: wp.int32,
        dist_mat: wp.array3d(dtype=wp.float32),
        los_mat: wp.array3d(dtype=wp.int32),
        q_count: wp.array2d(dtype=wp.int32),
        q_timestamp: wp.array3d(dtype=wp.float32),
        q_hops: wp.array3d(dtype=wp.int32),
        q_visited_mask: wp.array3d(dtype=wp.int32),
        ds_m_f: wp.array2d(dtype=wp.float32),
        ds_m_u: wp.array2d(dtype=wp.float32),
        local_ds_m_f: wp.array3d(dtype=wp.float32),
        local_ds_m_u: wp.array3d(dtype=wp.float32),
        belief_sharing_mode: wp.int32,               # 1 = local_decpomdp, 0 = shared_team_ideal
        target_cells: wp.array2d(dtype=wp.int32),    # (E, 25) for 256D Critic state ONLY
        cell_reported: wp.array2d(dtype=wp.int32),
        gcs_delivered: wp.array2d(dtype=wp.int32),
        last_act_norm: wp.array2d(dtype=wp.float32),
        env_step_count: wp.array(dtype=wp.int32),
        step_time_s: wp.array(dtype=wp.float32),
        max_steps: wp.int32,
        # Zero-copy GPU outputs for PyTorch
        obs_out: wp.array3d(dtype=wp.float32),       # (E, 6, 96)
        state_out: wp.array2d(dtype=wp.float32),     # (E, 256)
        mask_out: wp.array3d(dtype=wp.uint8),        # (E, 6, 7)
    ):
        """Build 96D NoOracle obs, 256D CTDE global state, and 7D adaptive mask directly on GPU."""
        e = wp.tid()
        now = wp.float32(env_step_count[e])
        t_norm = wp.float32(env_step_count[e]) / wp.float32(wp.max(max_steps, wp.int32(1)))
        gp = gcs_pos[e]

        # --- A. 96D Local Observation & 7D Adaptive Action Mask per UAV ---
        for u in range(NUM_UAVS):
            for f_i in range(OBS_DIM):
                obs_out[e, u, f_i] = wp.float32(0.0)
            for m_i in range(DISC_ACTION_DIM):
                mask_out[e, u, m_i] = wp.uint8(0)
            mask_out[e, u, 0] = wp.uint8(1)  # Silent always valid

            pu = uav_pos[e, u]
            vu = uav_vel[e, u]
            spd = wp.sqrt(vu[0] * vu[0] + vu[1] * vu[1])
            cos_t = vu[0] / spd if spd > wp.float32(1e-4) else wp.float32(1.0)
            sin_t = vu[1] / spd if spd > wp.float32(1e-4) else wp.float32(0.0)

            # 1) Ego kinematics (8)
            obs_out[e, u, 0] = pu[0] / wp.float32(MAP_SIZE_M)
            obs_out[e, u, 1] = pu[1] / wp.float32(MAP_SIZE_M)
            obs_out[e, u, 2] = vu[0] / wp.float32(MAX_SPEED_MPS)
            obs_out[e, u, 3] = vu[1] / wp.float32(MAX_SPEED_MPS)
            obs_out[e, u, 4] = cos_t
            obs_out[e, u, 5] = sin_t
            obs_out[e, u, 6] = uav_battery_j[e, u] / wp.float32(BATTERY_CAPACITY_J)
            obs_out[e, u, 7] = wp.float32(uav_alive[e, u])

            # 2) GCS geometry & link (4)
            obs_out[e, u, 8] = (gp[0] - pu[0]) / wp.float32(MAP_SIZE_M)
            obs_out[e, u, 9] = (gp[1] - pu[1]) / wp.float32(MAP_SIZE_M)
            obs_out[e, u, 10] = dist_mat[e, u, wp.static(GCS_IDX)] / wp.float32(MAP_SIZE_M)
            obs_out[e, u, 11] = wp.float32(los_mat[e, u, wp.static(GCS_IDX)])

            # 3) Queue & HoL packet (11)
            has_pkt = wp.int32(1) if q_count[e, u] > wp.int32(0) else wp.int32(0)
            hol_age = wp.max(wp.float32(0.0), now - q_timestamp[e, u, 0]) if has_pkt == wp.int32(1) else wp.float32(0.0)
            hol_ttl = wp.max(wp.float32(0.0), wp.float32(REPORT_TTL_S) - hol_age) if has_pkt == wp.int32(1) else wp.float32(0.0)
            hol_hop = wp.float32(q_hops[e, u, 0]) / wp.float32(6.0) if has_pkt == wp.int32(1) else wp.float32(0.0)
            vmask = q_visited_mask[e, u, 0] if has_pkt == wp.int32(1) else wp.int32(0)

            obs_out[e, u, 12] = wp.float32(q_count[e, u]) / wp.float32(MAX_QUEUE)
            obs_out[e, u, 13] = wp.float32(has_pkt)
            obs_out[e, u, 14] = wp.min(wp.float32(1.0), hol_age / wp.float32(wp.max(max_steps, wp.int32(1))))
            obs_out[e, u, 15] = hol_ttl / wp.float32(REPORT_TTL_S)
            obs_out[e, u, 16] = wp.min(wp.float32(1.0), hol_hop)
            for b_i in range(NUM_UAVS):
                obs_out[e, u, 17 + b_i] = wp.float32(1.0) if ((vmask >> b_i) & wp.int32(1)) != wp.int32(0) else wp.float32(0.0)

            # 4) 8-ray building/boundary LiDAR (8, indices 23..30)
            r_lidar = wp.float32(120.0)
            for ray_i in range(8):
                ang = wp.float32(ray_i) * wp.float32(0.785398163)
                dx = wp.cos(ang)
                dy = wp.sin(ang)
                min_d = r_lidar
                if dx > wp.float32(1e-4):
                    min_d = wp.min(min_d, (wp.float32(MAP_SIZE_M) - pu[0]) / dx)
                elif dx < wp.float32(-1e-4):
                    min_d = wp.min(min_d, -pu[0] / dx)
                if dy > wp.float32(1e-4):
                    min_d = wp.min(min_d, (wp.float32(MAP_SIZE_M) - pu[1]) / dy)
                elif dy < wp.float32(-1e-4):
                    min_d = wp.min(min_d, -pu[1] / dy)
                for o in range(num_obstacles):
                    ob = obstacles[e, o]
                    proj = (ob[0] - pu[0]) * dx + (ob[1] - pu[1]) * dy
                    if proj > wp.float32(0.0) and proj < min_d + ob[2]:
                        rx = ob[0] - pu[0] - proj * dx
                        ry = ob[1] - pu[1] - proj * dy
                        perp2 = rx * rx + ry * ry
                        if perp2 < ob[2] * ob[2]:
                            hit_d = proj - wp.sqrt(ob[2] * ob[2] - perp2)
                            if hit_d > wp.float32(0.0):
                                min_d = wp.min(min_d, hit_d)
                obs_out[e, u, 23 + ray_i] = wp.clamp(min_d / r_lidar, wp.float32(0.0), wp.float32(1.0))

            # 5) 5 Peers telemetry (40, indices 31..70) + Peer Action Masks
            peer_slot = wp.int32(0)
            for p in range(NUM_UAVS):
                if p == u:
                    continue
                d_up = dist_mat[e, u, p]
                in_rng = (uav_alive[e, p] == wp.int32(1)) and (d_up <= wp.float32(NOMINAL_RANGE_M))
                vis_p = wp.int32(1) if ((vmask >> p) & wp.int32(1)) != wp.int32(0) else wp.int32(0)
                base_p = wp.int32(31) + peer_slot * wp.int32(8)
                if in_rng:
                    pp = uav_pos[e, p]
                    obs_out[e, u, base_p + 0] = (pp[0] - pu[0]) / wp.float32(MAP_SIZE_M)
                    obs_out[e, u, base_p + 1] = (pp[1] - pu[1]) / wp.float32(MAP_SIZE_M)
                    obs_out[e, u, base_p + 2] = d_up / wp.float32(MAP_SIZE_M)
                    obs_out[e, u, base_p + 3] = wp.float32(1.0)
                    obs_out[e, u, base_p + 4] = wp.float32(los_mat[e, u, p])
                    obs_out[e, u, base_p + 5] = wp.float32(q_count[e, p]) / wp.float32(MAX_QUEUE)
                    obs_out[e, u, base_p + 6] = dist_mat[e, p, wp.static(GCS_IDX)] / wp.float32(MAP_SIZE_M)
                    obs_out[e, u, base_p + 7] = wp.float32(vis_p)
                peer_link_ok = ((los_mat[e, u, p] == wp.int32(1)) and (d_up <= wp.float32(NOMINAL_RANGE_M))) or (d_up <= wp.float32(75.0))
                if uav_alive[e, u] == wp.int32(1) and has_pkt == wp.int32(1) and (uav_alive[e, p] == wp.int32(1)) and peer_link_ok and vis_p == wp.int32(0) and q_count[e, p] < wp.int32(MAX_QUEUE):
                    mask_out[e, u, wp.int32(1) + peer_slot] = wp.uint8(1)
                peer_slot = peer_slot + wp.int32(1)

            d_ug = dist_mat[e, u, GCS_IDX]
            gcs_link_ok = ((los_mat[e, u, GCS_IDX] == wp.int32(1)) and (d_ug <= wp.float32(NOMINAL_RANGE_M))) or (d_ug <= wp.float32(75.0))
            if uav_alive[e, u] == wp.int32(1) and has_pkt == wp.int32(1) and gcs_link_ok:
                mask_out[e, u, wp.static(GCS_IDX)] = wp.uint8(1)

            # 6) Local DS Belief 3x3 patch (18, indices 71..88) + 4 quadrant m_U (4, indices 89..92)
            gx = wp.clamp(wp.int32(pu[0] / wp.float32(CELL_SIZE_M)), wp.int32(0), wp.int32(GRID_W - 1))
            gy = wp.clamp(wp.int32(pu[1] / wp.float32(CELL_SIZE_M)), wp.int32(0), wp.int32(GRID_H - 1))
            p_idx = wp.int32(71)
            for dy_c in range(-1, 2):
                for dx_c in range(-1, 2):
                    ny = gy + dy_c
                    nx = gx + dx_c
                    if ny >= wp.int32(0) and ny < wp.int32(GRID_H) and nx >= wp.int32(0) and nx < wp.int32(GRID_W):
                        cid = ny * wp.int32(GRID_W) + nx
                        mf_v = local_ds_m_f[e, u, cid] if belief_sharing_mode == wp.int32(1) else ds_m_f[e, cid]
                        mu_v = local_ds_m_u[e, u, cid] if belief_sharing_mode == wp.int32(1) else ds_m_u[e, cid]
                        obs_out[e, u, p_idx] = mf_v
                        obs_out[e, u, p_idx + wp.int32(1)] = mu_v
                    p_idx = p_idx + wp.int32(2)

            # 4 quadrant m_U averages
            q0 = wp.float32(0.0); c0 = wp.float32(0.0)
            q1 = wp.float32(0.0); c1 = wp.float32(0.0)
            q2 = wp.float32(0.0); c2 = wp.float32(0.0)
            q3 = wp.float32(0.0); c3 = wp.float32(0.0)
            for cy in range(GRID_H):
                for cx in range(GRID_W):
                    cid = cy * wp.int32(GRID_W) + cx
                    mu_v = local_ds_m_u[e, u, cid] if belief_sharing_mode == wp.int32(1) else ds_m_u[e, cid]
                    if cy <= gy and cx <= gx:
                        q0 = q0 + mu_v; c0 = c0 + wp.float32(1.0)
                    if cy <= gy and cx >= gx:
                        q1 = q1 + mu_v; c1 = c1 + wp.float32(1.0)
                    if cy >= gy and cx <= gx:
                        q2 = q2 + mu_v; c2 = c2 + wp.float32(1.0)
                    if cy >= gy and cx >= gx:
                        q3 = q3 + mu_v; c3 = c3 + wp.float32(1.0)
            obs_out[e, u, 89] = q0 / wp.max(c0, wp.float32(1.0))
            obs_out[e, u, 90] = q1 / wp.max(c1, wp.float32(1.0))
            obs_out[e, u, 91] = q2 / wp.max(c2, wp.float32(1.0))
            obs_out[e, u, 92] = q3 / wp.max(c3, wp.float32(1.0))

            # 7) MAC feedback & time (3, indices 93..95)
            obs_out[e, u, 93] = wp.float32(0.0)
            obs_out[e, u, 94] = wp.float32(0.0)
            obs_out[e, u, 95] = t_norm

        # --- B. 256D Centralized Critic Global State (CTDE) ---
        for u in range(NUM_UAVS):
            base_u = u * wp.int32(16)
            pu = uav_pos[e, u]
            vu = uav_vel[e, u]
            spd = wp.sqrt(vu[0] * vu[0] + vu[1] * vu[1])
            cos_t = vu[0] / spd if spd > wp.float32(1e-4) else wp.float32(1.0)
            sin_t = vu[1] / spd if spd > wp.float32(1e-4) else wp.float32(0.0)
            has_pkt = wp.int32(1) if q_count[e, u] > wp.int32(0) else wp.int32(0)
            hol_age = wp.max(wp.float32(0.0), now - q_timestamp[e, u, 0]) if has_pkt == wp.int32(1) else wp.float32(0.0)
            hol_ttl = wp.max(wp.float32(0.0), wp.float32(REPORT_TTL_S) - hol_age) if has_pkt == wp.int32(1) else wp.float32(0.0)

            state_out[e, base_u + 0] = pu[0] / wp.float32(MAP_SIZE_M)
            state_out[e, base_u + 1] = pu[1] / wp.float32(MAP_SIZE_M)
            state_out[e, base_u + 2] = vu[0] / wp.float32(MAX_SPEED_MPS)
            state_out[e, base_u + 3] = vu[1] / wp.float32(MAX_SPEED_MPS)
            state_out[e, base_u + 4] = cos_t
            state_out[e, base_u + 5] = sin_t
            state_out[e, base_u + 6] = uav_battery_j[e, u] / wp.float32(BATTERY_CAPACITY_J)
            state_out[e, base_u + 7] = wp.float32(uav_alive[e, u])
            state_out[e, base_u + 8] = dist_mat[e, u, wp.static(GCS_IDX)] / wp.float32(MAP_SIZE_M)
            state_out[e, base_u + 9] = wp.float32(los_mat[e, u, wp.static(GCS_IDX)])
            state_out[e, base_u + 10] = wp.float32(q_count[e, u]) / wp.float32(MAX_QUEUE)
            state_out[e, base_u + 11] = wp.min(wp.float32(1.0), hol_age / wp.float32(wp.max(max_steps, wp.int32(1))))
            state_out[e, base_u + 12] = hol_ttl / wp.float32(REPORT_TTL_S)
            state_out[e, base_u + 13] = wp.float32(q_hops[e, u, 0]) / wp.float32(6.0) if has_pkt == wp.int32(1) else wp.float32(0.0)
            state_out[e, base_u + 14] = wp.float32(0.0)
            state_out[e, base_u + 15] = last_act_norm[e, u]

        # Pairwise link matrix (15 x 2 = 30, indices 96..125)
        pair_idx = wp.int32(96)
        for i in range(NUM_UAVS):
            for j in range(i + 1, wp.static(NUM_UAVS)):
                d_ij = dist_mat[e, i, j]
                state_out[e, pair_idx] = d_ij / wp.float32(MAP_SIZE_M)
                state_out[e, pair_idx + wp.int32(1)] = (
                    wp.float32(1.0)
                    if (d_ij <= wp.float32(NOMINAL_RANGE_M) and los_mat[e, i, j] == wp.int32(1))
                    else wp.float32(0.0)
                )
                pair_idx = pair_idx + wp.int32(2)

        # Pooled Global DS Map: 5x5 x 2 = 50 (indices 126..175)
        for by in range(5):
            for bx in range(5):
                sf = wp.float32(0.0)
                su = wp.float32(0.0)
                for dy_p in range(4):
                    for dx_p in range(4):
                        cid = (by * wp.int32(4) + dy_p) * wp.int32(GRID_W) + (bx * wp.int32(4) + dx_p)
                        sf = sf + ds_m_f[e, cid]
                        su = su + ds_m_u[e, cid]
                slot = by * wp.int32(5) + bx
                state_out[e, wp.int32(126) + slot] = sf * wp.float32(0.0625)
                state_out[e, wp.int32(151) + slot] = su * wp.float32(0.0625)

        # Privileged target state (25 x 3 = 75, indices 176..250)
        for k in range(25):
            cid = target_cells[e, k]
            tx = (wp.float32(cid % wp.int32(GRID_W)) + wp.float32(0.5)) * wp.float32(CELL_SIZE_M) / wp.float32(MAP_SIZE_M)
            ty = (wp.float32(cid // wp.int32(GRID_W)) + wp.float32(0.5)) * wp.float32(CELL_SIZE_M) / wp.float32(MAP_SIZE_M)
            t_st = wp.float32(0.0)
            if gcs_delivered[e, cid] == wp.int32(1):
                t_st = wp.float32(1.0)
            elif cell_reported[e, cid] == wp.int32(1):
                t_st = wp.float32(0.5)
            base_k = wp.int32(176) + k * wp.int32(3)
            state_out[e, base_k + 0] = tx
            state_out[e, base_k + 1] = ty
            state_out[e, base_k + 2] = t_st

        # Global scalars (5, indices 251..255)
        alive_cnt = wp.float32(0.0)
        for u in range(NUM_UAVS):
            alive_cnt = alive_cnt + wp.float32(uav_alive[e, u])
        state_out[e, 251] = t_norm
        state_out[e, 252] = wp.float32(0.0)
        state_out[e, 253] = wp.float32(0.0)
        state_out[e, 254] = wp.float32(0.0)
        state_out[e, 255] = alive_cnt / wp.float32(NUM_UAVS)
