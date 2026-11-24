"""Complete End-to-End 2.5D Urban NLOS Multi-UAV Environment (REAL Warp GPU Backend + NumPy Reference).

Closes Gap A, Gap B, and Gap C:
- `backend="warp_slotted_fp32"` (or `"warp"`): Runs the complete 7-kernel pure-FP32 NVIDIA Warp
  pipeline on CUDA with zero-copy PyTorch tensors and optional CUDA Graph capture (`wp.ScopedCapture`).
- `backend="numpy"`: Pure-NumPy CPU reference with identical `belief_sharing_mode` (`local_decpomdp`
  default vs `shared_team_ideal`) and building LOS occlusion for exact CPU–GPU parity verification.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None

from uav_network.communication.gpu.phy_sinr_fp32 import (
    CCA_THRESHOLD_W,
    FSPL_1M_DB,
    GCS_IDX,
    LOS_EXPONENT,
    NLOS_EXTRA_LOSS_DB,
    NOISE_FLOOR_W,
    NUM_NODES,
    NUM_UAVS,
    SINR_THRESHOLD_LIN,
    TX_POWER_W,
)
from uav_network.environment.gpu.ds_aoi_belief_fp32 import (
    CELL_SIZE_M,
    GRID_H,
    GRID_W,
    NUM_CELLS,
    WarpSlottedUavEnvGraph,
    ds_closed_form,
    zeng_rotary_power_w,
)
from uav_network.environment.gpu.slotted_2d_warp_kernels import (
    BATTERY_CAPACITY_J,
    CRUISE_Z_M,
    DISC_ACTION_DIM,
    MAP_SIZE_M,
    MAX_ACCEL_MPS2,
    MAX_QUEUE,
    MAX_SPEED_MPS,
    NOMINAL_RANGE_M,
    OBS_DIM,
    REPORT_TTL_S,
    SAFETY_DIST_M,
    STATE_DIM,
)

CONT_ACTION_DIM = 2
GCS_POS = np.array([40.0, 40.0, 10.0], dtype=np.float32)
SENSING_RADIUS_M = 60.0
MAX_STEPS_DEFAULT = 550


@dataclass
class ScenarioSnapshot:
    seed: int
    obstacles: np.ndarray        # (22, 4) float32 -> (cx, cy, radius, height)
    target_cells: np.ndarray     # (25,) int32
    cell_has_target: np.ndarray  # (400,) int32 -> -1 if empty else 0..24
    init_uav_pos: np.ndarray     # (6, 3) float32
    gcs_pos: np.ndarray          # (3,) float32


def generate_scenario_snapshot(
    seed: int,
    num_obstacles: int = 22,
    num_targets: int = 25,
    map_size: float = MAP_SIZE_M,
) -> ScenarioSnapshot:
    rng = np.random.default_rng(seed)
    gcs = GCS_POS.copy()

    obstacles = np.zeros((num_obstacles, 4), dtype=np.float32)
    placed = 0
    attempts = 0
    while placed < num_obstacles and attempts < 8000:
        attempts += 1
        r = float(rng.uniform(25.0, 45.0))
        h = float(rng.uniform(60.0, 120.0))
        cx = float(rng.uniform(r + 20.0, map_size - r - 20.0))
        cy = float(rng.uniform(r + 20.0, map_size - r - 20.0))
        if math.hypot(cx - gcs[0], cy - gcs[1]) < r + 140.0:
            continue
        ok = True
        for k in range(placed):
            ox, oy, orad, _ = obstacles[k]
            if math.hypot(cx - ox, cy - oy) < r + orad + 35.0:
                ok = False
                break
        if ok:
            obstacles[placed] = (cx, cy, r, h)
            placed += 1
    if placed < num_obstacles:
        raise RuntimeError(f"Failed to place {num_obstacles} obstacles for seed={seed} (placed={placed})")

    valid_cells: List[int] = []
    for cy in range(GRID_H):
        cell_y = (cy + 0.5) * CELL_SIZE_M
        for cx in range(GRID_W):
            cell_x = (cx + 0.5) * CELL_SIZE_M
            if math.hypot(cell_x - gcs[0], cell_y - gcs[1]) < 160.0:
                continue
            if any(
                math.hypot(cell_x - float(obstacles[k, 0]), cell_y - float(obstacles[k, 1]))
                < float(obstacles[k, 2]) + 8.0
                for k in range(num_obstacles)
            ):
                continue
            valid_cells.append(cy * GRID_W + cx)

    chosen = rng.choice(valid_cells, size=num_targets, replace=False).astype(np.int32)
    cell_has_target = np.full((NUM_CELLS,), -1, dtype=np.int32)
    for t_idx, cid in enumerate(chosen):
        cell_has_target[int(cid)] = t_idx

    init_pos = np.zeros((NUM_UAVS, 3), dtype=np.float32)
    for u in range(NUM_UAVS):
        angle = 0.25 + u * (math.pi * 0.42 / max(NUM_UAVS - 1, 1))
        rad = 65.0 + (u % 2) * 32.0
        ux = float(np.clip(gcs[0] + rad * math.cos(angle), 20.0, map_size - 20.0))
        uy = float(np.clip(gcs[1] + rad * math.sin(angle), 20.0, map_size - 20.0))
        init_pos[u] = (ux, uy, CRUISE_Z_M)

    return ScenarioSnapshot(seed, obstacles, chosen, cell_has_target, init_pos, gcs)


def _segment_hits_circle(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, r: float) -> bool:
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-6:
        return False
    t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / l2))
    px, py = ax + t * dx - cx, ay + t * dy - cy
    return (px * px + py * py) <= r * r


def check_los_2d(ax: float, ay: float, bx: float, by: float, obstacles: np.ndarray) -> int:
    for k in range(obstacles.shape[0]):
        if _segment_hits_circle(ax, ay, bx, by, float(obstacles[k, 0]), float(obstacles[k, 1]), float(obstacles[k, 2])):
            return 0
    return 1


def rx_power_w(dist_m: float, is_los: int) -> float:
    d = max(1.0, dist_m)
    pl_db = FSPL_1M_DB + 10.0 * LOS_EXPONENT * math.log10(d) + (0.0 if is_los else NLOS_EXTRA_LOSS_DB)
    return TX_POWER_W * (10.0 ** (-pl_db / 10.0))


def build_local_obs_and_mask(
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    uav_battery_j: np.ndarray,
    uav_alive: np.ndarray,
    gcs_pos: np.ndarray,
    obstacles: np.ndarray,
    dist_mat: np.ndarray,
    los_mat: np.ndarray,
    q_count: np.ndarray,
    q_timestamp: np.ndarray,
    q_hops: np.ndarray,
    q_visited_mask: np.ndarray,
    ds_m_f_per_uav: np.ndarray,   # (6, 400) local or broadcast team DS
    ds_m_u_per_uav: np.ndarray,   # (6, 400)
    last_ack: np.ndarray,
    channel_busy_ewma: np.ndarray,
    step_t: int,
    max_steps: int = MAX_STEPS_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single Source of Truth: exact 96D NoOracle obs + 7D adaptive action mask."""
    obs = np.zeros((NUM_UAVS, OBS_DIM), dtype=np.float32)
    mask = np.zeros((NUM_UAVS, DISC_ACTION_DIM), dtype=np.uint8)
    mask[:, 0] = 1

    t_norm = float(step_t) / float(max(max_steps, 1))
    now_s = float(step_t)

    for u in range(NUM_UAVS):
        ds_f_2d = ds_m_f_per_uav[u].reshape(GRID_H, GRID_W)
        ds_u_2d = ds_m_u_per_uav[u].reshape(GRID_H, GRID_W)
        idx = 0
        ux, uy = float(uav_pos[u, 0]), float(uav_pos[u, 1])
        vx, vy = float(uav_vel[u, 0]), float(uav_vel[u, 1])
        spd = math.hypot(vx, vy)
        cos_t, sin_t = (vx / spd, vy / spd) if spd > 1e-4 else (1.0, 0.0)

        obs[u, 0:8] = [
            ux / MAP_SIZE_M, uy / MAP_SIZE_M, vx / MAX_SPEED_MPS, vy / MAX_SPEED_MPS,
            cos_t, sin_t, float(uav_battery_j[u]) / BATTERY_CAPACITY_J, float(uav_alive[u]),
        ]
        obs[u, 8:12] = [
            (float(gcs_pos[0]) - ux) / MAP_SIZE_M, (float(gcs_pos[1]) - uy) / MAP_SIZE_M,
            float(dist_mat[u, GCS_IDX]) / MAP_SIZE_M, float(los_mat[u, GCS_IDX]),
        ]

        has_pkt = 1 if int(q_count[u]) > 0 else 0
        hol_age = max(0.0, now_s - float(q_timestamp[u, 0])) if has_pkt else 0.0
        hol_ttl = max(0.0, REPORT_TTL_S - hol_age) if has_pkt else 0.0
        hol_hop = float(q_hops[u, 0]) / 6.0 if has_pkt else 0.0
        vmask = int(q_visited_mask[u, 0]) if has_pkt else 0
        obs[u, 12:17] = [
            float(q_count[u]) / float(MAX_QUEUE), float(has_pkt),
            min(1.0, hol_age / float(max_steps)), hol_ttl / REPORT_TTL_S, min(1.0, hol_hop),
        ]
        for b_i in range(NUM_UAVS):
            obs[u, 17 + b_i] = 1.0 if ((vmask >> b_i) & 1) else 0.0

        r_lidar = 120.0
        for ray_i in range(8):
            ang = ray_i * (math.pi / 4.0)
            dx, dy = math.cos(ang), math.sin(ang)
            min_d = r_lidar
            if dx > 1e-4:
                min_d = min(min_d, (MAP_SIZE_M - ux) / dx)
            elif dx < -1e-4:
                min_d = min(min_d, -ux / dx)
            if dy > 1e-4:
                min_d = min(min_d, (MAP_SIZE_M - uy) / dy)
            elif dy < -1e-4:
                min_d = min(min_d, -uy / dy)
            for k in range(obstacles.shape[0]):
                ocx, ocy, orad = float(obstacles[k, 0]), float(obstacles[k, 1]), float(obstacles[k, 2])
                proj = (ocx - ux) * dx + (ocy - uy) * dy
                if 0.0 < proj < min_d + orad:
                    perp2 = (ocx - ux - proj * dx) ** 2 + (ocy - uy - proj * dy) ** 2
                    if perp2 < orad * orad:
                        hit_d = proj - math.sqrt(orad * orad - perp2)
                        if hit_d > 0.0:
                            min_d = min(min_d, hit_d)
            obs[u, 23 + ray_i] = max(0.0, min(1.0, min_d / r_lidar))

        peer_slot = 0
        idx = 31
        for p in range(NUM_UAVS):
            if p == u:
                continue
            d_up = float(dist_mat[u, p])
            in_range = int(uav_alive[p]) == 1 and d_up <= NOMINAL_RANGE_M
            visited_p = 1 if ((vmask >> p) & 1) else 0
            if in_range:
                obs[u, idx : idx + 8] = [
                    (float(uav_pos[p, 0]) - ux) / MAP_SIZE_M, (float(uav_pos[p, 1]) - uy) / MAP_SIZE_M,
                    d_up / MAP_SIZE_M, 1.0, float(los_mat[u, p]),
                    float(q_count[p]) / float(MAX_QUEUE), float(dist_mat[p, GCS_IDX]) / MAP_SIZE_M, float(visited_p),
                ]
            peer_link_ok = (int(los_mat[u, p]) == 1 and d_up <= NOMINAL_RANGE_M) or (d_up <= 75.0)
            if int(uav_alive[u]) == 1 and has_pkt == 1 and int(uav_alive[p]) == 1 and peer_link_ok and visited_p == 0 and int(q_count[p]) < MAX_QUEUE:
                mask[u, 1 + peer_slot] = 1
            idx += 8
            peer_slot += 1

        d_ug = float(dist_mat[u, GCS_IDX])
        gcs_link_ok = (int(los_mat[u, GCS_IDX]) == 1 and d_ug <= NOMINAL_RANGE_M) or (d_ug <= 75.0)
        if int(uav_alive[u]) == 1 and has_pkt == 1 and gcs_link_ok:
            mask[u, GCS_IDX] = 1

        gx = max(0, min(GRID_W - 1, int(ux / CELL_SIZE_M)))
        gy = max(0, min(GRID_H - 1, int(uy / CELL_SIZE_M)))
        p_idx = 71
        for dy_c in (-1, 0, 1):
            for dx_c in (-1, 0, 1):
                ny, nx = gy + dy_c, gx + dx_c
                if 0 <= ny < GRID_H and 0 <= nx < GRID_W:
                    obs[u, p_idx] = float(ds_f_2d[ny, nx])
                    obs[u, p_idx + 1] = float(ds_u_2d[ny, nx])
                p_idx += 2
        obs[u, 89] = float(ds_u_2d[: gy + 1, : gx + 1].mean())
        obs[u, 90] = float(ds_u_2d[: gy + 1, gx:].mean())
        obs[u, 91] = float(ds_u_2d[gy:, : gx + 1].mean())
        obs[u, 92] = float(ds_u_2d[gy:, gx:].mean())
        obs[u, 93:96] = [float(last_ack[u]), float(channel_busy_ewma[u]), t_norm]

    return obs, mask


def build_global_state(
    uav_pos: np.ndarray,
    uav_vel: np.ndarray,
    uav_battery_j: np.ndarray,
    uav_alive: np.ndarray,
    dist_mat: np.ndarray,
    los_mat: np.ndarray,
    q_count: np.ndarray,
    q_timestamp: np.ndarray,
    q_hops: np.ndarray,
    ds_m_f: np.ndarray,
    ds_m_u: np.ndarray,
    target_cells: np.ndarray,
    cell_reported: np.ndarray,
    gcs_delivered: np.ndarray,
    last_ack: np.ndarray,
    last_act_norm: np.ndarray,
    step_t: int,
    pdr_so_far: float,
    mean_aoi_norm: float,
    tx_congestion_rate: float,
    max_steps: int = MAX_STEPS_DEFAULT,
) -> np.ndarray:
    state = np.zeros((STATE_DIM,), dtype=np.float32)
    now_s = float(step_t)
    for u in range(NUM_UAVS):
        vx, vy = float(uav_vel[u, 0]), float(uav_vel[u, 1])
        spd = math.hypot(vx, vy)
        cos_t, sin_t = (vx / spd, vy / spd) if spd > 1e-4 else (1.0, 0.0)
        has_pkt = int(q_count[u]) > 0
        hol_age = max(0.0, now_s - float(q_timestamp[u, 0])) if has_pkt else 0.0
        hol_ttl = max(0.0, REPORT_TTL_S - hol_age) if has_pkt else 0.0
        b = u * 16
        state[b : b + 16] = [
            float(uav_pos[u, 0]) / MAP_SIZE_M, float(uav_pos[u, 1]) / MAP_SIZE_M,
            vx / MAX_SPEED_MPS, vy / MAX_SPEED_MPS, cos_t, sin_t,
            float(uav_battery_j[u]) / BATTERY_CAPACITY_J, float(uav_alive[u]),
            float(dist_mat[u, GCS_IDX]) / MAP_SIZE_M, float(los_mat[u, GCS_IDX]),
            float(q_count[u]) / float(MAX_QUEUE), min(1.0, hol_age / float(max_steps)),
            hol_ttl / REPORT_TTL_S, float(q_hops[u, 0]) / 6.0 if has_pkt else 0.0,
            float(last_ack[u]), float(last_act_norm[u]),
        ]
    idx = 96
    for i in range(NUM_UAVS):
        for j in range(i + 1, NUM_UAVS):
            d_ij = float(dist_mat[i, j])
            state[idx] = d_ij / MAP_SIZE_M
            state[idx + 1] = 1.0 if (d_ij <= NOMINAL_RANGE_M and int(los_mat[i, j]) == 1) else 0.0
            idx += 2
    state[126:151] = ds_m_f.reshape(5, 4, 5, 4).mean(axis=(1, 3)).ravel()
    state[151:176] = ds_m_u.reshape(5, 4, 5, 4).mean(axis=(1, 3)).ravel()
    idx = 176
    for k in range(25):
        cid = int(target_cells[k])
        tx = ((cid % GRID_W) + 0.5) * CELL_SIZE_M / MAP_SIZE_M
        ty = ((cid // GRID_W) + 0.5) * CELL_SIZE_M / MAP_SIZE_M
        t_st = 1.0 if int(gcs_delivered[cid]) == 1 else (0.5 if int(cell_reported[cid]) == 1 else 0.0)
        state[idx : idx + 3] = [tx, ty, t_st]
        idx += 3
    state[251:256] = [
        float(step_t) / float(max(max_steps, 1)), float(pdr_so_far),
        float(mean_aoi_norm), float(tx_congestion_rate), float(uav_alive.mean()),
    ]
    return state


class WarpSlotted2DBackend:
    """REAL End-to-End NVIDIA Warp GPU Backend (Pure FP32 + Zero-Copy PyTorch + CUDA Graph)."""

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        aoi_decay_eta: float,
        w1_ds: float,
        w2_aoi: float,
        w3_prop_kj: float,
        w4_safety: float,
        w5_drop: float,
        base_seed: int,
        belief_sharing_mode: str = "local_decpomdp",
        device: str = "cuda:0",
        use_cuda_graph: bool = True,
        auto_reset: bool = True,
    ) -> None:
        if wp is None:
            raise RuntimeError("nvidia-warp (`warp-lang`) is required for backend='warp_slotted_fp32'.")
        import torch
        from uav_network.communication.gpu.mac_slotted_fp32 import mac_slotted_step_fp32
        from uav_network.communication.gpu.phy_sinr_fp32 import compute_phy_matrices_fp32, compute_phy_matrices_if_done_fp32
        from uav_network.environment.gpu.ds_aoi_belief_fp32 import ds_sense_and_energy_step_fp32
        from uav_network.environment.gpu.slotted_2d_warp_kernels import (
            assemble_reward_and_done_fp32,
            build_obs_state_mask_fp32,
            motion_2d_safety_fp32,
            selective_auto_reset_fp32,
        )

        self.torch = torch
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.aoi_decay_eta = float(aoi_decay_eta)
        self.weights = (float(w1_ds), float(w2_aoi), float(w3_prop_kj), float(w4_safety), float(w5_drop))
        self.base_seed = int(base_seed)
        self.belief_mode_int = 1 if belief_sharing_mode == "local_decpomdp" else 0
        self.torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.wp_device = str(self.torch_device)
        self.use_cuda_graph = bool(use_cuda_graph and self.torch_device.type == "cuda")
        self.auto_reset = bool(auto_reset)

        self._k_motion = motion_2d_safety_fp32
        self._k_phy = compute_phy_matrices_fp32
        self._k_phy_if_done = compute_phy_matrices_if_done_fp32
        self._k_ds = ds_sense_and_energy_step_fp32
        self._k_mac = mac_slotted_step_fp32
        self._k_rew = assemble_reward_and_done_fp32
        self._k_reset = selective_auto_reset_fp32
        self._k_obs = build_obs_state_mask_fp32

        wp.init()
        self._alloc_static_gpu_buffers()
        self._stream = torch.cuda.Stream(device=self.torch_device) if self.use_cuda_graph else None
        self._graph_runner = None

    def _alloc_static_gpu_buffers(self) -> None:
        torch = self.torch
        e, d = self.num_envs, self.torch_device
        # Static zero-copy I/O PyTorch tensors (fixed pointers for CUDA Graph!)
        self.act_cont_t = torch.zeros((e, NUM_UAVS, CONT_ACTION_DIM), dtype=torch.float32, device=d)
        self.act_disc_t = torch.zeros((e, NUM_UAVS), dtype=torch.int32, device=d)
        self.obs_t = torch.zeros((e, NUM_UAVS, OBS_DIM), dtype=torch.float32, device=d)
        self.state_t = torch.zeros((e, STATE_DIM), dtype=torch.float32, device=d)
        self.mask_t = torch.zeros((e, NUM_UAVS, DISC_ACTION_DIM), dtype=torch.uint8, device=d)
        self.rewards_t = torch.zeros((e,), dtype=torch.float32, device=d)
        self.dones_t = torch.zeros((e,), dtype=torch.float32, device=d)

        self.act_cont_wp = wp.from_torch(self.act_cont_t)
        self.act_disc_wp = wp.from_torch(self.act_disc_t)
        self.obs_wp = wp.from_torch(self.obs_t)
        self.state_wp = wp.from_torch(self.state_t)
        self.mask_wp = wp.from_torch(self.mask_t)
        self.rewards_wp = wp.from_torch(self.rewards_t)
        self.dones_wp = wp.from_torch(self.dones_t)
        self.terminated_t = torch.zeros((e,), dtype=torch.float32, device=d)
        self.terminated_wp = wp.from_torch(self.terminated_t)

        # Static Warp simulation buffers (100% FP32/INT32)
        wd = self.wp_device
        self.init_uav_pos = wp.zeros((e, NUM_UAVS), dtype=wp.vec3f, device=wd)
        self.uav_pos = wp.zeros((e, NUM_UAVS), dtype=wp.vec3f, device=wd)
        self.uav_vel = wp.zeros((e, NUM_UAVS), dtype=wp.vec3f, device=wd)
        self.node_pos = wp.zeros((e, NUM_NODES), dtype=wp.vec3f, device=wd)
        self.gcs_pos = wp.zeros((e,), dtype=wp.vec3f, device=wd)
        self.obstacles = wp.zeros((e, 22), dtype=wp.vec4f, device=wd)
        self.target_cells = wp.zeros((e, 25), dtype=wp.int32, device=wd)
        self.cell_has_target = wp.zeros((e, NUM_CELLS), dtype=wp.int32, device=wd)

        self.uav_battery_j = wp.full((e, NUM_UAVS), BATTERY_CAPACITY_J, dtype=wp.float32, device=wd)
        self.uav_alive = wp.ones((e, NUM_UAVS), dtype=wp.int32, device=wd)
        self.last_act_norm = wp.zeros((e, NUM_UAVS), dtype=wp.float32, device=wd)
        self.env_step_count = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.step_time_s = wp.zeros((e,), dtype=wp.float32, device=wd)

        self.dist_mat = wp.zeros((e, NUM_NODES, NUM_NODES), dtype=wp.float32, device=wd)
        self.los_mat = wp.ones((e, NUM_NODES, NUM_NODES), dtype=wp.int32, device=wd)
        self.rx_pwr_mat = wp.zeros((e, NUM_NODES, NUM_NODES), dtype=wp.float32, device=wd)
        self.cca_hear_mat = wp.zeros((e, NUM_NODES, NUM_NODES), dtype=wp.int32, device=wd)

        self.ds_pos_cnt = wp.zeros((e, NUM_CELLS), dtype=wp.int32, device=wd)
        self.ds_neg_cnt = wp.zeros((e, NUM_CELLS), dtype=wp.int32, device=wd)
        self.ds_m_f = wp.zeros((e, NUM_CELLS), dtype=wp.float32, device=wd)
        self.ds_m_u = wp.ones((e, NUM_CELLS), dtype=wp.float32, device=wd)
        self.cell_reported = wp.zeros((e, NUM_CELLS), dtype=wp.int32, device=wd)
        self.gcs_delivered = wp.zeros((e, NUM_CELLS), dtype=wp.int32, device=wd)
        self.gcs_true_aoi_s = wp.full((e, NUM_CELLS), -1.0, dtype=wp.float32, device=wd)
        self.ep_radio_energy_j = wp.zeros((e,), dtype=wp.float32, device=wd)

        # Private per-UAV local DS maps (~93.8 MiB at E=2048)
        self.local_ds_pos = wp.zeros((e, NUM_UAVS, NUM_CELLS), dtype=wp.int32, device=wd)
        self.local_ds_neg = wp.zeros((e, NUM_UAVS, NUM_CELLS), dtype=wp.int32, device=wd)
        self.local_ds_m_f = wp.zeros((e, NUM_UAVS, NUM_CELLS), dtype=wp.float32, device=wd)
        self.local_ds_m_u = wp.ones((e, NUM_UAVS, NUM_CELLS), dtype=wp.float32, device=wd)
        self.local_cell_reported = wp.zeros((e, NUM_UAVS, NUM_CELLS), dtype=wp.int32, device=wd)

        self.q_count = wp.zeros((e, NUM_UAVS), dtype=wp.int32, device=wd)
        self.q_cell_id = wp.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=wp.int32, device=wd)
        self.q_timestamp = wp.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=wp.float32, device=wd)
        self.q_retries = wp.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=wp.int32, device=wd)
        self.q_hops = wp.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=wp.int32, device=wd)
        self.q_visited_mask = wp.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=wp.int32, device=wd)

        rng_np = np.array([self.base_seed * 10007 + i * 7919 + 1 for i in range(e)], dtype=np.uint32)
        self.rng_states = wp.array(rng_np, dtype=wp.uint32, device=wd)

        # Step scratch & episode KPI accumulators
        self.step_safety_viol = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.step_delta_u = wp.zeros((e,), dtype=wp.float32, device=wd)
        self.step_prop_kj = wp.zeros((e,), dtype=wp.float32, device=wd)
        self.step_aoi_score = wp.zeros((e,), dtype=wp.float32, device=wd)
        self.step_new_gcs_deliv = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.step_false_gcs_deliv = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.step_multihop_deliv = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.step_collisions = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.step_hd_drops = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.step_retry_drops = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.step_ttl_expired = wp.zeros((e,), dtype=wp.int32, device=wd)

        self.ep_return = wp.zeros((e,), dtype=wp.float32, device=wd)
        self.ep_true_deliv = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.ep_false_deliv = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.ep_multihop = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.ep_collisions = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.ep_hd_drops = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.ep_retry_drops = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.ep_ttl_expired = wp.zeros((e,), dtype=wp.int32, device=wd)
        self.ep_prop_kj = wp.zeros((e,), dtype=wp.float32, device=wd)
        self.ep_safety_viol = wp.zeros((e,), dtype=wp.int32, device=wd)

    def _launch_step_pipeline(self) -> None:
        e, wd = self.num_envs, self.wp_device
        w1, w2, w3, w4, w5 = self.weights
        wp.launch(
            self._k_motion, dim=e, device=wd,
            inputs=[self.act_cont_wp, self.uav_pos, self.uav_vel, self.uav_alive, self.node_pos, self.gcs_pos,
                    self.obstacles, 22, self.last_act_norm, self.step_safety_viol, 1.0],
        )
        wp.launch(
            self._k_phy, dim=e, device=wd,
            inputs=[self.node_pos, self.obstacles, 22, self.dist_mat, self.los_mat, self.rx_pwr_mat, self.cca_hear_mat],
        )
        wp.launch(
            self._k_ds, dim=e, device=wd,
            inputs=[
                self.uav_pos, self.uav_vel, self.uav_battery_j, self.uav_alive, self.obstacles, 22,
                self.cell_has_target, self.ds_pos_cnt, self.ds_neg_cnt, self.ds_m_f, self.ds_m_u, self.cell_reported,
                self.local_ds_pos, self.local_ds_neg, self.local_ds_m_f, self.local_ds_m_u, self.local_cell_reported,
                self.belief_mode_int, self.q_count, self.q_cell_id, self.q_timestamp, self.q_retries, self.q_hops,
                self.q_visited_mask, self.step_time_s, self.rng_states, self.step_delta_u, self.step_prop_kj,
                SENSING_RADIUS_M, 1.0,
            ],
        )
        wp.launch(
            self._k_mac, dim=e, device=wd,
            inputs=[
                self.act_disc_wp, self.rx_pwr_mat, self.cca_hear_mat, self.uav_alive, self.step_time_s,
                self.q_count, self.q_cell_id, self.q_timestamp, self.q_retries, self.q_hops, self.q_visited_mask,
                self.gcs_delivered, self.gcs_true_aoi_s, self.uav_battery_j, self.ep_radio_energy_j,
                self.cell_has_target, self.rng_states,
                self.step_aoi_score, self.step_new_gcs_deliv, self.step_false_gcs_deliv, self.step_multihop_deliv,
                self.step_collisions, self.step_hd_drops, self.step_retry_drops, self.step_ttl_expired,
                self.aoi_decay_eta,
            ],
        )
        wp.launch(
            self._k_rew, dim=e, device=wd,
            inputs=[
                self.step_delta_u, self.step_aoi_score, self.step_prop_kj, self.step_safety_viol,
                self.step_retry_drops, self.step_ttl_expired, self.step_new_gcs_deliv, self.step_false_gcs_deliv,
                self.step_multihop_deliv, self.step_collisions, self.step_hd_drops, self.uav_alive,
                self.env_step_count, self.step_time_s, self.rewards_wp, self.dones_wp, self.terminated_wp,
                self.ep_return, self.ep_true_deliv, self.ep_false_deliv, self.ep_multihop, self.ep_collisions,
                self.ep_hd_drops, self.ep_retry_drops, self.ep_ttl_expired, self.ep_prop_kj, self.ep_safety_viol,
                w1, w2, w3, w4, w5, self.max_steps,
            ],
        )
        if self.auto_reset:
            wp.launch(
                self._k_reset, dim=e, device=wd,
            inputs=[
                self.dones_wp, self.init_uav_pos, self.uav_pos, self.uav_vel, self.uav_battery_j, self.uav_alive,
                self.node_pos, self.last_act_norm, self.env_step_count, self.ds_pos_cnt, self.ds_neg_cnt, self.ds_m_f, self.ds_m_u, self.cell_reported,
                self.gcs_delivered, self.gcs_true_aoi_s, self.ep_radio_energy_j,
                self.local_ds_pos, self.local_ds_neg, self.local_ds_m_f, self.local_ds_m_u,
                self.local_cell_reported, self.q_count, self.ep_return, self.ep_true_deliv, self.ep_false_deliv,
                self.ep_multihop, self.ep_collisions, self.ep_hd_drops, self.ep_retry_drops, self.ep_ttl_expired,
                self.ep_prop_kj, self.ep_safety_viol,
            ],
        )
            wp.launch(
                self._k_phy_if_done, dim=e, device=wd,
                inputs=[self.dones_wp, self.node_pos, self.obstacles, 22, self.dist_mat, self.los_mat, self.rx_pwr_mat, self.cca_hear_mat],
            )
        wp.launch(
            self._k_obs, dim=e, device=wd,
            inputs=[
                self.uav_pos, self.uav_vel, self.uav_battery_j, self.uav_alive, self.gcs_pos, self.obstacles, 22,
                self.dist_mat, self.los_mat, self.q_count, self.q_timestamp, self.q_hops, self.q_visited_mask,
                self.ds_m_f, self.ds_m_u, self.local_ds_m_f, self.local_ds_m_u, self.belief_mode_int,
                self.target_cells, self.cell_reported, self.gcs_delivered, self.last_act_norm,
                self.env_step_count, self.step_time_s, self.max_steps,
                self.obs_wp, self.state_wp, self.mask_wp,
            ],
        )

    def reset(self, seeds: List[int] | None = None, _reuse_snapshots: bool = False):
        import torch
        e = self.num_envs
        if not _reuse_snapshots or not getattr(self, "snapshots", None):
            snaps = [generate_scenario_snapshot(self.base_seed + i if seeds is None else int(seeds[i])) for i in range(e)]
            self.snapshots = snaps
            obs_arr = np.stack([s.obstacles for s in snaps], axis=0)
            tgt_arr = np.stack([s.target_cells for s in snaps], axis=0)
            cht_arr = np.stack([s.cell_has_target for s in snaps], axis=0)
            pos_arr = np.stack([s.init_uav_pos for s in snaps], axis=0)
            gcs_arr = np.stack([s.gcs_pos for s in snaps], axis=0)
            with (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.torch_device))) if self.torch_device.type == "cuda" else nullcontext()):
                wp.copy(self.obstacles, wp.array(obs_arr, dtype=wp.vec4f, device=self.wp_device))
                wp.copy(self.target_cells, wp.array(tgt_arr, dtype=wp.int32, device=self.wp_device))
                wp.copy(self.cell_has_target, wp.array(cht_arr, dtype=wp.int32, device=self.wp_device))
                wp.copy(self.init_uav_pos, wp.array(pos_arr, dtype=wp.vec3f, device=self.wp_device))
                wp.copy(self.gcs_pos, wp.array(gcs_arr, dtype=wp.vec3f, device=self.wp_device))

        with (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.torch_device))) if self.torch_device.type == "cuda" else nullcontext()):
            self.dones_t.fill_(1.0)
            self.terminated_t.zero_()
            self.step_time_s.zero_()
            wp.launch(
                self._k_reset, dim=e, device=self.wp_device,
                inputs=[
                    self.dones_wp, self.init_uav_pos, self.uav_pos, self.uav_vel, self.uav_battery_j, self.uav_alive,
                    self.node_pos, self.last_act_norm, self.env_step_count, self.ds_pos_cnt, self.ds_neg_cnt, self.ds_m_f, self.ds_m_u, self.cell_reported,
                    self.gcs_delivered, self.gcs_true_aoi_s, self.ep_radio_energy_j,
                    self.local_ds_pos, self.local_ds_neg, self.local_ds_m_f, self.local_ds_m_u,
                    self.local_cell_reported, self.q_count, self.ep_return, self.ep_true_deliv, self.ep_false_deliv,
                    self.ep_multihop, self.ep_collisions, self.ep_hd_drops, self.ep_retry_drops, self.ep_ttl_expired,
                    self.ep_prop_kj, self.ep_safety_viol,
                ],
            )
            self.dones_t.zero_()
            self.act_cont_t.zero_()
            wp.launch(
                self._k_motion, dim=e, device=self.wp_device,
                inputs=[self.act_cont_wp, self.uav_pos, self.uav_vel, self.uav_alive, self.node_pos, self.gcs_pos,
                        self.obstacles, 22, self.last_act_norm, self.step_safety_viol, 0.0],
            )
            wp.launch(
                self._k_phy, dim=e, device=self.wp_device,
                inputs=[self.node_pos, self.obstacles, 22, self.dist_mat, self.los_mat, self.rx_pwr_mat, self.cca_hear_mat],
            )
            wp.launch(
                self._k_obs, dim=e, device=self.wp_device,
                inputs=[
                    self.uav_pos, self.uav_vel, self.uav_battery_j, self.uav_alive, self.gcs_pos, self.obstacles, 22,
                    self.dist_mat, self.los_mat, self.q_count, self.q_timestamp, self.q_hops, self.q_visited_mask,
                    self.ds_m_f, self.ds_m_u, self.local_ds_m_f, self.local_ds_m_u, self.belief_mode_int,
                    self.target_cells, self.cell_reported, self.gcs_delivered, self.last_act_norm,
                    self.env_step_count, self.step_time_s, self.max_steps,
                    self.obs_wp, self.state_wp, self.mask_wp,
                ],
            )
        if self.use_cuda_graph and self._graph_runner is None:
            self._graph_runner = WarpSlottedUavEnvGraph(self._launch_step_pipeline, self._stream)
            self._graph_runner.capture()
            return self.reset(seeds=seeds, _reuse_snapshots=True)
        return self.obs_t, self.state_t, self.mask_t

    def step_torch(self, act_cont_t, act_disc_t, compute_reward_info: bool = True):
        """Zero-copy GPU step accepting PyTorch CUDA tensors and returning PyTorch CUDA tensors."""
        self.act_cont_t.copy_(act_cont_t, non_blocking=True)
        self.act_disc_t.copy_(act_disc_t, non_blocking=True)
        if self._graph_runner is not None:
            self._graph_runner.launch()
        else:
            self._launch_step_pipeline()
        if not compute_reward_info:
            return self.obs_t, self.state_t, self.mask_t, self.rewards_t, self.dones_t, {"terminated": self.terminated_t}
        w1, w2, w3, w4, w5 = self.weights
        to_t = wp.to_torch
        drop_events = to_t(self.step_retry_drops) + to_t(self.step_ttl_expired)
        info = {
            "r_ds": w1 * to_t(self.step_delta_u),
            "r_gcs_aoi": w2 * to_t(self.step_aoi_score),
            "r_propulsion": -w3 * to_t(self.step_prop_kj),
            "r_safety": -w4 * to_t(self.step_safety_viol).float(),
            "r_drop_ttl": -w5 * drop_events.float(),
            "terminated": self.terminated_t,
        }
        return self.obs_t, self.state_t, self.mask_t, self.rewards_t, self.dones_t, info


class Slotted2DUavEnv:
    """Unified Environment Wrapper supporting BOTH `backend='warp_slotted_fp32'` (GPU) and `backend='numpy'` (CPU)."""

    def __init__(
        self,
        num_envs: int = 1,
        max_steps: int = MAX_STEPS_DEFAULT,
        aoi_decay_eta: float = 0.02,
        w1_ds: float = 0.25,
        w2_aoi: float = 10.0,
        w3_prop_kj: float = 0.03,
        w4_safety: float = 20.0,
        w5_drop: float = 1.0,
        base_seed: int = 44,
        backend: str = "numpy",
        belief_sharing_mode: str = "local_decpomdp",
        device: str = "cuda:0",
        use_cuda_graph: bool = True,
        gpu_autoreset: bool = True,
    ) -> None:
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.aoi_decay_eta = float(aoi_decay_eta)
        self.w1_ds = float(w1_ds)
        self.w2_aoi = float(w2_aoi)
        self.w3_prop_kj = float(w3_prop_kj)
        self.w4_safety = float(w4_safety)
        self.w5_drop = float(w5_drop)
        self.base_seed = int(base_seed)
        self.backend = backend
        self.belief_sharing_mode = belief_sharing_mode

        self._gpu_backend = None
        if backend in ("warp", "warp_slotted_fp32"):
            self._gpu_backend = WarpSlotted2DBackend(
                num_envs=num_envs, max_steps=max_steps, aoi_decay_eta=aoi_decay_eta,
                w1_ds=w1_ds, w2_aoi=w2_aoi, w3_prop_kj=w3_prop_kj, w4_safety=w4_safety, w5_drop=w5_drop,
                base_seed=base_seed, belief_sharing_mode=belief_sharing_mode,
                device=device, use_cuda_graph=use_cuda_graph, auto_reset=gpu_autoreset,
            )
            return

        self.snapshots: List[ScenarioSnapshot] = []
        self._rngs: List[np.random.Generator] = []
        self._alloc_state()

    def _alloc_state(self) -> None:
        e = self.num_envs
        self.step_count = np.zeros((e,), dtype=np.int32)
        self.uav_pos = np.zeros((e, NUM_UAVS, 3), dtype=np.float32)
        self.uav_vel = np.zeros((e, NUM_UAVS, 2), dtype=np.float32)
        self.uav_battery_j = np.full((e, NUM_UAVS), BATTERY_CAPACITY_J, dtype=np.float64)
        self.uav_alive = np.ones((e, NUM_UAVS), dtype=np.int32)

        self.dist_mat = np.zeros((e, NUM_NODES, NUM_NODES), dtype=np.float32)
        self.los_mat = np.ones((e, NUM_NODES, NUM_NODES), dtype=np.int32)
        self.rx_pwr_mat = np.zeros((e, NUM_NODES, NUM_NODES), dtype=np.float32)
        self.cca_hear_mat = np.zeros((e, NUM_NODES, NUM_NODES), dtype=np.int32)

        self.ds_pos_cnt = np.zeros((e, NUM_CELLS), dtype=np.int32)
        self.ds_neg_cnt = np.zeros((e, NUM_CELLS), dtype=np.int32)
        self.ds_m_f = np.zeros((e, NUM_CELLS), dtype=np.float32)
        self.ds_m_u = np.ones((e, NUM_CELLS), dtype=np.float32)
        self.cell_reported = np.zeros((e, NUM_CELLS), dtype=np.int32)
        self.gcs_delivered = np.zeros((e, NUM_CELLS), dtype=np.int32)

        # Private per-UAV local DS maps (Gap C parity)
        self.local_ds_pos = np.zeros((e, NUM_UAVS, NUM_CELLS), dtype=np.int32)
        self.local_ds_neg = np.zeros((e, NUM_UAVS, NUM_CELLS), dtype=np.int32)
        self.local_ds_m_f = np.zeros((e, NUM_UAVS, NUM_CELLS), dtype=np.float32)
        self.local_ds_m_u = np.ones((e, NUM_UAVS, NUM_CELLS), dtype=np.float32)
        self.local_cell_reported = np.zeros((e, NUM_UAVS, NUM_CELLS), dtype=np.int32)

        self.q_count = np.zeros((e, NUM_UAVS), dtype=np.int32)
        self.q_cell_id = np.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=np.int32)
        self.q_timestamp = np.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=np.float32)
        self.q_retries = np.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=np.int32)
        self.q_hops = np.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=np.int32)
        self.q_visited_mask = np.zeros((e, NUM_UAVS, MAX_QUEUE), dtype=np.int32)

        self.last_ack = np.zeros((e, NUM_UAVS), dtype=np.float32)
        self.last_act_norm = np.zeros((e, NUM_UAVS), dtype=np.float32)
        self.channel_busy_ewma = np.zeros((e, NUM_UAVS), dtype=np.float32)

        self.ep_return = np.zeros((e,), dtype=np.float32)
        self.ep_true_deliv = np.zeros((e,), dtype=np.int32)
        self.ep_false_deliv = np.zeros((e,), dtype=np.int32)
        self.ep_multihop = np.zeros((e,), dtype=np.int32)
        self.ep_collisions = np.zeros((e,), dtype=np.int32)
        self.ep_hd_drops = np.zeros((e,), dtype=np.int32)
        self.ep_retry_drops = np.zeros((e,), dtype=np.int32)
        self.ep_ttl_expired = np.zeros((e,), dtype=np.int32)
        self.ep_prop_kj = np.zeros((e,), dtype=np.float32)
        self.ep_safety_viol = np.zeros((e,), dtype=np.int32)
        self.ep_aoi_samples: List[List[float]] = [[] for _ in range(e)]
        self.ep_tx_attempts = np.zeros((e,), dtype=np.int32)
        self.ep_tx_success = np.zeros((e,), dtype=np.int32)
        self.ep_network_j = np.zeros((e,), dtype=np.float32)

    def reset(self, seeds: List[int] | None = None):
        if self._gpu_backend is not None:
            return self._gpu_backend.reset(seeds=seeds)
        self._alloc_state()
        self.snapshots = []
        self._rngs = []
        for idx in range(self.num_envs):
            s = self.base_seed + idx if seeds is None else int(seeds[idx])
            snap = generate_scenario_snapshot(s)
            self.snapshots.append(snap)
            self._rngs.append(np.random.default_rng(s + 100003))
            self.uav_pos[idx] = snap.init_uav_pos
            self._update_phy_env(idx)
        return self._build_outputs()

    def sync_gpu_stats_to_host(self):
        """Snapshot Warp episode KPIs for validation, never during training."""
        if self._gpu_backend is None:
            return
        g = self._gpu_backend
        import warp as wp
        self.snapshots = g.snapshots
        for name in ("ep_return", "ep_true_deliv", "ep_false_deliv", "ep_multihop",
                     "ep_collisions", "ep_hd_drops", "ep_retry_drops", "ep_ttl_expired",
                     "ep_prop_kj", "ep_safety_viol", "ep_radio_energy_j",
                     "cell_reported", "ds_pos_cnt", "ds_neg_cnt"):
            source = getattr(g, name)
            host = wp.to_torch(source).detach().cpu().numpy().copy()
            setattr(self, "ep_network_j" if name == "ep_radio_energy_j" else name, host)
        ages = wp.to_torch(g.gcs_true_aoi_s).detach().cpu().numpy()
        self.ep_aoi_samples = [list(float(x) for x in row if x >= 0.0) for row in ages]

    def reset_indices(self, indices, seeds=None):
        """Reset only selected CPU environments; leave other worlds and RNG untouched.

        Warp's in-graph autoreset is handled by the GPU backend separately.
        """
        if self._gpu_backend is not None:
            raise RuntimeError("GPU backend performs selective autoreset inside Warp; do not reset_indices.")
        indices = [int(i) for i in indices]
        if len(indices) != len(set(indices)) or any(i < 0 or i >= self.num_envs for i in indices):
            raise ValueError("Invalid or duplicate environment indices.")
        if seeds is not None and len(seeds) != len(indices):
            raise ValueError("Need exactly one seed per reset environment.")
        for j, idx in enumerate(indices):
            next_seed = int(seeds[j]) if seeds is not None else self.base_seed + idx
            single = type(self)(
                num_envs=1, max_steps=self.max_steps, aoi_decay_eta=self.aoi_decay_eta,
                w1_ds=self.w1_ds, w2_aoi=self.w2_aoi, w3_prop_kj=self.w3_prop_kj,
                w4_safety=self.w4_safety, w5_drop=self.w5_drop, base_seed=next_seed,
                backend="numpy", belief_sharing_mode=self.belief_sharing_mode,
            )
            single.reset(seeds=[next_seed])
            for name, current in vars(self).items():
                candidate = getattr(single, name, None)
                if (isinstance(current, np.ndarray) and isinstance(candidate, np.ndarray)
                        and current.shape[0] == self.num_envs and candidate.shape == (1, *current.shape[1:])):
                    current[idx] = candidate[0]
            self.snapshots[idx] = single.snapshots[0]
            self._rngs[idx] = single._rngs[0]
            self.ep_aoi_samples[idx] = []
        return self._build_outputs()

    def _update_phy_env(self, e: int) -> None:
        snap = self.snapshots[e]
        nodes = np.vstack([self.uav_pos[e], snap.gcs_pos[None, :]])
        for i in range(NUM_NODES):
            self.dist_mat[e, i, i] = 0.0
            self.los_mat[e, i, i] = 1
            self.rx_pwr_mat[e, i, i] = TX_POWER_W
            self.cca_hear_mat[e, i, i] = 1
            for j in range(i + 1, NUM_NODES):
                d = float(np.linalg.norm(nodes[i] - nodes[j]))
                los = check_los_2d(float(nodes[i, 0]), float(nodes[i, 1]), float(nodes[j, 0]), float(nodes[j, 1]), snap.obstacles)
                prx = rx_power_w(d, los)
                hear = 1 if prx >= CCA_THRESHOLD_W else 0
                self.dist_mat[e, i, j] = self.dist_mat[e, j, i] = d
                self.los_mat[e, i, j] = self.los_mat[e, j, i] = los
                self.rx_pwr_mat[e, i, j] = self.rx_pwr_mat[e, j, i] = prx
                self.cca_hear_mat[e, i, j] = self.cca_hear_mat[e, j, i] = hear

    def _build_outputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_all = np.zeros((self.num_envs, NUM_UAVS, OBS_DIM), dtype=np.float32)
        state_all = np.zeros((self.num_envs, STATE_DIM), dtype=np.float32)
        mask_all = np.zeros((self.num_envs, NUM_UAVS, DISC_ACTION_DIM), dtype=np.uint8)
        for e in range(self.num_envs):
            snap = self.snapshots[e]
            if self.belief_sharing_mode == "local_decpomdp":
                ds_f_u = self.local_ds_m_f[e]
                ds_u_u = self.local_ds_m_u[e]
            else:
                ds_f_u = np.broadcast_to(self.ds_m_f[e][None, :], (NUM_UAVS, NUM_CELLS))
                ds_u_u = np.broadcast_to(self.ds_m_u[e][None, :], (NUM_UAVS, NUM_CELLS))
            o, m = build_local_obs_and_mask(
                self.uav_pos[e], self.uav_vel[e], self.uav_battery_j[e], self.uav_alive[e],
                snap.gcs_pos, snap.obstacles, self.dist_mat[e], self.los_mat[e],
                self.q_count[e], self.q_timestamp[e], self.q_hops[e], self.q_visited_mask[e],
                ds_f_u, ds_u_u, self.last_ack[e], self.channel_busy_ewma[e],
                int(self.step_count[e]), self.max_steps,
            )
            pdr = float(self.ep_tx_success[e]) / float(max(int(self.ep_tx_attempts[e]), 1))
            mean_aoi = (float(np.mean(self.ep_aoi_samples[e])) / REPORT_TTL_S) if self.ep_aoi_samples[e] else 0.0
            st = build_global_state(
                self.uav_pos[e], self.uav_vel[e], self.uav_battery_j[e], self.uav_alive[e],
                self.dist_mat[e], self.los_mat[e], self.q_count[e], self.q_timestamp[e], self.q_hops[e],
                self.ds_m_f[e], self.ds_m_u[e], snap.target_cells, self.cell_reported[e], self.gcs_delivered[e],
                self.last_ack[e], self.last_act_norm[e], int(self.step_count[e]),
                pdr, mean_aoi, float(self.channel_busy_ewma[e].mean()), self.max_steps,
            )
            obs_all[e], state_all[e], mask_all[e] = o, st, m
        return obs_all, state_all, mask_all

    def step(self, act_cont, act_disc_idx, active_mask=None):
        if np.shape(act_cont) != (self.num_envs, NUM_UAVS, CONT_ACTION_DIM):
            raise ValueError(f"Continuous action must have shape {(self.num_envs, NUM_UAVS, CONT_ACTION_DIM)}, got {np.shape(act_cont)}")
        if np.shape(act_disc_idx) != (self.num_envs, NUM_UAVS):
            raise ValueError(f"Discrete action must have shape {(self.num_envs, NUM_UAVS)}, got {np.shape(act_disc_idx)}")
        if self._gpu_backend is not None:
            import torch
            dev = self._gpu_backend.torch_device
            ac = torch.as_tensor(act_cont, dtype=torch.float32, device=dev)
            ad = torch.as_tensor(act_disc_idx, dtype=torch.int32, device=dev)
            if active_mask is not None and not bool(np.asarray(active_mask).all()):
                raise ValueError("GPU backend does not yet support an external active_mask.")
            return self._gpu_backend.step_torch(ac, ad)

        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.float32)
        info_r_ds = np.zeros((self.num_envs,), dtype=np.float32)
        info_r_aoi = np.zeros((self.num_envs,), dtype=np.float32)
        info_r_prop = np.zeros((self.num_envs,), dtype=np.float32)
        info_r_safe = np.zeros((self.num_envs,), dtype=np.float32)
        info_r_drop = np.zeros((self.num_envs,), dtype=np.float32)

        for e in range(self.num_envs):
            if active_mask is not None and not bool(active_mask[e]):
                dones[e] = 1.0
                continue
            snap = self.snapshots[e]
            rng = self._rngs[e]
            self.step_count[e] += 1
            now_s = float(self.step_count[e])

            safety_viol = 0
            prop_j = 0.0
            for u in range(NUM_UAVS):
                if self.uav_alive[e, u] == 0:
                    continue
                ax = float(np.clip(act_cont[e, u, 0], -1.0, 1.0)) * MAX_ACCEL_MPS2
                ay = float(np.clip(act_cont[e, u, 1], -1.0, 1.0)) * MAX_ACCEL_MPS2
                self.last_act_norm[e, u] = math.hypot(ax, ay) / (MAX_ACCEL_MPS2 * math.sqrt(2.0))
                vx = float(self.uav_vel[e, u, 0]) + ax
                vy = float(self.uav_vel[e, u, 1]) + ay
                spd = math.hypot(vx, vy)
                if spd > MAX_SPEED_MPS:
                    vx *= MAX_SPEED_MPS / spd
                    vy *= MAX_SPEED_MPS / spd
                nx = float(self.uav_pos[e, u, 0]) + vx
                ny = float(self.uav_pos[e, u, 1]) + vy

                if nx < 5.0 or nx > MAP_SIZE_M - 5.0 or ny < 5.0 or ny > MAP_SIZE_M - 5.0:
                    safety_viol += 1
                    nx = float(np.clip(nx, 5.0, MAP_SIZE_M - 5.0))
                    ny = float(np.clip(ny, 5.0, MAP_SIZE_M - 5.0))
                    vx, vy = 0.0, 0.0
                for k in range(snap.obstacles.shape[0]):
                    ox, oy, orad = float(snap.obstacles[k, 0]), float(snap.obstacles[k, 1]), float(snap.obstacles[k, 2])
                    d_ob = math.hypot(nx - ox, ny - oy)
                    if d_ob < orad + 4.0:
                        safety_viol += 1
                        scale = (orad + 5.0) / max(d_ob, 1e-4)
                        nx = ox + (nx - ox) * scale
                        ny = oy + (ny - oy) * scale
                        vx, vy = 0.0, 0.0

                self.uav_pos[e, u, 0], self.uav_pos[e, u, 1] = nx, ny
                self.uav_vel[e, u, 0], self.uav_vel[e, u, 1] = vx, vy
                p_w = zeng_rotary_power_w(math.hypot(vx, vy))
                prop_j += p_w
                rem = max(0.0, float(self.uav_battery_j[e, u]) - p_w)
                self.uav_battery_j[e, u] = rem
                if rem <= 0.0:
                    self.uav_alive[e, u] = 0

            # Resolve every pair after motion: a safety penalty alone is not a collision avoidance policy.
            # The +0.01m clearance avoids float32 rounding just under SAFETY_DIST_M.
            for i in range(NUM_UAVS):
                for j in range(i + 1, NUM_UAVS):
                    if self.uav_alive[e, i] and self.uav_alive[e, j]:
                        dx = float(self.uav_pos[e, j, 0] - self.uav_pos[e, i, 0])
                        dy = float(self.uav_pos[e, j, 1] - self.uav_pos[e, i, 1])
                        dist = math.hypot(dx, dy)
                        if dist < SAFETY_DIST_M:
                            safety_viol += 1
                            if dist < 1.0e-7:
                                dx, dy, dist = 1.0, 0.0, 1.0
                            target_x = float(self.uav_pos[e, i, 0]) + (SAFETY_DIST_M + 0.01) * dx / dist
                            target_y = float(self.uav_pos[e, i, 1]) + (SAFETY_DIST_M + 0.01) * dy / dist
                            self.uav_pos[e, j, 0] = np.clip(target_x, 5.0, MAP_SIZE_M - 5.0)
                            self.uav_pos[e, j, 1] = np.clip(target_y, 5.0, MAP_SIZE_M - 5.0)
                            self.uav_vel[e, j, :2] = 0.0

            self._update_phy_env(e)

            delta_u = 0.0
            for u in range(NUM_UAVS):
                if self.uav_alive[e, u] == 0:
                    continue
                ux, uy = float(self.uav_pos[e, u, 0]), float(self.uav_pos[e, u, 1])
                cx0 = max(0, int((ux - SENSING_RADIUS_M) / CELL_SIZE_M))
                cx1 = min(GRID_W - 1, int((ux + SENSING_RADIUS_M) / CELL_SIZE_M))
                cy0 = max(0, int((uy - SENSING_RADIUS_M) / CELL_SIZE_M))
                cy1 = min(GRID_H - 1, int((uy + SENSING_RADIUS_M) / CELL_SIZE_M))
                for cy in range(cy0, cy1 + 1):
                    cell_y = (cy + 0.5) * CELL_SIZE_M
                    for cx in range(cx0, cx1 + 1):
                        cell_x = (cx + 0.5) * CELL_SIZE_M
                        if math.hypot(cell_x - ux, cell_y - uy) > SENSING_RADIUS_M:
                            continue
                        if check_los_2d(ux, uy, cell_x, cell_y, snap.obstacles) == 0:
                            continue
                        cid = cy * GRID_W + cx
                        has_t = snap.cell_has_target[cid] >= 0
                        detected = (rng.random() < 0.82) if has_t else (rng.random() < 0.08)
                        if detected:
                            self.ds_pos_cnt[e, cid] += 1
                            self.local_ds_pos[e, u, cid] += 1
                        else:
                            self.ds_neg_cnt[e, cid] += 1
                            self.local_ds_neg[e, u, cid] += 1

                        old_u = float(self.ds_m_u[e, cid])
                        t_mf, _, t_mu = ds_closed_form(int(self.ds_pos_cnt[e, cid]), int(self.ds_neg_cnt[e, cid]))
                        self.ds_m_f[e, cid] = t_mf
                        self.ds_m_u[e, cid] = t_mu
                        if old_u > t_mu:
                            delta_u += (old_u - t_mu)

                        l_mf, _, l_mu = ds_closed_form(int(self.local_ds_pos[e, u, cid]), int(self.local_ds_neg[e, u, cid]))
                        self.local_ds_m_f[e, u, cid] = l_mf
                        self.local_ds_m_u[e, u, cid] = l_mu

                        use_mf, use_mu = (l_mf, l_mu) if self.belief_sharing_mode == "local_decpomdp" else (t_mf, t_mu)
                        already_rep = (
                            self.local_cell_reported[e, u, cid]
                            if self.belief_sharing_mode == "local_decpomdp"
                            else self.cell_reported[e, cid]
                        )
                        if detected and already_rep == 0 and use_mf >= 0.90 and use_mu <= 0.15:
                            cnt = int(self.q_count[e, u])
                            if cnt < MAX_QUEUE:
                                self.local_cell_reported[e, u, cid] = 1
                                self.cell_reported[e, cid] = 1
                                self.q_cell_id[e, u, cnt] = cid
                                self.q_timestamp[e, u, cnt] = now_s
                                self.q_retries[e, u, cnt] = 0
                                self.q_hops[e, u, cnt] = 0
                                self.q_visited_mask[e, u, cnt] = 1 << u
                                self.q_count[e, u] = cnt + 1

            aoi_score, n_drops = self._mac_slotted_step_env(e, act_disc_idx[e], now_s, rng)
            prop_kj = prop_j * 0.001
            r_ds = self.w1_ds * delta_u
            r_aoi = self.w2_aoi * aoi_score
            r_prop = -self.w3_prop_kj * prop_kj
            r_safe = -self.w4_safety * float(safety_viol)
            r_drop = -self.w5_drop * float(n_drops)
            step_r = r_ds + r_aoi + r_prop + r_safe + r_drop

            rewards[e] = step_r
            info_r_ds[e], info_r_aoi[e], info_r_prop[e], info_r_safe[e], info_r_drop[e] = (
                r_ds, r_aoi, r_prop, r_safe, r_drop
            )
            self.ep_return[e] += step_r
            self.ep_prop_kj[e] += prop_kj
            self.ep_safety_viol[e] += safety_viol
            if self.step_count[e] >= self.max_steps or int(self.uav_alive[e].sum()) == 0:
                dones[e] = 1.0

        obs, state, mask = self._build_outputs()
        infos = {
            "r_ds": info_r_ds, "r_gcs_aoi": info_r_aoi, "r_propulsion": info_r_prop,
            "r_safety": info_r_safe, "r_drop_ttl": info_r_drop,
        }
        return obs, state, mask, rewards, dones, infos

    def _spend_radio_energy(self, e: int, u: int, dst: int) -> None:
        airtime_s = 0.004328
        e_tx = (0.1 / 0.35 + 0.1) * airtime_s
        self.uav_battery_j[e, u] = max(0.0, float(self.uav_battery_j[e, u]) - e_tx)
        self.ep_network_j[e] += e_tx
        if 0 <= dst < NUM_UAVS:
            e_rx = 0.1 * airtime_s
            self.uav_battery_j[e, dst] = max(0.0, float(self.uav_battery_j[e, dst]) - e_rx)
            self.ep_network_j[e] += e_rx

    def _pop_queue_head(self, e: int, u: int) -> None:
        cnt = int(self.q_count[e, u])
        if cnt > 1:
            self.q_cell_id[e, u, : cnt - 1] = self.q_cell_id[e, u, 1:cnt]
            self.q_timestamp[e, u, : cnt - 1] = self.q_timestamp[e, u, 1:cnt]
            self.q_retries[e, u, : cnt - 1] = self.q_retries[e, u, 1:cnt]
            self.q_hops[e, u, : cnt - 1] = self.q_hops[e, u, 1:cnt]
            self.q_visited_mask[e, u, : cnt - 1] = self.q_visited_mask[e, u, 1:cnt]
        self.q_count[e, u] = max(0, cnt - 1)

    def _mac_slotted_step_env(
        self, e: int, dest_act: np.ndarray, now_s: float, rng: np.random.Generator
    ) -> Tuple[float, int]:
        snap = self.snapshots[e]
        aoi_score = 0.0
        drops = 0
        self.last_ack[e, :] = 0.0

        for u in range(NUM_UAVS):
            while self.q_count[e, u] > 0 and (now_s - float(self.q_timestamp[e, u, 0])) > REPORT_TTL_S:
                self._pop_queue_head(e, u)
                self.ep_ttl_expired[e] += 1
                drops += 1

        chosen_slot = np.full((NUM_UAVS,), -1, dtype=np.int32)
        chosen_dst = np.full((NUM_UAVS,), -1, dtype=np.int32)
        for u in range(NUM_UAVS):
            da = int(dest_act[u])
            if self.uav_alive[e, u] == 1 and self.q_count[e, u] > 0 and da > 0:
                dst = GCS_IDX if da == GCS_IDX else (da - 1 if (da - 1) < u else da)
                if dst < GCS_IDX and ((int(self.q_visited_mask[e, u, 0]) >> dst) & 1):
                    continue
                chosen_dst[u] = dst
                chosen_slot[u] = int(rng.integers(0, 4))

        for m in range(4):
            tx_active = np.zeros((NUM_UAVS,), dtype=np.int32)
            for u in range(NUM_UAVS):
                if chosen_slot[u] == m and chosen_dst[u] >= 0:
                    deferred = False
                    for prev in range(u):
                        if tx_active[prev] == 1 and self.cca_hear_mat[e, prev, u] == 1:
                            # Busy in final subslot means defer to next tick, never force a TX.
                            deferred = True
                            chosen_slot[u] = m + 1 if m < 3 else -1
                            break
                    if not deferred:
                        tx_active[u] = 1
            busy = float(tx_active.sum() > 0)
            self.channel_busy_ewma[e] = 0.8 * self.channel_busy_ewma[e] + 0.2 * busy

            for u in range(NUM_UAVS):
                if tx_active[u] == 0:
                    continue
                dst = int(chosen_dst[u])
                self.ep_tx_attempts[e] += 1
                self._spend_radio_energy(e, u, dst)
                if dst < GCS_IDX and tx_active[dst] == 1:
                    self.ep_hd_drops[e] += 1
                    self.q_retries[e, u, 0] += 1
                else:
                    sig = float(self.rx_pwr_mat[e, u, dst])
                    interf = sum(float(self.rx_pwr_mat[e, o, dst]) for o in range(NUM_UAVS) if o != u and tx_active[o] == 1)
                    sinr = sig / (NOISE_FLOOR_W + interf)
                    if sinr >= SINR_THRESHOLD_LIN:
                        cid = int(self.q_cell_id[e, u, 0])
                        ts = float(self.q_timestamp[e, u, 0])
                        hops = int(self.q_hops[e, u, 0]) + 1
                        vmask = int(self.q_visited_mask[e, u, 0]) | (1 << u)
                        if dst == GCS_IDX:
                            self._pop_queue_head(e, u)
                            self.last_ack[e, u] = 1.0
                            self.ep_tx_success[e] += 1
                            if self.gcs_delivered[e, cid] == 0:
                                self.gcs_delivered[e, cid] = 1
                                if snap.cell_has_target[cid] >= 0:
                                    age = max(0.0, now_s - ts)
                                    aoi_score += math.exp(-self.aoi_decay_eta * age)
                                    self.ep_true_deliv[e] += 1
                                    self.ep_aoi_samples[e].append(age)
                                    if hops >= 2:
                                        self.ep_multihop[e] += 1
                                else:
                                    self.ep_false_deliv[e] += 1
                        else:
                            d_cnt = int(self.q_count[e, dst])
                            if d_cnt < MAX_QUEUE:
                                self._pop_queue_head(e, u)
                                self.last_ack[e, u] = 1.0
                                self.ep_tx_success[e] += 1
                                self.local_cell_reported[e, dst, cid] = 1
                                self.q_cell_id[e, dst, d_cnt] = cid
                                self.q_timestamp[e, dst, d_cnt] = ts
                                self.q_retries[e, dst, d_cnt] = 0
                                self.q_hops[e, dst, d_cnt] = hops
                                self.q_visited_mask[e, dst, d_cnt] = vmask | (1 << dst)
                                self.q_count[e, dst] = d_cnt + 1
                            else:
                                self.q_retries[e, u, 0] += 1
                    else:
                        if interf > 0.0:
                            self.ep_collisions[e] += 1
                        self.q_retries[e, u, 0] += 1

                if self.q_count[e, u] > 0 and self.q_retries[e, u, 0] >= 5:
                    self._pop_queue_head(e, u)
                    self.ep_retry_drops[e] += 1
                    drops += 1
        return aoi_score, drops
