"""v5.2 numerical reference and 7-node oracle connectivity diagnostic.

This isolated audit correction is NOT a packet-level UavNetSim or Warp environment.
The diagnostic uses a global instantaneous connectivity graph and receiver-ledger
oracle, one deterministic sender sweep per second, and analytic frame airtime.
It does not simulate interference/SINR, CSMA/CA, event clocks, queues, TTL,
retries, serialized packets, path-vector routing, actor observations, or training.
Packing, queue capacity and GPU memory helpers are independent calculations only.
DS numerics, receiver-age reporting and per-UAV steady-flight/TX energy accounting
are corrected here; physical acceleration constraints remain unimplemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Topology & Clock Constants
# ---------------------------------------------------------------------------
ACTOR_N: int = 6
RADIO_N: int = 7
GCS_ID: int = 6
DISC_DIM: int = 7
NODE_MASK_7BIT: int = 0x7F
GRID_W: int = 20
GRID_H: int = 20
NUM_CELLS: int = GRID_W * GRID_H  # 400
CELL_SIZE_M: float = 50.0
MAP_SIZE_M: float = 1000.0

# UavNetSim header convention (verified in UavNetSim v1.0.1 & master config.py)
IP_HEADER_B: int = 20
MAC_HEADER_B: int = 14
PHY_HEADER_B: int = 24
LOWER_HEADERS_B: int = IP_HEADER_B + MAC_HEADER_B + PHY_HEADER_B  # 58 B
APP_HEADER_B: int = 12
ENTRY_WIRE_B: int = 8  # Matches 2 x uint32 (8 B/entry)
MAX_ENTRIES_PER_PKT: int = 62
QUEUE_CAP_PKTS: int = 64  # > 60 worst-case steady-state TTL packets

# Int32 microsecond clock for H=1500 s safety ceiling
SAFETY_HORIZON_S: int = 1500
SAFETY_HORIZON_US: int = SAFETY_HORIZON_S * 1_000_000  # 1,500,000,000 us
PACKET_TTL_S: int = 10
PACKET_TTL_US: int = PACKET_TTL_S * 1_000_000
INF_TIME_US: int = 2_000_000_000
INT32_MAX: int = 2_147_483_647

# Canonical 5-building + 2-blackout urban layout (restoring v5.0 canonical counts)
CANONICAL_BUILDINGS_5: List[Tuple[float, float, float, float]] = [
    # (cx, cy, radius_m, height_m) — R=45m, H in [70, 80]m, min spacing >= 110m
    (280.0, 280.0, 45.0, 75.0),
    (520.0, 320.0, 45.0, 80.0),
    (480.0, 560.0, 45.0, 72.0),
    (740.0, 520.0, 45.0, 78.0),
    (320.0, 740.0, 45.0, 70.0),
]
# 2 disjoint blackout zones: Manhattan radius r=3 cells -> 25 cells each (50 cells = 12.5% map)
CANONICAL_BLACKOUT_CENTERS_2: List[Tuple[int, int]] = [
    (15, 5),   # around (775m, 275m)
    (14, 15),  # around (725m, 775m)
]
CANONICAL_BLACKOUT_MANHATTAN_R: int = 3
GCS_POS_M: Tuple[float, float, float] = (50.0, 50.0, 15.0)
UAV_ALT_M: float = 50.0


# ---------------------------------------------------------------------------
# 1. Self-Consistent Dynamic Link Budget (Fixes N01 / N02)
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_V1: float = 3.0e8
SPEED_OF_LIGHT_EXACT: float = 299_792_458.0


def compute_link_budget_ranges_m(
    tx_power_w: float = 0.5,
    noise_power_w: float = 4.0e-11,
    snr_threshold_db: float = 6.0,
    carrier_freq_hz: float = 2.4e9,
    nlos_alpha: float = 3.0,
    nlos_excess_loss_db: float = 12.0,
    c_light: float = SPEED_OF_LIGHT_EXACT,
) -> Dict[str, float]:
    """Compute LoS (free-space) and NLoS (UavNetSim a2a.py urban) max ranges dynamically."""
    g_ref = (c_light / (4.0 * math.pi * carrier_freq_hz)) ** 2
    snr_lin = 10.0 ** (snr_threshold_db / 10.0)
    req_gain = (noise_power_w * snr_lin) / tx_power_w

    # LoS: g_ref * d^(-2) = req_gain
    d_los = math.sqrt(g_ref / req_gain)
    # NLoS: g_ref * d^(-nlos_alpha) * 10^(-excess/10) = req_gain
    ex_lin = 10.0 ** (-nlos_excess_loss_db / 10.0)
    d_nlos = ((g_ref * ex_lin) / req_gain) ** (1.0 / nlos_alpha)
    return {
        "tx_power_w": tx_power_w,
        "tx_power_dbm": 10.0 * math.log10(tx_power_w * 1000.0),
        "noise_power_w": noise_power_w,
        "snr_threshold_db": snr_threshold_db,
        "los_max_range_m": d_los,
        "nlos_max_range_m": d_nlos,
    }


# ---------------------------------------------------------------------------
# 2. Zeng, Xu & Zhang (IEEE TWC 2019, Eq. 12) Rotary-Wing Propulsion Power
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ZengRotaryWingParams:
    p0_w: float = 79.86       # Blade profile power in hover (W)
    pi_w: float = 88.63       # Induced power in hover (W)
    u_tip_mps: float = 120.0  # Rotor tip speed (m/s)
    v0_mps: float = 4.03      # Mean rotor induced velocity in hover (m/s)
    d0: float = 0.6           # Fuselage drag ratio
    rho: float = 1.225        # Air density (kg/m^3)
    s: float = 0.05           # Rotor solidity
    area_m2: float = 0.503    # Rotor disc area (m^2)
    avionics_w: float = 15.0  # Flight controller + onboard compute (W)
    rf_tx_pa_w: float = 3.0   # Active RF PA + baseband during TX duty cycle (W)
    e_max_j: float = 180_000.0  # 50 Wh usable (4S 4200 mAh LiPo w/ 20% reserve)


def zeng_propulsion_power_w(v_mps: float, p: ZengRotaryWingParams = ZengRotaryWingParams()) -> float:
    """Steady-level rotary-wing propulsion power P(V) in Watts (Zeng et al., IEEE TWC 2019)."""
    v = max(0.0, float(v_mps))
    p_blade = p.p0_w * (1.0 + 3.0 * (v ** 2) / (p.u_tip_mps ** 2))
    v2_norm = (v ** 2) / (2.0 * (p.v0_mps ** 2))
    p_ind = p.pi_w * math.sqrt(max(0.0, math.sqrt(1.0 + v2_norm ** 2) - v2_norm))
    p_par = 0.5 * p.d0 * p.rho * p.s * p.area_m2 * (v ** 3)
    return p_blade + p_ind + p_par


def step_energy_joules(
    v_mps: float,
    tx_airtime_s: float = 0.0,
    dt_s: float = 1.0,
    p: ZengRotaryWingParams = ZengRotaryWingParams(),
) -> float:
    """Total energy consumed in one step (propulsion + avionics + RF TX active time)."""
    p_flight = zeng_propulsion_power_w(v_mps, p) + p.avionics_w
    return p_flight * dt_s + p.rf_tx_pa_w * max(0.0, tx_airtime_s)


# ---------------------------------------------------------------------------
# 3. 2 x uint32 (8 B/entry) Bit-Packing with AoI Timestamp & No H=1500 Overflow
# ---------------------------------------------------------------------------
# word0 (32b): origin:3b (0..7) | cell:9b (0..511) | gen_ts_s:20b (0..1,048,575 s)
# word1 (32b): n_pos:16b (0..65535) | n_neg:16b (0..65535)
def pack_entry_2x_uint32(
    origin_id: int, cell_id: int, gen_ts_s: int, n_pos: int, n_neg: int
) -> Tuple[int, int]:
    if not (0 <= origin_id <= 7):
        raise ValueError(f"origin_id out of range: {origin_id}")
    if not (0 <= cell_id <= 511):
        raise ValueError(f"cell_id out of range: {cell_id}")
    if not (0 <= gen_ts_s <= 0xFFFFF):
        raise ValueError(f"gen_ts_s out of 20-bit range: {gen_ts_s}")
    if not (0 <= n_pos <= 0xFFFF) or not (0 <= n_neg <= 0xFFFF):
        raise ValueError(f"counters out of 16-bit range: ({n_pos}, {n_neg})")
    w0 = (origin_id & 0x7) | ((cell_id & 0x1FF) << 3) | ((gen_ts_s & 0xFFFFF) << 12)
    w1 = (n_pos & 0xFFFF) | ((n_neg & 0xFFFF) << 16)
    return w0, w1


def unpack_entry_2x_uint32(w0: int, w1: int) -> Tuple[int, int, int, int, int]:
    origin_id = w0 & 0x7
    cell_id = (w0 >> 3) & 0x1FF
    gen_ts_s = (w0 >> 12) & 0xFFFFF
    n_pos = w1 & 0xFFFF
    n_neg = (w1 >> 16) & 0xFFFF
    return origin_id, cell_id, gen_ts_s, n_pos, n_neg


def compute_v52_vram_gib(
    num_envs: int = 4096,
    queue_cap: int = QUEUE_CAP_PKTS,
    max_entries: int = MAX_ENTRIES_PER_PKT,
    bytes_per_entry: int = 8,
) -> Dict[str, float]:
    gib = 1024 ** 3
    payload_b = num_envs * RADIO_N * queue_cap * max_entries * bytes_per_entry
    watermarks_b = num_envs * ACTOR_N * RADIO_N * NUM_CELLS * 2 * 2  # uint16 (pos, neg)
    pkt_headers_b = num_envs * RADIO_N * queue_cap * 32
    routing_b = num_envs * RADIO_N * RADIO_N * 8
    mac_soa_b = num_envs * 3300
    total_b = payload_b + watermarks_b + pkt_headers_b + routing_b + mac_soa_b
    return {
        "total_e4096_gib": total_b / gib,
        "per_t4_e2048_gib": (total_b / 2.0) / gib,
        "payload_e4096_gib": payload_b / gib,
        "watermarks_e4096_gib": watermarks_b / gib,
    }


# ---------------------------------------------------------------------------
# 4. FP32-Safe Normalized 3D Segment-Cylinder Intersection & Blackout Check
# ---------------------------------------------------------------------------
def segment_hits_cylinder_3d_fp32(
    p0: Sequence[float],
    p1: Sequence[float],
    cx: float,
    cy: float,
    radius: float,
    height: float,
    eps: float = 1e-6,
) -> bool:
    x0, y0, z0 = np.float32(p0[0]), np.float32(p0[1]), np.float32(p0[2])
    x1, y1, z1 = np.float32(p1[0]), np.float32(p1[1]), np.float32(p1[2])
    cx32, cy32 = np.float32(cx), np.float32(cy)
    r32, h32 = np.float32(radius), np.float32(height)
    if (z0 > h32 and z1 > h32) or (z0 < np.float32(0.0) and z1 < np.float32(0.0)):
        return False
    dx, dy, dz = np.float32(x1 - x0), np.float32(y1 - y0), np.float32(z1 - z0)
    l_xy = np.float32(math.hypot(float(dx), float(dy)))
    fx, fy = np.float32(cx32 - x0), np.float32(cy32 - y0)
    if l_xy <= np.float32(eps):
        if float(fx * fx + fy * fy) <= float(r32 * r32):
            return min(float(z0), float(z1)) <= float(h32) and max(float(z0), float(z1)) >= 0.0
        return False
    ux, uy = np.float32(dx / l_xy), np.float32(dy / l_xy)
    d_perp = np.float32(fx * uy - fy * ux)
    sigma = np.float32(r32 * r32 - d_perp * d_perp)
    if sigma < np.float32(0.0):
        return False
    s_c = np.float32(fx * ux + fy * uy)
    half_chord = np.float32(math.sqrt(max(0.0, float(sigma))))
    t0 = max(0.0, float((s_c - half_chord) / l_xy))
    t1 = min(1.0, float((s_c + half_chord) / l_xy))
    if t0 > t1:
        return False
    za = float(z0) + t0 * float(dz)
    zb = float(z0) + t1 * float(dz)
    return min(za, zb) <= float(h32) and max(za, zb) >= 0.0


def is_in_blackout_cell(
    x_m: float,
    y_m: float,
    centers: Sequence[Tuple[int, int]] = CANONICAL_BLACKOUT_CENTERS_2,
    manhattan_r: int = CANONICAL_BLACKOUT_MANHATTAN_R,
) -> bool:
    cx = max(0, min(GRID_W - 1, int(x_m // CELL_SIZE_M)))
    cy = max(0, min(GRID_H - 1, int(y_m // CELL_SIZE_M)))
    return any(abs(cx - bx) + abs(cy - by) <= manhattan_r for bx, by in centers)


# ---------------------------------------------------------------------------
# 5. Canonical Dempster-Shafer Combination (B01) & True Receiver AoI (A01)
# ---------------------------------------------------------------------------
def canonical_ds_combine(
    m1: Tuple[float, float, float], m2: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Canonical DS on {T,F}; reject invalid BBAs and undefined total conflict."""
    for mass in (m1, m2):
        if len(mass) != 3 or any(not math.isfinite(x) or x < 0.0 for x in mass):
            raise ValueError("BBA must contain three finite nonnegative masses")
        if not math.isclose(math.fsum(mass), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("BBA masses must sum to one")
    t1, f1, u1 = m1
    t2, f2, u2 = m2
    # Sum surviving intersections directly: 1-K catastrophically cancels near K=1.
    t_num = math.fsum((t1*t2, t1*u2, u1*t2))
    f_num = math.fsum((f1*f2, f1*u2, u1*f2))
    u_num = u1*u2
    norm = math.fsum((t_num, f_num, u_num))
    if norm == 0.0:
        raise ValueError("Dempster combination undefined under total conflict")
    return t_num/norm, f_num/norm, u_num/norm


def counts_to_canonical_ds(
    n_pos: int, n_neg: int, ds_mf: float = 0.80, ds_me: float = 0.80
) -> Tuple[float, float, float]:
    """Stable closed form for repeated independent simple-support evidence.

    With a=(1-mf)^n_pos and b=(1-me)^n_neg, surviving masses are
    b(1-a), a(1-b), ab. Log scaling avoids cancellation and underflow.
    """
    for count in (n_pos, n_neg):
        if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
            raise ValueError("evidence counts must be nonnegative integers")
    for strength in (ds_mf, ds_me):
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("DS support strength must lie in [0,1]")
    a = (0.0 if n_pos == 0 else
         -math.inf if ds_mf == 1.0 else int(n_pos)*math.log1p(-ds_mf))
    b = (0.0 if n_neg == 0 else
         -math.inf if ds_me == 1.0 else int(n_neg)*math.log1p(-ds_me))
    scale = max(a,b)
    if scale == -math.inf:
        raise ValueError("Dempster combination undefined under total conflict")
    pa, pb = math.exp(a-scale), math.exp(b-scale)
    pu = math.exp(a+b-scale)
    t_num, f_num = pb*(-math.expm1(a)), pa*(-math.expm1(b))
    norm = math.fsum((t_num,f_num,pu))
    return t_num/norm, f_num/norm, pu/norm


# ---------------------------------------------------------------------------
# 6. 7-Node Oracle Connectivity Diagnostic (No Packet MAC or MARL Environment)
# ---------------------------------------------------------------------------
@dataclass
class RewardWeightsV52:
    w_loc_team: float = 0.50   # Applied to TEAM union uncertainty reduction (prevents 6x double counting R02!)
    w_p2p: float = 0.25
    r_tp: float = 25.0
    r_fp: float = 12.5
    fp_hold: float = 0.10      # Actively charged per step while FP remains latched at GCS (fixes R03)
    w_aoi: float = 6.0
    t_ref_s: float = 60.0
    lambda_b: float = 5.0      # Actively charged per second of TX+ACK airtime (fixes R03)
    confirm_th: float = 0.90


def run_7node_episode_comparison(
    policy_mode: str,
    seed: int = 20261125,
    max_steps: int = 300,
    tx_power_w: float = 0.5,
) -> Dict[str, object]:
    """Run scripted 6-UAV + GCS oracle connectivity diagnostic, not packet MAC."""
    rng = np.random.default_rng(seed)
    lb = compute_link_budget_ranges_m(tx_power_w=tx_power_w)
    r_los, r_nlos = lb["los_max_range_m"], lb["nlos_max_range_m"]
    w = RewardWeightsV52()
    zeng = ZengRotaryWingParams()

    # Place 15 targets outside buildings
    true_targets = np.zeros(NUM_CELLS, dtype=bool)
    placed = 0
    while placed < 15:
        cid = int(rng.integers(0, NUM_CELLS))
        if true_targets[cid]:
            continue
        cx_m = (cid % GRID_W + 0.5) * CELL_SIZE_M
        cy_m = (cid // GRID_W + 0.5) * CELL_SIZE_M
        if any(math.hypot(cx_m - bx, cy_m - by) <= br for bx, by, br, _ in CANONICAL_BUILDINGS_5):
            continue
        true_targets[cid] = True
        placed += 1

    # Start 6 UAVs in a semi-circle near GCS
    pos = np.array([
        [100.0 + 40.0 * i, 120.0 + 30.0 * (i % 2), UAV_ALT_M] for i in range(ACTOR_N)
    ], dtype=np.float64)
    # Simple lawnmower waypoints covering the 1km map
    vel_dirs = np.array([
        [1.0, 0.25], [0.8, 0.6], [0.4, 0.9], [0.95, 0.3], [0.5, 0.85], [0.7, 0.7]
    ], dtype=np.float64)
    vel_dirs /= np.linalg.norm(vel_dirs, axis=1, keepdims=True)
    battery_j = np.full(ACTOR_N, zeng.e_max_j, dtype=np.float64)

    # Provenance ledgers: [RADIO_N, ACTOR_N, NUM_CELLS, 2] and latest gen_ts_s [RADIO_N, ACTOR_N, NUM_CELLS]
    ledger = np.zeros((RADIO_N, ACTOR_N, NUM_CELLS, 2), dtype=np.int32)
    gen_ts = np.full((RADIO_N, ACTOR_N, NUM_CELLS), -1, dtype=np.int32)
    team_u_prev = np.ones(NUM_CELLS, dtype=np.float64)
    gcs_tp_latch = np.zeros(NUM_CELLS, dtype=np.int8)
    gcs_fp_latch = np.zeros(NUM_CELLS, dtype=np.int8)
    gcs_latest_u_s = np.full(NUM_CELLS, -1, dtype=np.int32)  # True receiver-side u_c^GCS(t)

    tot_loc = tot_p2p = tot_gcs_tp = tot_aoi = tot_fp = tot_fp_hold = tot_airtime_pen = 0.0
    delivered_aoi_samples: List[float] = []
    receiver_age_sum_s = 0
    receiver_age_cell_time_samples = 0
    steps_run = 0
    terminated = False
    truncated = False

    for t in range(1, max_steps + 1):
        steps_run = t
        # 1. Move UAVs at cruise speed ~15 m/s (bouncing off walls & buildings)
        v_cruise = 15.0
        pos_before_step = pos[:, :2].copy()
        for u in range(ACTOR_N):
            cand = pos[u, :2] + vel_dirs[u] * v_cruise
            if cand[0] < 25.0 or cand[0] > MAP_SIZE_M - 25.0:
                vel_dirs[u, 0] *= -1.0
                cand[0] = np.clip(cand[0], 25.0, MAP_SIZE_M - 25.0)
            if cand[1] < 25.0 or cand[1] > MAP_SIZE_M - 25.0:
                vel_dirs[u, 1] *= -1.0
                cand[1] = np.clip(cand[1], 25.0, MAP_SIZE_M - 25.0)
            if any(math.hypot(cand[0] - bx, cand[1] - by) <= br + 5.0 for bx, by, br, _ in CANONICAL_BUILDINGS_5):
                vel_dirs[u] = -vel_dirs[u]
                cand = pos[u, :2]
            pos[u, :2] = cand

        actual_speeds_mps = np.linalg.norm(pos[:, :2]-pos_before_step, axis=1)

        # 2. Local sensing (FoV 90 deg at z=50m -> r=50m)
        for u in range(ACTOR_N):
            ux, uy = pos[u, 0], pos[u, 1]
            c_min_x = max(0, int((ux - 50.0) // CELL_SIZE_M))
            c_max_x = min(GRID_W - 1, int((ux + 50.0) // CELL_SIZE_M))
            c_min_y = max(0, int((uy - 50.0) // CELL_SIZE_M))
            c_max_y = min(GRID_H - 1, int((uy + 50.0) // CELL_SIZE_M))
            for cy in range(c_min_y, c_max_y + 1):
                for cx in range(c_min_x, c_max_x + 1):
                    if math.hypot((cx + 0.5) * CELL_SIZE_M - ux, (cy + 0.5) * CELL_SIZE_M - uy) <= 50.0:
                        cid = cy * GRID_W + cx
                        det = bool(rng.random() < (0.85 if true_targets[cid] else 0.03))
                        ledger[u, u, cid, 0 if det else 1] += 1
                        gen_ts[u, u, cid] = t

        # Team-union local exploration reward (prevents 6x double-counting per R02)
        for cid in range(NUM_CELLS):
            n_p = int(sum(ledger[u, u, cid, 0] for u in range(ACTOR_N)))
            n_n = int(sum(ledger[u, u, cid, 1] for u in range(ACTOR_N)))
            if n_p > 0 or n_n > 0:
                _, _, u_now = counts_to_canonical_ds(n_p, n_n)
                du = max(0.0, team_u_prev[cid] - u_now)
                tot_loc += w.w_loc_team * du
                team_u_prev[cid] = u_now

        # 3. Oracle graph transfers, with analytic successful-frame airtime only.
        airtime_step_s = 0.0
        uav_tx_airtime_step_s = np.zeros(ACTOR_N, dtype=np.float64)
        if policy_mode == "full_comm":
            nodes = [tuple(pos[u]) for u in range(ACTOR_N)] + [GCS_POS_M]
            adj = np.zeros((RADIO_N, RADIO_N), dtype=bool)
            for i in range(RADIO_N):
                if is_in_blackout_cell(nodes[i][0], nodes[i][1]):
                    continue
                for j in range(i + 1, RADIO_N):
                    if is_in_blackout_cell(nodes[j][0], nodes[j][1]):
                        continue
                    d3 = math.dist(nodes[i], nodes[j])
                    blocked = any(
                        segment_hits_cylinder_3d_fp32(nodes[i], nodes[j], bx, by, br, bh)
                        for bx, by, br, bh in CANONICAL_BUILDINGS_5
                    )
                    if d3 <= (r_nlos if blocked else r_los):
                        adj[i, j] = adj[j, i] = True

            # BFS shortest hop to GCS; each connected UAV forwards its new entries along path
            parent = {GCS_ID: -1}
            q_bfs = [GCS_ID]
            while q_bfs:
                cur = q_bfs.pop(0)
                for nxt in range(RADIO_N):
                    if adj[cur, nxt] and nxt not in parent:
                        parent[nxt] = cur
                        q_bfs.append(nxt)

            for u in range(ACTOR_N):
                if u not in parent:
                    continue
                nxt = parent[u]
                # Find dirty cells to forward
                dirty = np.flatnonzero(
                    (ledger[u, :, :, 0] > ledger[nxt, :, :, 0])
                    | (ledger[u, :, :, 1] > ledger[nxt, :, :, 1])
                    | (gen_ts[u, :, :] > gen_ts[nxt, :, :])
                )
                if len(dirty) == 0:
                    continue
                k_sent = min(MAX_ENTRIES_PER_PKT, len(dirty))
                # Airtime for frame (70 + 8*k) B + 30 B ACK at 2 Mbps (4 us/B)
                frame_airtime_s = ((LOWER_HEADERS_B + APP_HEADER_B + ENTRY_WIRE_B*k_sent)*4.0)/1e6
                ack_airtime_s = 30*4.0/1e6
                airtime_step_s += frame_airtime_s+ack_airtime_s
                uav_tx_airtime_step_s[u] += frame_airtime_s
                if nxt != GCS_ID:
                    # Receiver transmits ACK; GCS energy is outside the UAV battery budget.
                    uav_tx_airtime_step_s[nxt] += ack_airtime_s
                for flat_idx in dirty[:k_sent]:
                    o = int(flat_idx // NUM_CELLS)
                    cid = int(flat_idx % NUM_CELLS)
                    n_p_old = int(ledger[nxt, :, cid, 0].sum())
                    n_n_old = int(ledger[nxt, :, cid, 1].sum())
                    _, _, u_before = counts_to_canonical_ds(n_p_old, n_n_old)

                    ledger[nxt, o, cid, 0] = max(ledger[nxt, o, cid, 0], ledger[u, o, cid, 0])
                    ledger[nxt, o, cid, 1] = max(ledger[nxt, o, cid, 1], ledger[u, o, cid, 1])
                    gen_ts[nxt, o, cid] = max(gen_ts[nxt, o, cid], gen_ts[u, o, cid])

                    n_p_new = int(ledger[nxt, :, cid, 0].sum())
                    n_n_new = int(ledger[nxt, :, cid, 1].sum())
                    m_t, _, u_after = counts_to_canonical_ds(n_p_new, n_n_new)

                    if nxt == GCS_ID:
                        # Update true receiver-side latest timestamp u_c^GCS(t)
                        gcs_latest_u_s[cid] = max(gcs_latest_u_s[cid], int(gen_ts[GCS_ID, o, cid]))
                        if m_t >= w.confirm_th:
                            if bool(true_targets[cid]) and gcs_tp_latch[cid] == 0:
                                gcs_tp_latch[cid] = 1
                                aoi_s = max(0.0, float(t - gcs_latest_u_s[cid]))
                                delivered_aoi_samples.append(aoi_s)
                                tot_gcs_tp += w.r_tp
                                tot_aoi += w.w_aoi * max(0.0, 1.0 - aoi_s / w.t_ref_s)
                            elif (not bool(true_targets[cid])) and gcs_fp_latch[cid] == 0:
                                gcs_fp_latch[cid] = 1
                                tot_fp -= w.r_fp
                    else:
                        tot_p2p += w.w_p2p * max(0.0, u_before - u_after)

        # Actively charge airtime and FP hold penalties every step (fixes R03!)
        tot_airtime_pen -= w.lambda_b * airtime_step_s
        tot_fp_hold -= w.fp_hold * float(gcs_fp_latch.sum())

        # Time-cell mean receiver age after each cell's first delivered observation.
        delivered_cells = gcs_latest_u_s >= 0
        receiver_age_sum_s += int((t-gcs_latest_u_s[delivered_cells]).sum())
        receiver_age_cell_time_samples += int(delivered_cells.sum())

        # 4. Update Zeng 2019 battery per UAV
        for u in range(ACTOR_N):
            battery_j[u] -= step_energy_joules(float(actual_speeds_mps[u]),
                                              float(uav_tx_airtime_step_s[u]), dt_s=1.0, p=zeng)

        if np.all(gcs_tp_latch[true_targets] == 1) or np.any(battery_j <= 0.0):
            terminated = True
            break

    if not terminated and steps_run >= max_steps:
        truncated = True

    total_return = tot_loc + tot_p2p + tot_gcs_tp + tot_aoi + tot_fp + tot_fp_hold + tot_airtime_pen
    return {
        "simulation_semantics": "oracle_connectivity_diagnostic_no_packet_mac",
        "policy_mode": policy_mode,
        "steps_run": steps_run,
        "terminated": terminated,
        "truncated": truncated,
        "total_return": round(total_return, 2),
        "r_local_team": round(tot_loc, 2),
        "r_p2p": round(tot_p2p, 2),
        "r_gcs_tp": round(tot_gcs_tp, 2),
        "r_gcs_aoi": round(tot_aoi, 2),
        "r_gcs_fp": round(tot_fp, 2),
        "r_fp_hold": round(tot_fp_hold, 2),
        "r_airtime": round(tot_airtime_pen, 3),
        "gcs_tp_confirmed": int(gcs_tp_latch[true_targets].sum()),
        "gcs_fp_latched": int(gcs_fp_latch.sum()),
        "mean_first_tp_confirmation_aoi_s": round(float(np.mean(delivered_aoi_samples)), 2) if delivered_aoi_samples else -1.0,
        "mean_gcs_aoi_over_delivered_cells_s": round(receiver_age_sum_s/receiver_age_cell_time_samples,2) if receiver_age_cell_time_samples else -1.0,
        "gcs_aoi_cell_time_samples": receiver_age_cell_time_samples,
        "min_battery_remaining_pct": round(float(battery_j.min() / zeng.e_max_j * 100.0), 2),
    }


def run_seeded_3d_mc_canonical_5_2(
    seed: int = 20261125,
    num_trials: int = 2000,
    tx_power_w: float = 0.5,
) -> Dict[str, float]:
    """Reproducible 3D LoS/NLoS + Blackout Monte Carlo on canonical 5 buildings + 2 blackouts."""
    rng = np.random.default_rng(seed)
    lb = compute_link_budget_ranges_m(tx_power_w=tx_power_w)
    r_los, r_nlos = lb["los_max_range_m"], lb["nlos_max_range_m"]

    direct = connected = multihop = blocked_pairs = total_pairs = 0
    for _ in range(num_trials):
        pts: List[Tuple[float, float, float]] = []
        while len(pts) < ACTOR_N:
            x = float(rng.uniform(25.0, MAP_SIZE_M - 25.0))
            y = float(rng.uniform(25.0, MAP_SIZE_M - 25.0))
            if any(math.hypot(x - bx, y - by) <= br + 10.0 for bx, by, br, _ in CANONICAL_BUILDINGS_5):
                continue
            pts.append((x, y, UAV_ALT_M))
        nodes = pts + [GCS_POS_M]
        adj = np.zeros((RADIO_N, RADIO_N), dtype=bool)
        for i in range(RADIO_N):
            if is_in_blackout_cell(nodes[i][0], nodes[i][1]):
                continue
            for j in range(i + 1, RADIO_N):
                if is_in_blackout_cell(nodes[j][0], nodes[j][1]):
                    continue
                d3 = math.dist(nodes[i], nodes[j])
                blk = any(
                    segment_hits_cylinder_3d_fp32(nodes[i], nodes[j], bx, by, br, bh)
                    for bx, by, br, bh in CANONICAL_BUILDINGS_5
                )
                total_pairs += 1
                if blk:
                    blocked_pairs += 1
                    ok = d3 <= r_nlos
                else:
                    ok = d3 <= r_los
                if ok:
                    adj[i, j] = adj[j, i] = True

        vis = {GCS_ID}
        fr = [GCS_ID]
        while fr:
            u = fr.pop()
            for v in range(RADIO_N):
                if adj[u, v] and v not in vis:
                    vis.add(v)
                    fr.append(v)
        for i in range(ACTOR_N):
            if adj[i, GCS_ID]:
                direct += 1
            if i in vis:
                connected += 1
                if not adj[i, GCS_ID]:
                    multihop += 1

    tot = num_trials * ACTOR_N
    return {
        "seed": seed,
        "num_trials": num_trials,
        "tx_power_w": tx_power_w,
        "tx_power_dbm": round(lb["tx_power_dbm"], 2),
        "los_max_range_m": round(r_los, 2),
        "nlos_max_range_m": round(r_nlos, 2),
        "direct_gcs_rate": round(direct / tot, 4),
        "connected_to_gcs_rate": round(connected / tot, 4),
        "multihop_share_of_connected": round(multihop / max(1, connected), 4),
        "pair_3d_building_occluded_rate": round(blocked_pairs / max(1, total_pairs), 4),
    }
