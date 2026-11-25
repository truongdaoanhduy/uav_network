"""v5.4 Incomplete 7-Radio/GCS NVIDIA Warp Scaffold & 80-Array SoA Manifest.

Designed for E_gpu = 2048 envs per NVIDIA Tesla T4 (E = 4096 across 2x T4) with:
  - Strict separation: ACTOR_N=6, RADIO_N=7, GCS_ID=6, Q=64, K=62, P_E2E=16
  - 2 x wp.uint32 (8 B/entry) payload packing carrying 20-bit generation timestamps
  - Exact integer-diff log-space canonical DS device function (FP32-safe up to n=65535)
  - Planned bounded MAC event loop (MAX_MAC_EVENTS_PER_STEP = 512); NOT YET IMPLEMENTED
  - Declared 80-array lower bound (490,001 B/env); missing state means NOT peak VRAM

Note: Gracefully loads without `warp` installed so the exact SoA byte-table & schema
can be verified on CPU; full CPU–Warp MAC/PHY parity is NOT implemented.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

try:
    import warp as wp  # type: ignore
    HAS_WARP = True
except ImportError:
    wp = None
    HAS_WARP = False

ACTOR_N = 6
RADIO_N = 7
GCS_ID = 6
QUEUE_CAP = 64
MAX_ENTRIES = 62
P_E2E = 16
NUM_CELLS = 400
MAX_MAC_EVENTS_PER_STEP = 512
INF_TIME_US = 2_000_000_000
SAFETY_HORIZON_US = 1_500_000_000

# Complete 80-array per-env SoA manifest: (group, name, shape, dtype_name, bytes_per_elem)
WARP_SOA_80_MANIFEST: List[Tuple[str, str, Tuple[int, ...], str, int]] = [
    # Clock/Ctrl (6)
    ("Clock/Ctrl", "sim_time_us", (1,), "int32", 4),
    ("Clock/Ctrl", "step_end_us", (1,), "int32", 4),
    ("Clock/Ctrl", "episode_id", (1,), "int32", 4),
    ("Clock/Ctrl", "mac_events_used", (1,), "int32", 4),
    ("Clock/Ctrl", "rng_state", (RADIO_N,), "uint32", 4),
    ("Clock/Ctrl", "mac_overflow_flag", (1,), "uint8", 1),
    # PHY/Kin (6)
    ("PHY/Kin", "node_pos_start", (RADIO_N, 3), "float32", 4),
    ("PHY/Kin", "node_pos_end", (RADIO_N, 3), "float32", 4),
    ("PHY/Kin", "node_blackout", (RADIO_N,), "uint8", 1),
    ("PHY/Kin", "uav_battery_j", (ACTOR_N,), "float32", 4),
    ("PHY/Kin", "uav_tx_airtime_us", (ACTOR_N,), "int32", 4),
    ("PHY/Kin", "chan_gain_lin", (RADIO_N, RADIO_N), "float32", 4),
    # MAC/PHY (19)
    ("MAC/PHY", "mac_state", (RADIO_N,), "int8", 1),
    ("MAC/PHY", "mac_backoff_frozen", (RADIO_N,), "uint8", 1),
    ("MAC/PHY", "mac_retx_cnt", (RADIO_N,), "int8", 1),
    ("MAC/PHY", "mac_tx_next_hop", (RADIO_N,), "int8", 1),
    ("MAC/PHY", "mac_tx_is_ack", (RADIO_N,), "uint8", 1),
    ("MAC/PHY", "mac_ack_pend_dst", (RADIO_N,), "int8", 1),
    ("MAC/PHY", "rx_locked_tx", (RADIO_N,), "int8", 1),
    ("MAC/PHY", "rx_is_ack", (RADIO_N,), "uint8", 1),
    ("MAC/PHY", "rx_corrupted", (RADIO_N,), "uint8", 1),
    ("MAC/PHY", "mac_active_slot", (RADIO_N,), "int16", 2),
    ("MAC/PHY", "mac_state_end_us", (RADIO_N,), "int32", 4),
    ("MAC/PHY", "mac_backoff_rem_us", (RADIO_N,), "int32", 4),
    ("MAC/PHY", "mac_tx_start_us", (RADIO_N,), "int32", 4),
    ("MAC/PHY", "mac_tx_end_us", (RADIO_N,), "int32", 4),
    ("MAC/PHY", "mac_tx_pkt_id", (RADIO_N,), "int32", 4),
    ("MAC/PHY", "mac_ack_pend_pkt_id", (RADIO_N,), "int32", 4),
    ("MAC/PHY", "rx_pkt_id", (RADIO_N,), "int32", 4),
    ("MAC/PHY", "rx_sig_w", (RADIO_N,), "float32", 4),
    ("MAC/PHY", "rx_max_interf_w", (RADIO_N,), "float32", 4),
    # Queue (17)
    ("Queue", "q_head", (RADIO_N,), "int16", 2),
    ("Queue", "q_tail", (RADIO_N,), "int16", 2),
    ("Queue", "q_count", (RADIO_N,), "int16", 2),
    ("Queue", "q_next_pkt_id", (RADIO_N,), "int32", 4),
    ("Queue", "q_pkt_id", (RADIO_N, QUEUE_CAP), "int32", 4),
    ("Queue", "q_create_us", (RADIO_N, QUEUE_CAP), "int32", 4),
    ("Queue", "q_expire_us", (RADIO_N, QUEUE_CAP), "int32", 4),
    ("Queue", "q_app_seq", (RADIO_N, QUEUE_CAP), "int32", 4),
    ("Queue", "q_frame_bytes", (RADIO_N, QUEUE_CAP), "int16", 2),
    ("Queue", "q_pkt_type", (RADIO_N, QUEUE_CAP), "int8", 1),
    ("Queue", "q_app_src", (RADIO_N, QUEUE_CAP), "int8", 1),
    ("Queue", "q_app_dst", (RADIO_N, QUEUE_CAP), "int8", 1),
    ("Queue", "q_hop_cnt", (RADIO_N, QUEUE_CAP), "int8", 1),
    ("Queue", "q_visited_mask", (RADIO_N, QUEUE_CAP), "uint8", 1),
    ("Queue", "q_entry_cnt", (RADIO_N, QUEUE_CAP), "int8", 1),
    ("Queue", "q_acked_peer", (RADIO_N, QUEUE_CAP), "int8", 1),
    ("Queue", "q_valid", (RADIO_N, QUEUE_CAP), "uint8", 1),
    # Payload (2)
    ("Payload", "q_payload_w0", (RADIO_N, QUEUE_CAP, MAX_ENTRIES), "uint32", 4),
    ("Payload", "q_payload_w1", (RADIO_N, QUEUE_CAP, MAX_ENTRIES), "uint32", 4),
    # Routing (7)
    ("Routing", "rt_next_hop", (RADIO_N, RADIO_N), "int8", 1),
    ("Routing", "rt_hop_dist", (RADIO_N, RADIO_N), "int8", 1),
    ("Routing", "rt_path_mask", (RADIO_N, RADIO_N), "uint8", 1),
    ("Routing", "rt_expire_us", (RADIO_N, RADIO_N), "int32", 4),
    ("Routing", "rt_hello_next_us", (RADIO_N,), "int32", 4),
    ("Routing", "rx_dedup_pkt_id", (RADIO_N, 16), "int32", 4),
    ("Routing", "rx_dedup_ptr", (RADIO_N,), "uint8", 1),
    # E2E-Pend (6)
    ("E2E-Pend", "e2e_pend_seq", (ACTOR_N, P_E2E), "int32", 4),
    ("E2E-Pend", "e2e_pend_dst", (ACTOR_N, P_E2E), "int8", 1),
    ("E2E-Pend", "e2e_pend_cnt", (ACTOR_N, P_E2E), "int8", 1),
    ("E2E-Pend", "e2e_pend_valid", (ACTOR_N, P_E2E), "uint8", 1),
    ("E2E-Pend", "e2e_pend_w0", (ACTOR_N, P_E2E, MAX_ENTRIES), "uint32", 4),
    ("E2E-Pend", "e2e_pend_w1", (ACTOR_N, P_E2E, MAX_ENTRIES), "uint32", 4),
    # Belief/GCS (10)
    ("Belief/GCS", "ledger_pos", (RADIO_N, ACTOR_N, NUM_CELLS), "uint16", 2),
    ("Belief/GCS", "ledger_neg", (RADIO_N, ACTOR_N, NUM_CELLS), "uint16", 2),
    ("Belief/GCS", "ledger_gen_ts_s", (RADIO_N, ACTOR_N, NUM_CELLS), "int32", 4),
    ("Belief/GCS", "ack_wm_pos", (ACTOR_N, RADIO_N, NUM_CELLS), "uint16", 2),
    ("Belief/GCS", "ack_wm_neg", (ACTOR_N, RADIO_N, NUM_CELLS), "uint16", 2),
    ("Belief/GCS", "gcs_latest_u_s", (NUM_CELLS,), "int32", 4),
    ("Belief/GCS", "gcs_tp_latch", (NUM_CELLS,), "uint8", 1),
    ("Belief/GCS", "gcs_fp_latch", (NUM_CELLS,), "uint8", 1),
    ("Belief/GCS", "true_targets", (NUM_CELLS,), "uint8", 1),
    ("Belief/GCS", "team_u_prev", (NUM_CELLS,), "float32", 4),
    # Stats (7)
    ("Stats", "stat_drop_qfull", (RADIO_N,), "int32", 4),
    ("Stats", "stat_drop_ttl", (RADIO_N,), "int32", 4),
    ("Stats", "stat_drop_retx", (RADIO_N,), "int32", 4),
    ("Stats", "stat_drop_noroute", (RADIO_N,), "int32", 4),
    ("Stats", "stat_mac_ack_ok", (RADIO_N,), "int32", 4),
    ("Stats", "stat_e2e_ack_rx", (ACTOR_N,), "int32", 4),
    ("Stats", "stat_gcs_data_rx", (1,), "int32", 4),
]


def compute_exact_80array_soa_bytes(num_envs: int = 2048) -> Dict[str, float]:
    """Compute declared byte footprint for 80 arrays only, NOT full GPU VRAM."""
    assert len(WARP_SOA_80_MANIFEST) == 80
    per_env_bytes = 0
    by_group: Dict[str, int] = {}
    for grp, _, shape, _, bpe in WARP_SOA_80_MANIFEST:
        n_elem = 1
        for d in shape:
            n_elem *= d
        b = n_elem * bpe
        per_env_bytes += b
        by_group[grp] = by_group.get(grp, 0) + b
    total_bytes = per_env_bytes * num_envs
    return {
        "array_count": 80,
        "bytes_per_env": per_env_bytes,
        "num_envs": num_envs,
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / (1024 ** 2), 3),
        "total_gib": round(total_bytes / (1024 ** 3), 4),
        "by_group_bytes_per_env": by_group,
    }


if HAS_WARP:
    @wp.func
    def counts_to_canonical_ds_wp(n_pos: int, n_neg: int, lmf: float, lme: float) -> wp.vec3:
        """FP32-safe integer-diff log-space canonical DS for NVIDIA Warp kernels."""
        if n_pos == 0 and n_neg == 0:
            return wp.vec3(0.0, 0.0, 1.0)
        a = 0.0 if n_pos == 0 else float(n_pos) * lmf
        b = 0.0 if n_neg == 0 else float(n_neg) * lme
        pa = 1.0
        pb = 1.0
        pu = 1.0
        if lmf == lme:
            d = float(n_pos - n_neg) * lmf
            if n_pos <= n_neg:
                pa = 1.0
                pb = wp.exp(-d)
                pu = wp.exp(b)
            else:
                pa = wp.exp(d)
                pb = 1.0
                pu = wp.exp(a)
        else:
            s = wp.max(a, b)
            pa = wp.exp(a - s)
            pb = wp.exp(b - s)
            pu = wp.exp(wp.min(a, b))
        t_num = 0.0 if n_pos == 0 else pb * (1.0 - wp.exp(a))
        f_num = 0.0 if n_neg == 0 else pa * (1.0 - wp.exp(b))
        z = t_num + f_num + pu
        return wp.vec3(t_num / z, f_num / z, pu / z)

    @wp.kernel
    def reset_done_envs_7node_kernel(
        done_mask: wp.array(dtype=wp.uint8),
        sim_time_us: wp.array(dtype=wp.int32),
        episode_id: wp.array(dtype=wp.int32),
        mac_overflow_flag: wp.array(dtype=wp.uint8),
        mac_state: wp.array2d(dtype=wp.int8),
        mac_state_end_us: wp.array2d(dtype=wp.int32),
        q_head: wp.array2d(dtype=wp.int16),
        q_tail: wp.array2d(dtype=wp.int16),
        q_count: wp.array2d(dtype=wp.int16),
        uav_battery_j: wp.array2d(dtype=wp.float32),
        e_max_j: float,
    ):
        """Partial atomic reset per env e without touching any other parallel world."""
        e = wp.tid()
        if done_mask[e] == wp.uint8(0):
            return
        sim_time_us[e] = 0
        episode_id[e] = episode_id[e] + 1
        mac_overflow_flag[e] = wp.uint8(0)
        for r in range(RADIO_N):
            mac_state[e, r] = wp.int8(0)
            mac_state_end_us[e, r] = INF_TIME_US
            q_head[e, r] = wp.int16(0)
            q_tail[e, r] = wp.int16(0)
            q_count[e, r] = wp.int16(0)
        for u in range(ACTOR_N):
            uav_battery_j[e, u] = e_max_j


def audit_gate3_warp_parity_contract() -> Dict[str, object]:
    """Fail-closed readiness audit: the 80-array schema is NOT an executable 7-radio environment.

    Bytes are lower bounds for named persisted fields, not peak GPU allocations.
    Do not turn this into a training launch until all blockers have been removed.
    """
    manifest_names = {name for _, name, _, _, _ in WARP_SOA_80_MANIFEST}
    # Minimum CPU-owned state missing in the Warp specification (fixed-shape only).
    required_additions = {
        "ack_wm_ts": (ACTOR_N * RADIO_N * NUM_CELLS * 4),
        "gcs_last_rx_us": (NUM_CELLS * 8),
        "uav_velocities_xy": (ACTOR_N * 2 * 4),
    }
    missing = {name: bytes_per_env for name, bytes_per_env in required_additions.items()
               if name not in manifest_names}
    declared = int(compute_exact_80array_soa_bytes(1)["bytes_per_env"])
    additional = sum(missing.values())
    return {
        "gate3_ready": False,
        "reason": "80-array SoA is only a declared schema, not a full CPU-equivalent Warp MAC environment",
        "warp_installed_locally": HAS_WARP,
        "existing_declared_arrays": len(WARP_SOA_80_MANIFEST),
        "declared_bytes_per_env": declared,
        "minimum_missing_fields_bytes_per_env": missing,
        "minimum_additional_bytes_per_env": additional,
        "minimum_known_bytes_per_env": declared + additional,
        "minimum_known_gib_per_2048_envs": round((declared + additional) * 2048 / (1024 ** 3), 4),
        "unquantified_extra_state": [
            "pending_hello_control_slot", "application_dedup_and_pending_expiry",
            "full_reset_of_ledgers_counters_routes_timers", "episode_sensor_rng",
            "safe_collision_trajectory_state", "actor_critic_runtime_buffers",
        ],
        "missing_executable_stages": [
            "CPU-parity sensor/dynamics/reward kernel", "7-node MAC/PHY event kernel",
            "receiver/relay/routing/ACK kernel", "bounded event-overflow contract",
            "atomic full-episode reset kernel", "NVIDIA CUDA T4 numeric parity tests",
        ],
    }

# v5.5 audit: current CPU reference requires these fields in addition to v5.4 SoA.
# This is a LOWER BOUND, NOT an assertion that all 7-radio GPU state is represented.
V55_REQUIRED_EXTRAS: List[Tuple[str, str, Tuple[int,...], str, int]] = [
    ('Belief/GCS','ack_wm_ts',(ACTOR_N,RADIO_N,NUM_CELLS),'int32',4),
    ('Belief/GCS','ledger_rx_age_s',(RADIO_N,ACTOR_N,NUM_CELLS),'float32',4),
    ('Belief/GCS','gcs_last_rx_us',(NUM_CELLS,),'int64',8),
    ('Stats','step_new_peer_cells',(1,),'int32',4),
    ('Clock/Ctrl','mission_epoch_us',(1,),'int64',8),
    ('Clock/Ctrl','gcs_confirm_rx_first_us',(NUM_CELLS,),'int64',8),
    ('MAC/PHY','half_duplex_billing_energy',(RADIO_N,),'float32',4),
    ('E2E-Pend','e2e_pend_expire_us',(ACTOR_N,P_E2E),'int64',8),
    ('Belief/GCS','actor_obs_5ch_pooled',(ACTOR_N,5,100),'float32',4),
]


def v55_minimum_gpu_memory_budget(num_envs:int=2048, replay_capacity:int=1_000_000) -> dict:
    """Measured = False. Conservative schema only; no CUDA allocation performed."""
    if num_envs<1 or replay_capacity<0: raise ValueError('invalid env count or replay capacity')
    old=compute_exact_80array_soa_bytes(num_envs)
    extra_per_env=sum((math.prod(shape)*bpe for _,_,shape,_,bpe in V55_REQUIRED_EXTRAS))
    # Existing 16-slot dedup IDs in manifest but CPU uses ring of 32 entries.
    dedup_extra=(32-16)*RADIO_N*4
    per_env=old['bytes_per_env']+extra_per_env+dedup_extra
    # GPUHybridReplayBuffer stores obs + next_obs, state + next_state, actions,
    # masks, reward, done; input state is concat obs and thus duplicated in replay.
    ob=14+5*100; st=ACTOR_N*ob; cont=2; disc=7
    replay_per_transition=(2*ACTOR_N*ob*2+2*st*2+
        ACTOR_N*cont*4+ACTOR_N*disc*4+2*ACTOR_N*disc+8)
    return {
        'num_envs':num_envs,
        'v54_80_arrays_bytes_per_env':old['bytes_per_env'],
        'v55_known_extra_bytes_per_env':extra_per_env+dedup_extra,
        'v55_minimum_state_bytes_per_env':per_env,
        'v55_minimum_state_gib':round(per_env*num_envs/(1024**3),4),
        'actor_observation_dim':ob,
        'central_state_dim':st,
        'replay_bytes_per_joint_transition':replay_per_transition,
        'replay_capacity':replay_capacity,
        'replay_gib':round(replay_capacity*replay_per_transition/(1024**3),4),
        'minimum_combined_gib':round((replay_capacity*replay_per_transition+per_env*num_envs)/(1024**3),4),
        'measured_cuda_peak':False,
        'cuda_full_mac_parity':False,
        'unbudgeted':['PyTorch actors/critics','optimizer states','minibatch activations',
                       'CUDA context','kernel scratch','interference state','event scheduler','routing housekeeping'],
    }
