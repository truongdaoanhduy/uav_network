"""v6.7 Parameterized Parallel Reset Kernels for `NetworkBuffersV67`."""
from __future__ import annotations

import numpy as np

from zhang_xue.gpu.state_v67 import (
    HAS_WARP,
    NEVER_US,
    ST_IDLE,
    NetworkBuffersV67,
    SimDimsV67,
    wp,
)

if HAS_WARP:
    @wp.kernel
    def reset_world_v67(
        s: NetworkBuffersV67,
        dims: SimDimsV67,
        done: wp.array(dtype=wp.uint8),
        seed: int,
    ):
        e = wp.tid()
        if done[e] == wp.uint8(0):
            return
        s.now[e] = 0
        s.mission_epoch_us[e] = 0
        s.episode[e] = s.episode[e] + 1
        s.overflow[e] = wp.uint8(0)
        s.event_count[e] = 0
        s.next_pid[e] = 1
        s.next_seq[e] = 1
        s.stats_gcs[e] = 0
        s.stats_peer[e] = 0
        s.stats_mac_ack[e] = 0
        s.stats_e2e[e] = 0
        s.stats_retry[e] = 0
        s.stats_drop[e] = 0
        s.stats_ttl[e] = 0
        s.stats_loop[e] = 0
        for r in range(dims.n_radio):
            rng = wp.uint32(seed + e * 131 + r * 17 + s.episode[e] * 997)
            rng = wp.uint32(rng * wp.uint32(1664525) + wp.uint32(1013904223))
            s.rng[e, r] = rng
            s.radio_alive[e, r] = wp.uint8(1)
            s.state[e, r] = wp.int8(ST_IDLE)
            s.deadline[e, r] = NEVER_US
            s.hello_due[e, r] = int((rng >> wp.uint32(16)) % wp.uint32(20000))
            s.hello_pending[e, r] = wp.uint8(0)
            s.hello_active_at_us[e, r] = -1
            for k in range(4):
                s.hello_offer_pose[e, r, k] = 0.0
                s.hello_active_pose[e, r, k] = 0.0
            s.active_slot[e, r] = wp.int8(-1)
            s.next_hop[e, r] = wp.int8(-1)
            s.tx_pkt_id[e, r] = -1
            s.ack_dst[e, r] = wp.int8(-1)
            s.ack_pid[e, r] = -1
            s.retries[e, r] = wp.uint8(0)
            s.backoff_rem[e, r] = 0
            s.frozen[e, r] = wp.uint8(0)
            s.rx_lock[e, r] = wp.int8(-1)
            s.rx_corrupt[e, r] = wp.uint8(0)
            s.rx_interf[e, r] = 0.0
            s.rx_sig[e, r] = 0.0
            s.tx_airtime[e, r] = 0
            s.q_count[e, r] = wp.uint8(0)
            s.received_ptr[e, r] = wp.uint8(0)
            for k in range(16):
                s.received_pid[e, r, k] = -1
            for q in range(dims.q_cap):
                s.q_order[e, r, q] = wp.uint8(0)
                s.q_used[e, r, q] = wp.uint8(0)
            for d in range(dims.n_radio):
                s.gain[e, r, d] = 0.0
                s.route_hop[e, r, d] = wp.int8(-1)
                s.route_dist[e, r, d] = wp.int8(127)
                s.route_mask[e, r, d] = wp.uint32(0)
                s.route_expiry[e, r, d] = 0
                s.hello_dv_hop[e, r, d] = wp.int8(-1)
                s.hello_dv_dist[e, r, d] = wp.int8(127)
                s.hello_dv_mask[e, r, d] = wp.uint32(0)
                s.hello_rx_at_us[e, r, d] = -1
                for k in range(4):
                    s.hello_rx_pose[e, r, d, k] = 0.0
                s.delivered_ptr[e, r, d] = wp.uint8(0)
                s.delivered_ack_ptr[e, r, d] = wp.uint8(0)
                for k in range(16):
                    s.delivered_seq[e, r, d, k] = -1
                    s.delivered_ack_seq[e, r, d, k] = -1
            if r < dims.n_uav:
                for d in range(dims.n_radio):
                    s.dirty_dest[e, r, d] = wp.uint8(0)
                for p in range(dims.p_cap):
                    s.pending_valid[e, r, p] = wp.uint8(0)
                    s.pending_dst[e, r, p] = wp.int8(-1)
                    s.pending_seq[e, r, p] = -1
                    s.pending_expiry[e, r, p] = 0
                    s.pending_entries[e, r, p] = wp.uint8(0)

    @wp.kernel
    def reset_cells_v67(
        s: NetworkBuffersV67,
        dims: SimDimsV67,
        done: wp.array(dtype=wp.uint8),
    ):
        e, cid = wp.tid()
        if done[e] == wp.uint8(0) or cid >= dims.num_cells:
            return
        s.gcs_latest[e, cid] = wp.int16(-1)
        s.gcs_last_rx_us[e, cid] = -1
        for r in range(dims.n_radio):
            for o in range(dims.n_uav):
                s.ledger_pos[e, r, o, cid] = wp.uint16(0)
                s.ledger_neg[e, r, o, cid] = wp.uint16(0)
                s.ledger_gen[e, r, o, cid] = wp.int16(-1)
                s.ledger_rx_age_s[e, r, o, cid] = wp.float16(0.0)
        for u in range(dims.n_uav):
            for d in range(dims.n_radio):
                for o in range(dims.n_uav):
                    s.ack_origin_gen[e, u, d, o * dims.num_cells + cid] = wp.int16(-1)


def reset_all_v67(
    state: NetworkBuffersV67,
    dims: SimDimsV67,
    done: np.ndarray,
    num_envs: int,
    seed: int,
    device: str,
) -> None:
    d_gpu = wp.array(np.asarray(done, dtype=np.uint8), dtype=wp.uint8, device=device)
    wp.launch(reset_world_v67, dim=num_envs, inputs=[state, dims, d_gpu, int(seed)], device=device)
    wp.launch(
        reset_cells_v67,
        dim=(num_envs, int(dims.num_cells)),
        inputs=[state, dims, d_gpu],
        device=device,
    )
