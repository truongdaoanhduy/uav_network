"""Parallel, episode-isolated reset for v5.7 7-radio SoA; no whole-ledger loops."""
from __future__ import annotations

import warp as wp
from zhang_xue.compat.warp_mac import NetworkBuffers

wp.set_module_options({"enable_backward": False})


@wp.kernel(enable_backward=False)
def reset_world(s: NetworkBuffers, done: wp.array(dtype=wp.uint8), seed: int):
    e = wp.tid()
    if done[e] == wp.uint8(0):
        return
    s.now[e] = 0
    s.mission_epoch_us[e] = 0
    s.episode[e] = s.episode[e] + 1
    s.overflow[e] = 0
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
    for r in range(7):
        s.rng[e, r] = wp.uint32(seed + e * 7919 + r * 101 + 1)
        s.radio_alive[e, r] = wp.uint8(1)
        s.state[e, r] = 0
        s.deadline[e, r] = 2_000_000_000
        s.hello_due[e,r]=2000*r + (17*e)%1000
        s.hello_pending[e, r] = 0
        s.active_slot[e, r] = -1
        s.next_hop[e, r] = -1
        s.tx_pkt_id[e, r] = -1
        s.ack_dst[e, r] = -1
        s.ack_pid[e, r] = -1
        s.retries[e, r] = 0
        s.backoff_rem[e, r] = 0
        s.frozen[e, r] = 0
        s.rx_lock[e, r] = -1
        s.rx_corrupt[e, r] = 0
        s.rx_sig[e, r] = 0.0
        s.rx_interf[e, r] = 0.0
        s.tx_airtime[e, r] = 0
        s.q_head[e, r] = 0
        s.q_count[e, r] = 0
        for dst in range(7):
            s.route_hop[e, r, dst] = -1
            s.route_dist[e, r, dst] = 127
            s.route_mask[e, r, dst] = 0
            s.route_expiry[e, r, dst] = 0
            s.hello_dv_hop[e, r, dst] = -1
            s.hello_dv_dist[e, r, dst] = 127
            s.hello_dv_mask[e, r, dst] = 0


@wp.kernel(enable_backward=False)
def reset_ledger(s: NetworkBuffers, done: wp.array(dtype=wp.uint8)):
    tid = wp.tid()
    e = tid // 16800
    if done[e] == wp.uint8(0):
        return
    r = (tid // 2400) % 7
    origin = (tid // 400) % 6
    cid = tid % 400
    s.ledger_pos[e, r, origin, cid] = wp.uint16(0)
    s.ledger_neg[e, r, origin, cid] = wp.uint16(0)
    s.ledger_gen[e, r, origin, cid] = -1
    s.ledger_rx_age_s[e, r, origin, cid] = 0.0


@wp.kernel(enable_backward=False)
def reset_ack_watermarks(s: NetworkBuffers, done: wp.array(dtype=wp.uint8)):
    tid = wp.tid()
    e = tid // 16800
    if done[e] == wp.uint8(0):
        return
    u = (tid // 2800) % 6
    dst = (tid // 400) % 7
    cid = tid % 400
    s.ack_pos[e, u, dst, cid] = wp.uint16(0)
    s.ack_neg[e, u, dst, cid] = wp.uint16(0)
    s.ack_gen[e, u, dst, cid] = -1


@wp.kernel(enable_backward=False)
def reset_dedup(s: NetworkBuffers, done: wp.array(dtype=wp.uint8)):
    tid = wp.tid()
    e = tid // 784
    if done[e] == wp.uint8(0):
        return
    r = (tid // 112) % 7
    org = (tid // 16) % 7
    k = tid % 16
    s.delivered_seq[e, r, org, k] = -1
    s.delivered_ack_seq[e, r, org, k] = -1
    if k == 0:
        s.delivered_ptr[e, r, org] = 0
        s.delivered_ack_ptr[e, r, org] = 0


@wp.kernel(enable_backward=False)
def reset_pending(s: NetworkBuffers, done: wp.array(dtype=wp.uint8)):
    tid = wp.tid()
    e = tid // 96
    if done[e] == wp.uint8(0):
        return
    u=(tid//16)%6;p=tid%16
    s.pending_valid[e, u, p] = 0
    s.pending_seq[e, u, p] = -1
    s.pending_expiry[e, u, p] = 0
    s.pending_entries[e, u, p] = 0
    s.pending_dst[e, u, p] = -1


@wp.kernel(enable_backward=False)
def reset_gcs(s: NetworkBuffers, done: wp.array(dtype=wp.uint8)):
    tid = wp.tid()
    e = tid // 400
    if done[e] == wp.uint8(0):
        return
    cid = tid % 400
    s.gcs_latest[e, cid] = -1
    s.gcs_last_rx_us[e, cid] = -1


def reset_all(state, done, num_envs: int, seed: int, device: str):
    arr = wp.array(done, dtype=wp.uint8, device=device)
    for kernel, dim, needs_seed in (
        (reset_world, num_envs, True),
        (reset_ledger, num_envs * 16800, False),
        (reset_ack_watermarks, num_envs * 16800, False),
        (reset_dedup, num_envs * 784, False),
        (reset_pending, num_envs * 96, False),
        (reset_gcs, num_envs * 400, False),
    ):
        args = [state, arr]
        if needs_seed:
            args.append(seed)
        wp.launch(kernel, dim=dim, inputs=args, device=device)
