"""v5.7 Executable Seven-Radio Warp CSMA/CA Simulator (6 UAVs + 1 GCS).

Upgrades over v5.6:
  - Removed dead duplicate kernels (B14) so only modular reset/inject/event kernels compile.
  - Fixed weak-signal RX lock: requires received power >= cca_th_w (B1).
  - Fixed multi-interferer SINR: computes TRUE SUM of all concurrent transmitters on lock and mid-packet (B2).
  - Fixed immutable HELLO DV snapshot (`hello_dv_*` captured at HELLO enqueue, read at RX) for CPU-Warp parity.
  - Fixed duplicate DATA handling: re-sends E2E_ACK on duplicate DATA, separate dedup for DATA vs E2E_ACK (B5).
  - Fixed `stats_peer` to count novel peer cells (not packets) and added `ledger_rx_age_s`, `gcs_last_rx_us`,
    `mission_epoch_us`, and `radio_alive` (B4, B6).
  - Fixed `stats_drop` race via `wp.atomic_add` (B7), fixed LCG low-bit bias via upper-bit shift,
    fixed `rx_corrupt`/`frozen`/`backoff_rem` resets, added early `break` on `k >= entries`, and clamped timestamps.
"""
from __future__ import annotations

import math
import numpy as np

try:
    import warp as wp
    HAS_WARP = True
except ImportError:
    wp = None
    HAS_WARP = False

N_RADIO = 7
N_UAV = 6
GCS = 6
Q = 64
K = 62
CELLS = 400
DATA = 0
E2E_ACK = 1
HELLO = 2
IDLE = 0
DIFS = 1
BACKOFF = 2
TX = 3
WAIT_ACK = 4
SIFS = 5
TX_ACK = 6
NEVER = 2_000_000_000

if HAS_WARP:
    wp.set_module_options({"enable_backward": False})

    @wp.struct
    class NetworkBuffers:
        now: wp.array(dtype=wp.int32)
        mission_epoch_us: wp.array(dtype=wp.int32)
        episode: wp.array(dtype=wp.int32)
        overflow: wp.array(dtype=wp.int32)
        event_count: wp.array(dtype=wp.int32)
        next_pid: wp.array(dtype=wp.int32)
        next_seq: wp.array(dtype=wp.int32)
        rng: wp.array2d(dtype=wp.uint32)
        radio_alive: wp.array2d(dtype=wp.uint8)
        gain: wp.array3d(dtype=wp.float32)
        state: wp.array2d(dtype=wp.int32)
        deadline: wp.array2d(dtype=wp.int32)
        hello_due: wp.array2d(dtype=wp.int32)
        hello_pending: wp.array2d(dtype=wp.int32)
        hello_dv_hop: wp.array3d(dtype=wp.int32)
        hello_dv_dist: wp.array3d(dtype=wp.int32)
        hello_dv_mask: wp.array3d(dtype=wp.int32)
        active_slot: wp.array2d(dtype=wp.int32)
        next_hop: wp.array2d(dtype=wp.int32)
        tx_pkt_id: wp.array2d(dtype=wp.int32)
        ack_dst: wp.array2d(dtype=wp.int32)
        ack_pid: wp.array2d(dtype=wp.int32)
        retries: wp.array2d(dtype=wp.int32)
        backoff_rem: wp.array2d(dtype=wp.int32)
        frozen: wp.array2d(dtype=wp.int32)
        rx_lock: wp.array2d(dtype=wp.int32)
        rx_corrupt: wp.array2d(dtype=wp.int32)
        rx_interf: wp.array2d(dtype=wp.float32)
        rx_sig: wp.array2d(dtype=wp.float32)
        tx_airtime: wp.array2d(dtype=wp.int32)
        q_head: wp.array2d(dtype=wp.int32)
        q_count: wp.array2d(dtype=wp.int32)
        q_type: wp.array3d(dtype=wp.int32)
        q_pid: wp.array3d(dtype=wp.int32)
        q_src: wp.array3d(dtype=wp.int32)
        q_dst: wp.array3d(dtype=wp.int32)
        q_seq: wp.array3d(dtype=wp.int32)
        q_expiry: wp.array3d(dtype=wp.int32)
        q_visited: wp.array3d(dtype=wp.int32)
        q_hops: wp.array3d(dtype=wp.int32)
        q_entries: wp.array3d(dtype=wp.int32)
        q_word0: wp.array4d(dtype=wp.uint32)
        q_word1: wp.array4d(dtype=wp.uint32)
        route_hop: wp.array3d(dtype=wp.int32)
        route_dist: wp.array3d(dtype=wp.int32)
        route_mask: wp.array3d(dtype=wp.int32)
        route_expiry: wp.array3d(dtype=wp.int32)
        # Separate bounded dedup rings for DATA and E2E_ACK (prevents key collision)
        delivered_seq: wp.array4d(dtype=wp.int32)
        delivered_ptr: wp.array3d(dtype=wp.int32)
        delivered_ack_seq: wp.array4d(dtype=wp.int32)
        delivered_ack_ptr: wp.array3d(dtype=wp.int32)
        ledger_pos: wp.array4d(dtype=wp.uint16)
        ledger_neg: wp.array4d(dtype=wp.uint16)
        ledger_gen: wp.array4d(dtype=wp.int32)
        ledger_rx_age_s: wp.array4d(dtype=wp.float32)
        ack_pos: wp.array4d(dtype=wp.uint16)
        ack_neg: wp.array4d(dtype=wp.uint16)
        ack_gen: wp.array4d(dtype=wp.int32)
        pending_seq: wp.array3d(dtype=wp.int32)
        pending_dst: wp.array3d(dtype=wp.int32)
        pending_valid: wp.array3d(dtype=wp.int32)
        pending_expiry: wp.array3d(dtype=wp.int32)
        pending_entries: wp.array3d(dtype=wp.int32)
        pending_word0: wp.array4d(dtype=wp.uint32)
        pending_word1: wp.array4d(dtype=wp.uint32)
        gcs_latest: wp.array2d(dtype=wp.int32)
        gcs_last_rx_us: wp.array2d(dtype=wp.int32)
        stats_gcs: wp.array(dtype=wp.int32)
        stats_peer: wp.array(dtype=wp.int32)
        stats_mac_ack: wp.array(dtype=wp.int32)
        stats_e2e: wp.array(dtype=wp.int32)
        stats_retry: wp.array(dtype=wp.int32)
        stats_drop: wp.array(dtype=wp.int32)
        stats_ttl: wp.array(dtype=wp.int32)
        stats_loop: wp.array(dtype=wp.int32)

    @wp.func
    def _random_slot(s: NetworkBuffers, e: int, r: int, cw: int) -> int:
        # Upper-bit extraction from LCG avoids the notorious low-bit periodicity on power-of-2 CW+1
        x = s.rng[e, r] * wp.uint32(1664525) + wp.uint32(1013904223)
        s.rng[e, r] = x
        hi = x >> wp.uint32(16)
        return wp.int32(hi % wp.uint32(cw + 1))

    @wp.func
    def _busy(s: NetworkBuffers, e: int, r: int, tx_power: float, cca: float) -> bool:
        for m in range(7):
            if m != r and s.radio_alive[e, m] != wp.uint8(0) and (s.state[e, m] == TX or s.state[e, m] == TX_ACK):
                if tx_power * s.gain[e, m, r] >= cca:
                    return True
        return False

    @wp.func
    def _sum_interference_at(s: NetworkBuffers, e: int, rx: int, exclude_tx: int, tx_power: float) -> float:
        total = float(0.0)
        for k in range(7):
            if k != rx and k != exclude_tx and s.radio_alive[e, k] != wp.uint8(0):
                if s.state[e, k] == TX or s.state[e, k] == TX_ACK:
                    total = total + tx_power * s.gain[e, k, rx]
        return total

    @wp.func
    def _qpush(
        s: NetworkBuffers, e: int, r: int, typ: int, src: int, dst: int, seq: int, exp: int,
        mask: int, hops: int,
    ) -> int:
        n = s.q_count[e, r]
        if n >= Q:
            wp.atomic_add(s.stats_drop, e, 1)
            return -1
        slot = (s.q_head[e, r] + n) % Q
        s.q_count[e, r] = n + 1
        s.q_type[e, r, slot] = typ
        s.q_src[e, r, slot] = src
        s.q_dst[e, r, slot] = dst
        s.q_seq[e, r, slot] = seq
        s.q_expiry[e, r, slot] = wp.min(NEVER, exp)
        s.q_visited[e, r, slot] = mask
        s.q_hops[e, r, slot] = hops
        s.q_entries[e, r, slot] = 0
        s.q_pid[e, r, slot] = wp.atomic_add(s.next_pid, e, 1)
        return slot

    @wp.func
    def _qpop(s: NetworkBuffers, e: int, r: int):
        if s.q_count[e, r] > 0:
            s.q_head[e, r] = (s.q_head[e, r] + 1) % Q
            s.q_count[e, r] = s.q_count[e, r] - 1

    @wp.func
    def _promote_queued_slot(s: NetworkBuffers, e: int, r: int, offset: int):
        """Make selected routable/priority entry the MAC head without losing payload.

        All queue buffers belong to one world/thread in the event kernel; this
        bounded swap is race-free for MAC, but reorders older DATA. CPU/Warp
        parity must still be verified by event-trace comparison on GPU.
        """
        if offset <= 0 or offset >= s.q_count[e, r]:
            return
        head = s.q_head[e, r]
        chosen = (head + offset) % Q
        # Copy *all* packet metadata; swapping only q_type would corrupt
        # packet identity, ACK matching, or E2E deduplication.
        tmp = s.q_type[e, r, head]
        s.q_type[e, r, head] = s.q_type[e, r, chosen]
        s.q_type[e, r, chosen] = tmp
        tmp = s.q_pid[e, r, head]
        s.q_pid[e, r, head] = s.q_pid[e, r, chosen]
        s.q_pid[e, r, chosen] = tmp
        tmp = s.q_src[e, r, head]
        s.q_src[e, r, head] = s.q_src[e, r, chosen]
        s.q_src[e, r, chosen] = tmp
        tmp = s.q_dst[e, r, head]
        s.q_dst[e, r, head] = s.q_dst[e, r, chosen]
        s.q_dst[e, r, chosen] = tmp
        tmp = s.q_seq[e, r, head]
        s.q_seq[e, r, head] = s.q_seq[e, r, chosen]
        s.q_seq[e, r, chosen] = tmp
        tmp = s.q_expiry[e, r, head]
        s.q_expiry[e, r, head] = s.q_expiry[e, r, chosen]
        s.q_expiry[e, r, chosen] = tmp
        tmp = s.q_visited[e, r, head]
        s.q_visited[e, r, head] = s.q_visited[e, r, chosen]
        s.q_visited[e, r, chosen] = tmp
        tmp = s.q_hops[e, r, head]
        s.q_hops[e, r, head] = s.q_hops[e, r, chosen]
        s.q_hops[e, r, chosen] = tmp
        tmp = s.q_entries[e, r, head]
        s.q_entries[e, r, head] = s.q_entries[e, r, chosen]
        s.q_entries[e, r, chosen] = tmp
        for k in range(K):
            w0 = s.q_word0[e, r, head, k]
            s.q_word0[e, r, head, k] = s.q_word0[e, r, chosen, k]
            s.q_word0[e, r, chosen, k] = w0
            w1 = s.q_word1[e, r, head, k]
            s.q_word1[e, r, head, k] = s.q_word1[e, r, chosen, k]
            s.q_word1[e, r, chosen, k] = w1

    @wp.func
    def _start(s: NetworkBuffers, e: int, r: int, t: int, is_ack: bool, tx_power: float, cca: float):
        duration = 120
        if is_ack:
            s.state[e, r] = TX_ACK
            s.next_hop[e, r] = s.ack_dst[e, r]
        else:
            slot = s.active_slot[e, r]
            typ = HELLO
            count = 0
            if slot >= 0:
                typ = s.q_type[e, r, slot]
                count = s.q_entries[e, r, slot]
            if typ == DATA:
                duration = (70 + 8 * count) * 4
            else:
                duration = 360
            s.state[e, r] = TX
            s.tx_pkt_id[e, r] = s.q_pid[e, r, slot] if slot >= 0 else -1
        s.deadline[e, r] = wp.min(NEVER, t + duration)
        if s.rx_lock[e, r] >= 0:
            s.rx_corrupt[e, r] = 1
        for m in range(7):
            if m != r and s.radio_alive[e, m] != wp.uint8(0):
                sig = tx_power * s.gain[e, r, m]
                if sig >= cca:
                    if s.state[e, m] == DIFS:
                        s.state[e, m] = IDLE
                        s.deadline[e, m] = NEVER
                    elif s.state[e, m] == BACKOFF:
                        s.backoff_rem[e, m] = wp.max(0, s.deadline[e, m] - t)
                        s.frozen[e, m] = 1
                        s.state[e, m] = IDLE
                        s.deadline[e, m] = NEVER
                if s.state[e, m] != TX and s.state[e, m] != TX_ACK:
                    # B1 fix: require signal >= cca_th_w before locking receiver
                    if s.rx_lock[e, m] < 0 and sig >= cca:
                        s.rx_lock[e, m] = r
                        s.rx_sig[e, m] = sig
                        # B2 fix: initialize interference with TRUE SUM of all concurrent transmitters
                        s.rx_interf[e, m] = _sum_interference_at(s, e, m, r, tx_power)
                        s.rx_corrupt[e, m] = 0
                    elif s.rx_lock[e, m] >= 0:
                        locked = s.rx_lock[e, m]
                        if locked != r:
                            # B2 fix: worst-case overlapping SUM interference across all concurrent TX
                            cur_interf = _sum_interference_at(s, e, m, locked, tx_power)
                            if cur_interf > s.rx_interf[e, m]:
                                s.rx_interf[e, m] = cur_interf
                else:
                    if s.rx_lock[e, m] >= 0:
                        s.rx_corrupt[e, m] = 1

    @wp.func
    def _handle_hello(s: NetworkBuffers, e: int, rx: int, tx: int, t: int):
        exp = wp.min(NEVER, t + 2_000_000)
        s.route_hop[e, rx, tx] = tx
        s.route_dist[e, rx, tx] = 1
        s.route_mask[e, rx, tx] = (1 << rx) | (1 << tx)
        s.route_expiry[e, rx, tx] = exp
        # Parity fix: read from immutable HELLO DV snapshot captured at enqueue time
        for d in range(7):
            if d != rx and d != tx and s.hello_dv_hop[e, tx, d] >= 0:
                path = s.hello_dv_mask[e, tx, d]
                hops = s.hello_dv_dist[e, tx, d] + 1
                if (path & (1 << rx)) == 0 and hops <= 6:
                    if (
                        s.route_hop[e, rx, d] < 0
                        or s.route_expiry[e, rx, d] <= t
                        or hops < s.route_dist[e, rx, d]
                        or s.route_hop[e, rx, d] == tx
                    ):
                        s.route_hop[e, rx, d] = tx
                        s.route_dist[e, rx, d] = hops
                        s.route_mask[e, rx, d] = path | (1 << rx)
                        s.route_expiry[e, rx, d] = exp

    @wp.func
    def _handle_payload(s: NetworkBuffers, e: int, rx: int, tx: int, t: int):
        slot = s.active_slot[e, tx]
        if slot < 0:
            return
        typ = s.q_type[e, tx, slot]
        src = s.q_src[e, tx, slot]
        dst = s.q_dst[e, tx, slot]
        seq = s.q_seq[e, tx, slot]
        # Immediate MAC ACK is scheduled BEFORE relay TTL check (matches CPU parity)
        if s.state[e, rx] == IDLE or s.state[e, rx] == DIFS or s.state[e, rx] == BACKOFF:
            if s.state[e, rx] == BACKOFF:
                s.backoff_rem[e, rx] = wp.max(0, s.deadline[e, rx] - t)
                s.frozen[e, rx] = 1
            s.ack_dst[e, rx] = tx
            s.ack_pid[e, rx] = s.q_pid[e, tx, slot]
            s.state[e, rx] = SIFS
            s.deadline[e, rx] = wp.min(NEVER, t + 10)

        # Separate dedup ring for DATA vs E2E_ACK
        duplicate = False
        if typ == DATA:
            for j in range(16):
                if s.delivered_seq[e, rx, src, j] == seq:
                    duplicate = True
            if not duplicate:
                j = s.delivered_ptr[e, rx, src] % 16
                s.delivered_seq[e, rx, src, j] = seq
                s.delivered_ptr[e, rx, src] = (j + 1) % 16
        else:
            for j in range(16):
                if s.delivered_ack_seq[e, rx, src, j] == seq:
                    duplicate = True
            if not duplicate:
                j = s.delivered_ack_ptr[e, rx, src] % 16
                s.delivered_ack_seq[e, rx, src, j] = seq
                s.delivered_ack_ptr[e, rx, src] = (j + 1) % 16

        if rx != dst:
            if duplicate:
                return
            if s.q_hops[e, tx, slot] >= 6 or (s.q_visited[e, tx, slot] & (1 << rx)) != 0:
                s.stats_loop[e] = s.stats_loop[e] + 1
                return
            if s.q_expiry[e, tx, slot] <= t:
                s.stats_ttl[e] = s.stats_ttl[e] + 1
                return
            fwd = _qpush(
                s, e, rx, typ, src, dst, seq, s.q_expiry[e, tx, slot],
                s.q_visited[e, tx, slot] | (1 << rx), s.q_hops[e, tx, slot] + 1,
            )
            if fwd >= 0:
                entries = s.q_entries[e, tx, slot]
                s.q_entries[e, rx, fwd] = entries
                for k in range(K):
                    if k >= entries:
                        break
                    s.q_word0[e, rx, fwd, k] = s.q_word0[e, tx, slot, k]
                    s.q_word1[e, rx, fwd, k] = s.q_word1[e, tx, slot, k]
            return

        if typ == DATA:
            if not duplicate:
                if rx == GCS:
                    s.stats_gcs[e] = s.stats_gcs[e] + 1
                entries = s.q_entries[e, tx, slot]
                epoch_us = s.mission_epoch_us[e]
                rx_mission_s = wp.max(0.0, float(t - epoch_us) * 1.0e-6)
                for k in range(K):
                    if k >= entries:
                        break
                    w0 = s.q_word0[e, tx, slot, k]
                    w1 = s.q_word1[e, tx, slot, k]
                    org = wp.int32(w0 & wp.uint32(7))
                    cid = wp.int32((w0 >> wp.uint32(3)) & wp.uint32(511))
                    gen = wp.int32((w0 >> wp.uint32(12)) & wp.uint32(1048575))
                    p = wp.uint16(w1 & wp.uint32(65535))
                    n = wp.uint16((w1 >> wp.uint32(16)) & wp.uint32(65535))
                    if org >= 0 and org < 6 and cid >= 0 and cid < 400:
                        new_info = (p > s.ledger_pos[e, rx, org, cid]) or (n > s.ledger_neg[e, rx, org, cid])
                        if p > s.ledger_pos[e, rx, org, cid]:
                            s.ledger_pos[e, rx, org, cid] = p
                        if n > s.ledger_neg[e, rx, org, cid]:
                            s.ledger_neg[e, rx, org, cid] = n
                        if gen > s.ledger_gen[e, rx, org, cid]:
                            s.ledger_gen[e, rx, org, cid] = gen
                            s.ledger_rx_age_s[e, rx, org, cid] = wp.max(0.0, rx_mission_s - float(gen))
                        # B6 fix: count novel cells at peer (not packet count)
                        if rx != GCS and new_info:
                            s.stats_peer[e] = s.stats_peer[e] + 1
                        # B12 fix: update gcs_last_rx_us when gen >= gcs_latest and new_info
                        if rx == GCS and (gen > s.gcs_latest[e, cid] or (gen == s.gcs_latest[e, cid] and new_info)):
                            s.gcs_latest[e, cid] = gen
                            s.gcs_last_rx_us[e, cid] = t
            # B5 fix: ALWAYS re-send E2E_ACK even on duplicate DATA (original ACK may have been lost)
            _qpush(s, e, rx, E2E_ACK, rx, src, seq, wp.min(NEVER, t + 10_000_000), 1 << rx, 0)
        elif typ == E2E_ACK and rx < 6:
            if duplicate:
                return
            for pslot in range(16):
                if s.pending_valid[e, rx, pslot] != 0 and s.pending_seq[e, rx, pslot] == seq:
                    dst_acked = s.pending_dst[e, rx, pslot]
                    count = s.pending_entries[e, rx, pslot]
                    for k in range(K):
                        if k >= count:
                            break
                        w0 = s.pending_word0[e, rx, pslot, k]
                        w1 = s.pending_word1[e, rx, pslot, k]
                        cid = wp.int32((w0 >> wp.uint32(3)) & wp.uint32(511))
                        gen = wp.int32((w0 >> wp.uint32(12)) & wp.uint32(1048575))
                        pn = wp.uint16(w1 & wp.uint32(65535))
                        nn = wp.uint16((w1 >> wp.uint32(16)) & wp.uint32(65535))
                        if cid >= 0 and cid < 400:
                            s.ack_pos[e, rx, dst_acked, cid] = wp.max(s.ack_pos[e, rx, dst_acked, cid], pn)
                            s.ack_neg[e, rx, dst_acked, cid] = wp.max(s.ack_neg[e, rx, dst_acked, cid], nn)
                            s.ack_gen[e, rx, dst_acked, cid] = wp.max(s.ack_gen[e, rx, dst_acked, cid], gen)
                    s.pending_valid[e, rx, pslot] = 0
                    s.stats_e2e[e] = s.stats_e2e[e] + 1
                    break

    @wp.func
    def _finish(s: NetworkBuffers, e: int, r: int, t: int, tx_power: float, noise: float, snr: float):
        is_ack = s.state[e, r] == TX_ACK
        slot = s.active_slot[e, r]
        typ = HELLO
        if slot >= 0:
            typ = s.q_type[e, r, slot]
        duration = 120
        if not is_ack:
            duration = 360
            if typ == DATA:
                duration = (70 + 8 * s.q_entries[e, r, slot]) * 4
        s.tx_airtime[e, r] = s.tx_airtime[e, r] + duration
        for rx in range(7):
            if rx != r and s.rx_lock[e, rx] == r:
                ok = (
                    s.radio_alive[e, rx] != wp.uint8(0)
                    and s.rx_corrupt[e, rx] == 0
                    and s.rx_sig[e, rx] > 0.0
                    and s.gain[e, r, rx] > 0.0
                    and s.rx_sig[e, rx] / (noise + s.rx_interf[e, rx]) >= snr
                )
                s.rx_lock[e, rx] = -1
                s.rx_corrupt[e, rx] = 0
                s.rx_sig[e, rx] = 0.0
                s.rx_interf[e, rx] = 0.0
                if ok:
                    if is_ack:
                        if rx == s.ack_dst[e, r] and s.state[e, rx] == WAIT_ACK and s.ack_pid[e, r] == s.tx_pkt_id[e, rx]:
                            s.stats_mac_ack[e] = s.stats_mac_ack[e] + 1
                            _qpop(s, e, rx)
                            s.active_slot[e, rx] = -1
                            s.retries[e, rx] = 0
                            s.frozen[e, rx] = 0
                            s.backoff_rem[e, rx] = 0
                            s.state[e, rx] = IDLE
                            s.deadline[e, rx] = NEVER
                    elif typ == HELLO:
                        _handle_hello(s, e, rx, r, t)
                    elif s.next_hop[e, r] == rx:
                        _handle_payload(s, e, rx, r, t)
        if is_ack:
            s.ack_dst[e, r] = -1
            s.ack_pid[e, r] = -1
            s.state[e, r] = IDLE
            s.deadline[e, r] = NEVER
        elif typ == HELLO:
            s.hello_pending[e, r] = 0
            s.active_slot[e, r] = -1
            s.frozen[e, r] = 0
            s.backoff_rem[e, r] = 0
            s.state[e, r] = IDLE
            s.deadline[e, r] = NEVER
        else:
            s.state[e, r] = WAIT_ACK
            s.deadline[e, r] = wp.min(NEVER, t + 180)


# Stable list of backing arrays; reset/populates are executed *on device*.
SHAPES = {
    "now": ((1,), np.int32),
    "mission_epoch_us": ((1,), np.int32),
    "episode": ((1,), np.int32),
    "overflow": ((1,), np.int32),
    "event_count": ((1,), np.int32),
    "next_pid": ((1,), np.int32),
    "next_seq": ((1,), np.int32),
    "rng": ((7,), np.uint32),
    "radio_alive": ((7,), np.uint8),
    "gain": ((7, 7), np.float32),
    "state": ((7,), np.int32),
    "deadline": ((7,), np.int32),
    "hello_due": ((7,), np.int32),
    "hello_pending": ((7,), np.int32),
    "hello_dv_hop": ((7, 7), np.int32),
    "hello_dv_dist": ((7, 7), np.int32),
    "hello_dv_mask": ((7, 7), np.int32),
    "active_slot": ((7,), np.int32),
    "next_hop": ((7,), np.int32),
    "tx_pkt_id": ((7,), np.int32),
    "ack_dst": ((7,), np.int32),
    "ack_pid": ((7,), np.int32),
    "retries": ((7,), np.int32),
    "backoff_rem": ((7,), np.int32),
    "frozen": ((7,), np.int32),
    "rx_lock": ((7,), np.int32),
    "rx_corrupt": ((7,), np.int32),
    "rx_interf": ((7,), np.float32),
    "rx_sig": ((7,), np.float32),
    "tx_airtime": ((7,), np.int32),
    "q_head": ((7,), np.int32),
    "q_count": ((7,), np.int32),
    "q_type": ((7, 64), np.int32),
    "q_pid": ((7, 64), np.int32),
    "q_src": ((7, 64), np.int32),
    "q_dst": ((7, 64), np.int32),
    "q_seq": ((7, 64), np.int32),
    "q_expiry": ((7, 64), np.int32),
    "q_visited": ((7, 64), np.int32),
    "q_hops": ((7, 64), np.int32),
    "q_entries": ((7, 64), np.int32),
    "q_word0": ((7, 64, 62), np.uint32),
    "q_word1": ((7, 64, 62), np.uint32),
    "route_hop": ((7, 7), np.int32),
    "route_dist": ((7, 7), np.int32),
    "route_mask": ((7, 7), np.int32),
    "route_expiry": ((7, 7), np.int32),
    "delivered_seq": ((7, 7, 16), np.int32),
    "delivered_ptr": ((7, 7), np.int32),
    "delivered_ack_seq": ((7, 7, 16), np.int32),
    "delivered_ack_ptr": ((7, 7), np.int32),
    "ledger_pos": ((7, 6, 400), np.uint16),
    "ledger_neg": ((7, 6, 400), np.uint16),
    "ledger_gen": ((7, 6, 400), np.int32),
    "ledger_rx_age_s": ((7, 6, 400), np.float32),
    "ack_pos": ((6, 7, 400), np.uint16),
    "ack_neg": ((6, 7, 400), np.uint16),
    "ack_gen": ((6, 7, 400), np.int32),
    "pending_seq": ((6, 16), np.int32),
    "pending_dst": ((6, 16), np.int32),
    "pending_valid": ((6, 16), np.int32),
    "pending_expiry": ((6, 16), np.int32),
    "pending_entries": ((6, 16), np.int32),
    "pending_word0": ((6, 16, 62), np.uint32),
    "pending_word1": ((6, 16, 62), np.uint32),
    "gcs_latest": ((400,), np.int32),
    "gcs_last_rx_us": ((400,), np.int32),
    "stats_gcs": ((1,), np.int32),
    "stats_peer": ((1,), np.int32),
    "stats_mac_ack": ((1,), np.int32),
    "stats_e2e": ((1,), np.int32),
    "stats_retry": ((1,), np.int32),
    "stats_drop": ((1,), np.int32),
    "stats_ttl": ((1,), np.int32),
    "stats_loop": ((1,), np.int32),
}


def backing_bytes_per_environment() -> int:
    return sum(math.prod(sh) * np.dtype(dtype).itemsize for sh, dtype in SHAPES.values())


class WarpSevenRadioNetwork:
    """Batched all-on-device 7-radio MAC; no per-world Python packet scheduler."""

    def __init__(self, num_envs: int, device: str = "cpu", seed: int = 20261125):
        if not HAS_WARP:
            raise RuntimeError("warp-lang is needed for MAC kernel execution")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.envs = num_envs
        self.device = device
        self.seed = seed
        self.state = NetworkBuffers()
        self.data = {}
        dt_map = {
            np.dtype(np.int32): wp.int32,
            np.dtype(np.uint32): wp.uint32,
            np.dtype(np.uint16): wp.uint16,
            np.dtype(np.uint8): wp.uint8,
            np.dtype(np.float32): wp.float32,
        }
        for name, (shape, dtype) in SHAPES.items():
            dt = dt_map[np.dtype(dtype)]
            arr = wp.zeros((num_envs, *shape) if shape != (1,) else (num_envs,), dtype=dt, device=device)
            self.data[name] = arr
            setattr(self.state, name, arr)
        self.reset()

    def reset(self, done: np.ndarray | None = None):
        if done is None:
            done = np.ones(self.envs, dtype=np.uint8)
        d = np.asarray(done, dtype=np.uint8)
        if d.shape != (self.envs,):
            raise ValueError("reset mask shape mismatch")
        from v56_warp_reset_kernels import reset_all

        reset_all(self.state, d, self.envs, self.seed, self.device)

    def _device_int2d(self, x, shape):
        if HAS_WARP and isinstance(x, wp.array):
            if x.shape != shape or x.dtype != wp.int32:
                raise ValueError("Warp int32 shape or dtype mismatch")
            return x
        x = np.asarray(x, dtype=np.int32)
        if x.shape != shape:
            raise ValueError("input shape mismatch")
        return wp.array(x, dtype=wp.int32, device=self.device)

    def set_channel_gain(self, gain):
        if HAS_WARP and isinstance(gain, wp.array):
            if gain.shape != (self.envs, 7, 7) or gain.dtype != wp.float32:
                raise ValueError("gain shape/dtype mismatch")
            wp.copy(self.data["gain"], gain)
        else:
            g = np.asarray(gain, dtype=np.float32)
            if g.shape != (self.envs, 7, 7):
                raise ValueError("gain shape mismatch")
            wp.copy(self.data["gain"], wp.array(g, dtype=wp.float32, device=self.device))

    def set_radio_alive(self, alive):
        if HAS_WARP and isinstance(alive, wp.array):
            wp.copy(self.data["radio_alive"], alive)
        else:
            a = np.asarray(alive, dtype=np.uint8)
            if a.shape != (self.envs, 7):
                raise ValueError("radio_alive shape mismatch")
            wp.copy(self.data["radio_alive"], wp.array(a, dtype=wp.uint8, device=self.device))

    def sense(self, cell: np.ndarray, positive: np.ndarray, t_s: int):
        c = self._device_int2d(cell, (self.envs, 6))
        p = self._device_int2d(positive, (self.envs, 6))
        from v56_warp_injection_kernels import _add_observation

        wp.launch(_add_observation, dim=(self.envs, 6), inputs=[self.state, c, p, int(t_s)], device=self.device)

    def inject(self, actions: np.ndarray, t_s: int = 0):
        if not (HAS_WARP and isinstance(actions, wp.array)):
            a = np.asarray(actions, dtype=np.int32)
            if np.any((a < 0) | (a > 6)):
                raise ValueError("actions must contain indices 0..6")
        else:
            a = actions
        a = self._device_int2d(a, (self.envs, 6))
        from v56_warp_injection_kernels import _inject

        wp.launch(
            _inject,
            dim=(self.envs, 6),
            inputs=[self.state, a, wp.zeros((self.envs,), dtype=wp.int32, device=self.device)],
            device=self.device,
        )

    def run(
        self,
        duration_us: int = 1_000_000,
        max_events: int = 4096,
        tx_power_w: float = 0.5,
        cca_threshold_w: float = 2e-11,
        noise_w: float = 4e-11,
        snr_threshold: float = 3.9810717,
        experimental: bool = False,
    ):
        if duration_us < 1 or duration_us > 1_000_000 or max_events < 1:
            raise ValueError("duration_us 1..1_000_000; max_events >=1")
        if not experimental:
            raise RuntimeError("Gate 3 blocked: 7-radio event MAC kernel lacks T4 CUDA parity benchmark")
        from v56_warp_event_kernels import _step

        wp.launch(
            _step,
            dim=self.envs,
            inputs=[
                self.state,
                int(duration_us),
                int(max_events),
                float(tx_power_w),
                float(cca_threshold_w),
                float(noise_w),
                float(snr_threshold),
            ],
            device=self.device,
        )

    def snapshot(self, names=("now", "overflow", "stats_gcs", "stats_mac_ack", "stats_e2e", "q_count", "route_hop")):
        return {n: self.data[n].numpy().copy() for n in names}
