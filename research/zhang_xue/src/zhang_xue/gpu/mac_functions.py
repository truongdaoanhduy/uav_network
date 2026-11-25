"""Warp MAC helper functions; no automatic gradients for event simulation."""
from .state import *

if HAS_WARP:
    wp.set_module_options({"enable_backward": False})

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
