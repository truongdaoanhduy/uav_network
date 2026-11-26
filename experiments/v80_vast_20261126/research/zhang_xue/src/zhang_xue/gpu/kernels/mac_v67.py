"""v6.7/v7.1 Parameterized Warp MAC Device Functions (Zero-Copy FIFO, Data-Mule, UavNetSim PHY)."""
from __future__ import annotations

from zhang_xue.gpu.state_v67 import (
    HAS_WARP,
    NEVER_US,
    PKT_DATA,
    PKT_E2E_ACK,
    PKT_TTL_US,
    ROUTE_TTL_US,
    ST_BACKOFF,
    ST_DIFS,
    ST_IDLE,
    ST_SIFS,
    ST_TX,
    ST_TX_ACK,
    ST_WAIT_ACK,
    TX_MAC_ACK_US,
    NetworkBuffersV67,
    SimDimsV67,
    wp,
)

if HAS_WARP:
    @wp.func
    def _rand_slot_v67(
        s: NetworkBuffersV67, dims: SimDimsV67, e: int, r: int, retries: int
    ) -> int:
        x = wp.uint32(s.rng[e, r] * wp.uint32(1664525) + wp.uint32(1013904223))
        s.rng[e, r] = x
        ret_clamped = wp.clamp(retries, 0, dims.max_retx)
        cw = ((dims.cw_min + 1) << ret_clamped) - 1
        return int((x >> wp.uint32(16)) % wp.uint32(cw + 1))

    @wp.func
    def _qpush_v67(
        s: NetworkBuffersV67, dims: SimDimsV67, e: int, r: int,
        ptype: int, pid: int, src: int, dst: int, seq: int, exp: int,
        vis: wp.uint32, hops: int, ent: int,
    ) -> int:
        cnt = int(s.q_count[e, r])
        if cnt >= dims.q_cap:
            wp.atomic_add(s.stats_drop, e, 1)
            return -1
        phys = int(-1)
        for p in range(dims.q_cap):
            if s.q_used[e, r, p] == wp.uint8(0):
                phys = p
                break
        if phys < 0:
            wp.atomic_add(s.stats_drop, e, 1)
            return -1
        s.q_used[e, r, phys] = wp.uint8(1)
        s.q_order[e, r, cnt] = wp.uint8(phys)
        s.q_count[e, r] = wp.uint8(cnt + 1)
        s.q_type[e, r, phys] = wp.uint8(ptype)
        s.q_pid[e, r, phys] = pid
        s.q_src[e, r, phys] = wp.int8(src)
        s.q_dst[e, r, phys] = wp.int8(dst)
        s.q_seq[e, r, phys] = seq
        s.q_expiry[e, r, phys] = exp
        s.q_visited[e, r, phys] = vis
        s.q_hops[e, r, phys] = wp.uint8(hops)
        s.q_entries[e, r, phys] = wp.uint8(ent)
        return phys

    @wp.func
    def _qpop_pos_v67(s: NetworkBuffersV67, e: int, r: int, pos: int):
        cnt = int(s.q_count[e, r])
        if pos < 0 or pos >= cnt:
            return
        phys = int(s.q_order[e, r, pos])
        s.q_used[e, r, phys] = wp.uint8(0)
        for i in range(pos, cnt - 1):
            s.q_order[e, r, i] = s.q_order[e, r, i + 1]
        s.q_count[e, r] = wp.uint8(cnt - 1)

    @wp.func
    def _promote_pos_v67(s: NetworkBuffersV67, e: int, r: int, pos: int):
        if pos <= 0:
            return
        phys = s.q_order[e, r, pos]
        for i in range(pos, 0, -1):
            s.q_order[e, r, i] = s.q_order[e, r, i - 1]
        s.q_order[e, r, 0] = phys

    @wp.func
    def _sum_interf_v67(
        s: NetworkBuffersV67, dims: SimDimsV67, e: int, rx: int, excl: int, tx_p: float,
    ) -> float:
        tot = float(0.0)
        for o in range(dims.n_radio):
            if o != excl and o != rx and s.radio_alive[e, o] != wp.uint8(0):
                st = int(s.state[e, o])
                if st == ST_TX or st == ST_TX_ACK:
                    tot += tx_p * s.gain[e, o, rx]
        return tot

    @wp.func
    def _busy_v67(
        s: NetworkBuffersV67, dims: SimDimsV67, e: int, r: int, tx_p: float, cca_th: float,
    ) -> int:
        if int(s.state[e, r]) == ST_TX or int(s.state[e, r]) == ST_TX_ACK:
            return 1
        tot = float(0.0)
        for o in range(dims.n_radio):
            if o != r and s.radio_alive[e, o] != wp.uint8(0):
                st = int(s.state[e, o])
                if st == ST_TX or st == ST_TX_ACK:
                    tot += tx_p * s.gain[e, o, r]
        return 1 if tot >= cca_th else 0

    @wp.func
    def _dedup_seen_v67(
        seq_arr: wp.array4d(dtype=wp.int32), ptr_arr: wp.array3d(dtype=wp.uint8),
        e: int, rx: int, src: int, seq: int,
    ) -> int:
        for k in range(16):
            if seq_arr[e, rx, src, k] == seq:
                return 1
        ptr = int(ptr_arr[e, rx, src])
        seq_arr[e, rx, src, ptr] = seq
        ptr_arr[e, rx, src] = wp.uint8((ptr + 1) & 15)
        return 0

    @wp.func
    def _start_v67(
        s: NetworkBuffersV67, dims: SimDimsV67, e: int, tx: int, dur: int,
        nh: int, is_ack: int, pid: int, tx_p: float, cca_th: float, noise_w: float, snr_th: float,
    ):
        # Half-duplex: starting TX corrupts any frame currently locked on this radio.
        if int(s.rx_lock[e, tx]) >= 0:
            s.rx_corrupt[e, tx] = wp.uint8(1)
        s.state[e, tx] = wp.int8(ST_TX_ACK if is_ack == 1 else ST_TX)
        s.deadline[e, tx] = s.now[e] + dur
        s.next_hop[e, tx] = wp.int8(nh)
        s.tx_pkt_id[e, tx] = pid
        rx_lock_th = cca_th if dims.rx_lock_on_cca != 0 else (noise_w * snr_th)
        for rx in range(dims.n_radio):
            if rx == tx or s.radio_alive[e, rx] == wp.uint8(0):
                continue
            p_rx = tx_p * s.gain[e, tx, rx]
            if _busy_v67(s, dims, e, rx, tx_p, cca_th) != 0:
                rst = int(s.state[e, rx])
                if rst == ST_DIFS:
                    s.state[e, rx] = wp.int8(ST_IDLE)
                    s.deadline[e, rx] = NEVER_US
                elif rst == ST_BACKOFF:
                    rem = s.deadline[e, rx] - s.now[e]
                    s.backoff_rem[e, rx] = rem if rem > 0 else 0
                    s.frozen[e, rx] = wp.uint8(1)
                    s.state[e, rx] = wp.int8(ST_IDLE)
                    s.deadline[e, rx] = NEVER_US
            if int(s.state[e, rx]) == ST_TX or int(s.state[e, rx]) == ST_TX_ACK:
                continue
            if int(s.rx_lock[e, rx]) < 0:
                if p_rx >= rx_lock_th:
                    s.rx_lock[e, rx] = wp.int8(tx)
                    s.rx_corrupt[e, rx] = wp.uint8(0)
                    s.rx_sig[e, rx] = p_rx
                    s.rx_interf[e, rx] = _sum_interf_v67(s, dims, e, rx, tx, tx_p)
            else:
                interf = _sum_interf_v67(s, dims, e, rx, int(s.rx_lock[e, rx]), tx_p)
                if interf > s.rx_interf[e, rx]:
                    s.rx_interf[e, rx] = interf

    @wp.func
    def _handle_hello_v67(s: NetworkBuffersV67, dims: SimDimsV67, e: int, rx: int, tx: int):
        exp = s.now[e] + ROUTE_TTL_US
        s.route_hop[e, rx, tx] = wp.int8(tx)
        s.route_dist[e, rx, tx] = wp.int8(1)
        s.route_mask[e, rx, tx] = (wp.uint32(1) << wp.uint32(rx)) | (wp.uint32(1) << wp.uint32(tx))
        s.route_expiry[e, rx, tx] = exp
        for dst in range(dims.n_radio):
            if dst == rx or dst == tx:
                continue
            if int(s.hello_dv_hop[e, tx, dst]) < 0:
                if int(s.route_hop[e, rx, dst]) == tx:
                    s.route_hop[e, rx, dst] = wp.int8(-1)
                    s.route_expiry[e, rx, dst] = 0
                continue
            pm = s.hello_dv_mask[e, tx, dst]
            if (pm & (wp.uint32(1) << wp.uint32(rx))) != wp.uint32(0):
                if int(s.route_hop[e, rx, dst]) == tx:
                    s.route_hop[e, rx, dst] = wp.int8(-1)
                    s.route_expiry[e, rx, dst] = 0
                continue
            ch = int(s.hello_dv_dist[e, tx, dst]) + 1
            if ch > dims.max_hops:
                continue
            rh = int(s.route_hop[e, rx, dst])
            if rh < 0 or s.route_expiry[e, rx, dst] <= s.now[e] or ch < int(s.route_dist[e, rx, dst]) or rh == tx:
                s.route_hop[e, rx, dst] = wp.int8(tx)
                s.route_dist[e, rx, dst] = wp.int8(ch)
                s.route_mask[e, rx, dst] = (pm | (wp.uint32(1) << wp.uint32(rx))) & dims.node_mask_all
                s.route_expiry[e, rx, dst] = exp

    @wp.func
    def _deliver_data_v67(s: NetworkBuffersV67, dims: SimDimsV67, e: int, rx: int, tx_node: int, slot: int, src: int, seq: int):
        if _dedup_seen_v67(s.delivered_seq, s.delivered_ptr, e, rx, src, seq) == 0:
            if rx == dims.gcs_id:
                wp.atomic_add(s.stats_gcs, e, 1)
            else:
                wp.atomic_add(s.stats_peer, e, 1)
            # The caller saved tx_node before clearing rx_lock; never index with -1.
            ent = int(s.q_entries[e, tx_node, slot])
            rx_now_s = float(wp.max(0, s.now[e] - s.mission_epoch_us[e])) * 1.0e-6
            for k in range(dims.k_entries):
                if k >= ent:
                    break
                w0 = s.q_word0[e, tx_node, slot, k]
                w1 = s.q_word1[e, tx_node, slot, k]
                if dims.scalable_wire == 0:
                    org = int(w0 & wp.uint32(0x7))
                    cid = int((w0 >> wp.uint32(3)) & wp.uint32(0x1FF))
                    gen = int((w0 >> wp.uint32(12)) & wp.uint32(0xFFFFF))
                else:
                    org = int(w0 & wp.uint32(0x1F))
                    cid = int((w0 >> wp.uint32(5)) & wp.uint32(0xFFF))
                    gen = int((w0 >> wp.uint32(17)) & wp.uint32(0x7FFF))
                np_v = wp.uint16(w1 & wp.uint32(0xFFFF))
                nn_v = wp.uint16((w1 >> wp.uint32(16)) & wp.uint32(0xFFFF))
                if org >= dims.n_uav or cid >= dims.num_cells:
                    continue
                if gen > 32767:
                    gen = 32767
                old_g = int(s.ledger_gen[e, rx, org, cid])
                new_info = int(0)
                if dims.aoi_ewma != 0:
                    if gen > old_g:
                        new_info = 1
                        s.ledger_pos[e, rx, org, cid] = np_v
                        s.ledger_neg[e, rx, org, cid] = nn_v
                        s.ledger_gen[e, rx, org, cid] = wp.int16(gen)
                        s.ledger_rx_age_s[e, rx, org, cid] = wp.float16(wp.max(0.0, rx_now_s - float(gen)))
                else:
                    if np_v > s.ledger_pos[e, rx, org, cid] or nn_v > s.ledger_neg[e, rx, org, cid]:
                        new_info = 1
                    if np_v > s.ledger_pos[e, rx, org, cid]:
                        s.ledger_pos[e, rx, org, cid] = np_v
                    if nn_v > s.ledger_neg[e, rx, org, cid]:
                        s.ledger_neg[e, rx, org, cid] = nn_v
                    if gen > old_g:
                        s.ledger_gen[e, rx, org, cid] = wp.int16(gen)
                        s.ledger_rx_age_s[e, rx, org, cid] = wp.float16(wp.max(0.0, rx_now_s - float(gen)))
                if rx == dims.gcs_id and (gen > int(s.gcs_latest[e, cid]) or (gen == int(s.gcs_latest[e, cid]) and new_info != 0)):
                    s.gcs_latest[e, cid] = wp.int16(gen)
                    s.gcs_last_rx_us[e, cid] = s.now[e]
        pid = s.next_pid[e]
        s.next_pid[e] = pid + 1
        _qpush_v67(s, dims, e, rx, PKT_E2E_ACK, pid, rx, src, seq, s.now[e] + PKT_TTL_US, wp.uint32(1) << wp.uint32(rx), 0, 0)

    @wp.func
    def _deliver_e2e_ack_v67(s: NetworkBuffersV67, dims: SimDimsV67, e: int, rx: int, src: int, seq: int):
        if _dedup_seen_v67(s.delivered_ack_seq, s.delivered_ack_ptr, e, rx, src, seq) != 0:
            return
        wp.atomic_add(s.stats_e2e, e, 1)
        if rx >= dims.n_uav:
            return
        for p in range(dims.p_cap):
            if s.pending_valid[e, rx, p] != wp.uint8(0) and s.pending_seq[e, rx, p] == seq:
                dst = int(s.pending_dst[e, rx, p])
                ent = int(s.pending_entries[e, rx, p])
                for k in range(dims.k_entries):
                    if k >= ent:
                        break
                    w0 = s.pending_word0[e, rx, p, k]
                    if dims.scalable_wire == 0:
                        org = int(w0 & wp.uint32(0x7))
                        cid = int((w0 >> wp.uint32(3)) & wp.uint32(0x1FF))
                        gen = int((w0 >> wp.uint32(12)) & wp.uint32(0xFFFFF))
                    else:
                        org = int(w0 & wp.uint32(0x1F))
                        cid = int((w0 >> wp.uint32(5)) & wp.uint32(0xFFF))
                        gen = int((w0 >> wp.uint32(17)) & wp.uint32(0x7FFF))
                    if org >= dims.n_uav or cid >= dims.num_cells:
                        continue
                    if gen > 32767:
                        gen = 32767
                    idx = org * dims.num_cells + cid
                    if gen > int(s.ack_origin_gen[e, rx, dst, idx]):
                        s.ack_origin_gen[e, rx, dst, idx] = wp.int16(gen)
                s.pending_valid[e, rx, p] = wp.uint8(0)
                break
