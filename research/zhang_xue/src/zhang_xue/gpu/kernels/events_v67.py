"""v6.7/v7.1 Parameterized Warp CSMA/CA Discrete-Event Step Kernel."""
from __future__ import annotations

from zhang_xue.gpu.kernels.mac_v67 import (
    _busy_v67,
    _deliver_data_v67,
    _deliver_e2e_ack_v67,
    _handle_hello_v67,
    _promote_pos_v67,
    _qpop_pos_v67,
    _qpush_v67,
    _rand_slot_v67,
    _start_v67,
)
from zhang_xue.gpu.state_v67 import (
    HAS_WARP,
    HELLO_INT_US,
    HELLO_V89_TX_US,
    NEVER_US,
    PKT_DATA,
    PKT_E2E_ACK,
    PKT_HELLO,
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
    def _select_routable_slot_v67(
        s: NetworkBuffersV67, e: int, r: int, t_min: int,
    ) -> int:
        """Prune expired packets, prefer PKT_E2E_ACK, and keep unroutable packets (store-carry-forward)."""
        pos = int(0)
        while pos < int(s.q_count[e, r]):
            phys = int(s.q_order[e, r, pos])
            if s.q_expiry[e, r, phys] <= t_min:
                _qpop_pos_v67(s, e, r, pos)
                wp.atomic_add(s.stats_ttl, e, 1)
            else:
                pos = pos + 1
        cnt = int(s.q_count[e, r])
        best_pos = int(-1)
        for p in range(cnt):
            phys = int(s.q_order[e, r, p])
            dst = int(s.q_dst[e, r, phys])
            nh = int(s.route_hop[e, r, dst]) if s.route_expiry[e, r, dst] > t_min else -1
            if nh < 0:
                continue
            if int(s.q_type[e, r, phys]) == PKT_E2E_ACK:
                _promote_pos_v67(s, e, r, p)
                return phys
            if best_pos < 0:
                best_pos = p
        if best_pos >= 0:
            phys = int(s.q_order[e, r, best_pos])
            _promote_pos_v67(s, e, r, best_pos)
            return phys
        return -1

    @wp.func
    def _arm_idle_v67(
        s: NetworkBuffersV67, dims: SimDimsV67, e: int, r: int,
        t: int, tx_p: float, cca_th: float,
    ):
        if s.radio_alive[e, r] == wp.uint8(0) or int(s.state[e, r]) != ST_IDLE:
            return
        has_work = 1 if s.hello_pending[e, r] != wp.uint8(0) else 0
        if has_work == 0 and int(s.q_count[e, r]) > 0:
            if _select_routable_slot_v67(s, e, r, t) >= 0:
                has_work = 1
        if has_work != 0 and _busy_v67(s, dims, e, r, tx_p, cca_th) == 0:
            s.state[e, r] = wp.int8(ST_DIFS)
            s.deadline[e, r] = t + dims.difs_us

    @wp.func
    def _finish_tx_v67(
        s: NetworkBuffersV67, dims: SimDimsV67, e: int, tx: int,
        noise_w: float, snr_th: float,
    ):
        st = int(s.state[e, tx])
        s.state[e, tx] = wp.int8(ST_IDLE)
        s.deadline[e, tx] = NEVER_US
        if st == ST_TX_ACK:
            s.tx_airtime[e, tx] = s.tx_airtime[e, tx] + TX_MAC_ACK_US
            nh = int(s.next_hop[e, tx])
            # ACK preambles lock every listener, including radios other than the
            # destination. Every matching lock must end with the physical frame.
            for rx in range(dims.n_radio):
                if int(s.rx_lock[e, rx]) != tx:
                    continue
                sinr = s.rx_sig[e, rx] / (noise_w + s.rx_interf[e, rx])
                ok = (s.rx_corrupt[e, rx] == wp.uint8(0)) and (sinr >= snr_th) and (s.gain[e, tx, rx] > 0.0)
                s.rx_lock[e, rx] = wp.int8(-1)
                s.rx_corrupt[e, rx] = wp.uint8(0)
                if rx == nh and ok and int(s.state[e, rx]) == ST_WAIT_ACK and int(s.tx_pkt_id[e, rx]) == int(s.ack_pid[e, tx]):
                    s.state[e, nh] = wp.int8(ST_IDLE)
                    s.deadline[e, nh] = NEVER_US
                    s.retries[e, nh] = wp.uint8(0)
                    slot = int(s.active_slot[e, nh])
                    if slot >= 0:
                        for pos in range(int(s.q_count[e, nh])):
                            if int(s.q_order[e, nh, pos]) == slot:
                                _qpop_pos_v67(s, e, nh, pos)
                                break
                    s.active_slot[e, nh] = wp.int8(-1)
                    wp.atomic_add(s.stats_mac_ack, e, 1)
            return

        slot = int(s.active_slot[e, tx])
        is_hello = 1 if slot < 0 else 0
        nh = int(s.next_hop[e, tx])
        dur = (HELLO_V89_TX_US if dims.enable_peer_telemetry != 0 else 360) if is_hello == 1 else (360 if int(s.q_type[e, tx, slot]) == PKT_E2E_ACK else (280 + 32 * int(s.q_entries[e, tx, slot])))
        s.tx_airtime[e, tx] = s.tx_airtime[e, tx] + dur

        for rx in range(dims.n_radio):
            if rx == tx or int(s.rx_lock[e, rx]) != tx:
                continue
            sinr = s.rx_sig[e, rx] / (noise_w + s.rx_interf[e, rx])
            ok = (s.rx_corrupt[e, rx] == wp.uint8(0)) and (sinr >= snr_th) and (s.gain[e, tx, rx] > 0.0)
            s.rx_lock[e, rx] = wp.int8(-1)
            s.rx_corrupt[e, rx] = wp.uint8(0)
            if not ok:
                continue
            if is_hello == 1:
                _handle_hello_v67(s, dims, e, rx, tx)
                continue
            if rx != nh:
                continue
            rst = int(s.state[e, rx])
            if rst == ST_IDLE or rst == ST_DIFS or rst == ST_BACKOFF:
                if rst == ST_BACKOFF:
                    rem = s.deadline[e, rx] - s.now[e]
                    s.backoff_rem[e, rx] = rem if rem > 0 else 0
                    s.frozen[e, rx] = wp.uint8(1)
                s.ack_dst[e, rx] = wp.int8(tx)
                s.ack_pid[e, rx] = s.q_pid[e, tx, slot]
                s.state[e, rx] = wp.int8(ST_SIFS)
                s.deadline[e, rx] = s.now[e] + dims.sifs_us
            # Link-layer reception and application TTL are separate: a decoded
            # expired frame still receives its physical MAC ACK, but no custody.
            if s.now[e] >= s.q_expiry[e, tx, slot]:
                wp.atomic_add(s.stats_ttl, e, 1)
                continue
            dst = int(s.q_dst[e, tx, slot])
            src = int(s.q_src[e, tx, slot])
            seq = int(s.q_seq[e, tx, slot])
            ptype = int(s.q_type[e, tx, slot])
            if dst == rx:
                if ptype == PKT_DATA:
                    _deliver_data_v67(s, dims, e, rx, tx, slot, src, seq)
                else:
                    _deliver_e2e_ack_v67(s, dims, e, rx, src, seq)
            else:
                hops = int(s.q_hops[e, tx, slot]) + 1
                vis = s.q_visited[e, tx, slot]
                if hops <= dims.max_hops and (vis & (wp.uint32(1) << wp.uint32(rx))) == wp.uint32(0) and s.now[e] < s.q_expiry[e, tx, slot]:
                    hop_pid = s.q_pid[e, tx, slot]
                    seen = int(0)
                    for k in range(16):
                        if s.received_pid[e, rx, k] == hop_pid:
                            seen = 1
                            break
                    if seen != 0:
                        continue
                    ent = int(s.q_entries[e, tx, slot])
                    ns = _qpush_v67(
                        s, dims, e, rx, ptype, s.next_pid[e], src, dst, seq,
                        s.q_expiry[e, tx, slot], (vis | (wp.uint32(1) << wp.uint32(rx))) & dims.node_mask_all, hops, ent,
                    )
                    s.next_pid[e] = s.next_pid[e] + 1
                    if ns >= 0:
                        # Failed queue admission is not custody: a later MAC
                        # retry may succeed after the relay gains queue space.
                        ptr = int(s.received_ptr[e, rx])
                        s.received_pid[e, rx, ptr] = hop_pid
                        s.received_ptr[e, rx] = wp.uint8((ptr + 1) & 15)
                        if ptype == PKT_DATA:
                            for k in range(ent):
                                s.q_word0[e, rx, ns, k] = s.q_word0[e, tx, slot, k]
                                s.q_word1[e, rx, ns, k] = s.q_word1[e, tx, slot, k]
                else:
                    if (vis & (wp.uint32(1) << wp.uint32(rx))) != wp.uint32(0):
                        wp.atomic_add(s.stats_loop, e, 1)
                    else:
                        wp.atomic_add(s.stats_ttl, e, 1)

        if is_hello == 0:
            s.state[e, tx] = wp.int8(ST_WAIT_ACK)
            s.deadline[e, tx] = s.now[e] + dims.ack_timeout_us

    @wp.func
    def _prepare_starts_v93(
        s: NetworkBuffersV67, dims: SimDimsV67, e: int, t: int,
        tx_p: float, cca_th: float,
    ) -> wp.uint32:
        """Two phases: finish every DIFS, then select starts before any TX."""
        for r in range(dims.n_radio):
            if s.radio_alive[e, r] == wp.uint8(0):
                continue
            if int(s.state[e, r]) == ST_DIFS and s.deadline[e, r] == t:
                if _busy_v67(s, dims, e, r, tx_p, cca_th) != 0:
                    s.state[e, r] = wp.int8(ST_IDLE)
                    s.deadline[e, r] = NEVER_US
                else:
                    rem = s.backoff_rem[e, r] if s.frozen[e, r] != wp.uint8(0) else (_rand_slot_v67(s, dims, e, r, int(s.retries[e, r])) * dims.slot_us)
                    s.frozen[e, r] = wp.uint8(0)
                    s.state[e, r] = wp.int8(ST_BACKOFF)
                    s.deadline[e, r] = t + rem
        ready = wp.uint32(0)
        for r in range(dims.n_radio):
            if s.radio_alive[e, r] == wp.uint8(0) or s.deadline[e, r] != t:
                continue
            st = int(s.state[e, r])
            if st == ST_SIFS:
                ready = ready | (wp.uint32(1) << wp.uint32(r))
            elif st == ST_BACKOFF and _busy_v67(s, dims, e, r, tx_p, cca_th) == 0:
                if s.hello_pending[e, r] != wp.uint8(0) or _select_routable_slot_v67(s, e, r, t) >= 0:
                    ready = ready | (wp.uint32(1) << wp.uint32(r))
        return ready

    @wp.kernel
    def step_v67(
        s: NetworkBuffersV67,
        dims: SimDimsV67,
        dur_us: int,
        max_ev: int,
        tx_p: float,
        cca_th: float,
        noise_w: float,
        snr_th: float,
    ):
        e = wp.tid()
        # Invalid transitions remain unusable until reset, including graph/capture
        # callers that deliberately disable synchronous host guard reads.
        if s.overflow[e] != wp.uint8(0):
            return
        if s.now[e] < 0 or s.now[e] >= NEVER_US - dur_us:
            s.overflow[e] = wp.uint8(1)
            return
        end_t = s.now[e] + dur_us
        ev = int(0)
        completed = int(0)
        for r in range(dims.n_radio):
            _arm_idle_v67(s, dims, e, r, s.now[e], tx_p, cca_th)
        while ev <= max_ev:
            t_min = end_t + 1
            for r in range(dims.n_radio):
                if s.radio_alive[e, r] == wp.uint8(0):
                    continue
                if s.hello_due[e, r] < t_min:
                    t_min = s.hello_due[e, r]
                if s.deadline[e, r] < t_min:
                    t_min = s.deadline[e, r]
            if t_min > end_t:
                completed = 1
                break
            if ev == max_ev:
                break
            s.now[e] = t_min
            ev = ev + 1
            for r in range(dims.n_radio):
                if s.radio_alive[e, r] == wp.uint8(0):
                    continue
                if s.hello_due[e, r] <= t_min:
                    s.hello_pending[e, r] = wp.uint8(1)
                    for d in range(dims.n_radio):
                        if s.route_expiry[e, r, d] <= t_min:
                            s.route_hop[e, r, d] = wp.int8(-1)
                            s.route_dist[e, r, d] = wp.int8(127)
                            s.route_mask[e, r, d] = wp.uint32(0)
                            s.route_expiry[e, r, d] = 0
                        s.hello_dv_hop[e, r, d] = s.route_hop[e, r, d]
                        s.hello_dv_dist[e, r, d] = s.route_dist[e, r, d]
                        s.hello_dv_mask[e, r, d] = s.route_mask[e, r, d]
                    s.hello_due[e, r] = t_min + HELLO_INT_US
                if s.deadline[e, r] == t_min and (int(s.state[e, r]) == ST_TX or int(s.state[e, r]) == ST_TX_ACK):
                    _finish_tx_v67(s, dims, e, r, noise_w, snr_th)
            ready_mask = _prepare_starts_v93(s, dims, e, t_min, tx_p, cca_th)
            for r in range(dims.n_radio):
                if s.radio_alive[e, r] == wp.uint8(0):
                    continue
                st = int(s.state[e, r])
                if st == ST_BACKOFF and s.deadline[e, r] == t_min:
                    if (ready_mask & (wp.uint32(1) << wp.uint32(r))) == wp.uint32(0):
                        s.state[e, r] = wp.int8(ST_IDLE)
                        s.deadline[e, r] = NEVER_US
                    elif s.hello_pending[e, r] != wp.uint8(0):
                        s.hello_pending[e, r] = wp.uint8(0)
                        s.active_slot[e, r] = wp.int8(-1)
                        pid = s.next_pid[e]
                        s.next_pid[e] = pid + 1
                        _start_v67(s, dims, e, r, HELLO_V89_TX_US if dims.enable_peer_telemetry != 0 else 360, -1, 0, pid, tx_p, cca_th, noise_w, snr_th, ready_mask)
                    elif int(s.q_count[e, r]) > 0:
                        phys0 = _select_routable_slot_v67(s, e, r, t_min)
                        if phys0 < 0:
                            s.state[e, r] = wp.int8(ST_IDLE)
                            s.deadline[e, r] = NEVER_US
                        else:
                            dst = int(s.q_dst[e, r, phys0])
                            nh = int(s.route_hop[e, r, dst])
                            s.active_slot[e, r] = wp.int8(phys0)
                            dur = 360 if int(s.q_type[e, r, phys0]) == PKT_E2E_ACK else (280 + 32 * int(s.q_entries[e, r, phys0]))
                            _start_v67(s, dims, e, r, dur, nh, 0, int(s.q_pid[e, r, phys0]), tx_p, cca_th, noise_w, snr_th, ready_mask)
                    else:
                        s.state[e, r] = wp.int8(ST_IDLE)
                        s.deadline[e, r] = NEVER_US
                elif st == ST_SIFS and s.deadline[e, r] == t_min:
                    _start_v67(s, dims, e, r, TX_MAC_ACK_US, int(s.ack_dst[e, r]), 1, int(s.ack_pid[e, r]), tx_p, cca_th, noise_w, snr_th, ready_mask)
                elif st == ST_WAIT_ACK and s.deadline[e, r] == t_min:
                    ret = int(s.retries[e, r]) + 1
                    s.retries[e, r] = wp.uint8(ret)
                    wp.atomic_add(s.stats_retry, e, 1)
                    if ret >= dims.max_retx:
                        wp.atomic_add(s.stats_drop, e, 1)
                        bad_nh = int(s.next_hop[e, r])
                        if bad_nh >= 0:
                            for d in range(dims.n_radio):
                                if int(s.route_hop[e, r, d]) == bad_nh:
                                    s.route_hop[e, r, d] = wp.int8(-1)
                                    s.route_expiry[e, r, d] = 0
                        slot = int(s.active_slot[e, r])
                        for pos in range(int(s.q_count[e, r])):
                            if int(s.q_order[e, r, pos]) == slot:
                                _qpop_pos_v67(s, e, r, pos)
                                break
                        s.active_slot[e, r] = wp.int8(-1)
                        s.retries[e, r] = wp.uint8(0)
                    s.state[e, r] = wp.int8(ST_IDLE)
                    s.deadline[e, r] = NEVER_US
                _arm_idle_v67(s, dims, e, r, t_min, tx_p, cca_th)
        if completed == 0:
            s.overflow[e] = wp.uint8(1)
        else:
            s.now[e] = end_t
        s.event_count[e] = s.event_count[e] + ev
