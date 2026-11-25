"""v5.7 Event Kernel for 7-Radio Warp CSMA/CA.

Fixes over v5.6:
  - Added `wp.set_module_options({"enable_backward": False})` and `@wp.kernel(enable_backward=False)`
    to eliminate the ~69s autodiff adjoint compilation explosion.
  - Hoisted pending-expiry cleanup OUT of the inner microsecond `while` loop (removes 96 * max_events
    redundant global memory scans per env per step).
  - Captures immutable HELLO DV snapshot (`hello_dv_*`) when HELLO timer fires.
  - Respects `radio_alive` (silences battery-depleted UAV radios).
  - Resets `retries = 0` when dropping TTL-expired packets; prevents HELLO from inheriting DATA retries.
  - Resets `frozen = 0, backoff_rem = 0` on WAIT_ACK timeout.
"""
import warp as wp
from v56_warp_7radio_mac import (
    NetworkBuffers,
    _qpop,
    _promote_queued_slot,
    _busy,
    _random_slot,
    _start,
    _finish,
    IDLE,
    DIFS,
    BACKOFF,
    TX,
    TX_ACK,
    SIFS,
    WAIT_ACK,
    NEVER,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel(enable_backward=False)
def _step(
    s: NetworkBuffers,
    duration_us: int,
    max_events: int,
    tx_power: float,
    cca: float,
    noise: float,
    snr: float,
):
    e = wp.tid()
    t = s.now[e]
    limit = wp.min(NEVER, t + duration_us)
    # Hoisted once at step start instead of inside every microsecond event
    for u in range(6):
        for p in range(16):
            if s.pending_valid[e, u, p] != 0 and s.pending_expiry[e, u, p] <= t:
                s.pending_valid[e, u, p] = 0

    event = int(0)
    while event < max_events and t < limit:
        for r in range(7):
            if s.radio_alive[e, r] == wp.uint8(0):
                continue
            if s.state[e, r] == IDLE:
                if s.ack_dst[e, r] >= 0:
                    s.state[e, r] = SIFS
                    s.deadline[e, r] = wp.min(NEVER, t + 10)
                else:
                    slot = wp.int32(-1)
                    hop = wp.int32(-1)
                    if s.hello_pending[e, r] != 0:
                        slot = -2
                    elif s.q_count[e, r] > 0:
                        # Expired head entries can be removed without payload copying.
                        while s.q_count[e, r] > 0:
                            head_slot = s.q_head[e, r]
                            if s.q_expiry[e, r, head_slot] <= t or s.q_dst[e, r, head_slot] == r:
                                _qpop(s, e, r)
                                s.stats_ttl[e] = s.stats_ttl[e] + 1
                                s.retries[e, r] = 0
                                continue
                            break
                        # B3-Warp fix: scan ALL queued entries, ACK first, then
                        # route-ready DATA. Unroutable head must not stall the radio.
                        # This is bounded by Q=64, not the number of MAC events.
                        chosen_offset = wp.int32(-1)
                        chosen_hop = wp.int32(-1)
                        for want_ack in range(2):
                            for offset in range(64):
                                if offset >= s.q_count[e, r]:
                                    break
                                cand_slot = (s.q_head[e, r] + offset) % 64
                                if s.q_expiry[e, r, cand_slot] <= t:
                                    continue
                                dst = s.q_dst[e, r, cand_slot]
                                if dst < 0 or dst >= 7 or dst == r:
                                    continue
                                is_e2e = s.q_type[e, r, cand_slot] == 1
                                if (want_ack == 0 and not is_e2e) or (want_ack == 1 and is_e2e):
                                    continue
                                if s.route_hop[e, r, dst] >= 0 and s.route_expiry[e, r, dst] > t:
                                    chosen_offset = offset
                                    chosen_hop = s.route_hop[e, r, dst]
                                    break
                            if chosen_offset >= 0:
                                break
                        if chosen_offset >= 0:
                            if chosen_offset > 0:
                                _promote_queued_slot(s, e, r, chosen_offset)
                                # Retry state belongs to the old head, not to the new packet.
                                s.retries[e, r] = 0
                                s.frozen[e, r] = 0
                                s.backoff_rem[e, r] = 0
                            slot = s.q_head[e, r]
                            hop = chosen_hop
                    if slot != -1 and not _busy(s, e, r, tx_power, cca):
                        s.active_slot[e, r] = slot
                        s.next_hop[e, r] = hop
                        s.state[e, r] = DIFS
                        s.deadline[e, r] = wp.min(NEVER, t + 50)

        nxt = limit
        for r in range(7):
            if s.radio_alive[e, r] == wp.uint8(0):
                continue
            nxt = wp.min(nxt, s.deadline[e, r])
            nxt = wp.min(nxt, s.hello_due[e, r])
        if nxt >= limit:
            t = limit
            break
        if nxt < t:
            s.overflow[e] = 1
            return
        t = nxt
        event = event + 1

        # 1. Periodic HELLO/DV enqueue with immutable DV snapshot
        for r in range(7):
            if s.radio_alive[e, r] != wp.uint8(0) and s.hello_due[e, r] <= t:
                s.hello_pending[e, r] = 1
                s.hello_due[e, r] = wp.min(NEVER, t + 500_000)
                for d in range(7):
                    if s.route_hop[e, r, d] >= 0 and s.route_expiry[e, r, d] > t:
                        s.hello_dv_hop[e, r, d] = s.route_hop[e, r, d]
                        s.hello_dv_dist[e, r, d] = s.route_dist[e, r, d]
                        s.hello_dv_mask[e, r, d] = s.route_mask[e, r, d]
                    else:
                        s.hello_dv_hop[e, r, d] = -1
                        s.hello_dv_dist[e, r, d] = 127
                        s.hello_dv_mask[e, r, d] = 0

        # 2. Finish transmissions first (matches CPU v5.7 two-phase ordering)
        for r in range(7):
            if s.radio_alive[e, r] != wp.uint8(0) and s.deadline[e, r] == t:
                if s.state[e, r] == TX or s.state[e, r] == TX_ACK:
                    _finish(s, e, r, t, tx_power, noise, snr)

        # 3. Advance DIFS / BACKOFF / SIFS / WAIT_ACK
        for r in range(7):
            if s.radio_alive[e, r] != wp.uint8(0) and s.deadline[e, r] == t:
                st = s.state[e, r]
                if st == DIFS:
                    rem = s.backoff_rem[e, r]
                    if s.frozen[e, r] == 0:
                        # HELLO uses base CW_min=31 (never inherits DATA retry count)
                        retx = 0 if s.active_slot[e, r] == -2 else s.retries[e, r]
                        cw = (32 << wp.min(4, retx)) - 1
                        rem = _random_slot(s, e, r, cw) * 20
                    s.frozen[e, r] = 0
                    s.backoff_rem[e, r] = 0
                    if rem == 0:
                        _start(s, e, r, t, False, tx_power, cca)
                    else:
                        s.state[e, r] = BACKOFF
                        s.deadline[e, r] = wp.min(NEVER, t + rem)
                elif st == BACKOFF:
                    _start(s, e, r, t, False, tx_power, cca)
                elif st == SIFS:
                    _start(s, e, r, t, True, tx_power, cca)
                elif st == WAIT_ACK:
                    s.stats_retry[e] = s.stats_retry[e] + 1
                    s.retries[e, r] = s.retries[e, r] + 1
                    if s.retries[e, r] >= 5 or (
                        s.active_slot[e, r] >= 0 and s.q_expiry[e, r, s.active_slot[e, r]] <= t
                    ):
                        _qpop(s, e, r)
                        wp.atomic_add(s.stats_drop, e, 1)
                        s.retries[e, r] = 0
                        s.active_slot[e, r] = -1
                    s.frozen[e, r] = 0
                    s.backoff_rem[e, r] = 0
                    s.state[e, r] = IDLE
                    s.deadline[e, r] = NEVER

    # Clean up expired pending records once at step end
    for u in range(6):
        for p in range(16):
            if s.pending_valid[e, u, p] != 0 and s.pending_expiry[e, u, p] <= t:
                s.pending_valid[e, u, p] = 0
    s.now[e] = t
    s.event_count[e] = s.event_count[e] + event
    if t < limit:
        s.overflow[e] = 1
