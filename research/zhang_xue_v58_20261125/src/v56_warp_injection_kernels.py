"""v5.7 Injection & Sensing Kernels for 7-Radio Warp CSMA/CA.

Fixes over v5.6:
  - Added `wp.set_module_options({"enable_backward": False})` and `@wp.kernel(enable_backward=False)`.
  - Fixed GPU data race on `stats_drop[e]` via `wp.atomic_add` (B7).
  - Fixed uint16 overflow saturation at 65535 in `_add_observation` (B18).
  - Added `_add_observations_batch` for multi-cell FOV sensing per UAV per step.
  - Clamped pending expiry to `NEVER` to prevent int32 wrap.
"""
import warp as wp
from v56_warp_7radio_mac import NetworkBuffers, _qpush, DATA, NEVER

wp.set_module_options({"enable_backward": False})


@wp.kernel(enable_backward=False)
def _inject(
    s: NetworkBuffers,
    actions: wp.array2d(dtype=wp.int32),
    episode_sec: wp.array(dtype=wp.int32),
):
    e, u = wp.tid()
    if s.radio_alive[e, u] == wp.uint8(0):
        return
    ai = actions[e, u]
    if ai < 1 or ai > 6:
        return
    dst = 6
    if ai < 6:
        dst = ai - 1
        if dst >= u:
            dst = dst + 1
    # At most one outstanding application packet per sender/receiver.
    for p in range(16):
        if s.pending_valid[e, u, p] != 0 and s.pending_dst[e, u, p] == dst and s.pending_expiry[e, u, p] > s.now[e]:
            return
    if s.q_count[e, u] >= 64:
        wp.atomic_add(s.stats_drop, e, 1)
        return
    for p in range(16):
        if s.pending_valid[e, u, p] == 0 or s.pending_expiry[e, u, p] <= s.now[e]:
            n = int(0)
            exp = wp.min(NEVER, s.now[e] + 10_000_000)
            seq = wp.atomic_add(s.next_seq, e, 1)
            slot = _qpush(s, e, u, DATA, u, dst, seq, exp, 1 << u, 0)
            if slot < 0:
                return
            for cell in range(400):
                pos = s.ledger_pos[e, u, u, cell]
                neg = s.ledger_neg[e, u, u, cell]
                gen = s.ledger_gen[e, u, u, cell]
                if n < 62 and (
                    pos > s.ack_pos[e, u, dst, cell]
                    or neg > s.ack_neg[e, u, dst, cell]
                    or gen > s.ack_gen[e, u, dst, cell]
                ):
                    if pos > wp.uint16(0) or neg > wp.uint16(0):
                        w0 = wp.uint32(u) | (wp.uint32(cell) << wp.uint32(3)) | (wp.uint32(wp.max(gen, 0)) << wp.uint32(12))
                        w1 = wp.uint32(pos) | (wp.uint32(neg) << wp.uint32(16))
                        s.q_word0[e, u, slot, n] = w0
                        s.q_word1[e, u, slot, n] = w1
                        s.pending_word0[e, u, p, n] = w0
                        s.pending_word1[e, u, p, n] = w1
                        n = n + 1
            if n == 0:
                s.q_count[e, u] = s.q_count[e, u] - 1
                return
            s.q_entries[e, u, slot] = n
            s.pending_valid[e, u, p] = 1
            s.pending_seq[e, u, p] = seq
            s.pending_dst[e, u, p] = dst
            s.pending_entries[e, u, p] = n
            s.pending_expiry[e, u, p] = exp
            return


@wp.kernel(enable_backward=False)
def _add_observation(
    s: NetworkBuffers,
    obs_cell: wp.array2d(dtype=wp.int32),
    obs_positive: wp.array2d(dtype=wp.int32),
    ts_s: int,
):
    e, u = wp.tid()
    if s.radio_alive[e, u] == wp.uint8(0):
        return
    cid = obs_cell[e, u]
    if cid >= 0 and cid < 400:
        if obs_positive[e, u] != 0:
            cur = wp.int32(s.ledger_pos[e, u, u, cid])
            if cur < 65535:
                s.ledger_pos[e, u, u, cid] = wp.uint16(cur + 1)
        else:
            cur = wp.int32(s.ledger_neg[e, u, u, cid])
            if cur < 65535:
                s.ledger_neg[e, u, u, cid] = wp.uint16(cur + 1)
        s.ledger_gen[e, u, u, cid] = ts_s
        s.ledger_rx_age_s[e, u, u, cid] = 0.0


@wp.kernel(enable_backward=False)
def _add_observations_batch(
    s: NetworkBuffers,
    obs_cells: wp.array3d(dtype=wp.int32),      # (E, 6, M) with -1 padding
    obs_positives: wp.array3d(dtype=wp.int32),  # (E, 6, M)
    max_cells_per_step: int,
    ts_s: int,
):
    """Multi-cell FOV sensing injection (fixes B18 1-cell/step limit on Warp)."""
    e, u = wp.tid()
    if s.radio_alive[e, u] == wp.uint8(0):
        return
    for k in range(max_cells_per_step):
        cid = obs_cells[e, u, k]
        if cid >= 0 and cid < 400:
            if obs_positives[e, u, k] != 0:
                cur = wp.int32(s.ledger_pos[e, u, u, cid])
                if cur < 65535:
                    s.ledger_pos[e, u, u, cid] = wp.uint16(cur + 1)
            else:
                cur = wp.int32(s.ledger_neg[e, u, u, cid])
                if cur < 65535:
                    s.ledger_neg[e, u, u, cid] = wp.uint16(cur + 1)
            s.ledger_gen[e, u, u, cid] = ts_s
            s.ledger_rx_age_s[e, u, u, cid] = 0.0
