"""v6.7/v7.1 Parameterized Warp Sense, Dirty-Dest, Compact-3 Resolver & Injection Kernels."""
from __future__ import annotations

from zhang_xue.gpu.kernels.mac_v67 import _qpush_v67
from zhang_xue.gpu.state_v67 import (
    FAULT_GENERATION,
    HAS_WARP,
    PKT_DATA,
    PKT_TTL_US,
    NetworkBuffersV67,
    SimDimsV67,
    wp,
)

if HAS_WARP:
    @wp.func
    def _pack_w0_v67(scalable: int, org: int, cid: int, gen: int) -> wp.uint32:
        if scalable == 0:
            o = wp.uint32(org) & wp.uint32(0x7)
            c = (wp.uint32(cid) & wp.uint32(0x1FF)) << wp.uint32(3)
            g = (wp.uint32(gen) & wp.uint32(0xFFFFF)) << wp.uint32(12)
            return o | c | g
        o = wp.uint32(org) & wp.uint32(0x1F)
        c = (wp.uint32(cid) & wp.uint32(0xFFF)) << wp.uint32(5)
        g = (wp.uint32(gen) & wp.uint32(0x7FFF)) << wp.uint32(17)
        return o | c | g

    @wp.kernel
    def validate_sensing_generations_v67(
        s: NetworkBuffersV67,
        dims: SimDimsV67,
        cell_id: wp.array2d(dtype=wp.int32),
        t_s: int,
    ):
        # One owner writes each environment's fault byte. A separate launch
        # validates every UAV before the parallel evidence kernel can mutate it.
        e = wp.tid()
        if s.overflow[e] != wp.uint8(0):
            return
        for u in range(dims.n_uav):
            cid = cell_id[e, u]
            if cid >= 0 and cid < dims.num_cells:
                if t_s <= int(s.ledger_gen[e, u, u, cid]):
                    s.overflow[e] = wp.uint8(FAULT_GENERATION)
                    return

    @wp.kernel
    def sense_batch_v67(
        s: NetworkBuffersV67,
        dims: SimDimsV67,
        cell_id: wp.array2d(dtype=wp.int32),
        detected: wp.array2d(dtype=wp.int32),
        t_s: int,
    ):
        e, u = wp.tid()
        if u >= dims.n_uav or s.overflow[e] != wp.uint8(0):
            return
        cid = cell_id[e, u]
        if cid < 0 or cid >= dims.num_cells:
            return
        det = detected[e, u]
        if dims.aoi_ewma != 0:
            last_t = int(s.ledger_gen[e, u, u, cid])
            ep = float(int(s.ledger_pos[e, u, u, cid])) / 256.0
            en = float(int(s.ledger_neg[e, u, u, cid])) / 256.0
            if last_t >= 0 and t_s > last_t:
                decay = wp.exp(-dims.aoi_eta * float(t_s - last_t))
                ep = ep * decay
                en = en * decay
            if det != 0:
                ep = ep + 1.0
            else:
                en = en + 1.0
            s.ledger_pos[e, u, u, cid] = wp.uint16(wp.clamp(int(wp.round(ep * 256.0)), 0, 65535))
            s.ledger_neg[e, u, u, cid] = wp.uint16(wp.clamp(int(wp.round(en * 256.0)), 0, 65535))
        else:
            if det != 0:
                s.ledger_pos[e, u, u, cid] = wp.uint16(wp.min(65535, int(s.ledger_pos[e, u, u, cid]) + 1))
            else:
                s.ledger_neg[e, u, u, cid] = wp.uint16(wp.min(65535, int(s.ledger_neg[e, u, u, cid]) + 1))
        s.ledger_gen[e, u, u, cid] = wp.int16(wp.min(32767, wp.max(0, t_s)))

    @wp.kernel
    def compute_dirty_dest_v67(
        s: NetworkBuffersV67,
        dims: SimDimsV67,
    ):
        e, u, dst = wp.tid()
        if u >= dims.n_uav or dst >= dims.n_radio:
            return
        if u == dst or s.radio_alive[e, u] == wp.uint8(0):
            s.dirty_dest[e, u, dst] = wp.uint8(0)
            return
        p_idx = dst if dst < u else (dst - 1)
        if s.pending_valid[e, u, p_idx] != wp.uint8(0) and s.pending_expiry[e, u, p_idx] > s.now[e]:
            s.dirty_dest[e, u, dst] = wp.uint8(0)
            return
        found = wp.uint8(0)
        for cid in range(dims.num_cells):
            if int(s.ledger_gen[e, u, u, cid]) > int(s.ack_origin_gen[e, u, dst, u * dims.num_cells + cid]):
                found = wp.uint8(1)
                break
        if found == wp.uint8(0) and dims.enable_data_mule != 0:
            for org in range(dims.n_uav):
                if org == u or org == dst:
                    continue
                base = org * dims.num_cells
                for cid in range(dims.num_cells):
                    if int(s.ledger_gen[e, u, org, cid]) > int(s.ack_origin_gen[e, u, dst, base + cid]):
                        found = wp.uint8(1)
                        break
                if found != wp.uint8(0):
                    break
        s.dirty_dest[e, u, dst] = found

    @wp.kernel
    def resolve_compact3_v67(
        s: NetworkBuffersV67,
        dims: SimDimsV67,
        actions: wp.array2d(dtype=wp.int32),
        pos_xy: wp.array3d(dtype=wp.float32),
        dest_out: wp.array2d(dtype=wp.int32),
    ):
        e, u = wp.tid()
        if u >= dims.n_uav:
            return
        act = actions[e, u]
        if act <= 0 or s.radio_alive[e, u] == wp.uint8(0):
            dest_out[e, u] = -1
            return
        if act == 1:
            dest_out[e, u] = dims.gcs_id if s.dirty_dest[e, u, dims.gcs_id] != wp.uint8(0) else -1
            return
        if dims.enable_peer == 0:
            dest_out[e, u] = -1
            return
        if dims.decentralized_peer_select != 0:
            best = int(-1)
            best_hops = int(128)
            # Route knowledge and dirty watermarks belong to this UAV. Hidden
            # peer positions/aliveness cannot influence the local policy.
            for p in range(dims.n_uav):
                if p == u or s.dirty_dest[e, u, p] == wp.uint8(0):
                    continue
                if s.route_expiry[e, u, p] <= s.now[e] or int(s.route_hop[e, u, p]) < 0:
                    continue
                hops = int(s.route_dist[e, u, p])
                if hops > 0 and hops <= dims.max_hops and hops < best_hops:
                    best_hops = hops
                    best = p
            dest_out[e, u] = best
            return
        ux, uy = pos_xy[e, u, 0], pos_xy[e, u, 1]
        best = int(-1)
        best_d2 = float(1.0e30)
        for p in range(dims.n_uav):
            if p == u or s.radio_alive[e, p] == wp.uint8(0) or s.dirty_dest[e, u, p] == wp.uint8(0):
                continue
            dx = pos_xy[e, p, 0] - ux
            dy = pos_xy[e, p, 1] - uy
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = p
        dest_out[e, u] = best

    @wp.kernel
    def inject_v67(
        s: NetworkBuffersV67,
        dims: SimDimsV67,
        dest: wp.array2d(dtype=wp.int32),
    ):
        e, u = wp.tid()
        if u >= dims.n_uav or s.radio_alive[e, u] == wp.uint8(0):
            return
        dst = dest[e, u]
        if dst < 0 or dst >= dims.n_radio or dst == u:
            return
        p_idx = dst if dst < u else (dst - 1)
        now = s.now[e]
        # One outstanding E2E request per destination: without this guard,
        # the next policy step replaces pending_seq/words and late E2E ACKs
        # cannot advance the correct multi-origin watermark.
        if s.pending_valid[e, u, p_idx] != wp.uint8(0):
            if s.pending_expiry[e, u, p_idx] > now:
                return
            s.pending_valid[e, u, p_idx] = wp.uint8(0)
        cnt = int(0)
        for pass_id in range(2):
            if pass_id == 1 and dims.enable_data_mule == 0:
                break
            for org in range(dims.n_uav):
                if pass_id == 0 and org != u:
                    continue
                if pass_id == 1 and (org == u or org == dst):
                    continue
                base = org * dims.num_cells
                for cid in range(dims.num_cells):
                    g = int(s.ledger_gen[e, u, org, cid])
                    if g >= 0 and g > int(s.ack_origin_gen[e, u, dst, base + cid]):
                        w0 = _pack_w0_v67(dims.scalable_wire, org, cid, g)
                        w1 = wp.uint32(s.ledger_pos[e, u, org, cid]) | (wp.uint32(s.ledger_neg[e, u, org, cid]) << wp.uint32(16))
                        s.pending_word0[e, u, p_idx, cnt] = w0
                        s.pending_word1[e, u, p_idx, cnt] = w1
                        cnt = cnt + 1
                        if cnt >= dims.k_entries:
                            break
                if cnt >= dims.k_entries:
                    break
            if cnt >= dims.k_entries:
                break
        if cnt == 0:
            return
        pid = wp.atomic_add(s.next_pid, e, 1)
        seq = wp.atomic_add(s.next_seq, e, 1)
        slot = _qpush_v67(
            s, dims, e, u, PKT_DATA, pid, u, dst, seq,
            now + PKT_TTL_US, wp.uint32(1) << wp.uint32(u), 0, cnt,
        )
        if slot >= 0:
            for k in range(cnt):
                s.q_word0[e, u, slot, k] = s.pending_word0[e, u, p_idx, k]
                s.q_word1[e, u, slot, k] = s.pending_word1[e, u, p_idx, k]
            s.pending_seq[e, u, p_idx] = seq
            s.pending_dst[e, u, p_idx] = wp.int8(dst)
            s.pending_valid[e, u, p_idx] = wp.uint8(1)
            s.pending_expiry[e, u, p_idx] = now + PKT_TTL_US
            s.pending_entries[e, u, p_idx] = wp.uint8(cnt)
