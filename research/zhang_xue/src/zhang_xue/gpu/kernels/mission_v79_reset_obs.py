"""V79 Warp GPU masked CSMA/CA warmup, selective reset, and observation kernels."""
from __future__ import annotations

from zhang_xue.gpu.kernels.events_v67 import _arm_idle_v67, _finish_tx_v67, _select_routable_slot_v67, _prepare_starts_v93
from zhang_xue.gpu.kernels.mac_v67 import _busy_v67, _qpop_pos_v67, _rand_slot_v67, _start_v67
from zhang_xue.gpu.kernels.mission_v79_kernels import GRID, N_UAV, NUM_CELLS, _fuse_uq88_v79, _mix_seed_v103
from zhang_xue.gpu.kernels.phy_v67 import _in_blackout_v67
from zhang_xue.gpu.state_v67 import (
    HAS_WARP, HELLO_INT_US, HELLO_V89_TX_US, NEVER_US, PKT_E2E_ACK,
    ST_BACKOFF, ST_DIFS, ST_IDLE, ST_SIFS, ST_TX, ST_TX_ACK, ST_WAIT_ACK, TX_MAC_ACK_US,
    NetworkBuffersV67, SimDimsV67, wp,
)

if HAS_WARP:
    @wp.kernel
    def step_v67_masked(
        s: NetworkBuffersV67, dims: SimDimsV67, active_mask: wp.array(dtype=wp.uint8),
        dur_us: int, max_ev: int, tx_p: float, cca_th: float, noise_w: float, snr_th: float,
    ):
        """Run CSMA/CA event loop ONLY for worlds where active_mask[e] != 0."""
        e = wp.tid()
        if active_mask[e] == wp.uint8(0) or s.overflow[e] != wp.uint8(0):
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

    @wp.kernel
    def _reset_world_v79(
        pos: wp.array3d(dtype=wp.float32), vel: wp.array3d(dtype=wp.float32),
        energy: wp.array2d(dtype=wp.float32), energy_comp: wp.array2d(dtype=wp.float32),
        truth: wp.array2d(dtype=wp.uint8), u_prev: wp.array2d(dtype=wp.float32),
        gcs_tp: wp.array2d(dtype=wp.uint8), gcs_tp_active: wp.array2d(dtype=wp.uint8),
        gcs_tp_drops: wp.array(dtype=wp.int32),
        gcs_fp: wp.array2d(dtype=wp.uint8), gcs_fp_ever: wp.array2d(dtype=wp.uint8),
        ep_step: wp.array(dtype=wp.int32), ep_id: wp.array(dtype=wp.int32),
        done: wp.array(dtype=wp.uint8), terminated: wp.array(dtype=wp.uint8),
        truncated: wp.array(dtype=wp.uint8), failed: wp.array(dtype=wp.uint8),
        unsafe_mask: wp.array2d(dtype=wp.uint8), reward: wp.array(dtype=wp.float32),
        intervention: wp.array(dtype=wp.float32), pos_xy: wp.array3d(dtype=wp.float32),
        legal: wp.array(dtype=wp.int32), legal_size: int, n_targets: int,
        battery_max: float, uav_alt: float, seed: int,
    ):
        e = wp.tid()
        if done[e] == wp.uint8(0):
            return
        ep_step[e] = 0
        ep_id[e] = ep_id[e] + 1
        failed[e] = wp.uint8(0)
        terminated[e] = wp.uint8(0)
        truncated[e] = wp.uint8(0)
        reward[e] = 0.0
        intervention[e] = 0.0
        gcs_tp_drops[e] = 0
        for c in range(NUM_CELLS):
            truth[e, c] = wp.uint8(0)
            u_prev[e, c] = 1.0
            gcs_tp[e, c] = wp.uint8(0)
            gcs_tp_active[e, c] = wp.uint8(0)
            gcs_fp[e, c] = wp.uint8(0)
            gcs_fp_ever[e, c] = wp.uint8(0)
        for u in range(N_UAV):
            pos[e, u, 0] = 90.0 + 120.0 * float(u)
            pos[e, u, 1] = 110.0 + 30.0 * float(u % 2)
            pos[e, u, 2] = uav_alt
            pos_xy[e, u, 0] = pos[e, u, 0]
            pos_xy[e, u, 1] = pos[e, u, 1]
            vel[e, u, 0] = 0.0
            vel[e, u, 1] = 0.0
            energy[e, u] = battery_max
            energy_comp[e, u] = 0.0
            unsafe_mask[e, u] = wp.uint8(0)
        pos[e, N_UAV, 0] = 50.0
        pos[e, N_UAV, 1] = 50.0
        pos[e, N_UAV, 2] = 15.0
        rng = wp.rand_init(_mix_seed_v103(seed ^ (e * 2654435761) ^ (ep_id[e] * 11939)))
        count = int(0)
        for _ in range(300):
            if count >= n_targets:
                break
            cid = legal[wp.randi(rng, 0, legal_size)]
            if truth[e, cid] == wp.uint8(0):
                truth[e, cid] = wp.uint8(1)
                count += 1
        if count != n_targets:
            failed[e] = wp.uint8(1)

    @wp.kernel
    def _finalize_reset_v79(
        net: NetworkBuffersV67, done: wp.array(dtype=wp.uint8),
    ):
        """Anchor mission_epoch_us after masked warmup and clear done flag on GPU."""
        e = wp.tid()
        if done[e] == wp.uint8(0):
            return
        net.mission_epoch_us[e] = net.now[e]
        for r in range(N_UAV + 1):
            net.tx_airtime[e, r] = 0
        done[e] = wp.uint8(0)

    @wp.kernel
    def _observe_core_v79(
        net: NetworkBuffersV67, dims: SimDimsV67, pos: wp.array3d(dtype=wp.float32),
        vel: wp.array3d(dtype=wp.float32), batt: wp.array2d(dtype=wp.float32),
        blackout_xy: wp.array2d(dtype=wp.int32), n_bo: int, blackout_r: int,
        obs: wp.array3d(dtype=wp.float32), mask: wp.array3d(dtype=wp.uint8),
        battery_max: float, qcap: float, v_max: float,
    ):
        e, u = wp.tid()
        x = pos[e, u, 0]
        y = pos[e, u, 1]
        gx = pos[e, N_UAV, 0]
        gy = pos[e, N_UAV, 1]
        dx = (gx - x) / 1000.0
        dy = (gy - y) / 1000.0
        soc = wp.clamp(batt[e, u] / battery_max, 0.0, 1.0)
        in_bo = _in_blackout_v67(x, y, GRID, 50.0, blackout_xy, n_bo, blackout_r)
        own_radio = 1 if (net.radio_alive[e, u] != wp.uint8(0) and in_bo == 0 and soc > 0.0) else 0
        obs[e, u, 0] = x / 1000.0
        obs[e, u, 1] = y / 1000.0
        obs[e, u, 2] = vel[e, u, 0] / v_max
        obs[e, u, 3] = vel[e, u, 1] / v_max
        obs[e, u, 4] = soc
        obs[e, u, 5] = float(own_radio)
        obs[e, u, 6] = float(net.q_count[e, u]) / qcap
        obs[e, u, 7] = dx
        obs[e, u, 8] = dy
        obs[e, u, 9] = wp.sqrt(dx * dx + dy * dy)
        valid_gcs = net.route_hop[e, u, N_UAV] >= 0 and net.route_expiry[e, u, N_UAV] > net.now[e]
        obs[e, u, 10] = 1.0 if valid_gcs else 0.0
        # Exact continuous dirty_gcs fraction [0, 1] matching CPU observations.py:83
        dirty_cnt = int(0)
        for org in range(N_UAV):
            if org != u and dims.enable_data_mule == 0:
                continue
            base = org * NUM_CELLS
            for cid in range(NUM_CELLS):
                if int(net.ledger_gen[e, u, org, cid]) > int(net.ack_origin_gen[e, u, N_UAV, base + cid]):
                    dirty_cnt += 1
        obs[e, u, 11] = wp.min(1.0, float(dirty_cnt) / float(NUM_CELLS))
        # S10 fix: Exact ACK watermark fraction matching CPU observations.py
        wm_cnt = int(0)
        if dims.enable_data_mule != 0:
            for dst in range(N_UAV + 1):
                for idx in range(N_UAV * NUM_CELLS):
                    if int(net.ack_origin_gen[e, u, dst, idx]) >= 0:
                        wm_cnt += 1
            obs[e, u, 12] = float(wm_cnt) / float((N_UAV + 1) * N_UAV * NUM_CELLS)
        else:
            own_base = u * NUM_CELLS
            for dst in range(N_UAV + 1):
                for cid in range(NUM_CELLS):
                    if int(net.ack_origin_gen[e, u, dst, own_base + cid]) >= 0:
                        wm_cnt += 1
            obs[e, u, 12] = float(wm_cnt) / float((N_UAV + 1) * NUM_CELLS)
        rcount = int(0)
        for peer in range(N_UAV + 1):
            if peer == u:
                continue
            if net.route_hop[e, u, peer] >= 0 and net.route_expiry[e, u, peer] > net.now[e]:
                rcount += 1
        obs[e, u, 13] = float(rcount) / 7.0
        mask[e, u, 0] = wp.uint8(1)
        mask[e, u, 1] = wp.uint8(1) if (own_radio != 0 and valid_gcs and net.dirty_dest[e, u, N_UAV] != wp.uint8(0)) else wp.uint8(0)
        allowed = wp.uint8(0)
        # A valid GCS route does not make a peer transmission physically illegal.
        # Keep the selective GCS-first rule in the non-RL heuristic, not in the POMDP action mask.
        if own_radio != 0 and dims.enable_peer != 0:
            for peer in range(N_UAV):
                if peer == u:
                    continue
                hops = int(net.route_dist[e, u, peer])
                if (net.route_hop[e, u, peer] >= 0 and net.route_expiry[e, u, peer] > net.now[e]
                        and hops > 0 and hops <= dims.max_hops and net.dirty_dest[e, u, peer] != wp.uint8(0)):
                    allowed = wp.uint8(1)
        mask[e, u, 2] = allowed

    @wp.kernel
    def _observe_map_v79(  # S06: semantic pooling (max m_T, min m_F, mean m_U) is a non-simplex actor feature
        net: NetworkBuffersV67, pos: wp.array3d(dtype=wp.float32),
        ep_step: wp.array(dtype=wp.int32), obs: wp.array3d(dtype=wp.float32),
        logmf: float, logme: float, eta: float, ref_age: float,
        static_raw_count: int,
    ):
        e, u, bid = wp.tid()
        cy = bid // 10
        cx = bid % 10
        t_max = float(0.0)
        f_min = float(1.0)
        u_avg = float(0.0)
        age_max = float(0.0)
        now = int(ep_step[e])
        for yy in range(2):
            for xx in range(2):
                cid = (2 * cy + yy) * 20 + 2 * cx + xx
                ds = _fuse_uq88_v79(net, e, u, cid, now, logmf, logme, eta, static_raw_count)
                t_max = wp.max(t_max, ds[0])
                f_min = wp.min(f_min, ds[1])
                u_avg += 0.25 * ds[2]
                latest = int(-1)
                for org in range(N_UAV):
                    if int(net.ledger_pos[e, u, org, cid]) + int(net.ledger_neg[e, u, org, cid]) > 0:
                        latest = wp.max(latest, int(net.ledger_gen[e, u, org, cid]))
                age = 1.0 if latest < 0 else wp.clamp(float(now - latest) / ref_age, 0.0, 1.0)
                age_max = wp.max(age_max, age)
        obs[e, u, 14 + bid] = t_max
        obs[e, u, 114 + bid] = f_min
        obs[e, u, 214 + bid] = u_avg
        obs[e, u, 314 + bid] = age_max
        px = int(wp.clamp(wp.floor(pos[e, u, 0] / 50.0), 0.0, 19.0)) // 2
        py = int(wp.clamp(wp.floor(pos[e, u, 1] / 50.0), 0.0, 19.0)) // 2
        obs[e, u, 414 + bid] = 1.0 if (px == cx and py == cy) else 0.0

    @wp.kernel
    def _clear_masked_diagnostics_v79(
        mask: wp.array(dtype=wp.uint8),
        terminal_reason: wp.array(dtype=wp.uint8),
        safety_reason: wp.array(dtype=wp.uint8),
        reward_components: wp.array2d(dtype=wp.float32),
    ):
        """Reset selected-world diagnostics and reward decomposition atomically."""
        e=wp.tid()
        if mask[e] != wp.uint8(0):
            terminal_reason[e]=wp.uint8(0)
            safety_reason[e]=wp.uint8(0)
            for k in range(8):
                reward_components[e, k] = 0.0
