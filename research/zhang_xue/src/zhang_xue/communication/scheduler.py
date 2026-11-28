from .common import *


class _NetworkSchedulerMixin:

    def _try_arm_idle_radios(self) -> None:
        for r_id in range(self.radio_n):
            if not self.radio_alive[r_id]:
                continue
            r = self.radios[r_id]
            if r.mac_state != ST_IDLE:
                continue
            # Pending MAC ACK has strict priority
            if r.ack_pend_dst >= 0:
                r.mac_state = ST_SIFS_ACK
                r.state_end_us = self.sim_time_us + self.protocol.sifs_us
                continue
            # Control plane must still be able to discover routes when DATA has no route.
            if r.active_pkt is None and r.pending_hello is not None:
                candidate = r.pending_hello
                r.pending_hello = None
                if self.sim_time_us < candidate.expire_us:
                    r.active_pkt = candidate
                    r.active_next_hop = -1
                    r.retx_cnt = 0
                    r.backoff_frozen = False
                    r.backoff_rem_us = 0
            # B3 fix: Drop expired packets and avoid HOL blocking of E2E_ACK or routable DATA
            if r.active_pkt is None and r.queue:
                # Prune expired packets across queue
                kept = deque()
                while r.queue:
                    p = r.queue.popleft()
                    if self.sim_time_us >= p.expire_us:
                        self.stats["drop_ttl"] += 1
                    else:
                        kept.append(p)
                r.queue = kept
                # Select first routable packet, prioritizing E2E_ACK over DATA
                chosen_idx = -1
                chosen_nh = -1
                for want_ack in (True, False):
                    for idx, cand in enumerate(r.queue):
                        is_e2e = (cand.pkt_type == PKT_E2E_ACK)
                        if want_ack and not is_e2e:
                            continue
                        if (not want_ack) and is_e2e:
                            continue
                        if cand.pkt_type == PKT_HELLO_DV:
                            chosen_idx = idx
                            chosen_nh = -1
                            break
                        nh = self._lookup_next_hop(r_id, cand.app_dst)
                        if nh >= 0:
                            chosen_idx = idx
                            chosen_nh = nh
                            break
                    if chosen_idx >= 0:
                        break
                if chosen_idx >= 0:
                    del_list = list(r.queue)
                    r.active_pkt = del_list.pop(chosen_idx)
                    r.queue = deque(del_list)
                    r.active_next_hop = chosen_nh
                    r.retx_cnt = 0
                    r.backoff_frozen = False
                    r.backoff_rem_us = 0
            if r.active_pkt is not None:
                if self.sim_time_us >= r.active_pkt.expire_us:
                    r.active_pkt = None
                    r.active_next_hop = -1
                    self.stats["drop_ttl"] += 1
                    continue
                if not self._is_channel_busy_at(r_id):
                    r.mac_state = ST_DIFS
                    r.state_end_us = self.sim_time_us + self.protocol.difs_us


    def _enqueue_hello_dv(self, r_id: int) -> None:
        self._prune_expired_routes(r_id)
        dv = {dst: (hops, pmask) for dst, (_, hops, pmask, _) in self.routes[r_id].items()}
        pid = self._next_pkt_id
        self._next_pkt_id += 1
        pose = None
        if getattr(self, "obs_schema_v2", False):
            # The V89 wire schema also includes the fixed GCS pose/zero velocity.
            vx, vy = self.velocities[r_id] if r_id < self.actor_n else (0.0, 0.0)
            pose = (
                float(self.positions[r_id, 0]),
                float(self.positions[r_id, 1]),
                float(vx),
                float(vy),
            )
        pkt = Packet(
            pkt_id=pid,
            episode_id=self.episode_id,
            pkt_type=PKT_HELLO_DV,
            app_src=r_id,
            app_dst=-1,
            app_seq=0,
            create_us=self.sim_time_us,
            expire_us=min(INF_TIME_US, self.sim_time_us + HELLO_INTERVAL_US),
            dv_table=dv,
            peer_pose=pose,
        )
        # Periodic HELLO/DV replaces any older unsent HELLO rather than consuming Q.
        self.radios[r_id].pending_hello = pkt


    def _prepare_tx_starts(self) -> int:
        """Finish DIFS and select contenders from one pre-start channel state."""
        for r_id, r in enumerate(self.radios):
            if not self.radio_alive[r_id] or r.state_end_us != self.sim_time_us:
                continue
            if r.mac_state != ST_DIFS:
                continue
            if self._is_channel_busy_at(r_id):
                r.mac_state = ST_IDLE
                r.state_end_us = INF_TIME_US
                continue
            if r.backoff_frozen:
                rem = r.backoff_rem_us
            else:
                cw = (self.protocol.cw_min + 1) * (1 << r.retx_cnt) - 1
                rem = int(self.rng.integers(0, cw + 1)) * self.protocol.slot_us
            r.backoff_frozen = False
            r.backoff_rem_us = 0
            r.mac_state = ST_BACKOFF
            r.state_end_us = self.sim_time_us + rem
        ready_mask = 0
        for r_id, r in enumerate(self.radios):
            if not self.radio_alive[r_id] or r.state_end_us != self.sim_time_us:
                continue
            if r.mac_state == ST_SIFS_ACK or (
                r.mac_state == ST_BACKOFF and not self._is_channel_busy_at(r_id)
            ):
                ready_mask |= 1 << r_id
        return ready_mask


    def run_until_us(self, target_us: int, max_events: int = 2000) -> bool:
        """Advance discrete-event CSMA/CA network to target_us. Returns True if completed cleanly."""
        if target_us < self.sim_time_us:
            raise ValueError(f"Cannot run network clock backward: {target_us} < {self.sim_time_us}")
        self._prune_app_tracking()
        for _ in range(max_events):
            self._try_arm_idle_radios()
            t_next = target_us
            for r_id in range(self.radio_n):
                if not self.radio_alive[r_id]:
                    continue
                if self.radios[r_id].state_end_us < t_next:
                    t_next = self.radios[r_id].state_end_us
                if self.hello_next_us[r_id] < t_next:
                    t_next = self.hello_next_us[r_id]
            if t_next >= target_us:
                self.sim_time_us = target_us
                self._prune_app_tracking()
                return True

            self.sim_time_us = t_next
            self.stats["events_processed"] += 1

            # 1. Process periodic HELLO/DV timers (alive radios only)
            for r_id in range(self.radio_n):
                if self.radio_alive[r_id] and self.hello_next_us[r_id] == t_next:
                    self._enqueue_hello_dv(r_id)
                    self.hello_next_us[r_id] = min(INF_TIME_US, t_next + HELLO_INTERVAL_US)

            # 2a. B13 fix: Finish all due TX/TX_ACK first (matches Warp two-phase ordering)
            for r_id in range(self.radio_n):
                if not self.radio_alive[r_id]:
                    continue
                r = self.radios[r_id]
                if r.state_end_us == t_next and r.mac_state in (ST_TX_DATA, ST_TX_MAC_ACK):
                    self._finish_tx(r_id)

            # 2b. Match Warp V93: choose every simultaneous start before TX
            # changes carrier sense, including zero-slot DIFS completions.
            ready_mask = self._prepare_tx_starts()
            for r_id in range(self.radio_n):
                if not self.radio_alive[r_id]:
                    continue
                r = self.radios[r_id]
                if r.state_end_us != t_next:
                    continue
                st = r.mac_state
                if st == ST_BACKOFF:
                    if ready_mask & (1 << r_id):
                        r.backoff_rem_us = 0
                        self._start_tx(r_id, is_ack=False, ready_mask=ready_mask)
                    else:
                        r.mac_state = ST_IDLE
                        r.state_end_us = INF_TIME_US
                elif st == ST_SIFS_ACK:
                    self._start_tx(r_id, is_ack=True, ready_mask=ready_mask)
                elif st == ST_WAIT_ACK:
                    self.stats["mac_ack_timeout"] += 1
                    r.retx_cnt += 1
                    if r.retx_cnt >= self.protocol.max_retries or (r.active_pkt is not None and self.sim_time_us >= r.active_pkt.expire_us):
                        self.stats["drop_retx"] += 1
                        bad_nh = r.active_next_hop
                        if bad_nh >= 0:
                            stale = [d for d, cur in self.routes[r_id].items() if cur[0] == bad_nh]
                            for d in stale:
                                del self.routes[r_id][d]
                        r.active_pkt = None
                        r.active_next_hop = -1
                        r.retx_cnt = 0
                        r.backoff_frozen = False
                        r.backoff_rem_us = 0
                        r.mac_state = ST_IDLE
                        r.state_end_us = INF_TIME_US
                    else:
                        # B9 fix: re-lookup next hop on MAC retry in case neighbor moved out of range
                        if r.active_pkt is not None and r.active_pkt.pkt_type != PKT_HELLO_DV:
                            nh = self._lookup_next_hop(r_id, r.active_pkt.app_dst)
                            if nh >= 0:
                                r.active_next_hop = nh
                            else:
                                # Route expired/dead: return packet to queue head for store-carry-forward
                                r.queue.appendleft(r.active_pkt)
                                r.active_pkt = None
                                r.active_next_hop = -1
                                r.retx_cnt = 0
                        r.backoff_frozen = False
                        r.backoff_rem_us = 0
                        r.mac_state = ST_IDLE
                        r.state_end_us = INF_TIME_US
        return False
