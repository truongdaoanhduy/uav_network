from .common import *


class _NetworkLedgerMixin:

    def record_local_observation(self, actor_id: int, cell_id: int, detected: bool, t_s: int) -> None:
        if not (0 <= actor_id < ACTOR_N):
            raise ValueError("GCS cannot generate sensor observations")
        if detected:
            self.ledger_pos[actor_id, actor_id, cell_id] = min(
                65535, int(self.ledger_pos[actor_id, actor_id, cell_id]) + 1
            )
        else:
            self.ledger_neg[actor_id, actor_id, cell_id] = min(
                65535, int(self.ledger_neg[actor_id, actor_id, cell_id]) + 1
            )
        self.ledger_gen_ts_s[actor_id, actor_id, cell_id] = int(t_s)


    def count_all_dirty_entries_non_oracle(self, src: int, dst: int) -> int:
        """Total dirty cells before MAX_ENTRIES_PER_PKT (62) packetization truncation (B11 fix)."""
        if not (0 <= src < ACTOR_N) or not (0 <= dst < RADIO_N) or src == dst:
            return 0
        own_p = self.ledger_pos[src, src]
        own_n = self.ledger_neg[src, src]
        own_t = self.ledger_gen_ts_s[src, src]
        wm_p = self.ack_wm_pos[src, dst]
        wm_n = self.ack_wm_neg[src, dst]
        wm_t = self.ack_wm_ts[src, dst]
        return int(np.count_nonzero((own_p > wm_p) | (own_n > wm_n) | ((own_t > wm_t) & (own_t >= 0))))


    def get_dirty_entries_non_oracle(self, src: int, dst: int) -> List[Tuple[int, int]]:
        """Strictly local dirty selection using sender's own ledger vs. sender's E2E ACK watermark."""
        if not (0 <= src < ACTOR_N) or not (0 <= dst < RADIO_N) or src == dst:
            return []
        own_p = self.ledger_pos[src, src]
        own_n = self.ledger_neg[src, src]
        own_t = self.ledger_gen_ts_s[src, src]
        wm_p = self.ack_wm_pos[src, dst]
        wm_n = self.ack_wm_neg[src, dst]
        wm_t = self.ack_wm_ts[src, dst]
        dirty_cells = np.flatnonzero((own_p > wm_p) | (own_n > wm_n) | ((own_t > wm_t) & (own_t >= 0)))
        entries: List[Tuple[int, int]] = []
        for cid in dirty_cells[:MAX_ENTRIES_PER_PKT]:
            entries.append(
                pack_entry_2x_uint32(
                    src, int(cid), max(0, int(own_t[cid])), int(own_p[cid]), int(own_n[cid])
                )
            )
        return entries


    def enqueue_app_data(self, src: int, dst: int, entries: Sequence[Tuple[int, int]]) -> Optional[int]:
        if not (0 <= src < ACTOR_N):
            raise ValueError("Only actors 0..5 can originate DATA packets")
        if not entries:
            return None
        r = self.radios[src]
        if len(r.queue) >= self.queue_cap:
            self.stats["drop_qfull"] += 1
            return None
        pid = self._next_pkt_id
        self._next_pkt_id += 1
        seq = self._next_app_seq
        self._next_app_seq += 1
        packed = tuple(entries[:MAX_ENTRIES_PER_PKT])
        pkt = Packet(
            pkt_id=pid,
            episode_id=self.episode_id,
            pkt_type=PKT_DATA,
            app_src=src,
            app_dst=dst,
            app_seq=seq,
            create_us=self.sim_time_us,
            expire_us=min(INF_TIME_US, self.sim_time_us + PACKET_TTL_US),
            hop_cnt=0,
            visited_mask=(1 << src),
            packed_entries=packed,
        )
        r.queue.append(pkt)
        self.e2e_pending[(src, seq)] = (dst, packed)
        self.e2e_pending_expires_us[(src, seq)] = pkt.expire_us
        self.stats["data_enqueued"] += 1
        return pid


    def _prune_app_tracking(self) -> None:
        """Free delivered/pending app records after their bounded lifetime."""
        now = self.sim_time_us
        for key in [k for k, exp in self.e2e_pending_expires_us.items() if exp <= now]:
            self.e2e_pending_expires_us.pop(key, None)
            self.e2e_pending.pop(key, None)
        for key in [k for k, exp in self.delivered_app_expires_us.items() if exp <= now]:
            self.delivered_app_expires_us.pop(key, None)


    def _enqueue_packet_internal(self, node_id: int, pkt: Packet) -> bool:
        r = self.radios[node_id]
        if len(r.queue) >= self.queue_cap:
            self.stats["drop_qfull"] += 1
            return False
        r.queue.append(pkt)
        return True


    def _prune_expired_routes(self, node_id: int) -> None:
        now = self.sim_time_us
        rt = self.routes[node_id]
        dead = [d for d, (_, _, _, exp) in rt.items() if exp <= now]
        for d in dead:
            del rt[d]


    def _lookup_next_hop(self, node_id: int, dst: int) -> int:
        self._prune_expired_routes(node_id)
        if dst == -1:
            return -1  # Broadcast HELLO
        entry = self.routes[node_id].get(dst)
        return entry[0] if entry is not None else -1
