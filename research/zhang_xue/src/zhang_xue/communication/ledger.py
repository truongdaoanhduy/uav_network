from .common import *


class _NetworkLedgerMixin:

    def record_local_observation(self, actor_id: int, cell_id: int, detected: bool, t_s: int) -> None:
        if not (0 <= actor_id < self.actor_n):
            raise ValueError("GCS cannot generate sensor observations")
        from ..belief.dempster_shafer import (
            decode_uq88, deng_entropy, encode_uq88, ewma_effective_counts_update,
            fuse_ewma_provenance_uq88, fuse_raw_provenance_counts,
        )
        if self.use_uq88_ewma:
            last_ts = int(self.ledger_gen_ts_s[actor_id, actor_id, cell_id])
            ep = decode_uq88(self.ledger_pos[actor_id, actor_id, cell_id])
            en = decode_uq88(self.ledger_neg[actor_id, actor_id, cell_id])
            ep_new, en_new = ewma_effective_counts_update(ep, en, last_ts, float(t_s), detected, self.aoi_eta)
            self.ledger_pos[actor_id, actor_id, cell_id] = encode_uq88(ep_new)
            self.ledger_neg[actor_id, actor_id, cell_id] = encode_uq88(en_new)
            self.ledger_gen_ts_s[actor_id, actor_id, cell_id] = int(t_s)
            mt, mf_v, mu = fuse_ewma_provenance_uq88(
                self.ledger_pos[actor_id, :, cell_id], self.ledger_neg[actor_id, :, cell_id],
                self.ledger_gen_ts_s[actor_id, :, cell_id], float(t_s),
                mf=self.ds_mf, me=self.ds_me,
                eta=self.aoi_eta if self.ds_fusion_mode == "sum" else 0.0,
                mode=self.ds_fusion_mode,
            )
        else:
            if detected:
                self.ledger_pos[actor_id, actor_id, cell_id] = min(
                    65535, int(self.ledger_pos[actor_id, actor_id, cell_id]) + 1
                )
            else:
                self.ledger_neg[actor_id, actor_id, cell_id] = min(
                    65535, int(self.ledger_neg[actor_id, actor_id, cell_id]) + 1
                )
            self.ledger_gen_ts_s[actor_id, actor_id, cell_id] = int(t_s)
            mt, mf_v, mu = fuse_raw_provenance_counts(
                self.ledger_pos[actor_id, :, cell_id], self.ledger_neg[actor_id, :, cell_id],
                self.ledger_gen_ts_s[actor_id, :, cell_id],
                mf=self.ds_mf, me=self.ds_me, mode=self.ds_fusion_mode,
            )
        # Keep peer_h_min synchronized with self-sensed knowledge so peer_info_gain
        # cannot be rewarded for information the UAV already gathered locally.
        h_self = deng_entropy(mt, mf_v, mu)
        if h_self < self.peer_h_min[actor_id, cell_id]:
            self.peer_h_min[actor_id, cell_id] = h_self


    def _dirty_mask_for_origin(self, src: int, dst: int, origin: int) -> np.ndarray:
        own_p = self.ledger_pos[src, origin]
        own_n = self.ledger_neg[src, origin]
        own_t = self.ledger_gen_ts_s[src, origin]
        if self.enable_data_mule_forwarding and self.ack_wm_origin_ts is not None:
            wm_p = self.ack_wm_origin_pos[src, dst, origin]
            wm_n = self.ack_wm_origin_neg[src, dst, origin]
            wm_t = self.ack_wm_origin_ts[src, dst, origin]
        else:
            wm_p = self.ack_wm_pos[src, dst]
            wm_n = self.ack_wm_neg[src, dst]
            wm_t = self.ack_wm_ts[src, dst]
        if self.aoi_ewma_end_to_end:
            return (own_t > wm_t) & (own_t >= 0)
        return (own_p > wm_p) | (own_n > wm_n) | ((own_t > wm_t) & (own_t >= 0))

    def _origin_order(self, src: int, dst: int, *, advance: bool = False) -> List[int]:
        if not self.enable_data_mule_forwarding:
            return [src]
        origins = [src] + [o for o in range(self.actor_n) if o != src and o != dst]
        if len(origins) > 1:
            key = (src, dst)
            cur = self._origin_cursor.get(key, 0) % len(origins)
            origins = origins[cur:] + origins[:cur]
            if advance:
                self._origin_cursor[key] = (cur + 1) % len(origins)
        # Own traffic also takes a turn: 62 own entries cannot starve relay origins.
        return origins

    def count_all_dirty_entries_non_oracle(self, src: int, dst: int) -> int:
        """Total dirty (origin, cell) entries before MAX_ENTRIES_PER_PKT (62) truncation."""
        if not (0 <= src < self.actor_n) or not (0 <= dst < self.radio_n) or src == dst:
            return 0
        return sum(int(np.count_nonzero(self._dirty_mask_for_origin(src, dst, o))) for o in self._origin_order(src, dst))

    def get_dirty_entries_non_oracle(self, src: int, dst: int) -> List[Tuple[int, int]]:
        """Read-only payload preview from local evidence and local E2E ACK watermarks."""
        if not (0 <= src < self.actor_n) or not (0 <= dst < self.radio_n) or src == dst:
            return []
        entries: List[Tuple[int, int]] = []
        for origin in self._origin_order(src, dst):
            dirty_cells = np.flatnonzero(self._dirty_mask_for_origin(src, dst, origin))
            own_p = self.ledger_pos[src, origin]
            own_n = self.ledger_neg[src, origin]
            own_t = self.ledger_gen_ts_s[src, origin]
            # Sort by freshest generation before applying the packet entry limit.
            if dirty_cells.size > 1:
                order = np.argsort(-own_t[dirty_cells], kind="stable")
                dirty_cells = dirty_cells[order]
            for cid in dirty_cells:
                entries.append(
                    pack_entry_2x_uint32(
                        origin, int(cid), max(0, int(own_t[cid])), int(own_p[cid]), int(own_n[cid]),
                        max_origins=self.actor_n, max_cells=self.num_cells,
                    )
                )
                if len(entries) >= MAX_ENTRIES_PER_PKT:
                    return entries
        return entries


    def enqueue_app_data(self, src: int, dst: int, entries: Sequence[Tuple[int, int]]) -> Optional[int]:
        if not (0 <= src < self.actor_n):
            raise ValueError("Only actors 0..N-1 can originate DATA packets")
        if not entries:
            return None
        r = self.radios[src]
        packed = tuple(entries[:MAX_ENTRIES_PER_PKT])
        # Optional queue coalescing: avoid flooding the FIFO with identical unroutable snapshots.
        if self.coalesce_duplicate_queue:
            snapshot = sorted(packed)
            kept = deque()
            for existing in r.queue:
                if self.sim_time_us >= existing.expire_us:
                    self.stats["drop_ttl"] += 1
                    if existing.pkt_type == PKT_DATA and existing.app_src == src:
                        key = (src, existing.app_seq)
                        self.e2e_pending.pop(key, None)
                        self.e2e_pending_expires_us.pop(key, None)
                else:
                    kept.append(existing)
            r.queue = kept
            for existing in r.queue:
                if (
                    existing.pkt_type == PKT_DATA
                    and existing.app_src == src
                    and existing.app_dst == dst
                    and sorted(existing.packed_entries) == snapshot
                ):
                    # Coalescing does not renew packet identity or its TTL.
                    key = (src, existing.app_seq)
                    self.e2e_pending[key] = (dst, existing.packed_entries)
                    self.e2e_pending_expires_us[key] = existing.expire_us
                    return existing.pkt_id
        if len(r.queue) >= self.queue_cap:
            self.stats["drop_qfull"] += 1
            return None
        pid = self._next_pkt_id
        self._next_pkt_id += 1
        seq = self._next_app_seq
        self._next_app_seq += 1
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
        # Only an accepted new packet consumes an origin's turn; previews are pure.
        self._origin_order(src, dst, advance=True)
        self.stats["data_enqueued"] += 1
        if dst == self.gcs_id:
            self.stats["data_enqueued_gcs"] += 1
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
