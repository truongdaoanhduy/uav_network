from .common import *


class _NetworkDeliveryMixin:

    def _handle_rx_frame(self, rx: int, prev_hop: int, pkt: Packet) -> None:
        if pkt.episode_id != self.episode_id:
            return  # Reject stale packet from prior episode

        if pkt.pkt_type == PKT_HELLO_DV:
            # Update 1-hop neighbor & path-vector DV routes (reject loops via 7-bit mask)
            exp = min(INF_TIME_US, self.sim_time_us + ROUTE_TTL_US)
            self.routes[rx][prev_hop] = (prev_hop, 1, (1 << rx) | (1 << prev_hop), exp)
            if pkt.dv_table:
                for dst, (hops, pmask) in pkt.dv_table.items():
                    if dst == rx or (pmask & (1 << rx)) != 0:
                        continue  # Path-vector loop prevention!
                    cand_hops = hops + 1
                    if cand_hops > MAX_HOPS:
                        continue
                    cur = self.routes[rx].get(dst)
                    if cur is None or cur[3] <= self.sim_time_us or cand_hops < cur[1] or cur[0] == prev_hop:
                        self.routes[rx][dst] = (prev_hop, cand_hops, (pmask | (1 << rx)) & NODE_MASK_7BIT, exp)
            return

        # Unicast DATA or E2E_ACK: only intended next hop processes it
        if self.radios[prev_hop].active_next_hop != rx:
            return

        # Schedule immediate SIFS -> TX_MAC_ACK (bypasses FIFO queue)
        rr = self.radios[rx]
        if rr.mac_state in (ST_IDLE, ST_DIFS, ST_BACKOFF):
            if rr.mac_state == ST_BACKOFF:
                rr.backoff_rem_us = max(0, rr.state_end_us - self.sim_time_us)
                rr.backoff_frozen = True
            rr.ack_pend_dst = prev_hop
            rr.ack_pend_pkt_id = pkt.pkt_id
            rr.mac_state = ST_SIFS_ACK
            rr.state_end_us = self.sim_time_us + self.protocol.sifs_us

        # Deduplicate packet ID (handles DATA received + MAC ACK lost -> retry)
        if pkt.pkt_id in rr.dedup_ids:
            self.stats["dedup_hits"] += 1
            return
        rr.dedup_ids.append(pkt.pkt_id)

        # Check if rx is final application destination
        if pkt.app_dst == rx:
            if pkt.pkt_type == PKT_DATA:
                app_key = (rx, pkt.app_src, pkt.app_seq)
                already_delivered = app_key in self.delivered_app_expires_us
                # Retry at a relay gets a different hop-local pkt_id: dedup using app identity.
                if not already_delivered:
                    self.delivered_app_expires_us[app_key] = min(INF_TIME_US, self.sim_time_us + PACKET_TTL_US)
                    self.stats["data_delivered_dst"] += 1
                    if rx == GCS_ID:
                        self.stats["data_delivered_gcs"] += 1
                    # Merge packed 2x uint32 entries into receiver's provenance ledger.
                    for w0, w1 in pkt.packed_entries:
                        origin, cid, gen_ts, n_p, n_n = unpack_entry_2x_uint32(w0, w1)
                        new_info = (n_p > self.ledger_pos[rx, origin, cid]
                                    or n_n > self.ledger_neg[rx, origin, cid])
                        if n_p > self.ledger_pos[rx, origin, cid]:
                            self.ledger_pos[rx, origin, cid] = n_p
                        if n_n > self.ledger_neg[rx, origin, cid]:
                            self.ledger_neg[rx, origin, cid] = n_n
                        if gen_ts > self.ledger_gen_ts_s[rx, origin, cid]:
                            self.ledger_gen_ts_s[rx, origin, cid] = gen_ts
                            # Delivery-time-only discount, no artificial refresh on old retries.
                            self.ledger_rx_age_s[rx, origin, cid] = max(
                                0., (self.sim_time_us - self.mission_epoch_us) / 1e6 - gen_ts)
                        if rx != GCS_ID and new_info:
                            self.step_new_peer_cells += 1
                            # Approximate marginal uncertainty reduction weight: diminishes rapidly with existing counts
                            prev_total = int(self.ledger_pos[rx, :, cid].sum() + self.ledger_neg[rx, :, cid].sum())
                            self.step_peer_info_gain += 1.0 / float(max(1, prev_total))
                        if rx == GCS_ID and (
                            gen_ts > self.gcs_latest_u_s[cid]
                            or (gen_ts == self.gcs_latest_u_s[cid] and new_info)
                        ):
                            # Stale retries cannot refresh receiver AoI; same-second novel evidence updates rx time (B12).
                            self.gcs_latest_u_s[cid] = gen_ts
                            self.gcs_last_rx_us[cid] = self.sim_time_us
                else:
                    self.stats["dedup_hits"] += 1
                # E2E ACK re-sent even on duplicates: original ACK may have been lost.
                ack_pid = self._next_pkt_id
                self._next_pkt_id += 1
                e2e_ack = Packet(
                    pkt_id=ack_pid,
                    episode_id=self.episode_id,
                    pkt_type=PKT_E2E_ACK,
                    app_src=rx,
                    app_dst=pkt.app_src,
                    app_seq=pkt.app_seq,
                    create_us=self.sim_time_us,
                    expire_us=min(INF_TIME_US, self.sim_time_us + PACKET_TTL_US),
                    hop_cnt=0,
                    visited_mask=(1 << rx),
                )
                self._enqueue_packet_internal(rx, e2e_ack)
            elif pkt.pkt_type == PKT_E2E_ACK:
                # Advance sender's non-oracle E2E ACK watermark!
                self.stats["e2e_ack_rx"] += 1
                pend = self.e2e_pending.pop((rx, pkt.app_seq), None)
                self.e2e_pending_expires_us.pop((rx, pkt.app_seq), None)
                if pend is not None:
                    dst_acked, entries = pend
                    for w0, w1 in entries:
                        _, cid, gen_ts, n_p, n_n = unpack_entry_2x_uint32(w0, w1)
                        if n_p > self.ack_wm_pos[rx, dst_acked, cid]:
                            self.ack_wm_pos[rx, dst_acked, cid] = n_p
                        if n_n > self.ack_wm_neg[rx, dst_acked, cid]:
                            self.ack_wm_neg[rx, dst_acked, cid] = n_n
                        if gen_ts > self.ack_wm_ts[rx, dst_acked, cid]:
                            self.ack_wm_ts[rx, dst_acked, cid] = gen_ts
            return

        # Intermediate relay: DO NOT merge into relay's own belief; check loop & hop budget
        if pkt.hop_cnt + 1 >= MAX_HOPS or (pkt.visited_mask & (1 << rx)) != 0:
            self.stats["drop_loop"] += 1
            return
        if self.sim_time_us >= pkt.expire_us:
            self.stats["drop_ttl"] += 1
            return
        fwd = Packet(
            pkt_id=self._next_pkt_id,
            episode_id=pkt.episode_id,
            pkt_type=pkt.pkt_type,
            app_src=pkt.app_src,
            app_dst=pkt.app_dst,
            app_seq=pkt.app_seq,
            create_us=pkt.create_us,
            expire_us=pkt.expire_us,
            hop_cnt=pkt.hop_cnt + 1,
            visited_mask=(pkt.visited_mask | (1 << rx)) & NODE_MASK_7BIT,
            packed_entries=pkt.packed_entries,
        )
        self._next_pkt_id += 1
        self._enqueue_packet_internal(rx, fwd)
