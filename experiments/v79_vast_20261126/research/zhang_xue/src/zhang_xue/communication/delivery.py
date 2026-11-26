from .common import *


class _NetworkDeliveryMixin:

    def _handle_rx_frame(self, rx: int, prev_hop: int, pkt: Packet) -> None:
        if pkt.episode_id != self.episode_id:
            return  # Reject stale packet from prior episode

        if pkt.pkt_type == PKT_HELLO_DV:
            # Update 1-hop neighbor & path-vector DV routes (reject loops via bitmask)
            exp = min(INF_TIME_US, self.sim_time_us + ROUTE_TTL_US)
            self.routes[rx][prev_hop] = (prev_hop, 1, (1 << rx) | (1 << prev_hop), exp)
            advertised = pkt.dv_table or {}
            # Revoke any multi-hop routes previously learned via prev_hop that disappeared from its DV table
            stale_dsts = [
                dst for dst, cur in self.routes[rx].items()
                if dst != prev_hop and cur[0] == prev_hop and (
                    dst not in advertised or (advertised[dst][1] & (1 << rx)) != 0
                )
            ]
            for dst in stale_dsts:
                del self.routes[rx][dst]
            if advertised:
                for dst, (hops, pmask) in advertised.items():
                    if dst == rx or (pmask & (1 << rx)) != 0:
                        continue  # Path-vector loop prevention!
                    cand_hops = hops + 1
                    if cand_hops > self.radio_n - 1:
                        continue
                    cur = self.routes[rx].get(dst)
                    if cur is None or cur[3] <= self.sim_time_us or cand_hops < cur[1] or cur[0] == prev_hop:
                        self.routes[rx][dst] = (prev_hop, cand_hops, (pmask | (1 << rx)) & self.node_mask_all, exp)
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

        # A decoded frame still receives its physical MAC ACK after application expiry.
        # Never merge expired evidence or advance an expired E2E request's watermark.
        if self.sim_time_us >= pkt.expire_us:
            self.stats["drop_ttl"] += 1
            return

        # Deduplicate packet ID (handles DATA received + MAC ACK lost -> retry)
        if pkt.pkt_id in rr.dedup_ids:
            self.stats["dedup_hits"] += 1
            return

        # Check if rx is final application destination
        if pkt.app_dst == rx:
            rr.dedup_ids.append(pkt.pkt_id)
            if pkt.pkt_type == PKT_DATA:
                app_key = (rx, pkt.app_src, pkt.app_seq)
                already_delivered = app_key in self.delivered_app_expires_us
                # Retry at a relay gets a different hop-local pkt_id: dedup using app identity.
                if not already_delivered:
                    self.delivered_app_expires_us[app_key] = min(INF_TIME_US, self.sim_time_us + PACKET_TTL_US)
                    self.stats["data_delivered_dst"] += 1
                    if rx == self.gcs_id:
                        self.stats["data_delivered_gcs"] += 1
                    # Merge packed 2x uint32 entries into receiver's provenance ledger.
                    from ..belief.dempster_shafer import (
                        DENG_MAX_BITS, counts_to_canonical_ds_stable, deng_entropy, fuse_ewma_provenance_uq88,
                    )
                    rx_now_s = max(0.0, (self.sim_time_us - self.mission_epoch_us) / 1e6)
                    for w0, w1 in pkt.packed_entries:
                        origin, cid, gen_ts, n_p, n_n = unpack_entry_2x_uint32(w0, w1, scalable=self.scalable_wire)
                        if self.aoi_ewma_end_to_end:
                            new_info = gen_ts > int(self.ledger_gen_ts_s[rx, origin, cid])
                            if new_info:
                                self.ledger_pos[rx, origin, cid] = n_p
                                self.ledger_neg[rx, origin, cid] = n_n
                                self.ledger_gen_ts_s[rx, origin, cid] = gen_ts
                                self.ledger_rx_age_s[rx, origin, cid] = max(0.0, rx_now_s - gen_ts)
                        else:
                            new_info = (n_p > self.ledger_pos[rx, origin, cid]
                                        or n_n > self.ledger_neg[rx, origin, cid])
                            if n_p > self.ledger_pos[rx, origin, cid]:
                                self.ledger_pos[rx, origin, cid] = n_p
                            if n_n > self.ledger_neg[rx, origin, cid]:
                                self.ledger_neg[rx, origin, cid] = n_n
                            if gen_ts > self.ledger_gen_ts_s[rx, origin, cid]:
                                self.ledger_gen_ts_s[rx, origin, cid] = gen_ts
                                self.ledger_rx_age_s[rx, origin, cid] = max(0.0, rx_now_s - gen_ts)
                        if rx != self.gcs_id and new_info:
                            self.step_new_peer_cells += 1
                            # v6.2: Monotonic Deng-Entropy reduction (telescoping sum <= 1.0, immune to farming/conflict)
                            if self.aoi_ewma_end_to_end:
                                mt, mf_v, mu = fuse_ewma_provenance_uq88(
                                    self.ledger_pos[rx, :, cid], self.ledger_neg[rx, :, cid],
                                    self.ledger_gen_ts_s[rx, :, cid], rx_now_s,
                                    mf=self.ds_mf, me=self.ds_me,
                                    eta=self.aoi_eta if self.ds_fusion_mode == "sum" else 0.0,
                                    mode=self.ds_fusion_mode,
                                )
                            else:
                                tot_p = int(self.ledger_pos[rx, :, cid].sum())
                                tot_n = int(self.ledger_neg[rx, :, cid].sum())
                                mt, mf_v, mu = counts_to_canonical_ds_stable(tot_p, tot_n, self.ds_mf, self.ds_me)
                            h_post = deng_entropy(mt, mf_v, mu)
                            delta_i = max(0.0, (self.peer_h_min[rx, cid] - h_post) / DENG_MAX_BITS)
                            if h_post < self.peer_h_min[rx, cid]:
                                self.peer_h_min[rx, cid] = h_post
                            self.step_peer_info_gain += delta_i
                        if rx == self.gcs_id and new_info:
                            self.gcs_last_novel_rx_us[cid] = self.sim_time_us
                        if rx == self.gcs_id and (
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
                        origin, cid, gen_ts, n_p, n_n = unpack_entry_2x_uint32(w0, w1, scalable=self.scalable_wire)
                        if self.enable_data_mule_forwarding and self.ack_wm_origin_ts is not None:
                            if n_p > self.ack_wm_origin_pos[rx, dst_acked, origin, cid]:
                                self.ack_wm_origin_pos[rx, dst_acked, origin, cid] = n_p
                            if n_n > self.ack_wm_origin_neg[rx, dst_acked, origin, cid]:
                                self.ack_wm_origin_neg[rx, dst_acked, origin, cid] = n_n
                            if gen_ts > self.ack_wm_origin_ts[rx, dst_acked, origin, cid]:
                                self.ack_wm_origin_ts[rx, dst_acked, origin, cid] = gen_ts
                        if origin == rx:
                            if n_p > self.ack_wm_pos[rx, dst_acked, cid]:
                                self.ack_wm_pos[rx, dst_acked, cid] = n_p
                            if n_n > self.ack_wm_neg[rx, dst_acked, cid]:
                                self.ack_wm_neg[rx, dst_acked, cid] = n_n
                            if gen_ts > self.ack_wm_ts[rx, dst_acked, cid]:
                                self.ack_wm_ts[rx, dst_acked, cid] = gen_ts
            return

        # Intermediate relay: DO NOT merge into relay's own belief; check loop & hop budget
        if pkt.hop_cnt >= self.radio_n - 1 or (pkt.visited_mask & (1 << rx)) != 0:
            self.stats["drop_loop"] += 1
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
            visited_mask=(pkt.visited_mask | (1 << rx)) & self.node_mask_all,
            packed_entries=pkt.packed_entries,
        )
        self._next_pkt_id += 1
        # MAC ACK confirms decoding, not relay buffer admission. On queue overflow,
        # record the drop but let a retry after a lost ACK acquire custody later.
        if self._enqueue_packet_internal(rx, fwd):
            rr.dedup_ids.append(pkt.pkt_id)
