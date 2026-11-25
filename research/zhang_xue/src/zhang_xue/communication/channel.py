from .common import *


class _NetworkChannelMixin:

    def _is_channel_busy_at(self, node_id: int) -> bool:
        for j in range(RADIO_N):
            if j == node_id:
                continue
            if self.radios[j].mac_state in (ST_TX_DATA, ST_TX_MAC_ACK):
                p_rx = self.tx_power_w * self.chan_gain[j, node_id]
                if p_rx >= self.cca_th_w:
                    return True
        return False


    def _compute_interference_at(self, rx: int, exclude_tx: int) -> float:
        interf = 0.0
        for m in range(RADIO_N):
            if m == rx or m == exclude_tx:
                continue
            if self.radios[m].mac_state in (ST_TX_DATA, ST_TX_MAC_ACK):
                interf += self.tx_power_w * self.chan_gain[m, rx]
        return interf


    def _start_tx(self, tx: int, is_ack: bool) -> None:
        r = self.radios[tx]
        # Half-duplex: if tx was receiving something, corrupt its RX immediately
        if r.rx_locked_tx >= 0:
            r.rx_corrupted = True
            self.stats["half_duplex_drops"] += 1

        if is_ack:
            dur = TX_MAC_ACK_US
            r.mac_state = ST_TX_MAC_ACK
            r.state_end_us = self.sim_time_us + dur
            r.tx_is_ack = True
            r.tx_pkt_id = r.ack_pend_pkt_id
            r.tx_ack_dst = r.ack_pend_dst
            pkt_ref = None
            pkt_id = r.ack_pend_pkt_id
        else:
            assert r.active_pkt is not None
            dur = r.active_pkt.tx_duration_us
            r.mac_state = ST_TX_DATA
            r.state_end_us = self.sim_time_us + dur
            r.tx_is_ack = False
            r.tx_pkt_id = r.active_pkt.pkt_id
            r.tx_ack_dst = -1
            pkt_ref = r.active_pkt
            pkt_id = r.active_pkt.pkt_id

        # Notify all other radios: freeze backoff / update RX lock / update mid-packet interference
        for rx in range(RADIO_N):
            if rx == tx:
                continue
            rr = self.radios[rx]
            p_rx = self.tx_power_w * self.chan_gain[tx, rx]

            # Freeze ongoing DIFS or BACKOFF if CCA busy
            if p_rx >= self.cca_th_w:
                if rr.mac_state == ST_DIFS:
                    rr.mac_state = ST_IDLE
                    rr.state_end_us = INF_TIME_US
                elif rr.mac_state == ST_BACKOFF:
                    rr.backoff_rem_us = max(0, rr.state_end_us - self.sim_time_us)
                    rr.backoff_frozen = True
                    rr.mac_state = ST_IDLE
                    rr.state_end_us = INF_TIME_US

            # Half-duplex check: transmitting radio cannot lock onto incoming signal
            if rr.mac_state in (ST_TX_DATA, ST_TX_MAC_ACK):
                continue

            if not self.radio_alive[rx]:
                continue
            if rr.rx_locked_tx < 0:
                # B1 fix: require p_rx >= cca_th_w so signals 26,000x below noise floor cannot lock RX
                if p_rx >= self.cca_th_w:
                    rr.rx_locked_tx = tx
                    rr.rx_locked_pkt = pkt_ref
                    rr.rx_locked_is_ack = is_ack
                    rr.rx_locked_pkt_id = pkt_id
                    rr.rx_corrupted = False
                    rr.rx_sig_w = p_rx
                    rr.rx_max_interf_w = self._compute_interference_at(rx, exclude_tx=tx)
            else:
                # Mid-packet interferer! Update worst-case interference on existing RX lock
                interf = self._compute_interference_at(rx, exclude_tx=rr.rx_locked_tx)
                if interf > rr.rx_max_interf_w:
                    rr.rx_max_interf_w = interf


    def _finish_tx(self, tx: int) -> None:
        r = self.radios[tx]
        was_ack = r.tx_is_ack
        pkt = r.active_pkt
        ack_dst = r.tx_ack_dst
        ack_pkt_id = r.tx_pkt_id

        # Charge TX airtime to actual transmitting radio (GCS airtime tracked separately, never billed to UAVs)
        dur = TX_MAC_ACK_US if was_ack else (pkt.tx_duration_us if pkt is not None else 0)
        self.step_tx_airtime_us[tx] += dur

        # Deliver to any radio locked onto tx whose SINR >= threshold
        for rx in range(RADIO_N):
            if rx == tx:
                continue
            rr = self.radios[rx]
            if rr.rx_locked_tx != tx:
                continue
            # Evaluate SINR with worst-case overlapping interference
            sinr = rr.rx_sig_w / (self.noise_power_w + rr.rx_max_interf_w)
            ok = (not rr.rx_corrupted) and (sinr >= self.snr_th_lin) and (self.chan_gain[tx, rx] > 0.0)
            if was_ack and self.force_drop_next_mac_ack_from == tx:
                ok = False
                self.force_drop_next_mac_ack_from = None
            # Clear RX lock
            rr.rx_locked_tx = -1
            rr.rx_locked_pkt = None
            rr.rx_corrupted = False

            if not ok:
                self.stats["sinr_drops"] += 1
                continue

            if was_ack:
                if rx == ack_dst and rr.mac_state == ST_WAIT_ACK and rr.tx_pkt_id == ack_pkt_id:
                    # Successful MAC ACK receipt!
                    self.stats["mac_ack_ok"] += 1
                    rr.active_pkt = None
                    rr.active_next_hop = -1
                    rr.retx_cnt = 0
                    rr.backoff_frozen = False
                    rr.backoff_rem_us = 0
                    rr.mac_state = ST_IDLE
                    rr.state_end_us = INF_TIME_US
            else:
                assert pkt is not None
                self._handle_rx_frame(rx=rx, prev_hop=tx, pkt=pkt)

        # Transition transmitter state
        if was_ack:
            r.ack_pend_dst = -1
            r.ack_pend_pkt_id = -1
            r.tx_is_ack = False
            r.mac_state = ST_IDLE
            r.state_end_us = INF_TIME_US
        else:
            assert pkt is not None
            if pkt.pkt_type == PKT_HELLO_DV:
                # Broadcast HELLO needs no MAC ACK
                r.active_pkt = None
                r.active_next_hop = -1
                r.retx_cnt = 0
                r.backoff_frozen = False
                r.mac_state = ST_IDLE
                r.state_end_us = INF_TIME_US
            else:
                r.mac_state = ST_WAIT_ACK
                r.state_end_us = self.sim_time_us + self.protocol.ack_timeout_us
