"""Behavior regressions for packet TTL, queue custody and origin selection."""
from __future__ import annotations

import numpy as np

from zhang_xue.communication.common import (
    CsmaProtocolConfig, INF_TIME_US, PACKET_TTL_US, PKT_DATA, Packet,
    ST_SIFS_ACK, ST_TX_MAC_ACK, unpack_entry_2x_uint32,
)
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork


def _net(num_uavs=2, **kwargs):
    net = SevenRadioCsmaCaNetwork(
        num_uavs=num_uavs, buildings=[], blackout_centers=[], seed=71, **kwargs,
    )
    net.hello_next_us = [INF_TIME_US] * net.radio_n
    net.chan_gain.fill(1e-4)
    np.fill_diagonal(net.chan_gain, 0.0)
    for src in range(net.radio_n):
        for dst in range(net.radio_n):
            if src != dst:
                net.routes[src][dst] = (dst, 1, (1 << src) | (1 << dst), INF_TIME_US)
    return net


def _enqueue_own(net, src=0, dst=None, t_s=0):
    dst = net.gcs_id if dst is None else dst
    net.record_local_observation(src, 0, True, t_s)
    entries = net.get_dirty_entries_non_oracle(src, dst)
    net.enqueue_app_data(src, dst, entries)
    return net.radios[src].queue[-1]


def test_packet_expiring_during_difs_never_transmits():
    net = _net(num_uavs=1)
    pkt = _enqueue_own(net)
    pkt.expire_us = 1

    assert net.run_until_us(10_000)

    assert net.stats["data_delivered_gcs"] == 0
    assert net.stats["drop_ttl"] == 1
    assert net.step_tx_airtime_us[0] == 0
    assert net.radios[0].active_pkt is None


def test_packet_expiring_in_flight_is_mac_acked_without_application_delivery():
    net = _net(num_uavs=1, protocol=CsmaProtocolConfig(cw_min=1))
    pkt = _enqueue_own(net)
    pkt.expire_us = 100  # TX starts by 70 us; its 312 us airtime crosses expiry.

    assert net.run_until_us(10_000)

    assert net.stats["data_delivered_gcs"] == 0
    assert net.stats["drop_ttl"] == 1
    assert net.stats["mac_ack_ok"] == 1
    assert net.stats["e2e_ack_rx"] == 0
    assert net.ledger_gen_ts_s[net.gcs_id, 0, 0] == -1


def test_queue_full_relay_can_accept_retry_after_buffer_becomes_available():
    net = _net(queue_cap=1, protocol=CsmaProtocolConfig(cw_min=1))
    net.routes[0][net.gcs_id] = (1, 2, net.node_mask_all, INF_TIME_US)
    del net.routes[1][0]
    net.radios[1].queue.append(Packet(
        99, 0, PKT_DATA, 1, 0, 99, 0, PACKET_TTL_US, visited_mask=1 << 1,
    ))
    _enqueue_own(net)
    net.force_drop_next_mac_ack_from = 1

    assert net.run_until_us(400)
    assert net.stats["drop_qfull"] == 1
    # Queue overflow does not suppress the physical acknowledgement of reception.
    assert net.radios[1].mac_state in (ST_SIFS_ACK, ST_TX_MAC_ACK)
    net.radios[1].queue.clear()

    assert net.run_until_us(100_000)

    assert net.stats["mac_ack_timeout"] >= 1
    assert net.stats["data_delivered_gcs"] == 1
    assert net.stats["drop_qfull"] == 1


def test_live_coalescing_preserves_original_packet_and_pending_deadline():
    net = _net(coalesce_duplicate_queue=True)
    del net.routes[0][net.gcs_id]
    pkt = _enqueue_own(net)
    expiry = pkt.expire_us
    assert net.run_until_us(1_000_000)

    entries = net.get_dirty_entries_non_oracle(0, net.gcs_id)
    assert net.enqueue_app_data(0, net.gcs_id, entries) == pkt.pkt_id

    assert len(net.radios[0].queue) == 1
    assert pkt.expire_us == expiry
    assert net.e2e_pending_expires_us[(0, pkt.app_seq)] == expiry


def test_expired_coalesced_snapshot_gets_fresh_identity_and_working_e2e_ack():
    net = _net(coalesce_duplicate_queue=True)
    del net.routes[0][net.gcs_id]
    pkt = _enqueue_own(net)
    assert net.run_until_us(PACKET_TTL_US + 1)
    assert not net.e2e_pending
    entries = net.get_dirty_entries_non_oracle(0, net.gcs_id)

    pid = net.enqueue_app_data(0, net.gcs_id, entries)

    assert pid != pkt.pkt_id
    replacement = net.radios[0].queue[0]
    assert replacement.app_seq != pkt.app_seq
    assert len(net.radios[0].queue) == 1
    assert (0, replacement.app_seq) in net.e2e_pending
    net.routes[0][net.gcs_id] = (net.gcs_id, 1, net.node_mask_all, INF_TIME_US)
    assert net.run_until_us(net.sim_time_us + 10_000)
    assert net.ack_wm_ts[0, net.gcs_id, 0] == 0
    assert net.count_all_dirty_entries_non_oracle(0, net.gcs_id) == 0


def test_repeated_payload_previews_do_not_change_origin_selection():
    net = _net(num_uavs=3, enable_data_mule_forwarding=True)
    for origin in (1, 2):
        net.ledger_gen_ts_s[0, origin, :62] = 1
        net.ledger_pos[0, origin, :62] = 1
    expected = net.get_dirty_entries_non_oracle(0, net.gcs_id)

    for _ in range(6):
        assert net.get_dirty_entries_non_oracle(0, net.gcs_id) == expected

    assert net.count_all_dirty_entries_non_oracle(0, net.gcs_id) == 124
    assert net.stats["data_enqueued"] == 0


def test_successful_enqueues_rotate_own_and_relay_origins_under_sustained_load():
    net = _net(num_uavs=3, enable_data_mule_forwarding=True)
    for origin in range(3):
        net.ledger_gen_ts_s[0, origin, :62] = 1
        net.ledger_pos[0, origin, :62] = 1
    selected = []
    for _ in range(6):
        entries = net.get_dirty_entries_non_oracle(0, net.gcs_id)
        assert net.enqueue_app_data(0, net.gcs_id, entries) is not None
        selected.append({unpack_entry_2x_uint32(*entry)[0] for entry in entries})

    assert selected == [{0}, {1}, {2}, {0}, {1}, {2}]


def test_coalescing_ignores_origin_order_and_does_not_consume_another_turn():
    net = _net(num_uavs=2, queue_cap=1, enable_data_mule_forwarding=True,
               coalesce_duplicate_queue=True)
    for origin in range(2):
        net.ledger_gen_ts_s[0, origin, 0] = 1
        net.ledger_pos[0, origin, 0] = 1
    entries = net.get_dirty_entries_non_oracle(0, net.gcs_id)
    pid = net.enqueue_app_data(0, net.gcs_id, entries)
    next_preview = net.get_dirty_entries_non_oracle(0, net.gcs_id)
    assert next_preview == entries[::-1]

    assert net.enqueue_app_data(0, net.gcs_id, next_preview) == pid

    assert len(net.radios[0].queue) == 1
    assert net.stats["drop_qfull"] == 0
    assert net.get_dirty_entries_non_oracle(0, net.gcs_id) == next_preview
