"""Real Warp kernels; CPU by default, UAV_WARP_TEST_DEVICE=cuda:N for CUDA gates."""
from __future__ import annotations

import os

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
from zhang_xue.gpu.state_v67 import NEVER_US, ST_TX, ST_TX_ACK, ST_WAIT_ACK


def _network(*, envs=1, uavs=2, grid=2):
    return WarpCompressedRadioNetworkV67(envs, uavs, grid, device=os.environ.get("UAV_WARP_TEST_DEVICE", "cpu"))


def _write(net, name, values):
    target = net.data[name]
    values = np.asarray(values, dtype=target.numpy().dtype)
    wp.copy(target, wp.array(values, dtype=target.dtype, device=net.device))


def test_one_cell_grid_preserves_cell_buffer_ranks_and_reset():
    net = _network(envs=2, grid=1)
    net.sense_batch([[0, -1], [0, -1]], [[True, False], [True, False]], 1)
    received = np.full((2, 3, 16), -1, dtype=np.int32)
    received[:, 0, 0] = 17
    _write(net, "received_pid", received)
    _write(net, "received_ptr", [[1, 0, 0], [1, 0, 0]])
    net.reset(np.array([True, False]))
    snapshot = net.snapshot(("ledger_gen", "gcs_latest", "gcs_last_rx_us", "received_pid", "received_ptr"))
    assert snapshot["gcs_latest"].shape == (2, 1)
    assert snapshot["gcs_last_rx_us"].shape == (2, 1)
    assert snapshot["ledger_gen"][0, 0, 0, 0] == -1
    assert snapshot["ledger_gen"][1, 0, 0, 0] == 1
    assert snapshot["received_pid"][0, 0, 0] == -1
    assert snapshot["received_pid"][1, 0, 0] == 17
    assert snapshot["received_ptr"].tolist() == [[0, 0, 0], [1, 0, 0]]


def _single_hello(net):
    _write(net, "radio_alive", [[1, 0, 0]])
    _write(net, "hello_due", [[0, NEVER_US, NEVER_US]])


def test_exact_event_budget_completes_when_no_more_events_are_due():
    net = _network()
    _single_hello(net)
    net.step(duration_us=1, max_events=1)
    snapshot = net.snapshot(("now", "overflow", "event_count", "deadline"))
    assert snapshot["now"].tolist() == [1]
    assert snapshot["overflow"].tolist() == [0]
    assert snapshot["event_count"].tolist() == [1]
    assert snapshot["deadline"][0, 0] == 50


def test_genuine_overflow_stays_blocked_without_host_readback_until_reset():
    net = _network()
    _single_hello(net)
    net.step(duration_us=100, max_events=1, check_host_guards=False)
    before = net.snapshot(("now", "overflow", "event_count", "state", "deadline"))
    assert before["overflow"].tolist() == [1]
    assert before["now"].tolist() == [0]
    net.step(duration_us=100, max_events=128, check_host_guards=False)
    after = net.snapshot(tuple(before))
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
    with pytest.raises(RuntimeError, match="budget exceeded"):
        net.step(duration_us=100, max_events=128)
    net.reset()
    net.step(duration_us=1, max_events=1)
    assert net.snapshot(("overflow",))["overflow"].tolist() == [0]


def test_clock_guard_runs_on_device_when_host_readback_is_disabled():
    net = _network()
    _write(net, "now", [NEVER_US - 5])
    net.step(duration_us=10, check_host_guards=False)
    snapshot = net.snapshot(("now", "overflow"))
    assert snapshot["now"].tolist() == [NEVER_US - 5]
    assert snapshot["overflow"].tolist() == [1]


def test_mac_ack_releases_every_overhearing_receiver_lock():
    net = _network()
    _write(net, "hello_due", [[NEVER_US] * 3])
    _write(net, "state", [[ST_WAIT_ACK, ST_TX_ACK, 0]])
    _write(net, "deadline", [[100, 0, NEVER_US]])
    _write(net, "next_hop", [[-1, 0, -1]])
    _write(net, "rx_lock", [[1, -1, 1]])
    _write(net, "rx_sig", [[1.0, 0.0, 1.0]])
    _write(net, "rx_corrupt", [[0, 0, 1]])
    net.set_channel_gain(np.ones((1, 3, 3), dtype=np.float32))
    net.step(duration_us=1, max_events=4)
    snapshot = net.snapshot(("rx_lock", "rx_corrupt", "stats_mac_ack"))
    assert snapshot["rx_lock"].tolist() == [[-1, -1, -1]]
    assert snapshot["rx_corrupt"].tolist() == [[0, 0, 0]]
    assert snapshot["stats_mac_ack"].tolist() == [1]


def test_hello_does_not_advertise_expired_routes():
    net = _network()
    _write(net, "hello_due", [[0, NEVER_US, NEVER_US]])
    routes = np.full((1, 3, 3), -1, dtype=np.int8)
    routes[0, 0, 2] = 1
    _write(net, "route_hop", routes)
    _write(net, "route_expiry", np.zeros((1, 3, 3), dtype=np.int32))
    net.step(duration_us=1, max_events=4)
    snapshot = net.snapshot(("hello_dv_hop", "route_hop"))
    assert snapshot["hello_dv_hop"][0, 0, 2] == -1
    assert snapshot["route_hop"][0, 0, 2] == -1


def _isolated_link(net, *, ack_reachable=True):
    _write(net, "hello_due", [[NEVER_US] * 3])
    gains = np.zeros((1, 3, 3), dtype=np.float32)
    gains[0, 0, 1] = 1.0e-6
    if ack_reachable:
        gains[0, 1, 0] = 1.0e-6
    net.set_channel_gain(gains)
    routes = np.full((1, 3, 3), -1, dtype=np.int8)
    routes[0, 0, 1] = 1
    routes[0, 1, 0] = 0
    _write(net, "route_hop", routes)
    distances = np.full((1, 3, 3), 127, dtype=np.int8)
    distances[0, 0, 1] = 1
    distances[0, 1, 0] = 1
    _write(net, "route_dist", distances)
    _write(net, "route_expiry", np.full((1, 3, 3), 10_000_000, dtype=np.int32))
    net.sense_batch([[0, -1]], [[True, False]], 1)


def test_injected_queue_starts_without_waiting_for_a_hello_event():
    net = _network()
    _isolated_link(net)
    net.inject_compact3([[2, 0]], np.zeros((1, 2, 2), dtype=np.float32))
    net.step(duration_us=10_000, max_events=128)
    snapshot = net.snapshot(("stats_peer", "stats_e2e", "q_count"))
    assert snapshot["stats_peer"].tolist() == [1]
    assert snapshot["stats_e2e"].tolist() == [1]
    assert snapshot["q_count"].tolist() == [[0, 0, 0]]


def test_lost_mac_ack_does_not_duplicate_relay_custody_and_counts_retry_drop():
    net = _network()
    _isolated_link(net, ack_reachable=False)
    routes = net.snapshot(("route_hop",))["route_hop"]
    routes[0, 0, 2] = 1
    _write(net, "route_hop", routes)
    net.inject_compact3([[1, 0]], np.zeros((1, 2, 2), dtype=np.float32))
    net.step(duration_us=100_000, max_events=512)
    snapshot = net.snapshot(("q_count", "stats_retry", "stats_drop", "q_seq"))
    assert snapshot["q_count"].tolist() == [[0, 1, 0]]
    assert snapshot["stats_retry"].tolist() == [5]
    assert snapshot["stats_drop"].tolist() == [1]
    assert snapshot["q_seq"][0, 1, 0] == 1


def test_expired_in_flight_data_is_link_acked_without_application_delivery():
    net = _network()
    _isolated_link(net)
    net.inject_compact3([[2, 0]], np.zeros((1, 2, 2), dtype=np.float32))
    _write(net, "state", [[ST_TX, 0, 0]])
    _write(net, "deadline", [[400, NEVER_US, NEVER_US]])
    _write(net, "active_slot", [[0, -1, -1]])
    _write(net, "next_hop", [[1, -1, -1]])
    _write(net, "tx_pkt_id", [[1, -1, -1]])
    _write(net, "rx_lock", [[-1, 0, -1]])
    _write(net, "rx_sig", [[0.0, 1.0e-7, 0.0]])
    expiry = net.snapshot(("q_expiry",))["q_expiry"]
    expiry[0, 0, 0] = 300
    _write(net, "q_expiry", expiry)
    net.step(duration_us=1_000, max_events=16)
    snapshot = net.snapshot(("ledger_gen", "stats_peer", "stats_mac_ack", "stats_ttl"))
    assert snapshot["ledger_gen"][0, 1, 0, 0] == -1
    assert snapshot["stats_peer"].tolist() == [0]
    assert snapshot["stats_mac_ack"].tolist() == [1]
    assert snapshot["stats_ttl"].tolist() == [1]


def test_local_peer_selection_ignores_hidden_peer_position_and_alive_state():
    net = _network(uavs=3)
    net.sense_batch([[0, -1, -1]], [[True, False, False]], 1)
    routes = np.full((1, 4, 4), -1, dtype=np.int8)
    routes[0, 0, 1:3] = [1, 2]
    _write(net, "route_hop", routes)
    distances = np.full((1, 4, 4), 127, dtype=np.int8)
    distances[0, 0, 1:3] = [1, 2]
    _write(net, "route_dist", distances)
    _write(net, "route_expiry", np.full((1, 4, 4), 10_000_000, dtype=np.int32))
    positions = np.array([[[0.0, 0.0], [100.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    net.inject_compact3([[2, 0, 0]], positions)
    first = net.snapshot(("q_dst",))["q_dst"][0, 0, 0]
    net.reset()
    net.sense_batch([[0, -1, -1]], [[True, False, False]], 1)
    _write(net, "route_hop", routes)
    _write(net, "route_dist", distances)
    _write(net, "route_expiry", np.full((1, 4, 4), 10_000_000, dtype=np.int32))
    net.set_radio_alive([[1, 0, 1, 1]])
    positions[0, 1:3] = [[1_000.0, 1_000.0], [0.1, 0.0]]
    net.inject_compact3([[2, 0, 0]], positions)
    second = net.snapshot(("q_dst",))["q_dst"][0, 0, 0]
    assert (first, second) == (1, 1)


def test_new_application_retry_cannot_refresh_gcs_with_identical_evidence():
    net = _network()
    _write(net, "hello_due", [[NEVER_US] * 3])
    gains = np.zeros((1, 3, 3), dtype=np.float32)
    gains[0, 0, 2] = 1.0e-6  # Reverse path is lost, including the E2E ACK.
    net.set_channel_gain(gains)
    routes = np.full((1, 3, 3), -1, dtype=np.int8)
    routes[0, 0, 2] = 2
    routes[0, 2, 0] = 0
    _write(net, "route_hop", routes)
    _write(net, "route_expiry", np.full((1, 3, 3), 100_000_000, dtype=np.int32))
    net.sense_batch([[0, -1]], [[True, False]], 1)
    positions = np.zeros((1, 2, 2), dtype=np.float32)
    net.inject_compact3([[1, 0]], positions)
    net.step(duration_us=10_100_000, max_events=1024)
    first = net.snapshot(("gcs_last_rx_us",))["gcs_last_rx_us"][0, 0]
    assert first >= 0
    # E2E ownership expired; the same dirty snapshot now has a new app sequence.
    # A renewed HELLO contact makes the forward route usable after retry revoke.
    _write(net, "route_hop", routes)
    _write(net, "route_expiry", np.full((1, 3, 3), 100_000_000, dtype=np.int32))
    net.inject_compact3([[1, 0]], positions)
    net.step(duration_us=100_000, max_events=1024)
    snapshot = net.snapshot(("stats_gcs", "gcs_last_rx_us"))
    assert snapshot["stats_gcs"].tolist() == [2]
    assert snapshot["gcs_last_rx_us"][0, 0] == first


def test_full_relay_queue_does_not_record_custody_before_admission():
    net = _network()
    _isolated_link(net, ack_reachable=False)
    routes = net.snapshot(("route_hop",))["route_hop"]
    routes[0, 0, 2] = 1
    _write(net, "route_hop", routes)
    for name, row in (
        ("q_used", np.ones(32)), ("q_order", np.arange(32)),
        ("q_dst", np.full(32, 2)), ("q_expiry", np.full(32, 10_000_000)),
        ("q_pid", np.arange(1_000, 1_032)),
    ):
        values = net.snapshot((name,))[name]
        values[0, 1] = row
        _write(net, name, values)
    _write(net, "q_count", [[0, 32, 0]])
    net.inject_compact3([[1, 0]], np.zeros((1, 2, 2), dtype=np.float32))
    net.step(duration_us=1_000, max_events=128)
    snapshot = net.snapshot(("received_ptr", "stats_drop"))
    assert snapshot["received_ptr"].tolist() == [[0, 0, 0]]
    assert snapshot["stats_drop"].tolist() == [1]
    used = net.snapshot(("q_used",))["q_used"]
    used[0, 1, 31] = 0
    _write(net, "q_used", used)
    _write(net, "q_count", [[1, 31, 0]])
    net.step(duration_us=100_000, max_events=512)
    snapshot = net.snapshot(("received_ptr", "q_count", "q_seq"))
    assert snapshot["received_ptr"].tolist() == [[0, 1, 0]]
    assert snapshot["q_count"].tolist() == [[0, 32, 0]]
    assert np.count_nonzero(snapshot["q_seq"][0, 1] == 1) == 1


def test_raw_count_mode_is_explicitly_unsupported_by_generation_only_ack_layout():
    with pytest.raises(NotImplementedError, match="generation-only.*EWMA"):
        WarpCompressedRadioNetworkV67(1, 2, 2, aoi_ewma=False, device="cpu")


@pytest.mark.parametrize("name", ["num_envs", "num_uavs", "grid_w"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_runtime_dimensions_reject_boolean_and_fractional_values(name, value):
    args = dict(num_envs=1, num_uavs=2, grid_w=2, device="cpu")
    args[name] = value
    with pytest.raises(ValueError, match="integer"):
        WarpCompressedRadioNetworkV67(**args)


@pytest.mark.parametrize("eta", [-0.01, np.nan, np.inf])
def test_runtime_rejects_invalid_ewma_decay_before_allocation(eta):
    with pytest.raises(ValueError, match="aoi_eta"):
        WarpCompressedRadioNetworkV67(1, 2, 2, aoi_eta=eta, device="cpu")


def test_cuda_memory_preflight_does_not_depend_on_optional_torch(monkeypatch):
    import sys
    from types import SimpleNamespace

    # Only device metadata is stubbed: this test launches no CUDA kernels.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(wp, "get_device", lambda _: SimpleNamespace(is_cuda=True, free_memory=0))
    with pytest.raises(MemoryError, match="Network SoA"):
        WarpCompressedRadioNetworkV67(1, 2, 2, device="cuda:0")


@pytest.mark.parametrize("timestamp", [2, 1])
def test_duplicate_or_backward_generation_rejects_whole_env_before_sensing(timestamp):
    net = _network(envs=2)
    net.sense_batch([[0, -1], [0, -1]], [[True, False], [True, False]], 2)
    before = net.snapshot(("ledger_pos", "ledger_neg", "ledger_gen"))
    with pytest.raises(ValueError, match="strictly increasing"):
        net.sense_batch([[0, 1], [1, -1]], [[False, True], [True, False]], timestamp)
    after = net.snapshot((*before, "overflow"))
    for name in before:
        np.testing.assert_array_equal(after[name][0], before[name][0])
    assert after["overflow"].tolist() == [3, 0]
    assert after["ledger_gen"][1, 0, 0, 1] == timestamp


def test_distinct_cells_and_sources_can_share_generation_timestamp():
    net = _network()
    net.sense_batch([[0, -1]], [[True, False]], 2)
    net.sense_batch([[1, 0]], [[False, True]], 2)
    snapshot = net.snapshot(("ledger_gen", "ledger_pos", "ledger_neg", "overflow"))
    assert snapshot["overflow"].tolist() == [0]
    assert snapshot["ledger_gen"][0, 0, 0, :2].tolist() == [2, 2]
    assert snapshot["ledger_gen"][0, 1, 1, 0] == 2
    assert snapshot["ledger_pos"][0, 0, 0, 0] == 256
    assert snapshot["ledger_neg"][0, 0, 0, 1] == 256


def test_unguarded_generation_fault_is_sticky_and_reset_recovers():
    net = _network()
    net.sense_batch([[0, -1]], [[True, False]], 2)
    before = net.snapshot(("ledger_pos", "ledger_neg", "ledger_gen", "now", "event_count"))
    net.sense_batch([[0, -1]], [[False, False]], 2, check_host_guards=False)
    net.sense_batch([[1, -1]], [[True, False]], 3, check_host_guards=False)
    net.step(duration_us=1_000, check_host_guards=False)
    after = net.snapshot((*before, "overflow"))
    assert after["overflow"].tolist() == [3]
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
    with pytest.raises(ValueError, match="strictly increasing"):
        net.step(duration_us=1_000)
    net.reset()
    net.sense_batch([[0, -1]], [[True, False]], 2)
    assert net.snapshot(("overflow",))["overflow"].tolist() == [0]
