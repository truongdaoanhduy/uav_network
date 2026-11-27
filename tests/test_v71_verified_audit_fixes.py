"""v7.1 RED->GREEN regression tests for Warp kernel parity, MAC/routing, DS-AoI, safety and memory."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from zhang_xue.communication.common import PKT_DATA, PKT_HELLO_DV, Packet, ST_TX_DATA
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.runtime_v67 import normalize_sensing_inputs
from zhang_xue.learners.actor_adapter import decode_tx_power_w
from zhang_xue.mobility.flight_safety_energy import kinematic_safety_shield
from scripts.maintenance.network_memory_report import memory_inventory

ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / "research/zhang_xue/src/zhang_xue/gpu"


def test_v71_warp_source_contracts_fix_v70_gaps():
    events = (GPU_DIR / "kernels/events_v67.py").read_text(encoding="utf-8")
    mac = (GPU_DIR / "kernels/mac_v67.py").read_text(encoding="utf-8")
    inject = (GPU_DIR / "kernels/inject_v67.py").read_text(encoding="utf-8")
    phy = (GPU_DIR / "kernels/phy_v67.py").read_text(encoding="utf-8")
    assert "_select_routable_slot_v67" in events
    # v75 MAC retries are validated at runtime via SimDimsV67.
    # Checking the obsolete compile-time spelling is a brittle false failure.
    assert "if ret >= dims.max_retx:" in events
    assert "if completed == 0:" in events
    assert "if int(s.rx_lock[e, tx]) >= 0:" in mac
    assert "cw = wp.min(1023, ((dims.cw_min + 1) << ret_clamped) - 1)" in mac
    assert "dims.cw_inclusive_upper" in mac
    assert "wp.uint32(org)" in inject and "wp.uint32(s.ledger_neg" in inject
    assert "pending_valid[e, u, p_idx]" in inject.split("def compute_dirty_dest_v67", 1)[1].split("def resolve_compact3_v67", 1)[0]
    assert "gain[e, j, i] =" not in phy.split("def compute_channel_gain_v67(", 1)[1]
    # Packet gate behavior is executed on shared CPU/CUDA kernels by
    # test_v72_packet_gate.py; source spelling is not delivery evidence.


def test_v71_normalize_sensing_inputs_accepts_bool_and_np_integer():
    c, d, t = normalize_sensing_inputs(
        np.array([[10, -1]], dtype=np.int32),
        np.array([[True, False]], dtype=bool),
        np.int64(7),
        envs=1, uavs=2, cells=400,
    )
    assert c.dtype == np.int32 and d.tolist() == [[1, 0]] and t == 7
    with pytest.raises(ValueError):
        normalize_sensing_inputs(np.array([[10, -1]]), np.array([[True, False]]), True, envs=1, uavs=2, cells=400)


def test_v71_memory_report_rejects_15_854gib_on_16gib_t4():
    rep = memory_inventory(env_counts=(2048, 4096))
    by_key = {(p["num_uavs"], p["grid_w"], p["num_envs"]): p for p in rep["profiles"]}
    assert by_key[(12, 40, 2048)]["single_t4_16gib_network_only_possible"] is False
    assert by_key[(12, 40, 2048)]["fits_t4_with_3gib_reserve"] is False
    assert by_key[(6, 20, 2048)]["fits_t4_with_3gib_reserve"] is True


def test_v71_aggregate_cca_interference_marks_channel_busy():
    net = SevenRadioCsmaCaNetwork(num_uavs=3, tx_power_w=0.1, noise_power_w=4.39e-13, cca_threshold_dbm=-82.0)
    net.chan_gain[:] = 0.0
    # Each single TX produces 0.6 * cca_th_w (< cca_th_w), but 2 simultaneous TXs exceed cca_th_w.
    g = (0.6 * net.cca_th_w) / 0.1
    net.chan_gain[0, 2] = g
    net.chan_gain[1, 2] = g
    net.radios[0].mac_state = ST_TX_DATA
    assert net._is_channel_busy_at(2) is False
    net.radios[1].mac_state = ST_TX_DATA
    assert net._is_channel_busy_at(2) is True


def test_v71_hello_revokes_disappeared_multihop_route():
    net = SevenRadioCsmaCaNetwork(num_uavs=3, tx_power_w=0.1)
    net.episode_id = 1
    net.sim_time_us = 100_000
    pkt1 = Packet(pkt_id=1, episode_id=1, pkt_type=PKT_HELLO_DV, app_src=1, app_dst=-1,
                  app_seq=1, create_us=100_000, expire_us=500_000, dv_table={2: (1, (1 << 1) | (1 << 2))})
    net._handle_rx_frame(rx=0, prev_hop=1, pkt=pkt1)
    assert 2 in net.routes[0] and net.routes[0][2][0] == 1
    net.sim_time_us = 200_000
    pkt2 = Packet(pkt_id=2, episode_id=1, pkt_type=PKT_HELLO_DV, app_src=1, app_dst=-1,
                  app_seq=2, create_us=200_000, expire_us=600_000, dv_table={})
    net._handle_rx_frame(rx=0, prev_hop=1, pkt=pkt2)
    assert 2 not in net.routes[0]


def test_v71_queue_coalescing_prevents_duplicate_flood_and_origin_fairness():
    net = SevenRadioCsmaCaNetwork(num_uavs=4, queue_cap=16, enable_data_mule_forwarding=True, aoi_ewma_end_to_end=True, coalesce_duplicate_queue=True)
    net.record_local_observation(0, 5, True, 1)
    entries = net.get_dirty_entries_non_oracle(0, net.gcs_id)
    pid1 = net.enqueue_app_data(0, net.gcs_id, entries)
    for _ in range(50):
        pid_n = net.enqueue_app_data(0, net.gcs_id, entries)
        assert pid_n == pid1
    assert len(net.radios[0].queue) == 1
    assert net.stats["drop_qfull"] == 0
    # Relay origin round-robin fairness
    o1 = net._origin_order(0, net.gcs_id, advance=True)[1]
    o2 = net._origin_order(0, net.gcs_id, advance=True)[1]
    assert o1 != o2


def test_v71_metrics_fp_single_penalty_ewma_nonnegative_and_decentralized_peer():
    cfg = EnvConfigV54(
        num_uavs=4, grid_w=20, map_size_m=1000.0, num_targets=15,
        comm_action_mode="compact_3", aoi_ewma_end_to_end=True,
        enable_fp_unlatch=True, reward_scale_mode="per_target_norm",
        continuous_aoi_penalty_w=0.1, compact3_decentralized_peer_select=True,
    )
    env = ZhangXue7NodePacketEnv(cfg=cfg, seed=7)
    env.reset(seed=7)
    # Remove all routes from UAV 0 -> decentralized SHARE_PEERS must return -1
    env.net.routes[0].clear()
    env.net.record_local_observation(0, 10, True, 1)
    assert env._resolve_comm_dest(0, 2) == -1
    obs, state, mask, reward, term, trunc, info = env.step(np.zeros((4, 2)), np.zeros(4, dtype=np.int64))
    assert "precision" in info and "recall" in info and "f1" in info and "false_confirm_rate" in info
    assert info["r_loc_team"] >= 0.0
    assert info["r_continuous_aoi"] <= 0.0


def test_v71_safety_shield_brakes_within_1m_of_building_and_decode_tx_power_scalable():
    # UAV at 0.5m outside building margin moving toward building must be braked
    buildings = [(200.0, 200.0, 20.0, 70.0)]
    pos = np.array([[200.0 - 22.5, 200.0]], dtype=np.float64)
    vel = np.array([[10.0, 0.0]], dtype=np.float64)
    cmd = np.array([[15.0, 0.0]], dtype=np.float64)
    v_safe, _ = kinematic_safety_shield(pos, vel, cmd, np.array([True]), buildings, 1000.0, 1.0, 15.0, 5.0, 50.0)
    assert v_safe[0, 0] < vel[0, 0]
    # decode_tx_power_w works for N=4 and N=8 (not hardcoded to 6)
    p4 = decode_tx_power_w(np.zeros((4, 3)))
    p8 = decode_tx_power_w(np.ones((8, 3)))
    assert p4.shape == (4,) and p8.shape == (8,)
