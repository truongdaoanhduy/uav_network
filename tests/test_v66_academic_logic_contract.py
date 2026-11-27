"""RED/GREEN checks for v6.6 independent scientific and execution audit.

This test suite verifies real I/O behavior, not string-presence claims.
"""
from __future__ import annotations

import numpy as np
import pytest

from zhang_xue.communication.common import CANONICAL_BUILDINGS_5, scale_canonical_geometry
from zhang_xue.environment.configuration import make_v65_datamule_config
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.gpu.runtime_v65 import WarpCompressedRadioNetwork, resolve_compact3_actions_np
from scripts.maintenance.q1_gpu_preflight import q1_gpu_readiness


def test_compact3_peer_choice_requires_receiver_specific_dirty_evidence():
    actions = np.array([[2, 2, 1]], dtype=np.int32)
    positions = np.array([[[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]]], dtype=np.float32)
    alive = np.array([[True, True, True]])
    dirty = np.zeros((1, 3, 4), dtype=bool)
    dirty[0, 0, 2] = True  # nearest peer 1 is up-to-date, only peer 2 is dirty
    dirty[0, 1, 0] = True
    dst = resolve_compact3_actions_np(actions, positions, alive, dirty_dest=dirty)
    assert dst.tolist() == [[2, 0, 3]]


def test_compact3_helper_masks_dead_source_and_radio_blackout():
    actions = np.array([[1, 2, 2]], dtype=np.int32)
    positions = np.array([[[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]]], dtype=np.float32)
    alive = np.array([[False, True, True]])
    dirty = np.ones((1, 3, 4), dtype=bool)
    can_tx = np.array([[True, False, True]], dtype=bool)
    dst = resolve_compact3_actions_np(actions, positions, alive, dirty_dest=dirty, can_tx=can_tx)
    assert dst.tolist() == [[-1, -1, 1]]


def test_2km_similar_geometry_scales_obstacle_radius_and_gcs_spawn():
    buildings, _, _ = scale_canonical_geometry(2000.0, 40, scale_obstacles=True)
    assert buildings[0][2] == CANONICAL_BUILDINGS_5[0][2] * 2
    cfg = make_v65_datamule_config(num_uavs=4, grid_w=40, map_size_m=2000.0)
    env = ZhangXue7NodePacketEnv(cfg, seed=79)
    env.reset(seed=79)
    assert env.positions[env.gcs_id, :2].tolist() == [100.0, 100.0]
    assert env.positions[0, :2].tolist() == [180.0, 220.0]


def test_gpu_readiness_marks_v65_buffer_allocation_as_non_executable():
    report = q1_gpu_readiness()
    assert report['v65_layout'] == WarpCompressedRadioNetwork.active_layout_id()
    assert report['v65_packet_step_implemented'] is False
    assert 'compressed_runtime_has_no_packet_step' in report['blockers']
    assert report['ready'] is False


def test_v65_gpu_budget_must_account_for_4d_data_mule_watermarks():
    from zhang_xue.gpu.runtime_v65 import v65_gpu_memory_budget
    from zhang_xue.gpu.layout_v62 import make_shapes_v63

    shapes = make_shapes_v63(6, 20)
    assert 'ack_wm_origin_pos' not in shapes
    assert 'ack_wm_origin_neg' not in shapes
    assert 'ack_wm_origin_ts' not in shapes
    budget = v65_gpu_memory_budget(num_envs_per_gpu=2048, num_uavs=6, grid_w=20)
    assert budget['datamule_ack_bytes_per_env'] == 806_400
    assert budget['datamule_ack_gib'] > 1.5
    assert budget['datamule_packet_fields_implemented'] is False


def test_reference_profile_can_match_uavnetsim_200_packet_queue_without_changing_main_baseline():
    from zhang_xue.configuration.loader import load_scenario
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    ref = load_scenario(root/'research/zhang_xue/configs/uavnetsim_reference.yaml')
    main = load_scenario(root/'research/zhang_xue/configs/scenario_7radio.yaml')
    assert ref.mac_queue_capacity == 200
    assert main.mac_queue_capacity == 64


def test_reproducible_map_direct_link_audit_counts_union_and_overlap():
    from scripts.maintenance.academic_scenario_audit import audit_direct_gcs_link
    r = audit_direct_gcs_link(1000.0, 20)
    assert r['cells'] == 400
    assert r['blackout_cells'] == 50
    assert r['nlos_cells'] == 137
    assert r['blackout_and_nlos_cells'] == 26
    assert r['no_direct_decodable_link_cells'] == 161
    assert r['no_direct_decodable_link_pct'] == pytest.approx(40.25)


def test_direct_link_audit_separates_snr_decode_from_cca_rx_lock():
    from scripts.maintenance.academic_scenario_audit import audit_direct_gcs_link
    r2 = audit_direct_gcs_link(2000.0, 40)
    r3 = audit_direct_gcs_link(3000.0, 60)
    assert r2['no_direct_decodable_link_cells'] == 635
    assert r2['no_direct_receivable_link_cells'] == 635
    assert r2['no_direct_cca_sensed_link_cells'] == 1092
    assert r3['no_direct_receivable_link_cells'] == 2079
    assert r3['no_direct_cca_sensed_link_cells'] == 2995
    assert r2['no_direct_receivable_link_pct'] == pytest.approx(39.688)


def test_two_contact_data_mule_delivers_cross_origin_but_no_mule_cannot():
    """Real packet/ACK behavior under intermittent contacts, not a flight-dynamics test.

    The topology change is an explicit radio-level discontinuity so this test
    cannot be presented as a physically flown trajectory or an RL improvement.
    """
    from zhang_xue.communication.network import SevenRadioCsmaCaNetwork

    first = np.array([[325., 275., 50.], [330., 330., 50.], [50., 50., 15.]])
    second = first.copy()
    second[1] = [325., 175., 50.]
    for enabled in (True, False):
        net = SevenRadioCsmaCaNetwork(
            tx_power_w=0.1, noise_power_w=4.38957709293155e-13,
            cca_threshold_dbm=-82.0, aoi_ewma_end_to_end=True,
            enable_data_mule_forwarding=enabled, num_uavs=2, seed=46,
        )
        net.reset(first, episode_id=9)
        assert net.run_until_us(600_000)
        assert net.routes[0].get(2) is None and net.routes[1].get(2) is None
        net.record_local_observation(0, 42, True, 1)
        assert net.enqueue_app_data(0, 1, net.get_dirty_entries_non_oracle(0, 1))
        assert net.run_until_us(1_600_000)
        assert net.ledger_pos[1, 0, 42] == 256
        assert net.ledger_pos[2, 0, 42] == 0
        net.update_positions(second)
        assert net.run_until_us(2_300_000)
        forwarded = net.get_dirty_entries_non_oracle(1, 2)
        assert len(forwarded) == (1 if enabled else 0)
        net.enqueue_app_data(1, 2, forwarded)
        assert net.run_until_us(3_300_000)
        assert int(net.ledger_pos[2, 0, 42]) == (256 if enabled else 0)
        assert net.stats['data_delivered_gcs'] == (1 if enabled else 0)


def test_v66_peer_ablation_profiles_match_everything_except_peer_switch():
    from dataclasses import asdict
    from pathlib import Path
    from zhang_xue.configuration.loader import load_scenario
    root = Path(__file__).resolve().parents[1] / 'research/zhang_xue/configs'
    for suffix in ('1km', '2km'):
        peer = load_scenario(root / f'scenario_v66_datamule_{suffix}.yaml')
        no_peer = load_scenario(root / f'scenario_v66_no_peer_{suffix}.yaml')
        left, right = asdict(peer), asdict(no_peer)
        assert left.pop('enable_peer_dissemination') is True
        assert right.pop('enable_peer_dissemination') is False
        assert left == right
        assert peer.enable_data_mule_forwarding and no_peer.enable_data_mule_forwarding
        assert peer.w_p2p == no_peer.w_p2p == 0
        if suffix == '2km':
            assert (peer.map_size_m, peer.grid_w, peer.num_targets) == (2000, 40, 60)


def test_setup_has_explicit_cpu_datamule_gate_flag():
    from pathlib import Path
    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash', 'setup.sh', '--cpu-gate-datamule', '--dry-run'],
                            cwd=root, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'cpu_gate_datamule=true' in result.stdout


def test_academic_scenario_audit_standalone_cli_works_in_clean_zip():
    from pathlib import Path
    import subprocess
    import sys
    import os
    root = Path(__file__).resolve().parents[1]
    isolated_env = dict(os.environ)
    isolated_env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, 'scripts/maintenance/academic_scenario_audit.py',
                             '--sizes-m', '1000'], cwd=root, text=True,
                            capture_output=True, timeout=15, env=isolated_env)
    assert result.returncode == 0, result.stderr
    assert 'no_direct_receivable_link_pct' in result.stdout
