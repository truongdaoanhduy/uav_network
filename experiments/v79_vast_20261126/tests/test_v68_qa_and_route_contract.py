"""v6.8: evidence reproducibility and scalable CPU path-vector limits."""
from __future__ import annotations

import sys
from pathlib import Path

from zhang_xue.communication.common import PKT_HELLO_DV, Packet
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork


def _hello_advertising_gcs() -> Packet:
    # A valid six-hop advertisement from node 1 towards GCS=8.
    return Packet(
        pkt_id=99, episode_id=0, pkt_type=PKT_HELLO_DV,
        app_src=1, app_dst=-1, app_seq=0, create_us=0,
        expire_us=1_000_000,
        dv_table={8: (6, sum(1 << n for n in (1, 2, 3, 4, 5, 8)))},
    )


def test_eight_uav_network_can_learn_seven_hop_loop_free_route():
    net = SevenRadioCsmaCaNetwork(
        num_uavs=8, buildings=[], blackout_centers=[], grid_w=20,
    )
    net._handle_rx_frame(0, 1, _hello_advertising_gcs())
    assert net.routes[0][8][1] == 7


def test_seven_radio_base_preserves_six_hop_cap():
    net = SevenRadioCsmaCaNetwork(
        num_uavs=6, buildings=[], blackout_centers=[], grid_w=20,
    )
    pkt = _hello_advertising_gcs()
    pkt.dv_table = {6: (6, sum(1 << n for n in (1, 2, 3, 4, 5, 6)))}
    net._handle_rx_frame(0, 1, pkt)
    assert 6 not in net.routes[0]


def test_qa_logger_captures_actual_subprocess_output(tmp_path: Path):
    from scripts.maintenance.run_qa_bundle import run_gate

    result = run_gate(
        'unit_example', [sys.executable, '-c', 'print("qa-real-output")'],
        tmp_path, cwd=tmp_path,
    )
    assert result['status'] == 'PASS'
    assert result['return_code'] == 0
    assert (tmp_path / 'logs/unit_example.txt').read_text().strip() == 'qa-real-output'
    assert result['argv'][0] == sys.executable


def test_qa_logger_reports_failing_command_without_masking(tmp_path: Path):
    from scripts.maintenance.run_qa_bundle import run_gate

    result = run_gate(
        'nonzero_example', [sys.executable, '-c', 'import sys; sys.exit(7)'],
        tmp_path, cwd=tmp_path,
    )
    assert result['status'] == 'FAIL'
    assert result['return_code'] == 7
    assert (tmp_path / 'logs/nonzero_example.txt').exists()


def test_setup_cli_exposes_qa_command():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(['bash', str(root / 'setup.sh'), '--help'],
                          cwd=root, text=True, capture_output=True, check=True)
    assert '--qa' in proc.stdout


def test_uq88_packet_counts_cannot_silently_saturate_in_episode():
    import pytest
    from zhang_xue.environment.configuration import EnvConfigV54

    for eta in (0.0, 0.001):
        with pytest.raises(ValueError, match='UQ8.8'):
            EnvConfigV54(aoi_ewma_end_to_end=True, ds_aoi_eta=eta, safety_horizon_s=900)
    EnvConfigV54(aoi_ewma_end_to_end=True, ds_aoi_eta=0.02, safety_horizon_s=900)
    EnvConfigV54(aoi_ewma_end_to_end=True, ds_aoi_eta=0.0, safety_horizon_s=100)


def test_fractional_dt_is_rejected_until_mission_aoi_uses_real_seconds():
    import pytest
    from zhang_xue.environment.configuration import EnvConfigV54

    with pytest.raises(ValueError, match='dt_s=1'):
        EnvConfigV54(dt_s=0.5)
    EnvConfigV54(dt_s=1.0)
