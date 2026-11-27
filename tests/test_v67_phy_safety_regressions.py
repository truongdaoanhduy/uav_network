"""Cross-layer PHY and per-UAV safety regressions against upstream UavNetSim."""
from __future__ import annotations

import numpy as np
import pytest

from zhang_xue.communication.common import Packet, PKT_HELLO_DV, INF_TIME_US
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.environment.configuration import make_v65_datamule_config
from zhang_xue.environment.core import ZhangXue7NodePacketEnv


def _run_hello_with_signal_w(signal_w: float) -> tuple[bool, bool]:
    noise_w = 4.38957709293155e-13
    net = SevenRadioCsmaCaNetwork(
        num_uavs=2, tx_power_w=0.1, noise_power_w=noise_w,
        cca_threshold_dbm=-82.0, seed=73,
    )
    net.reset(np.array([[150.,150.,50.], [210.,150.,50.], [50.,50.,15.]]), episode_id=1)
    net.chan_gain.fill(0)
    net.chan_gain[0,1] = net.chan_gain[1,0] = signal_w / 0.1
    net.radios[0].active_pkt = Packet(
        pkt_id=1, episode_id=1, pkt_type=PKT_HELLO_DV,
        app_src=0, app_dst=-1, app_seq=0,
        create_us=0, expire_us=INF_TIME_US,
        dv_table={},
    )
    # CCA determines whether the channel is sensed busy, not whether PHY can decode.
    cca_busy = net._is_channel_busy_at(1)
    net._start_tx(0, is_ack=False)
    rx_locked = net.radios[1].rx_locked_tx == 0
    net.sim_time_us = net.radios[0].state_end_us
    net._finish_tx(0)
    return (rx_locked and 0 in net.routes[1]), net._is_channel_busy_at(1)


def test_packet_above_sinr_but_below_cca_must_be_decodable():
    noise_w = 4.38957709293155e-13
    signal_w = 8.0 * noise_w  # 9 dB SNR; received -84.6 dBm, below -82 dBm CCA
    assert signal_w < 10**((-82.0-30)/10)
    delivered, _busy_after = _run_hello_with_signal_w(signal_w)
    assert delivered


def test_packet_below_sinr_must_not_be_decodable():
    noise_w = 4.38957709293155e-13
    delivered, _ = _run_hello_with_signal_w(noise_w)
    assert not delivered


def test_other_uav_can_sense_during_local_safety_violation(monkeypatch):
    import zhang_xue.environment.transition as transitions

    cfg = make_v65_datamule_config(
        num_uavs=2, num_targets=2, enable_3d_sensing_occlusion=False,
        safety_failure_terminate=False,
    )
    env = ZhangXue7NodePacketEnv(cfg, seed=54)
    env.reset(seed=54)
    # Test only the post-detection sensing policy; do not fake a safe flight path.
    monkeypatch.setattr(
        transitions, 'trajectory_safety_violations',
        lambda *args, **kwargs: {'boundary': [0], 'building': [], 'pairwise': []},
    )
    assert np.count_nonzero(env.net.ledger_gen_ts_s[1,1] >= 0) == 0
    _, _, _, _, _, _, info = env.step(np.zeros((2,2)), np.zeros(2, dtype=np.int32))
    assert info['safety_violation'] is True
    assert np.count_nonzero(env.net.ledger_gen_ts_s[1,1] >= 0) > 0
    assert np.count_nonzero(env.net.ledger_gen_ts_s[0,0] >= 0) == 0


def test_map_audit_does_not_conflate_cca_with_phy_decode():
    from scripts.maintenance.academic_scenario_audit import audit_direct_gcs_link
    r = audit_direct_gcs_link(2000.0, 40)
    assert r['no_direct_decodable_link_cells'] == 635
    assert r['no_direct_receivable_link_cells'] == 635
    assert r['no_direct_cca_sensed_link_cells'] == 1092
    assert r['no_direct_cca_sensed_link_cells'] > r['no_direct_receivable_link_cells']

@pytest.mark.parametrize('mode,expect_terminated', [
    ('any_depleted', True), ('all_depleted', False),
])
def test_battery_failure_is_an_explicit_team_mission_policy(mode, expect_terminated):
    cfg = make_v65_datamule_config(
        num_uavs=2, num_targets=2, safety_failure_terminate=False,
        battery_termination_mode=mode,
    )
    env = ZhangXue7NodePacketEnv(cfg, seed=41)
    env.reset(seed=41)
    env.battery_j[0] = 0.0
    _, _, _, _, terminated, _, _ = env.step(np.zeros((2,2)), np.zeros(2, dtype=np.int32))
    assert terminated is expect_terminated


def test_battery_failure_mode_rejects_silent_typo():
    with pytest.raises(ValueError, match='battery_termination_mode'):
        make_v65_datamule_config(battery_termination_mode='unknown')


def test_v67_peer_ablation_profiles_are_identical_except_switch():
    from dataclasses import asdict
    from pathlib import Path
    from zhang_xue.configuration.loader import load_scenario

    configs = Path(__file__).resolve().parents[1] / 'research/zhang_xue/configs'
    for size in ('1km', '2km'):
        on = load_scenario(configs / f'scenario_v67_datamule_{size}.yaml')
        off = load_scenario(configs / f'scenario_v67_no_peer_{size}.yaml')
        d1, d0 = asdict(on), asdict(off)
        assert d1.pop('enable_peer_dissemination') is True
        assert d0.pop('enable_peer_dissemination') is False
        assert d1 == d0
        assert on.battery_termination_mode == off.battery_termination_mode == 'all_depleted'


def test_v67_cpu_gate_is_explicit_in_setup():
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run(['bash', 'setup.sh', '--cpu-gate-v67', '--dry-run'],
                       cwd=root, capture_output=True, text=True, check=True)
    assert 'cpu_gate_v67=true' in p.stdout
