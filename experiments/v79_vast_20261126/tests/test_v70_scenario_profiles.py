"""Ablation scenarios must be paired with identical physical/reward settings."""
from pathlib import Path
import tempfile

import pytest
import yaml

from scripts.maintenance.paired_scenario_gate import audit_pair, inventory

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'research/zhang_xue/configs'


@pytest.mark.parametrize('size', ['1km', '2km'])
def test_v67_data_mule_peer_ablation_matches_every_other_key(size):
    a = CFG / f'scenario_v67_datamule_{size}.yaml'
    b = CFG / f'scenario_v67_no_peer_{size}.yaml'
    result = audit_pair(a, b)
    assert result['status'] == 'MATCHED', result
    assert result['deliberately_changed_fields'] == ['enable_peer_dissemination']
    assert result['effective_battery_termination_mode'] == 'all_depleted'


def test_cross_version_termination_mismatch_must_be_exposed():
    a = CFG / 'scenario_v66_datamule_1km.yaml'
    b = CFG / 'scenario_v67_datamule_1km.yaml'
    result = audit_pair(a, b)
    assert result['status'] == 'CONFOUNDED'
    assert 'battery_termination_mode' in result['confounders']


def test_paired_gate_detects_mutated_reward_and_sensing_parameters(tmp_path):
    a = CFG / 'scenario_v67_datamule_1km.yaml'
    b = tmp_path / 'no_peer.yaml'
    config = yaml.safe_load((CFG / 'scenario_v67_no_peer_1km.yaml').read_text())
    config['confirm_th'] = .99
    b.write_text(yaml.safe_dump(config))
    result = audit_pair(a, b)
    assert result['status'] == 'CONFOUNDED'
    assert 'confirm_th' in result['confounders']


def test_inventory_explicitly_reports_map_density_and_termination():
    data = inventory(CFG)
    assert 'scenario_v67_datamule_1km.yaml' in data
    assert 'scenario_v67_datamule_2km.yaml' in data
    a, b = data['scenario_v67_datamule_1km.yaml'], data['scenario_v67_datamule_2km.yaml']
    assert a['target_density_per_km2'] == pytest.approx(b['target_density_per_km2'])
    assert a['target_density_per_km2'] == 15.0
    assert a['battery_termination_mode'] == b['battery_termination_mode'] == 'all_depleted'
    assert a['uavs_per_km2'] != b['uavs_per_km2']


def test_3km_stress_profiles_validate_and_are_paired():
    a = CFG / 'scenario_v70_datamule_3km.yaml'
    b = CFG / 'scenario_v70_no_peer_3km.yaml'
    result = audit_pair(a, b)
    assert result['status'] == 'MATCHED'
    inv = inventory(CFG)
    assert inv[a.name]['target_density_per_km2'] == pytest.approx(15.0)
    assert inv[a.name]['uavs_per_km2'] == pytest.approx(6/9)
