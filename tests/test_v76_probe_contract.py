"""v76 RED/GREEN: prove that CLI/probe consumes new v75 Warp parameters."""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from uav_network.algorithms.configuration.loader import load_learner_config
from uav_network.experiments.capacity_probe import run_probe
from uav_network.experiments.radio_config import radio_config_manifest
from zhang_xue.configuration.loader import load_scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / 'research/zhang_xue/configs/scenario_v67_datamule_1km.yaml'
TARGET = ROOT / 'configs/learners/e4096_b8192.yaml'


def test_opt_in_queue_parity_fail_closed():
    s = load_scenario(SCENARIO)
    import pytest
    with pytest.raises(ValueError, match='queue parity'):
        radio_config_manifest(s, warp_q_cap=32, require_queue_parity=True)
    m = radio_config_manifest(s, warp_q_cap=64, require_queue_parity=True)
    assert m['comparability']['queue_capacities_match'] is True
    assert m['comparability']['queue_parity_required'] is True


def test_probe_passes_dynamic_mac_and_qcap_to_warp_constructor(monkeypatch, tmp_path):
    """Inspect the real code path even without a local Warp/CUDA runtime."""
    import uav_network.experiments.capacity_probe as probe
    sent = {}
    class SentinelNetwork:
        def __init__(self, *args, **kwargs):
            sent.update(kwargs)
            raise RuntimeError('V76 constructor sentinel after capturing validated kwargs')
    monkeypatch.setattr(probe, 'WarpCompressedRadioNetworkV67', SentinelNetwork)
    cfg = load_learner_config(TARGET)
    r = run_probe(cfg, scenario_path=SCENARIO, output_dir=tmp_path,
                  mode='functional', algorithm='matd3', network_device='cpu', learner_device='cpu',
                  functional_envs=2, functional_batch=4, functional_hidden=32,
                  scenario_overrides=('mac_difs_us=70',),
                  warp_q_cap=64, allow_dynamic_mac_timing=True, require_queue_parity=True)
    assert r['status'] == 'FAIL' and r['failure_phase'] == 'warp_radio_allocate_reset'
    assert 'constructor sentinel' in r['error']
    assert sent['q_cap'] == 64
    assert sent['protocol'].difs_us == 70
    assert sent['protocol'].max_retries > 0
    assert r['radio_config']['warp_mac_effective']['mac_difs_us'] == 70
    assert r['radio_config']['comparability']['queue_capacities_match'] is True
    assert r['radio_config']['dynamic_mac_timing_enabled'] is True


def test_probe_mismatched_queue_rejected_before_warp_allocation(monkeypatch, tmp_path):
    import uav_network.experiments.capacity_probe as probe
    class ForbiddenNetwork:
        def __init__(self,*a,**kw):
            raise AssertionError('must never allocate when queue differs')
    monkeypatch.setattr(probe,'WarpCompressedRadioNetworkV67',ForbiddenNetwork)
    cfg=load_learner_config(TARGET)
    r=run_probe(cfg,scenario_path=SCENARIO,output_dir=tmp_path,
                mode='functional',algorithm='masac',network_device='cpu',learner_device='cpu',
                functional_envs=2,functional_batch=4,functional_hidden=32,
                warp_q_cap=32,require_queue_parity=True)
    assert r['status']=='FAIL'
    assert r['failure_phase']=='scenario_and_real_cpu_fixture'
    assert 'queue parity' in r['error']


def test_kaggle_cli_exposes_opt_in_flags():
    out = subprocess.check_output([sys.executable, str(ROOT / 'scripts/gpu/run_capacity_probe.py'), '--help'], text=True)
    for flag in ('--warp-q-cap', '--allow-dynamic-mac-timing', '--require-queue-parity'):
        assert flag in out
