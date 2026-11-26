from zhang_xue.configuration.loader import load_scenario
from scripts.maintenance.scenario_conflict_audit import analyze
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
S=BASE/'research/zhang_xue/configs/scenario_v67_datamule_1km.yaml'


def test_audit_all_critical_domains_and_noise_delta():
    report=analyze(load_scenario(S),warp_q_cap=64,num_envs=4096)
    assert report['classification']=='static_source_audit_not_simulator_parity'
    assert report['radio_soa_estimate']['bytes_per_env']==598629
    ids={x['id'] for x in report['findings']}
    assert {'MAP-01','MAP-02','DS-01','PHY-01','PHY-02','NET-02',
            'REW-01','REW-02','ENE-01','MET-01','GPU-01','GPU-02'} <= ids
    assert 'NET-01' not in ids
    assert '+19.60 dB' in next(x['finding'] for x in report['findings'] if x['id']=='PHY-01')


def test_audit_default_queue_mismatch_explicit():
    report=analyze(load_scenario(S))
    q=next(x for x in report['findings'] if x['id']=='NET-01')
    assert 'queue=64' in q['finding']
    assert '--warp-q-cap 64' in q['recommended_action']
