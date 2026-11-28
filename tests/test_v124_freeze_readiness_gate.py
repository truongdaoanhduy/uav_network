"""C81/C86 prevent automatic freezing based on a PASS pytest/CUDA report."""
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
MOD=ROOT/'scripts/maintenance/v124_freeze_readiness_gate.py'
spec=importlib.util.spec_from_file_location('v124_freeze',MOD)
v=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def test_no_gpu_test_result_auto_freezes():
    out=v.assess({'pytest':{'status':'PASS'},'cuda':{'status':'PASS'}})
    assert out['scenario_frozen'] is False
    assert out['phase2_baseline_started'] is False
    assert out['status']=='BLOCKED_NOT_FROZEN'
    assert len(out['outstanding'])==8
    assert out['yaml_mismatch']==[]


def test_even_positive_manual_claim_requires_review():
    out=v.assess({k:{'independent_review_accepted':True} for k in v.BLOCKERS})
    assert out['status']=='REVIEW_REQUIRED_NOT_FROZEN'
    assert out['can_request_manual_review']
    assert out['scenario_frozen'] is False


def test_swapped_yaml_blocks_everything(tmp_path):
    p=tmp_path/'wrong.yml';p.write_text('foo: bar\n')
    out=v.assess(yaml_paths={'base':p,'rf':v.YAML_RF})
    assert out['yaml_mismatch']==['base']
    assert out['scenario_frozen'] is False
