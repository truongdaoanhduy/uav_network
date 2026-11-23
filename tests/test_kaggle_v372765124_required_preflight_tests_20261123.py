"""Kaggle 372765124: Runtime source must retain required unittest files.

Actual trace: `python -m unittest discover -s tests -p test_cuda.py -v`
returned status 5 (= zero discovered tests). This was caused by a source size
optimization that discarded the entire tests/ directory even though the source
runtime cloud.remote RUNS three test patterns as a mandatory GPU preflight.
"""
import ast
import base64
import io
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 import build_notebook

REQUIRED={
 'test_cuda.py','test_cuda_scenario_v2.py','test_training_audit_regressions.py',
 'test_benchmark_devices.py','test_performance.py','test_vector.py',
 'test_vectorized_critics.py','test_xyz_radio.py','test_xyz_sensing.py',
 'test_gpu_packets.py',
}


def source_files_from_real_notebook():
    notebook=build_notebook()
    code=notebook.cells[2].source
    m=re.search(r'payload = base64\.b85decode\(("[^"]+")\)',code)
    assert m
    blob=base64.b85decode(ast.literal_eval(m.group(1)))
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as src:
        return {member.name:src.extractfile(member).read() for member in src}


def test_required_preflight_files_and_import_closure_are_embedded():
    files=source_files_from_real_notebook()
    for name in REQUIRED:
        assert 'tests/'+name in files, f'Kaggle mandatory unittest preflight will crash: missing tests/{name}'
    assert not any(k.startswith('tests/') and k.split('/')[-1] not in REQUIRED for k in files)
    assert 'configs/matd3-nooracle-posterior-only-map800-h256x256-orthogonal-compile-tx08-hf-fullflow-20261123.yaml' in files
    assert 'uav_network/cloud/remote.py' in files
    assert len(files)<340


@pytest.mark.parametrize('pattern', ['test_cuda.py','test_cuda_scenario_v2.py','test_training_audit_regressions.py'])
def test_embedded_archive_unittest_discovery_finds_real_tests_without_gpu(pattern,tmp_path):
    files=source_files_from_real_notebook()
    root=tmp_path/'source';root.mkdir()
    for name,body in files.items():
        f=root/name;f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes(body)
    p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],
        cwd=root,env={**os.environ,'PYTHONPATH':str(root),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        capture_output=True,text=True,timeout=60)
    assert p.returncode==0, f'Failed mandatory Kaggle preflight for {pattern}: '+p.stderr[-2400:]
    assert 'Ran 0 tests' not in p.stderr, 'Kaggle exit status 5: missing test module'
    assert 'Ran ' in p.stderr, p.stderr[-300:]


def test_exact_cloud_remote_calls_all_three_real_unittest_suites_from_embedded_source(tmp_path,monkeypatch):
    """Exercise actual remote.run_job test sequence, stop before GPU bench/train.

    Tests execute in extracted Kaggle source. CUDA-only classes are skipped on
    local CPU; transport/GPU are isolated fakes (not claimed CUDA verified).
    """
    import json
    import types
    import torch
    from scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 import make_spec
    files=source_files_from_real_notebook()
    project=tmp_path/'kaggle_source';project.mkdir()
    for name,content in files.items():
        p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    # Import the copied remote module in a local isolated subprocess. This
    # captures mismatched imports and scripts that unit-file checks miss.
    runner='''from pathlib import Path
import os,json,sys,subprocess
from uav_network.cloud import remote
from uav_network.configuration import load_config
spec=json.loads(Path("spec.json").read_text())
root=Path.cwd()
results=root/"cuda_preflights";results.mkdir()
original_path=remote.Path
remote.Path=lambda p: results if str(p)=="/kaggle/working" else original_path(p)
remote.hardware_info=lambda: {"gpu":"CPU_TEST_ONLY"}
remote.torch.cuda.is_available=lambda:True
original_run=subprocess.run
patterns=[]
class STOP(Exception):pass
def wrapped(args,**kwargs):
    if len(args)>4 and args[1:4]==['-m','unittest','discover']:
        patterns.append(args[args.index('-p')+1])
        return original_run(args,**kwargs)
    if 'uav_network.cloud.execution_check' in args:
        assert patterns==['test_cuda.py','test_cuda_scenario_v2.py','test_training_audit_regressions.py'],patterns
        raise STOP
    raise AssertionError('Unexpected subprocess before GPU execution_check:'+repr(args[:8]))
subprocess.run=wrapped
try:remote.run_job(spec)
except STOP:pass
else:raise AssertionError('Expected GPU execution_check handoff')
assert len(patterns)==3
for name in ('cuda-tests.log','scenario-v2-cuda-tests.log','mission-regressions.log'):
    s=(results/name).read_text()
    assert 'Ran 0 tests' not in s,s[-500:]
    assert 'Ran ' in s,s[-500:]
print('CLOUD_REMOTE_PREFLIGHT_ALL_THREE_OK',patterns,flush=True)
'''
    (project/'spec.json').write_text(json.dumps(make_spec('unit-test-source-commit')))
    p=subprocess.run([sys.executable,'-c',runner],cwd=project,
        env={**os.environ,'PYTHONPATH':str(project),'TORCHINDUCTOR_COMPILE_THREADS':'1',
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        capture_output=True,text=True,timeout=85)
    assert p.returncode==0,(p.stdout[-1400:],p.stderr[-2400:])
    assert 'CLOUD_REMOTE_PREFLIGHT_ALL_THREE_OK' in p.stdout
