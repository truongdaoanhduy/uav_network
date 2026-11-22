"""The failed V7 policy is evaluated only under its pinned source, no retrain."""
import ast,base64,hashlib,io,json,lzma,tarfile
from pathlib import Path
import nbformat

from scripts.build_matd3_v7_recovery_kaggle_ui_20261122 import (
    build,V7_ROOT,COMMIT,RUN,EXPECTED_SHA,EVIDENCE_ROOT)


def test_v7_recovery_notebook_is_self_contained_safe_under_1mb_and_never_trains(tmp_path):
    import pytest
    if not V7_ROOT.is_dir() or not (EVIDENCE_ROOT/'job.json').is_file():
        pytest.skip('External archived Kaggle artifacts are local forensic fixtures')
    target=tmp_path/'v7_recovery.ipynb'
    meta=build(target)
    doc=nbformat.read(target,as_version=4)
    assert meta['bytes']<1000000 and meta['python_source_bytes']<1000000
    assert [cell.cell_type for cell in doc.cells]==['markdown','code','code']
    assert doc.metadata['v7_commit']==COMMIT
    assert doc.metadata['recover_run']==RUN
    code=doc.cells[2].source
    ast.parse(code)
    assert 'uav_network.experiments.one_session' in code
    assert 'subprocess.run([str(python), "-m", runner' not in code
    assert 'training.train(' not in code.replace('or training.train().','')
    assert EXPECTED_SHA in code
    assert 'test' in code.lower()
    assert 'source_version' not in code or 'source_version' in doc.metadata
    raw=base64.b85decode(code.split('payload = base64.b85decode("')[1].split('"')[0])
    assert hashlib.sha256(raw).hexdigest()==doc.metadata['source_sha256']
    source=lzma.decompress(raw)
    with tarfile.open(fileobj=io.BytesIO(source),mode='r:') as t:
        core=t.extractfile('uav_network/environment/observations.py').read().decode()
        assert 'return 7' in core
        assert 'return 8' not in core
        assert any(m.name.endswith('third_party/UavNetSim/simulator/simulator.py') for m in t)
    job=json.loads((EVIDENCE_ROOT/'job.json').read_text())
    assert job['runs']['matd3']['id']==RUN
    assert job['commit']==COMMIT
