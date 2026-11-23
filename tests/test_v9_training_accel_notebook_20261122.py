import ast
import base64
import hashlib
import nbformat
import re

from scripts.build_matd3_v9_training_accel_kaggle_ui_20261122 import build_notebook


def test_notebook_keeps_full_training_and_gpu_eval_science():
    n=build_notebook()
    assert [c.cell_type for c in n.cells]==['markdown','code','code']
    for cell in n.cells[1:]:ast.parse(cell.source)
    assert len(nbformat.writes(n).encode())<1_000_000
    c1=n.cells[1].source;c2=n.cells[2].source
    assert 'TRAINACCEL' in c1+c2
    assert 'UPDATES_PER_VECTOR_STEP' in c1
    assert 'VALIDATION_EPISODES = 100' in c1
    assert 'MAX_WARP_EVAL_WORLDS_PER_GPU = 256' in c1
    assert 'CHECKPOINTS_PER_GPU_BATCH = 2' in c1
    assert 'runtime.episodes=20000' in c2
    assert 'UAV_FAST_TRAINING' in c2
    assert 'one_session' in c2
    assert '4000' in c1+c2
    assert 'UAV_DEFERRED_EXPORT_HF' in c2
    assert 'scripts/requirements-network.txt' in c2
    assert 'NOORACLE-V9-TRAINACCEL' in n.metadata['performance_variant']
    assert n.metadata['baseline_training_run']=='e079eace'
    m=re.search(r'payload\s*=\s*base64\.b85decode\("([^"\n]+)"\)',c2)
    assert m
    package=base64.b85decode(m.group(1))
    assert hashlib.sha256(package).hexdigest()==n.metadata['source_sha256']
