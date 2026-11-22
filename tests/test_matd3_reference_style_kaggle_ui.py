"""Reference-layout notebook: three historical cells but no oracle model source."""
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import lzma
import io
import json
import re
import sys
import tarfile
import types
from pathlib import Path
from unittest import mock

import nbformat

from uav_network.configuration import load_config
from uav_network.cloud.source_bundle import build_source_archive
from scripts.build_matd3_reference_style_kaggle_ui import (
    REFERENCE_NOTEBOOK, build,
)


def test_kaggle_ui_has_original_three_cell_layout_and_unique_source():
    if not REFERENCE_NOTEBOOK.is_file():
        import pytest
        pytest.skip('Reference notebook5051433b29.ipynb is not present on this host')
    root=Path(__file__).resolve().parents[1]
    original=nbformat.read(REFERENCE_NOTEBOOK, as_version=4)
    result=build(REFERENCE_NOTEBOOK,root)
    assert len(result.cells)==len(original.cells)==3
    assert [x.cell_type for x in result.cells]==[x.cell_type for x in original.cells]==[
        'markdown','code','code']
    assert 'UPDATES_PER_VECTOR_STEP = 4' in result.cells[1].source
    assert 'VALIDATION_EPISODES = 100' in result.cells[1].source
    assert 'CHECKPOINTS_PER_GPU_BATCH = 2' in result.cells[1].source
    assert 'UserSecretsClient' in result.cells[1].source
    assert 'ENABLE_HF_FORENSICS = True' in result.cells[1].source
    assert 'UAV_DIAGNOSTIC_STORE' in result.cells[2].source
    assert 'UAV_DIAGNOSTIC_HF_REPO' in result.cells[2].source
    assert 'NOORACLE-V8' in result.cells[0].source
    # Kaggle Save Version enforces < 1,000,000 bytes of kernel source.
    # Verify *both* flattened Python and the serialized notebook file size.
    assert sum(len(c.source.encode('utf-8')) for c in result.cells if c.cell_type == 'code') < 1_000_000
    assert len(nbformat.writes(result).encode('utf-8')) < 1_000_000
    code=result.cells[2].source
    ast.parse(result.cells[1].source)
    ast.parse(code)
    source_match=re.search(r'payload = base64\.b85decode\("([A-Za-z0-9!#$%&()*+\-;<=>?@^_`{|}~]+)"\)',code)
    assert source_match, 'Missing executable source archive'
    blob=base64.b85decode(source_match.group(1))
    assert hashlib.sha256(blob).hexdigest()==result.metadata['source_sha256']
    assert blob.startswith(b'\xfd7zXZ\x00'), 'expected source to be compact xz'
    # This must remain the *exact same set of original source bytes* as the
    # repository GZIP bundle. A size reduction may not silently drop tests,
    # pinned networking code, or any of the CUDA preflight modules.
    assert lzma.decompress(blob) == gzip.decompress(build_source_archive(root))
    assert 'tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz")' in code
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:xz') as files:
        for name in ('uav_network/algorithms/matd3.py',
                     'uav_network/environment/gpu/bayesian.py',
                     'uav_network/communication/gpu/mac.py',
                     'uav_network/experiments/training.py'):
            assert files.extractfile(name).read() == (root/name).read_bytes()


def test_reference_style_ui_controls_drive_actual_nooracle_config():
    if not REFERENCE_NOTEBOOK.is_file():
        import pytest
        pytest.skip('Reference notebook5051433b29.ipynb is not present on this host')
    root=Path(__file__).resolve().parents[1]
    book=build(REFERENCE_NOTEBOOK,root)
    shell=book.cells[1].source
    code=book.cells[2].source
    injected=code[:code.index('if SPEC["stage"] in {"visualize", "archive"}:')]
    fake=types.SimpleNamespace(UserSecretsClient=lambda: types.SimpleNamespace(get_secret=lambda n:'test-secret'))
    # This invokes only small preflight & spec-generation snippets. It does
    # NOT run Kaggle, install dependencies, or expose real user secrets.
    with mock.patch.dict(sys.modules,{'kaggle_secrets':fake}),\
         mock.patch.dict('os.environ',{'KAGGLE_KERNEL_REF':'example/matd3-v7'},clear=False),\
         mock.patch('torch.cuda.is_available',return_value=True),\
         mock.patch('torch.cuda.device_count',return_value=2),\
         mock.patch('torch.cuda.get_device_name',return_value='Tesla T4'):
        context={}
        exec(shell,context)
        exec(injected,context)
    spec=context['SPEC']
    cfg=load_config(spec['config'],spec['overrides'])
    assert spec['kernel_ref']=='example/matd3-v7'
    assert spec['runs']['matd3']['name'].startswith('MATD3-EVAL100-ACTORBN-PLAINCRITIC-NOORACLE-V8')
    assert cfg.runtime.num_envs==4096
    assert cfg.matd3.updates_per_step==4
    assert cfg.matd3.batch_size==8192
    assert cfg.matd3.hidden_sizes==[256,512]
    assert cfg.matd3.actor_batch_norm is True
    assert cfg.matd3.critic_batch_norm is False
    assert cfg.sensor.report_cell_hypotheses is True
    assert cfg.runtime.episodes==20000
    assert cfg.tracking.periodic_evaluation_episodes==100
    assert spec['dense_eval_schedule']['final_test_scenarios']==4000
    assert spec['one_session_train_eval_test'] is True
    assert spec['require_full_episode_budget'] is True
