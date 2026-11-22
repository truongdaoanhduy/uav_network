"""Reproduce actual Kaggle cell [2] extraction, not a builder-only assertion."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

import pytest
from scripts.build_nooracle_gpu_train_speed_ab_20261123 import build_notebook, FILES


def test_exact_kaggle_bootstrap_extracts_and_overlays_in_clean_destination(tmp_path):
    nb=build_notebook()
    source=nb.cells[2].source
    # Use exactly the generated Kaggle execution cell without its venv/pip
    # section (CUDA/network dependency unavailable on the local CPU).
    anchor='venv=root/".venv"'
    assert source.count(anchor)==1
    preflight=source[:source.index(anchor)]
    assert 'Expected pinned TAR.XZ source archive' in preflight
    assert 'with tarfile.open(fileobj=io.BytesIO(blob),mode="r:xz") as tf:' in preflight
    assert 'lzma.decompress(blob)),mode="r:xz"' not in preflight
    # The notebook refuses an existing directory; simulate /kaggle/working in tmp_path.
    assert source.count('/kaggle/working/ct095_matd3_training_ab')==1
    preflight=preflight.replace('/kaggle/working/ct095_matd3_training_ab',str(tmp_path/'kaggle'))
    ast.parse(preflight)
    scope={}
    exec(compile(preflight,'kaggle-bootstrap-cell-2', 'exec'), scope)
    assert scope["sha"](scope["blob"])==nb.metadata["source_sha256"]
    assert scope["old_root"].is_dir() and scope["new_root"].is_dir()
    assert len(list(scope["old_root"].rglob("*")))>500
    for rel in FILES:
        assert hashlib.sha256((scope["old_root"]/rel).read_bytes()).hexdigest()==nb.metadata["baseline_file_sha256"][rel]
        assert hashlib.sha256((scope["new_root"]/rel).read_bytes()).hexdigest()==nb.metadata["candidate_file_sha256"][rel]
    for repo in (scope["old_root"],scope["new_root"]):
        worker=repo/'scripts/trial_real_matd3_train_gpu_worker_20261123.py'
        assert worker.is_file()
        assert hashlib.sha256(worker.read_bytes()).hexdigest()==nb.metadata["worker_sha256"]
        assert (repo/'configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml').exists()
    # No external operations required for archive extraction.


def test_exact_extracted_archives_can_complete_bounded_real_matd3_cpu_training(tmp_path):
    """The source *inside Kaggle ipynb* must run end-to-end, not just unpack."""
    import os,subprocess,sys
    n=build_notebook()
    src=n.cells[2].source
    preflight=src[:src.index('venv=root/".venv"')]
    preflight=preflight.replace('/kaggle/working/ct095_matd3_training_ab',str(tmp_path/'kaggle'))
    ns={};exec(compile(preflight,'kaggle-bootstrap-archive', 'exec'),ns)
    data={}
    for name,folder in [('baseline',ns['old_root']),('candidate',ns['new_root'])]:
        env=os.environ.copy()
        env.update(PYTHONPATH=str(folder),WANDB_MODE='disabled',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[sys.executable,'-m','scripts.trial_real_matd3_train_gpu_worker_20261123',
                 '--mode',name,'--output',str(tmp_path/(name+'_training')),
                 '--device','cpu','--steps','24','--envs','8','--batch','16',
                 '--updates','1','--horizon','12','--learning-starts','16']
        process=subprocess.run(command,cwd=folder,env=env,text=True,capture_output=True,timeout=90)
        assert process.returncode==0,(name,process.stdout[-1000:],process.stderr[-1300:])
        results=[json.loads(line.split(' ',1)[1]) for line in process.stdout.splitlines()
                 if line.startswith('MATD3_REAL_TRAIN_GPU_WORKER_JSON ')]
        assert len(results)==1
        data[name]=results[0]
        assert data[name]['training_completed_20k'] is False
        assert data[name]['stop_reason']=='env_step_budget'
        assert data[name]['total_environment_transitions']==192
        assert data[name]['actual_learner_updates']>0
    for key in ('science_sha256','model_sha256','episode_sha256','actual_learner_updates'):
        assert data['baseline'][key]==data['candidate'][key],key


def test_saved_notebook_is_identical_to_fixed_builder_and_not_stale():
    from scripts.build_nooracle_gpu_train_speed_ab_20261123 import OUT
    import nbformat
    stored=nbformat.read(OUT,as_version=4)
    built=build_notebook()
    assert stored.cells[2].source==built.cells[2].source
    assert 'with tarfile.open(fileobj=io.BytesIO(blob),mode="r:xz") as tf:' in stored.cells[2].source
    assert 'lzma.decompress(blob)),mode="r:xz"' not in stored.cells[2].source
    assert stored.metadata['source_sha256']==built.metadata['source_sha256']
