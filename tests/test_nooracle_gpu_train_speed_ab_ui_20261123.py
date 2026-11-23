"""Kaggle notebook is a source-pinned bounded *training* A/B pilot."""
from __future__ import annotations
import ast,base64,hashlib,io,lzma,re,tarfile
from pathlib import Path
import nbformat
from scripts.build_nooracle_gpu_train_speed_ab_20261123 import build_notebook, FILES, WORKER, ROOT


def test_gpu_training_ab_notebook_has_full_science_and_hard_step_limit():
    nb=build_notebook()
    nbformat.validate(nb)
    assert len(nb.cells)==4
    assert len(nbformat.writes(nb).encode())<1_000_000
    meta=nb.metadata
    assert meta['training_ab_bounded'] and meta['full_20000_training'] is False
    assert (meta['envs'],meta['batch'],meta['updates'],meta['replay_size'])==(4096,8192,4,1000000)
    assert meta['replay_cpu_fraction']==.75
    assert (meta['max_steps'],meta['seed'],meta['pairs'],meta['pilot_vector_steps'])==(1500,44,2,64)
    for c in nb.cells[1:]:
        assert c.cell_type=='code';ast.parse(c.source)
    assert 'torch.cuda.device_count() >= 2' in nb.cells[1].source
    assert 'LEARNING_STARTS=131072' in nb.cells[1].source
    runner=nb.cells[3].source
    assert 'TRAIN_AB_START' in runner
    assert 'baseline' in runner and 'candidate' in runner
    assert 'model_weights_match' in runner and 'episode_records_match' in runner
    assert 'science' not in runner or 'scientific_config_match' in runner
    assert 'UAV_TRAIN_ENV_STEP_LIMIT' in WORKER.read_text()
    assert 'UAV_TRAINING_ONLY' in WORKER.read_text()
    assert 'UAV_DIRECT_GPU_COLLECTOR_COPY' in WORKER.read_text()
    assert 'stop_reason' in runner and 'env_step_budget' in runner
    assert 'one_session' not in runner
    assert 'WANDB_API_KEY' in runner and 'env.pop(key,None)' in runner


def test_gpu_train_source_bundle_candidate_overlays_are_byte_exact():
    n=build_notebook()
    source=n.cells[2].source
    match=re.search(r'blob=base64.b85decode\((\'[A-Za-z0-9!#$%&()*+,-./:;<=>?@^_`{|}~]+\')\)',source)
    assert match
    bundle=base64.b85decode(ast.literal_eval(match.group(1)))
    assert hashlib.sha256(bundle).hexdigest()==n.metadata['source_sha256']
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(bundle)),mode='r:') as tf:
        assert 'configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml' in tf.getnames()
        for file in FILES:
            ref=tf.extractfile(file).read()
            current=Path(ROOT,file).read_bytes()
            assert hashlib.sha256(ref).hexdigest()==n.metadata['baseline_file_sha256'][file]
            assert hashlib.sha256(current).hexdigest()==n.metadata['candidate_file_sha256'][file]
            assert ref != current
    assert hashlib.sha256(WORKER.read_bytes()).hexdigest()==n.metadata['worker_sha256']
