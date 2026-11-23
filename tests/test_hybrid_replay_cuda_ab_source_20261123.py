"""No GPU needed: verify CPU path of exact minibatch parity A/B."""
import json,subprocess,sys,os
from pathlib import Path


def test_b8192_schema_and_identical_replay_minibatches_cpu(monkeypatch):
    results=[]
    for mode in ('off','on'):
        p=subprocess.run([sys.executable,'-m','scripts.benchmark_hybrid_replay_cuda_ab_20261123',
                          '--device','cpu','--staging',mode,
                          '--batch','128','--capacity','512','--warmup','2','--samples','4'],
                          cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,
                          env={**os.environ,'OMP_NUM_THREADS':'1'},timeout=20)
        assert p.returncode==0,p.stderr[-1200:]
        rows=[json.loads(x.split(' ',1)[1]) for x in p.stdout.splitlines() if x.startswith('HYBRID_REPLAY_CUDA_AB_JSON ')]
        assert len(rows)==1
        results+=rows
    assert results[0]['all_sampled_tensor_sha256']==results[1]['all_sampled_tensor_sha256']
    assert results[0]['rng_state_sha256']==results[1]['rng_state_sha256']
    assert results[0]['seen_bitmap_sha256']==results[1]['seen_bitmap_sha256']
    assert results[0]['cpu_fraction']==results[1]['cpu_fraction']==.75
