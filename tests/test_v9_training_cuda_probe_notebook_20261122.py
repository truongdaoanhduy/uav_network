"""Never benchmark fake kernel data or swap in the historical Oracle source."""
import ast
import base64
import hashlib
import json
import lzma
import nbformat
import re

from scripts.build_v9_training_cuda_ab_probe_20261122 import build, CHANGED_MODULES


def test_self_contained_cuda_source_hash_and_overlays():
    n=build()
    assert [x.cell_type for x in n.cells]==['markdown','code','code']
    assert len(nbformat.writes(n).encode())<1_000_000
    for x in n.cells[1:]:ast.parse(x.source)
    c=n.cells[2].source
    assert "--envs',str(ENVS)" in c
    assert "'--device','cuda'" in c
    assert 'for repeat in range(REPEATS)' in c
    assert "report['trajectory_parity']" in c
    assert 'CUDA_TRAJECTORY_PARITY' in c
    assert 'UAV_WARP_PROFILE_JSON' in c
    assert 'https://github.com/' not in c
    assert 'no training' not in c.lower() or n.metadata['has_train_commands'] is False
    assert n.metadata['scientific_environment_changed'] is False
    assert 'WANDB_API_KEY' not in c and 'HF_TOKEN' not in c
    encoded=re.search(r'encoded=json.loads\(lzma.decompress\(base64.b85decode\(("[^"]+"|\'[^\']+\')\)\)\)',c)
    assert encoded is not None
    overlay=json.loads(lzma.decompress(base64.b85decode(ast.literal_eval(encoded.group(1)))))
    assert set(CHANGED_MODULES).issubset(overlay)
    for path,body in overlay.items():
        assert hashlib.sha256(base64.b85decode(body)).hexdigest()==n.metadata['optimized_module_sha256'][path]
