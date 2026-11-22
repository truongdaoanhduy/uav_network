"""Guard Kaggle pilot: must be CUDA Eval only, source pinned, no trainer."""
import ast
import base64
import hashlib
import re
import lzma
from pathlib import Path

from scripts.build_eval8192_safe_gpu_pilot_ui_20261123 import build_notebook


def test_eval8192_pilot_embeds_correct_source_and_never_runs_training():
    n=build_notebook()
    assert len(n.cells)==4
    assert n.metadata['training_permitted'] is False
    assert n.metadata['smoke_only'] is True
    assert n.metadata['gpu_world_pool_size']==8192
    assert n.metadata['gpu_policies']==81
    assert n.metadata['seeds_per_policy']==100
    assert n.metadata['smoke_steps']==12
    assert n.metadata['production_steps']==1500
    assert n.metadata['production_final_test_episodes']==1000
    assert all(c.cell_type=='code' for c in n.cells[1:])
    for c in n.cells[1:]:
        ast.parse(c.source)
    assert "POLICIES_PER_GPU=81" in n.cells[1].source
    assert "PHYSICAL_WORLDS_PER_GPU=8192" in n.cells[1].source
    assert "GPU_SMOKE_STARTED" not in n.cells[2].source
    assert "GPU_SMOKE_RESULT" in n.cells[3].source
    assert "one_session" not in n.cells[3].source
    assert "train(" not in n.cells[3].source
    assert '--checkpoint-offset' in n.cells[3].source
    assert 'gpu*81' in n.cells[3].source or '81*gpu' in n.cells[3].source
    matches=re.findall(r'base64\.b85decode\("([^"\n]+)"\)',n.cells[2].source)
    assert len(matches)==2
    assert hashlib.sha256(base64.b85decode(matches[0])).hexdigest()==n.metadata['source_sha256']
    smoke=lzma.decompress(base64.b85decode(matches[1]))
    assert hashlib.sha256(smoke).hexdigest()==n.metadata['smoke_script_sha256']
    source=smoke.decode()
    assert "def smoke(" in source and "evaluate_group(" in source
    assert "train(" not in source and "one_session" not in source


def test_local_production_candidate_asserts_eighty_one_not_two():
    from scripts.build_matd3_v9_eval8192_candidate_ui_20261123 import build_notebook as production
    n=production()
    assert "assert CHECKPOINTS_PER_GPU_BATCH == 81" in n.cells[2].source
    assert "assert CHECKPOINTS_PER_GPU_BATCH == 2" not in n.cells[2].source
