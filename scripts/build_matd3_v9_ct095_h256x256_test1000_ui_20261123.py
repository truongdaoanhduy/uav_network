"""Standalone NoOracle V9 MATD3 E4096/U4, confirm .95, H256x256, Eval100/Test1000.

Retain 18d756ef's *evaluation protocol* (seeded WarpVectorEnv, two GPU
workers, deterministic actor-only validation and held-out sharded test), not its
incompatible actor weights or target-ID observation semantics.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path

import nbformat

from scripts.build_matd3_v9_training_accel_kaggle_ui_20261122 import (
    build_notebook as build_v2_notebook,
)

CONFIG_PATH = 'configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml'
OUTPUT = Path('notebooks/MATD3-NOORACLE-V9-CT095-H256x256-EVAL100-TEST1000-E4096-U4-B8192-S44-KAGGLE-UI.ipynb')
MARKER = 'SPEC["gpu_eval_candidate_limit"] = 512\n'


def build_notebook():
    nb = build_v2_notebook()
    nb.cells[0].source = (nb.cells[0].source
        .replace('H256×512', 'H256×256')
        .replace('H256x512', 'H256x256')
        .replace('Test4000', 'Test1000')
        .replace('4k held-out Test', '1k held-out Test')
        .replace('4,000', '1,000')
        .replace('4000 held-out', '1000 held-out'))
    nb.cells[0].source += (
        '\n\n**NEW CT095 / H256x256 / TEST1000 SCIENCE EXPERIMENT:** '
        'Bayesian confirmation threshold is **0.95** (was 0.99). '
        'MATD3 shared Actor/Plain Critic hidden sizes are **256×256** (was 256×512). '
        'Run 20,000 training episodes from freshly initialized weights, '
        'deterministic Val100 per Actor checkpoint, then independent held-out '
        'Final Test1000 of the validation-selected best model. '
        'NoOracle cell reports, actual Warp PHY/MAC, rewards, energy and '
        'seed families are preserved. Do not load H256x512 checkpoints. '
        'This differs scientifically from both historical W&B runs.\n'
    )
    old = 'MAX_WARP_TEST_WORLDS_PER_GPU = 256'
    assert nb.cells[1].source.count(old) == 1
    nb.cells[1].source = nb.cells[1].source.replace(old, 'MAX_WARP_TEST_WORLDS_PER_GPU = 500')
    # 1000 heldout worlds: GPU0 seeds base+0..499, GPU1 base+500..999.
    # Preserve reference-style 2x100 fixed-seed Validation per GPU launch.
    source = nb.cells[2].source
    assert source.count(MARKER) == 1
    patch = '''# CT095-H256x256-TEST1000: explicit values AFTER reference-style setup.
SPEC["config"] = "''' + CONFIG_PATH + '''"
SPEC["dense_eval_schedule"]["final_test_scenarios"] = 1000
SPEC["overrides"].extend([
    "sensor.confirmation_threshold=0.95",
    "matd3.hidden_sizes=[256,256]",
    "runtime.eval_episodes=1000",
    "tracking.group=MATD3-NOORACLE-V9-CT095-H256x256-E4096-U4-TEST1000-S44",
])
SPEC["runs"][ALGORITHM]["name"] = (
    f"MATD3-EVAL100-ACTORBN-PLAINCRITIC-NOORACLE-V9-CT095-"
    f"E4096-U{UPDATES_PER_VECTOR_STEP}-B8192-H256x256-S{TRAIN_SEED}-TEST1000-{_rid}"
)
assert SPEC["dense_eval_schedule"]["validation_scenarios_per_checkpoint"] == 100
assert MAX_WARP_TEST_WORLDS_PER_GPU == 500
assert CHECKPOINTS_PER_GPU_BATCH == 2
print("NOORACLE_V9_CT095_H256x256_TEST1000", SPEC["config"],flush=True)
'''
    source = source.replace(MARKER, patch + MARKER, 1)
    source = source.replace('MATD3_NO_ORACLE_TRAIN_VALIDATION100_TEST4000_COMPLETE',
                            'MATD3_NO_ORACLE_CT095_H256_TEST1000_COMPLETE')
    # This is runtime display text only, not the original embedded SHA-pinned archive.
    old_summary='"E4096/U4/B8192/H256x512/S44"'
    if old_summary in source:
        source=source.replace(old_summary,'"E4096/U4/B8192/H256x256/S44/CT095/TEST1000"')
    ast.parse(source)
    nb.cells[2].source = source
    nb.metadata['performance_variant']='NOORACLE-V9-CT095-H256x256-TEST1000'
    nb.metadata['scientific_scenario']='NOORACLE-V9-CT095'
    nb.metadata['change_scope']='confirmation_threshold_095+MATD3_H256x256+test1000;V2_MAC_optimized'
    nb.metadata['baseline_eval_reference_run']='18d756ef'
    nb.metadata['protocol'].update({'training_episodes':20000,'validation_episodes':100,
                                    'test_episodes':1000,'report_identity':'observed_cell_id'})
    rendered=nbformat.writes(nb).encode()
    if len(rendered)>=1_000_000:
        raise ValueError('Notebook over 1MB Kaggle source limit')
    nbformat.validate(nb)
    return nb


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args()
    notebook=build_notebook()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,args.output)
    print('NEW_NOTEBOOK',args.output,'SOURCE_SHA256',notebook.metadata['source_sha256'],
          'COMMIT',notebook.metadata['reference_commit'])

if __name__=='__main__':main()
