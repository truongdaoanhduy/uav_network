"""Build a separate CUDA-evaluation candidate; never overwrite approved notebook.

The science is CT095 H256x256 Test1000, but on each T4 the Validation world
pool is 8192 and checkpoints are assigned in contiguous 81x100 blocks.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import nbformat
from scripts.build_matd3_v9_ct095_h256x256_test1000_ui_20261123 import build_notebook as base_build

OUTPUT=Path('notebooks/MATD3-NOORACLE-V9-CT095-H256x256-EVAL8192-CONTIGUOUS-TEST1000-CANDIDATE-T4X2.ipynb')

def build_notebook():
    nb=base_build()
    pre=nb.cells[1].source
    for old,new in [('CHECKPOINTS_PER_GPU_BATCH = 2', 'CHECKPOINTS_PER_GPU_BATCH = 81'),
                    ('MAX_WARP_EVAL_WORLDS_PER_GPU = 256', 'MAX_WARP_EVAL_WORLDS_PER_GPU = 8192')]:
        assert pre.count(old)==1
        pre=pre.replace(old,new)
    pre=pre.replace('2 x 100 = 200 worlds/GPU, conservative T4 VRAM setting',
                    '81 x 100 = 8100 real worlds/GPU, 92 inactive filler slots')
    pre=pre.replace('FINAL TEST4000','FINAL TEST1000')
    nb.cells[1].source=pre
    source=nb.cells[2].source
    old='"--eval-world-pool-size", "0"'
    new='"--eval-world-pool-size", str(MAX_WARP_EVAL_WORLDS_PER_GPU)'
    assert source.count(old)==1
    source=source.replace(old,new)
    old='"--policies-per-batch", str(CHECKPOINTS_PER_GPU_BATCH),'
    new='''"--policies-per-batch", str(CHECKPOINTS_PER_GPU_BATCH),
        "--checkpoint-assignment", "contiguous_batches",'''
    assert source.count(old)==1
    source=source.replace(old,new)
    # Reference-style source still held its old 2-policy assertion.  Without
    # this edit, the 81-policy production notebook aborts before running.
    assert source.count('assert CHECKPOINTS_PER_GPU_BATCH == 2') == 1
    source=source.replace('assert CHECKPOINTS_PER_GPU_BATCH == 2',
                          'assert CHECKPOINTS_PER_GPU_BATCH == 81')
    source=source.replace('HELD-OUT TEST4000', 'HELD-OUT TEST1000')
    source=source.replace('HF BEST → TEST4000', 'HF BEST → TEST1000')
    ast.parse(source)
    nb.cells[2].source=source
    nb.cells[0].source += (
        '\n\n**Eval8192 CUDA candidate — NOT YET VALIDATED ON A T4.** '
        'GPU0 handles checkpoints 1–81, GPU1 handles 82–162, repeat '
        'for later batches. Each GPU constructs and reuses 8,192 physical '
        'Warp worlds; up to 8,100 valid 100-seed checkpoint episodes, '
        'with the remaining 92 worlds inert for metrics. '
        'The same 100 seed bank is used for every checkpoint. '
        'Test1000 remains 500 independent seeds per GPU.\n'
    )
    nb.metadata['performance_variant']='NOORACLE-V9-CT095-H256x256-EVAL8192-CONTIGUOUS-TEST1000'
    nb.metadata['eval_assignment']='contiguous_batches'
    nb.metadata['eval_world_pool_size_per_gpu']=8192
    nb.metadata['eval_real_worlds_per_full_batch']=8100
    nb.metadata['eval_unused_slots_per_full_batch']=92
    nb.metadata['requires_cuda_t4_validation']=True
    nb.metadata['baseline_safe_notebook_untouched']=True
    nbformat.validate(nb)
    if len(nbformat.writes(nb).encode()) >= 1_000_000:
        raise RuntimeError('Candidate notebook exceeds Kaggle 1MB')
    return nb

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args()
    nb=build_notebook()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,args.output)
    print('CANDIDATE_NOTEBOOK',args.output,'BYTES',args.output.stat().st_size,
          'SOURCE_SHA256',nb.metadata['source_sha256'])

if __name__=='__main__':
    main()
