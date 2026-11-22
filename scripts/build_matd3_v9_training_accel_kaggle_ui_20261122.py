"""Generate an independent complete-training V9 notebook with source-index acceleration.

Keeps scientific settings, E4096/U4/B8192, Val100/Test4000, W&B and HF.
Does not modify Kaggle notebooks or reuse the active e079eace W&B ID.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import nbformat

from scripts.build_matd3_reference_style_kaggle_ui import REFERENCE_NOTEBOOK, build

TAG="NOORACLE-V9-TRAINACCEL"
OUTPUT=Path('notebooks/MATD3-NOORACLE-V9-TRAINACCEL-CELLLOOKUP-OBSFAST-E4096-U4-B8192-H256x512-S44-KAGGLE-UI.ipynb')


def build_notebook(reference=REFERENCE_NOTEBOOK):
    root=Path(__file__).resolve().parents[1]
    n=build(reference,root)
    assert len(n.cells)==3
    assert [c.cell_type for c in n.cells]==['markdown','code','code']
    for cell in n.cells:
        cell.source=cell.source.replace('NOORACLE-V9',TAG)
        if cell.cell_type=='code':ast.parse(cell.source)
    n.cells[0].source += ("\n\n**TRAINACCEL: new independent full training run.** "
        "Same Bayesian sensor, 1,000m map, true Warp PHY/MAC, 140 hypothesis slots, "
        "20k episodes, 100-seed checkpoint Eval, 4k held-out Test, reward, "
        "replay, ActorBN / plain Critic, HF and W&B as V9 e079eace. "
        "Uses an ephemeral cell→target index for truth-side Bayesian sensing "
        "and reuses full-slot local report eligibility during decentralized "
        "Actor observation. The derived index is never visible to actors "
        "and does not enter checkpoint scientific state. "
        "CPU trajectory checksums match V9 pinned commit across 12x180 steps; "
        "T4 speed improvement is UNVERIFIED. The full 20k training is not "
        "automatically started by generating this file.\n")
    n.metadata['performance_variant']=TAG
    n.metadata['baseline_training_run']='e079eace'
    n.metadata['scientific_scenario']='NOORACLE-V9'
    n.metadata['reference_original_source_commit']='1346ad47'
    n.metadata['accelerated_source_commit']='329f527549722c664eeeb0d56b6b384ae5250f7c'
    n.metadata['change_scope']='target_cell_index_once_per_world_reset+reused_full_slot_observation_eligibility+safe_sparse_mac'
    n.metadata['requires_cuda_t4_validation']=True
    n.metadata['separate_wandb_run_required']=True
    rendered=nbformat.writes(n).encode()
    if len(rendered)>=1_000_000:raise RuntimeError(f'Training notebook exceeds Kaggle 1MB limit: {len(rendered)}')
    nbformat.validate(n)
    return n


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,default=REFERENCE_NOTEBOOK)
    ap.add_argument('--output',type=Path,default=OUTPUT)
    args=ap.parse_args()
    n=build_notebook(args.reference)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,args.output)
    print('TRAINING_ACCEL_NOTEBOOK',args.output,'BYTES',args.output.stat().st_size,
          'SOURCE_SHA256',n.metadata['source_sha256'])

if __name__=='__main__':main()
