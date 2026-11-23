"""New source-sealed, from-scratch NoOracle posterior/communication/actor ablation.

Do NOT change the earlier MAP1000/TX0.6 FASTGRID run or re-use its Actor.
The existing Kaggle T4×2 full Train20k/Val100 per Actor/Best/Test1000 flow
remains. Performance/quality improvements require *new* validation evidence.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import nbformat

import scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 as original_map1000
import scripts.build_map1000_tx06_autotune_fullflow_ui_20261123 as original_autotune
from scripts.build_map1000_tx06_autotune_fullflow_fe799_fix_20261123 import (
    build_notebook as previously_verified_builder,
    TARGET_KERNEL_REF,
)

ROOT=Path(__file__).resolve().parents[1]
CONFIG='configs/matd3-nooracle-map1000-tx06-prior0021875-reward02-actorstable-20261123.yaml'
NAME='MATD3-MAP1000-TX06-PRIOR0021875-REWARD02-ACTORNORM-T4X2-VAL100-TEST1000-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME
RUN_SUFFIX='-PRIOR0021875-REWARD02-ACTORNORM'


@contextmanager
def _science_config_context():
    """Patch builder-only global references, restore even on assertion errors."""
    old_map1000=original_map1000.CONFIG
    old_autotune=original_autotune.CONFIG
    try:
        original_map1000.CONFIG=CONFIG
        original_autotune.CONFIG=CONFIG
        yield
    finally:
        original_map1000.CONFIG=old_map1000
        original_autotune.CONFIG=old_autotune


def build_notebook():
    from uav_network.configuration import load_config
    c=load_config(CONFIG);c.validate()
    assert c.sensor.belief_prior==35/1600
    assert c.reward.communication_efficiency==.2
    assert c.matd3.output_normalization and c.matd3.actor_output_regularization==.001
    assert c.world.motion_safety=='reject' and c.network.tx_power_max_w==.6
    assert c.runtime.warm_start_checkpoint=='none' and c.runtime.episodes==20000
    with _science_config_context():
        n=previously_verified_builder()
    assert n.metadata['science_config']==CONFIG
    assert len(n.cells)==3
    # Human-readable science declaration in Kaggle. Zero changes to the
    # existing Warp packet/CBF/reward physics or actor-observation features.
    n.cells[0].source+='''

## NEW SCIENCE ABLATION — CALIBRATED MAP1000/TX0.6

This is **NOT** the old run `59042708` and **cannot resume any historical
checkpoint**. Starting from scratch is required because the reward, belief
prior and MATD3 continuous policy head are deliberately changed.

- **Calibrated prior**: `35/1600 = 0.021875` per cell instead of `0.5`.
  The actual scene distribution excludes forbidden cells, so this is an
  initial density calibration, NOT a known perfect Bayesian spatial prior.
- **Communication-efficiency penalty**: `0.2` instead of `1.0`. Keep
  false-confirmation, drop, expiry, physical TX energy and discovery/delivery
  rewards as defined by the existing science. This is an *ablation*, not a
  proven optimal reward coefficient.
- **Actor stabilization**: motion-head pre-tanh normalization ON; raw Actor
  output squared regularization `0.001`. The actual collision-safe motion
  mode stays **reject**; no retired XYZ DWA or CBF is introduced.
- New training and validation precision: true-cell confirmation precision,
  true-GCS-delivered report precision, false cell confirmations and false
  reports at GCS. These are **terminal scoring only**, NEVER actor inputs.
- **GPU protocol** remains T4×2, E4096/U4/B8192, seed44, 20k episodes,
  Val100 for *each* periodic Actor, independent Best Actor and Test1000.
- A clean held-out validation is required before claiming lower error rates,
  better reward alignment or fewer boundary rejections.
'''
    n.cells[1].source+='''
# New science settings — user-visible; enforced by sealed YAML below.
BELIEF_PRIOR=35/1600
REWARD_COMMUNICATION_EFFICIENCY=0.2
MATD3_ACTOR_OUTPUT_REGULARIZATION=0.001
MATD3_MOTION_OUTPUT_NORMALIZATION=True
REQUIRES_FRESH_TRAINING=True
assert BELIEF_PRIOR==0.021875 and REWARD_COMMUNICATION_EFFICIENCY==0.2
assert MATD3_ACTOR_OUTPUT_REGULARIZATION==0.001
assert MATD3_MOTION_OUTPUT_NORMALIZATION and REQUIRES_FRESH_TRAINING
'''
    # Mark the new scientific ablation *before* job.json is persisted so that
    # the name and provenance for W&B, HF and Kaggle cannot silently collide
    # with the old FASTGRID train. Preserve per-run UUID allocation.
    src=n.cells[2].source
    marker='spec_file = Path("/kaggle/working/job.json")'
    if src.count(marker)!=1:
        raise RuntimeError('Kaggle job spec marker changed; refusing unsafe notebook patch')
    src=src.replace(marker,'''# Isolate from previous MAP1000 NoOracle run identifiers.
SPEC["runs"]["matd3"]["name"]+='''+repr(RUN_SUFFIX)+'''
SPEC["science_ablation"]={
    "belief_prior":0.021875,
    "communication_efficiency":0.2,
    "actor_output_regularization":0.001,
    "actor_output_normalization":True,
    "checkpoint_resume":"none",
    "checkpoint_compatibility":"new-scientific-mdp-do-not-warmstart",
}
'''+marker,1)
    n.cells[2].source=src
    n.metadata.update({
        'science_config':CONFIG,
        'target_kernel_ref':TARGET_KERNEL_REF,
        'new_mdp_requires_training_from_scratch':True,
        'warm_start':'none',
        'reward_communication_efficiency':0.2,
        'belief_prior':0.021875,
        'actor_output_regularization':0.001,
        'actor_motion_output_normalization':True,
        'source_git_commit_verified_at_build':n.metadata.get('reference_commit'),
        'runtime_cuda_validation_complete':False,
        'fullflow_20k_gpu_run_complete':False,
    })
    nbformat.validate(n)
    if len(nbformat.writes(n).encode())>=850000:
        raise RuntimeError('Kaggle three-cell notebook exceeds safe 850k bound')
    return n


def main():
    notebook=build_notebook()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,OUT)
    print('CALIBRATED_MAP1000_NEW_FULLFLOW_NOTEBOOK',OUT,
          'BYTES',OUT.stat().st_size,
          'SOURCE_GIT_COMMIT',notebook.metadata['reference_commit'],
          'SCIENCE_CONFIG',CONFIG,
          'EMBEDDED_SHA256',notebook.metadata['source_sha256'],
          'NOTEBOOK_SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__=='__main__':main()
