"""Build separate FASTGRID full-flow Kaggle notebook without changing science.

This takes the well-tested 3-cell, SHA verified, <1MB reference-format builder
and opts into periodic-only Actor snapshot selection, low-risk learner scalar
telemetry improvements, plus sparse CUDA Event timing. It retains HF/W&B
forensics, 100-episode/checkpoint GPU validation and held-out Test1000.
"""
from pathlib import Path
import hashlib
import nbformat
from scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 import (
    ROOT,build_notebook
)

NAME='MATD3-NOORACLE-POSTERIORONLY-TX08-FASTGRID-OPTIMIZED-T4X2-E4096-U4-B8192-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME

INTRO_EXTRA='''

### FASTGRID speed ablation (separate W&B run)

**Changes to scheduling/implementation, NOT to the MAP800/Posterior-only science:**
- `UAV_PERIODIC_ACTORS_ONLY=true`: **do not save training-metric-best-candidate extras**.
  Keep initial, every 100 vector steps, and final. Every stored Actor is still
  validated on its 100 fixed seeds, and Final Test remains 1000 held-out episodes.
  The prior 602a1e82 run had 254 actors: 76 periodic and 178 off-grid extras.
- `UAV_FAST_GRAD_CLIP=true`: avoid duplicate per-gradient norm reductions;
  retains the configured clipping rule and nonfinite rejection.
- `UAV_BATCH_SCALAR_METRICS=true`: batch CUDA-to-host telemetry scalar copies
  on eligible tensors, avoiding repeated synchronization points.
- `UAV_CUDA_EVENT_PROFILE=true`: sparse CUDA Event timing for one update every
  existing 65,536 global-step diagnostic interval; yields
  `timing/probe_update_cuda_ms` without synchronizing all updates.
- **Deep W&B + Hugging Face forensics remain ENABLED.** No `UAV_FAST_TRAINING`.
  Warp/UavNetSim radio, replay75%CPU/25%GPU, E4096 U4 B8192, MATD3 network,
  reward, Bayesian Pd/Pf, TX0.8W and random seed44 remain unchanged.

**Risks/limits:** Existing 602a1e82 checkpoint must NOT be resumed; this is a
new ablation. No T4x2 speed claim is made before a fresh GPU benchmark.
`UAV_HYBRID_PINNED_STAGING` and `UAV_DIRECT_GPU_COLLECTOR_COPY` remain off,
because their GPU speed/correctness needs separate CUDA A/B testing.
'''

SPEED_FLAGS={
    'UAV_PERIODIC_ACTORS_ONLY':'true',
    'UAV_FAST_GRAD_CLIP':'true',
    'UAV_BATCH_SCALAR_METRICS':'true',
    'UAV_CUDA_EVENT_PROFILE':'true',
}


def build_speed_notebook():
    n=build_notebook()
    n.cells[0].source+=INTRO_EXTRA
    marker='# Kaggle runs only this Python bootstrap; source is inside the payload.\n'
    assert n.cells[2].source.count(marker)==1
    extra=("# FASTGRID: opt-in scheduling/telemetry-only performance profile.\n"
           +''.join(f"os.environ[{name!r}]={value!r}\n" for name,value in SPEED_FLAGS.items())
           +"SPEC['speed_profile']='periodic_grid_only_fast_scalar_grad_cuda_timing'\n"
           +"SPEC['runs']['matd3']['name'] += '-FASTGRID'\n"
           +"print('SPEED_PROFILE_FASTGRID',SPEC['speed_profile'],flush=True)\n")
    n.cells[2].source=n.cells[2].source.replace(marker,extra+marker,1)
    n.metadata['speed_profile']='fastgrid_periodic_only_batch_scalars_fast_grad_clip'
    n.metadata['checkpoint_policy']='periodic_only_every100_initial_final'
    n.metadata['t4_cuda_speed_verified']=False
    n.metadata['replay_cpu_fraction']=.75
    nbformat.validate(n)
    assert len(n.cells)==3
    assert len(nbformat.writes(n).encode('utf-8'))<850_000
    assert len(n.cells[2].source.encode('utf-8'))<825_000
    return n


def main():
    n=build_speed_notebook();OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUT)
    print('KAGGLE_FASTGRID_NOTEBOOK_READY',OUT,'BYTES',OUT.stat().st_size,
          'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest(),
          'SOURCE_SHA256',n.metadata['source_sha256'],flush=True)


if __name__=='__main__':main()
