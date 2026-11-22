"""Rebuild the *original 18d756ef Kaggle UI layout* for the corrected v9 run.

Three cells, like notebook5051433b29.ipynb:
1) reference-style experiment readme;
2) editable MATD3 config and real GPU/Secrets preflight;
3) self-contained, SHA-verified source + train/Val100/best/Test4000 runner.

Original reference cell layout is the template; its old oracle-assisted source
MUST NOT be used for the actual executed pipeline.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import lzma
import json
from pathlib import Path
import re
import subprocess

from uav_network.cloud.kernel import kernel_script
from uav_network.cloud.source_bundle import build_source_archive, source_sha256
from scripts.build_matd3_nooracle_one_session_ui import make_spec

REFERENCE_NOTEBOOK = Path(
    '/home/aduy/Downloads/M2_EXACT_REFERENCE_18d756ef_20261121/'
    'kaggle_notebook/notebook5051433b29.ipynb'
)


def preflight_cell(old: str) -> str:
    """Keep the original preflight API with honest safe Warp GPU eval budgets."""
    out = old.replace('NOTEBOOK_REF = "therwed/notebook5051433b29"',
                      'import os\nNOTEBOOK_REF = os.environ.get("KAGGLE_KERNEL_REF") or "YOUR_KAGGLE_USERNAME/YOUR_NOTEBOOK_SLUG"')
    out = out.replace('MAX_WARP_EVAL_WORLDS_PER_GPU = 8192',
                      'MAX_WARP_EVAL_WORLDS_PER_GPU = 256')
    out = out.replace('MAX_WARP_TEST_WORLDS_PER_GPU = 4096',
                      'MAX_WARP_TEST_WORLDS_PER_GPU = 256')
    out = out.replace('CHECKPOINTS_PER_GPU_BATCH = 81  # 81 x 100 = 8100 env/GPU; adjust to 64 if VRAM is tight',
                      'CHECKPOINTS_PER_GPU_BATCH = 2  # 2 x 100 = 200 worlds/GPU, conservative T4 VRAM setting')
    out = out.replace('TRAIN_SEED = 44  # Thí nghiệm mới, không dùng lại checkpoint CBF.',
                      'TRAIN_SEED = 44  # No-oracle v9: train mới, không nạp checkpoint observation v7.')
    out = out.replace('print("MATD3 EVAL100 / 100 seeds; ActorBN baseline (Critic plain)")',
                      'print("MATD3 EVAL100 / 100 seeds; ActorBN baseline (Critic plain), Bayesian NOORACLE-V9")')
    # Opt-in research evidence, with a 5-minute bounded archive cadence;
    # kept separate from W&B scalar plots and training learner updates.
    out += ('\nENABLE_HF_FORENSICS = True  # False = fast W&B-only mode\n'
            'UAV_HF_DIAGNOSTIC_REPO = "duy95/uav-search-diagnostics"\n')
    # Add a genuine code-checked pipeline marker before training.
    out += ('\nprint("TRAIN → VAL100 → BEST CHECKPOINT → FINAL TEST4000")\n'
            'print("GPU evaluation budget:", CHECKPOINTS_PER_GPU_BATCH, '
            '"policies/batch;", MAX_WARP_EVAL_WORLDS_PER_GPU, "worlds/GPU")\n')
    assert 'NOTEBOOK_REF' in out and 'UserSecretsClient' in out
    ast.parse(out)
    return out


def markdown_cell(old: str) -> str:
    out=old.replace('# MATD3 BenchMARL-style fixed-seed convergence — ActorBN baseline (Critic plain)',
                    '# MATD3 EVAL100 — ActorBN baseline (Critic plain) — NOORACLE-V9')
    out=out.replace('Run: E4096 / U4 / B8192 / 20,000 training episodes / seed44 / 6 UAV / 35 targets / 25 obstacles / safety reject.',
                    'Run: E4096 / U4 / B8192 / H256×512 / 20,000 training episodes / seed44 / 6 UAV / 35 targets / 25 obstacles / safety reject.')
    out=out.replace('**Multi-checkpoint Eval:** up to 81 checkpoints × 100 episodes = 8,100 Warp worlds PER GPU, \nwith torch.vmap over separate trained Actor weights. Actor-only inference, NO Critic/replay/backward. \nThe maximum is a requested upper bound, not a measured safe T4 configuration; reduce CHECKPOINTS_PER_GPU_BATCH to 64 if needed.',
                    '**Multi-checkpoint Eval:** 2 Actor checkpoints × 100 environments = 200 Warp worlds/GPU per batch, \nusing torch.vmap over Actor weights. The smaller batch is a conservative T4 memory setting, \nnot a measured throughput optimum. All model checkpoints still receive deterministic Val100.')
    out=out.replace('Critic BN experiment is an **ablation, not a proven improvement**. BN normalizes hidden Q features only, \nand Critic BN stats are frozen for actor-gradient/target-Q evaluation. \nAll reward, world, train ratio, model width, exploration and seed settings remain matched between notebooks.',
                    '**Only the scenario differs from the historical run:** report generation uses the observed 2D cell ID \nand Bayesian confidence, never ground-truth target ID. Ground truth is used only to simulate the sensor \nand to score reward/evaluation. Actor uses BN, Critic stays plain (no BN). \nThere is **no DS-map fusion at GCS** or **3D obstacle reconstruction** in this experiment.')
    out=out.replace('**Usage:** fill NOTEBOOK_REF, choose Kaggle GPU T4×2, enable Internet and Secrets WANDB_API_KEY + HF_TOKEN, then Save & Run All.',
                    '**Usage:** set NOTEBOOK_REF if Kaggle has not provided KAGGLE_KERNEL_REF, choose Kaggle GPU T4×2, \nenable Internet and Secrets WANDB_API_KEY + HF_TOKEN, then Save & Run All.')
    out += ('\n**New experiment:** starts MATD3 from random weights (obs-v8), not checkpoint `18d756ef`. '
            'Selection uses only Val100; independent Test4000 is reserved for the best Actor. '
            'For a fair experiment, CUDA T4×2 capacity and full 20,000-episode completion must still be verified.\n')
    return out


def runtime_spec_injection(spec: dict) -> str:
    """Match the original NOTEBOOK_REF/rid/knobs and keep no-oracle YAML pinned."""
    return '''# Original-style UI knobs are defined in the previous cell.
from uuid import uuid4
SPEC["kernel_ref"] = NOTEBOOK_REF
SPEC["account"] = "ui_" + NOTEBOOK_REF.split("/", 1)[0]
SPEC["submitted_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
_rid = uuid4().hex[:8]
SPEC["runs"] = {ALGORITHM: {
    "id": _rid,
    "name": f"{ALGORITHM.upper()}-EVAL100-ACTORBN-PLAINCRITIC-NOORACLE-V9-E4096-U{UPDATES_PER_VECTOR_STEP}-B8192-H256x512-S{TRAIN_SEED}-{_rid}"}}
SPEC["config"] = "configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml"
SPEC["overrides"] = [
    f"runtime.algorithm={ALGORITHM}",
    f"runtime.seed={TRAIN_SEED}",
    "runtime.episodes=20000",
    "runtime.max_train_seconds=0",
    "pql.enabled=false",
    "world.motion_safety=reject",
    "matd3.actor_batch_norm=true",
    "matd3.critic_batch_norm=false",
    "matd3.initialization=default",
    "matd3.actor_output_regularization=0.0",
    f"matd3.updates_per_step={UPDATES_PER_VECTOR_STEP}",
    f"tracking.periodic_evaluation_episodes={VALIDATION_EPISODES}",
    "tracking.deferred_gpu_eval_enabled=true",
    "tracking.deferred_gpu_eval_keep_all=true",
    "tracking.deferred_gpu_eval_collection_stride=100",
    "tracking.deferred_gpu_eval_target=0",
    "tracking.deferred_gpu_eval_candidate_limit=0",
    "tracking.online_gpu_eval_enabled=false",
    f"tracking.group=MATD3-EVAL100-ACTORBN-PLAINCRITIC-NOORACLE-V9-E4096-U{UPDATES_PER_VECTOR_STEP}-S{TRAIN_SEED}",
]
SPEC["validation_step_targets"] = list(range(0, 80_000_001, 100 * 4096))
SPEC["dense_eval_schedule"] = {
    "actor_snapshot_every_vector_steps": 100,
    "validation_every_vector_steps": 100,
    "training_episodes": 20000,
    "validation_scenarios_per_checkpoint": VALIDATION_EPISODES,
    "final_test_scenarios": 4000,
    "evaluation_deterministic": True,
    "evaluation_static": True,
}
SPEC["checkpoint_selection_rule"] = "exact_100_vector_step_checkpoints_plus_real_final"
SPEC["gpu_eval_candidate_limit"] = 512
SPEC["one_session_train_eval_test"] = True
SPEC["require_full_episode_budget"] = True
SPEC["training_only"] = True
SPEC["requested_episodes"] = 20000
SPEC["eval_test_reserved_seconds"] = 5400
os.environ["UAV_FAST_TRAINING"] = "false"
os.environ["UAV_DIAGNOSTIC_STORE"] = (
    "huggingface_public" if ENABLE_HF_FORENSICS else "off")
os.environ["UAV_DIAGNOSTIC_HF_REPO"] = UAV_HF_DIAGNOSTIC_REPO
os.environ["UAV_DEFERRED_EXPORT_HF"] = "true"
os.environ["UAV_WANDB_COMPACT_LOGS"] = "true"
# Preserve the historical native episode/global-step W&B paper panels.
# Raw per-step/episode/replay/PHY evidence belongs to public HF archives.
os.environ["UAV_WANDB_PLOTS_ONLY"] = "true"
os.environ["UAV_TRAIN_EPISODE_RETURN_LOG_EVERY"] = "64"
os.environ["UAV_WANDB_EPISODE_LOG_EVERY"] = "32"
# Finalized deferred Actor manifest is the handoff; validated best selection
# MUST only happen inside GPU Val100, before held-out Test4000.
print("DEFERRED_BEST_SELECTION", "TRAIN → COMPLETE ARCHIVE → GPU VAL100 → HF BEST → TEST4000",flush=True)
print("Kaggle UI:", "https://www.kaggle.com/code/" + SPEC["kernel_ref"], flush=True)
print("W&B:", f"https://wandb.ai/{SPEC['entity']}/{SPEC['project']}/runs/{_rid}", flush=True)
print("CONFIGURATION", SPEC["config"], "E4096/U4/B8192/H256x512/S44", flush=True)
print("FULLFLOW", "TRAIN → VAL100 → BEST → HELD-OUT TEST4000", flush=True)
'''


def patch_execution(source: str, spec: dict) -> str:
    literal = 'SPEC = ' + repr(spec) + '\n'
    assert source.count(literal)==1, 'Source bootstrap not based on provided spec'
    source = source.replace(literal, literal + runtime_spec_injection(spec), 1)
    # Keep all 520 source files, including CUDA regression tests. Kaggle
    # rejects source code >= 1 MB; XZ shrinks the same tarball without deleting
    # any modules, changing file bytes, or requiring a download at runtime.
    gz_open='tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz")'
    assert source.count(gz_open)==1
    source=source.replace(gz_open,'tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz")',1)
    # Preserve the original notebook's optional W&B dual-axis presentation.
    marker = 'spec_file.write_text(json.dumps(SPEC, indent=2))\n'
    assert source.count(marker)==1
    optional_view = '''# Match the historical notebook's optional native W&B dual-axis workspace.
# This does not fabricate evaluation data or modify the training learner.
_dual_view_path = Path("/kaggle/working") / f"wandb_dual_axis_view_{_rid}.json"
_dual_view_log = Path("/kaggle/working") / f"wandb_dual_axis_view_{_rid}.log"
_variant = "MATD3 Bayesian NOORACLE-V9, safety reject, ActorBN / plain Critic"
try:
    _cmd = [str(python), "-m", "uav_network.experiments.wandb_dualaxis_baseline_workspace",
            "--run-id", _rid, "--variant", _variant,
            "--apply", "--wait-seconds", "900", "--output", str(_dual_view_path)]
    with _dual_view_log.open("a") as _stream:
        subprocess.Popen(_cmd, cwd=root, stdout=_stream, stderr=subprocess.STDOUT)
    print("WANDB_DUAL_VIEW_REQUESTED", _dual_view_log, flush=True)
except Exception as _exc:
    print("WANDB_DUAL_VIEW_OPTIONAL_WARNING", type(_exc).__name__, str(_exc)[:200], flush=True)
'''
    source = source.replace(marker, marker+optional_view, 1)
    hardcoded = '''"--policies-per-batch", "2",
        "--eval-world-pool-size", "0",
        "--max-eval-worlds-per-gpu", "256",
        "--max-test-worlds-per-gpu", "256",'''
    configured = '''"--policies-per-batch", str(CHECKPOINTS_PER_GPU_BATCH),
        "--eval-world-pool-size", "0",
        "--max-eval-worlds-per-gpu", str(MAX_WARP_EVAL_WORLDS_PER_GPU),
        "--max-test-worlds-per-gpu", str(MAX_WARP_TEST_WORLDS_PER_GPU),'''
    assert source.count(hardcoded)==1
    source=source.replace(hardcoded,configured,1)
    source+='''
print("KAGGLE_TRAIN_EVAL_TEST_EXECUTION_COMPLETE", flush=True)
try:
    if _dual_view_path.exists():
        print("WANDB_EPISODE_AND_GLOBAL_STEP_VIEW",
              json.loads(_dual_view_path.read_text()).get("url", ""), flush=True)
    else:
        print("WANDB_DUAL_VIEW_OPTIONAL_WARNING: see", _dual_view_log, flush=True)
except Exception as _exc:
    print("WANDB_DUAL_VIEW_OPTIONAL_WARNING", type(_exc).__name__, str(_exc)[:200], flush=True)
'''
    ast.parse(source)
    return source


def build(reference: Path, repo_root: Path):
    import nbformat
    original=nbformat.read(reference,as_version=4)
    assert len(original.cells)==3
    assert [cell.cell_type for cell in original.cells]==['markdown','code','code']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo_root,text=True).strip()
    original_gzip_tarball=build_source_archive(repo_root)
    tarball=lzma.compress(gzip.decompress(original_gzip_tarball),
                          format=lzma.FORMAT_XZ,preset=6)
    spec=make_spec(commit)
    source=kernel_script(spec,source_archive=tarball,credential_source='kaggle_secrets')
    nb=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(markdown_cell(original.cells[0].source)),
        nbformat.v4.new_code_cell(preflight_cell(original.cells[1].source)),
        nbformat.v4.new_code_cell(patch_execution(source,spec)),
    ],metadata={
        'kernelspec':copy.deepcopy(original.metadata['kernelspec']),
        'language_info':copy.deepcopy(original.metadata['language_info']),
        'source_sha256':source_sha256(tarball),
        'reference_commit':commit,
        'reference_notebook':'notebook5051433b29.ipynb',
        'reference_run':'MATD3-EVAL100-ACTORBN-PLAINCRITIC-E4096-U4-B8192-H256x512-S44-18d756ef',
        'protocol':{'training_episodes':20000,'validation_episodes':100,
                    'test_episodes':4000,'backend':'warp_packet','belief':'bayesian',
                    'report_identity':'observed_cell_id'},
    })
    # Kaggle checks the Python source size (not the gzip archive size).
    # Notebook JSON is checked as well to tolerate conservative UI limits.
    python_bytes=sum(len(cell.source.encode('utf-8')) for cell in nb.cells
                     if cell.cell_type=='code')
    notebook_bytes=len(nbformat.writes(nb).encode('utf-8'))
    if python_bytes>=1_000_000 or notebook_bytes>=1_000_000:
        raise ValueError(f'Notebook exceeds Kaggle 1 MB: python={python_bytes}, '
                         f'ipynb={notebook_bytes}')
    return nb


def main():
    import nbformat
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,default=REFERENCE_NOTEBOOK)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    nb=build(args.reference,root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,args.output)
    print('NOTEBOOK',args.output,'BYTES',args.output.stat().st_size)
    print('CELLS',[(c.cell_type,len(c.source)) for c in nb.cells])
    print('TOTAL_PYTHON_SOURCE_BYTES',sum(len(c.source.encode('utf-8')) for c in nb.cells if c.cell_type=='code'))
    print('SOURCE_SHA256',nb.metadata['source_sha256'])
    print('SOURCE_COMMIT',nb.metadata['reference_commit'])

if __name__=='__main__':main()
