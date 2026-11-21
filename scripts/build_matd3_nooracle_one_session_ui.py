"""Build a self-contained, token-free Kaggle UI notebook for full MATD3 flow.

Kaggle UI: Import Notebook, choose GPU T4 x2, enable Internet, add
WANDB_API_KEY and HF_TOKEN Kaggle Secrets, then Save & Run All.
The source tarball is embedded and SHA256-verified by the bootstrap.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import uuid

from uav_network.cloud.source_bundle import build_source_archive, source_sha256
from uav_network.cloud.kernel import kernel_script


def make_spec(commit):
    return {
        'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
        'commit': commit,
        'account': 'kaggle_ui',
        'kernel_ref': 'kaggle-ui/matd3-nooracle-v7-eval100-test4000',
        'config': 'configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml',
        'overrides': [],
        'stage': 'train',
        'training_mode': 'base',
        'gpus_per_run': 2,
        'accelerator': 'NvidiaTeslaT4',
        'timeout_seconds': 39600,
        'session_ceiling_seconds': 39600,
        'eval_test_reserved_seconds': 5400,
        'training_only': True,
        'auto_continue': False,
        'cpu_visualize': False,
        'one_session_train_eval_test': True,
        'require_full_episode_budget': True,
        'checkpoint_repo': 'duy95/uav-search-checkpoints',
        'entity': 'truongdaoanhduy-uit',
        'project': 'uav_search_target',
        'requested_episodes': 20000,
        'gpu_eval_candidate_limit': 512,
        'validation_step_targets': list(range(0, 80_000_001, 100*4096)),
        'checkpoint_selection_rule': 'fixed_100_vector_steps_plus_actual_final',
        'dense_eval_schedule': {
            'actor_snapshot_every_vector_steps': 100,
            'validation_every_vector_steps': 100,
            'validation_scenarios_per_checkpoint': 100,
            'final_test_scenarios': 4000,
            'evaluation_deterministic': True,
        },
        'warm_start_artifacts': {},
        'runs': {'matd3': {
            'id': 'v7' + uuid.uuid4().hex[:6],
            'name': 'MATD3-EVAL100-ACTORBN-PLAINCRITIC-NOORACLE-V7-E4096-U4-B8192-H256x512-S44-20261122',
        }},
    }


def main():
    import nbformat
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    archive=build_source_archive(root)
    spec=make_spec(commit)
    source=kernel_script(spec, archive, credential_source='kaggle_secrets')
    # When Kaggle supplies its real run ref, preserve actual provenance.
    source=source.replace("SPEC = "+repr(spec),
        "SPEC = "+repr(spec)+"\n"
        "import re\n"
        "_real_ref = os.environ.get('KAGGLE_KERNEL_REF')\n"
        "if _real_ref and re.fullmatch(r'[\\w-]+/[\\w-]+',_real_ref):\n"
        "    SPEC['kernel_ref'] = _real_ref\n",1)
    # This notebook has no secrets: secrets are retrieved on Kaggle at runtime.
    intro='''# MATD3 no-oracle v7: Search → REPORT → GCS

Reproduces the **train → Validation100 → select best Actor → held-out Test4000**
flow of run `18d756ef`, on the corrected **Bayesian cell-report** simulator.
It does **not** implement 3D obstacle reconstruction or DS/GCS belief fusion.

**Before running:** Kaggle notebook **Settings → Accelerator → GPU T4 x2**;
Internet **On**; in **Add-ons → Secrets** add `WANDB_API_KEY` and `HF_TOKEN`
with access enabled for this notebook. Then select **Save Version → Save & Run All**.

This notebook contains a frozen checksum-verified source tarball, no API tokens.
Its run ID is unique. Training starts from random weights; the old actor cannot
be reused because observation semantics are now v7.

Configuration: 6 UAV / 35 target / 25 obstacle; world 1000m; camera Bayesian;
E4096, U4, B8192, H256×512, Actor BN, Critic plain; 1M hybrid replay CPU75%/
GPU25%, seed44, 20k episodes, 1500 steps maximum, snapshots every 100 vector
steps, fixed 100 validation scenarios per selected checkpoint, independent 4000
final-test scenarios, on the two T4 GPUs.

**Success criterion:** W&B `fullflow/complete=true`, `validation_complete=true`,
`test_complete=true`, and a nonempty `/kaggle/working/gpu_batch_eval/`.
If GPU memory, quota, wall clock or dependencies fail, this notebook must not
claim fullflow success. The complete 20k-episode budget is mandatory.
'''
    note='''The next cell contains the complete immutable Python bootstrap.
Do not replace it with the old `18d756ef` source: that source contains
oracle-assisted target-ID reporting. Validation and test are held out.
'''
    nb=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(intro),
        nbformat.v4.new_markdown_cell(note),
        nbformat.v4.new_code_cell(source,execution_count=None,outputs=[]),
    ],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                'language_info':{'name':'python'},'source_sha256':source_sha256(archive),
                'reference_commit':commit,'research_run':spec['runs']['matd3']['id']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,args.output)
    print('NOTEBOOK',args.output,'size_bytes',args.output.stat().st_size)
    print('SOURCE_ARCHIVE_SHA256',source_sha256(archive))
    print('RUN_ID',spec['runs']['matd3']['id'],'SOURCE_COMMIT',commit)
    print('SNAPSHOT_PERIOD_VECTOR_STEPS',100,'VALIDATION_SCENARIOS',100,'FINAL_TEST_SCENARIOS',4000)

if __name__=='__main__':main()
