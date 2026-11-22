"""Build a self-contained Kaggle 3-cell notebook matching therwed/notebook5051433b29.

No API keys, credentials, env files or remote source checkout are bundled.
Real full 20k training + every-snapshot deterministic Val100 + heldout Test1000.
A clean Kaggle GPU T4×2 + Kaggle Secrets connection is required to run.
"""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import ast
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import nbformat

from uav_network.cloud.source_bundle import build_source_archive, source_sha256
from uav_network.cloud.kernel import kernel_script
from scripts.launch_nooracle_map800_tx08_training_with_hf_forensics_20261123 import FORENSIC_PROFILE

ROOT=Path(__file__).resolve().parents[1]
SOURCE_REF='therwed/notebook5051433b29'
CONFIG='configs/matd3-nooracle-posterior-only-map800-h256x256-orthogonal-compile-tx08-hf-fullflow-20261123.yaml'
NAME='MATD3-POSTERIORONLY-MAP800-TX08-T4X2-FULLFLOW-UNDER1MB-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME


def minimal_runtime_source_archive():
    """Verified immutable runtime-only tar; fixes Kaggle's strict 1 MB source cap.

    The general-purpose cloud archive includes >200 nonruntime test files,
    every earlier experiment YAML and their fixtures. Embedding all of these
    expands the notebook to ~1.2 MB. Only omit tests and irrelevant YAMLs:
    keep ALL algorithm, environment, Warp, GPU evaluation, telemetry, HF/W&B
    code, pinned UavNetSim and required runtime scripts without modification.
    """
    original=build_source_archive(ROOT)
    original_names=set()
    included=set()
    raw=io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(original),mode='r:gz') as source, \
            tarfile.open(fileobj=raw,mode='w') as reduced:
        for entry in source:
            original_names.add(entry.name)
            if entry.name.startswith('tests/'):
                continue
            if entry.name.startswith('configs/') and entry.name!=CONFIG:
                continue
            if not entry.isfile():
                raise RuntimeError('Unexpected source archive member')
            data=source.extractfile(entry).read()
            reduced.addfile(entry,io.BytesIO(data))
            included.add(entry.name)
    required={CONFIG,'pyproject.toml','scripts/requirements-network.txt',
        'scripts/upstream-revision.txt','uav_network/environment/gpu/bayesian.py',
        'uav_network/experiments/training.py','uav_network/experiments/one_session.py',
        'uav_network/experiments/deferred_gpu_evaluation.py',
        'uav_network/experiments/forensic_transitions.py',
        'third_party/UavNetSim/simulator/simulator.py'}
    if required-included:
        raise RuntimeError('Runtime bundle missing required source: '+repr(required-included))
    if not included.issubset(original_names):
        raise RuntimeError('Runtime bundle added an unauthorized source file')
    payload=gzip.compress(raw.getvalue(),compresslevel=9,mtime=0)
    if len(payload)>640_000:
        raise RuntimeError('Kaggle runtime archive unexpectedly exceeds 640 KB')
    return payload


def make_spec(commit):
    return {
        'submitted_at_utc':'set-on-kaggle-at-runtime',
        'commit':commit,
        'account':'kaggle_ui',
        'kernel_ref':'kaggle-ui/imported-posterior-only-tx08',
        'stage':'train',
        'config':CONFIG,
        'training_mode':'base',
        'gpus_per_run':2,
        'accelerator':'NvidiaTeslaT4',
        'timeout_seconds':43200,
        'session_ceiling_seconds':43200,
        'eval_test_reserved_seconds':3600,
        'training_only':True,
        'auto_continue':False,
        'cpu_visualize':False,
        'one_session_train_eval_test':True,
        'require_full_episode_budget':True,
        'episode_budget_only':True,
        # Never silently reduce 100-episode evaluation to a 32-policy shortlist.
        'validate_all_snapshots':True,
        'gpu_eval_candidate_limit':0,
        'checkpoint_selection_rule':'EVERY_IMMUTABLE_ACTOR_SNAPSHOT_no_shortlist',
        'checkpoint_repo':'duy95/uav-search-checkpoints',
        'entity':'truongdaoanhduy-uit',
        'project':'uav_search_target',
        'requested_episodes':20000,
        'dense_eval_schedule':{
            'actor_snapshot_every_vector_steps':100,
            'validation_every_vector_steps':100,
            'validation_scenarios_per_checkpoint':100,
            'final_test_scenarios':1000,
            'evaluation_deterministic':True,
        },
        'overrides':[
            'runtime.algorithm=matd3',
            'runtime.seed=44',
            'runtime.episodes=20000',
            'runtime.max_train_seconds=0',
            'runtime.num_envs=4096',
            'runtime.compile_models=true',
            'matd3.updates_per_step=4',
            'matd3.batch_size=8192',
            'matd3.initialization=orthogonal',
            'matd3.critic_batch_norm=false',
            'matd3.actor_batch_norm=true',
            'world.motion_safety=reject',
            'world.map_size_m=800',
            'sensor.grid_cell_m=20',
            'sensor.confirmation_threshold=0.99',
            'sensor.confirmation_mode=posterior_only',
            'sensor.confirmation_min_likelihood_ratio=0',
            'sensor.report_cell_hypotheses=true',
            'network.tx_power_max_w=0.8',
            'network.power_control=learned',
            'reward.communication_reference_tx_power_w=0.4',
            'tracking.deferred_gpu_eval_enabled=true',
            'tracking.deferred_gpu_eval_keep_all=true',
            'tracking.deferred_gpu_eval_collection_stride=100',
            'tracking.online_gpu_eval_enabled=false',
            'tracking.periodic_evaluation_episodes=100',
            'pql.enabled=false',
        ],
        'warm_start_artifacts':{},
        'runs':{'matd3':{'id':'resolved_at_runtime',
            'name':'MATD3-POSTERIORONLY-MAP800-TX08-E4096-U4-B8192-H256x256-S44-NEW'}},
    }


INTRO='''# MATD3 · Posterior-only (KHÔNG LR) · MAP800 · TX0,8 W · T4×2

**Notebook Full Flow:** đúng định dạng 3 cell như Kaggle tham chiếu
[`therwed/notebook5051433b29`](https://www.kaggle.com/code/therwed/notebook5051433b29):
**Giới thiệu → thông số/cấp quyền → source nhúng + Train → GPU Validation → Best Actor → GPU Test.**
Đây là source code đóng băng, SHA-256 verified; **không download GitHub khi chạy**, không chứa token.

**Science mới:** 800×800 m, 40×40 ô Bayes (20m/ô), 6 UAV, 35 mục tiêu, 25 vật cản,
độ cao 0–150m, FoV 90°, Pd/Pf thay đổi theo độ cao. Xác nhận khi **mẫu camera DƯƠNG tính
và Posterior ≥ 0,99**; **không có LR≥4, không có trần xác nhận 50m**. Bayes vẫn cập nhật
bằng Pd/Pf. REPORT là ID ô đã quan sát, không dùng tọa độ target thật trong Actor.
*Rủi ro*: ở 150m, Pf=0,3 nên false confirmation có thể tăng; theo dõi log đầy đủ.

**Mạng và năng lượng:** Actor học TX liên tục **0,001–0,8 W** và lựa chọn im lặng /
relay qua UAV khác / GCS. TX/RX và bay cùng tiêu thụ pin. Reward chuẩn hoá RF theo
mốc 0,4W cố định, không vô tình làm năng lượng cao thành rẻ khi tăng Pmax.

**Train thật:** MATD3 ActorBN/PlainCritic H256×256, orthogonal, torch.compile Inductor,
E4096/U4/B8192, replay 1M (75% CPU/25% GPU), seed44, 20.000 episodes, tối đa
1.500 step/episode. Tạo Actor snapshot đầu, **mỗi 100 vector steps** và cuối.

**GPU Validation:** Đánh giá *mọi* checkpoint, cố định **100 seed độc lập/checkpoint**,
2 GPU chạy song song, tối đa **81 checkpoint×100=8.100 episode** trong **8.192 env
vật lý/GPU**; checkpoint phân theo lô liên tiếp, cân bằng hai GPU khi số checkpoint ít.
X-axis W&B là `gpu_eval_periodic/checkpoint_global_step`; Test không trộn vào đường Val.
**Final Test:** Best Actor duy nhất, **1.000 episode** seed riêng, 500 episode/GPU.

**Logs sâu:** W&B online và Hugging Face public diagnostic archives có SHA readback;
RF histogram, TX/RX Joule, relay/direct, buffer, collisions, PHY/SINR, false positives,
Actor gradients/pre-tanh/Q, replay minibatches, selected-world pre/post full belief,
training-step reward/progress/optimizer và eval/test metrics. Không ghi toàn bộ
4.096 env mỗi bước vì chi phí dữ liệu; raw sample worlds 0,1 mỗi 16 vector steps.

### Cách chạy

1. Kaggle → **Import Notebook**, chọn file `.ipynb` này.
2. Settings → Accelerator **GPU T4 x2**; Internet **On**.
3. Add-ons → Secrets: `WANDB_API_KEY` và `HF_TOKEN`, cấp quyền notebook.
4. **Save Version → Save & Run All**. Cell 1 tự nhận `KAGGLE_KERNEL_REF` khi có;
   không cần sửa thủ công notebook slug.

**Giới hạn quan trọng:** chưa chạy CUDA E4096 của bản posterior-only trên máy này.
8192 worlds/GPU yêu cầu VRAM thực tế lớn, có thể OOM trên T4; notebook không tự
hạ số env và giả vờ đạt yêu cầu. Kaggle có giới hạn phiên/giờ chạy; nếu không đủ
20.000 episodes hoặc chưa xong Eval/Test, trạng thái **KHÔNG** được đánh dấu hoàn tất.
Checkpoint version **11**, không tiếp tục weights v8/v9/v10 hay run `18d756ef`.
'''

CONFIG_CELL='''# User-editable Kaggle preflight — tương tự notebook5051433b29.
import os,re,torch
TRAIN_SEED = 44
TRAIN_EPISODES = 20000
ENVS = 4096
UPDATES_PER_VECTOR_STEP = 4
BATCH = 8192
VALIDATION_EPISODES = 100
FINAL_TEST_EPISODES = 1000
MAX_WARP_EVAL_WORLDS_PER_GPU = 8192
CHECKPOINTS_PER_GPU_BATCH = 81  # 81*100=8100 real episodes, 92 masked filler slots
POSTERIOR_THRESHOLD = .99
TX_POWER_MAX_W = .8
NOTEBOOK_REF = os.environ.get('KAGGLE_KERNEL_REF','longlonthiu/notebookbbd0aec150')
if not re.fullmatch(r'[\\w-]+/[\\w-]+',NOTEBOOK_REF):
    raise ValueError('KAGGLE_KERNEL_REF must look like username/notebook-slug')
assert (TRAIN_EPISODES,ENVS,UPDATES_PER_VECTOR_STEP,BATCH)==(20000,4096,4,8192)
assert VALIDATION_EPISODES==100 and FINAL_TEST_EPISODES==1000
assert CHECKPOINTS_PER_GPU_BATCH * VALIDATION_EPISODES <= MAX_WARP_EVAL_WORLDS_PER_GPU
if not torch.cuda.is_available() or torch.cuda.device_count()<2:
    raise RuntimeError('Chọn Kaggle Accelerator: GPU T4 x2 (không fallback CPU).')
for i in range(2):print('CUDA GPU',i,torch.cuda.get_device_name(i),flush=True)
from kaggle_secrets import UserSecretsClient
for name in ('WANDB_API_KEY','HF_TOKEN'):
    try:value=UserSecretsClient().get_secret(name)
    except Exception:raise RuntimeError('Thiếu hoặc chưa cấp quyền Kaggle Secret '+name) from None
    if not isinstance(value,str) or not value.strip():raise RuntimeError('Kaggle Secret rỗng: '+name)
    del value
print('PREFLIGHT_OK | Posterior only | No LR/50m | TX<=0.8W | 20k Train/Val100/Test1000',flush=True)
'''


def build_notebook():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    spec=make_spec(commit)
    archive=minimal_runtime_source_archive()
    src_sha=source_sha256(archive)
    script=kernel_script(spec,source_archive=archive,credential_source='kaggle_secrets')
    old='SPEC = '+repr(spec)+'\n'
    assert script.count(old)==1, 'Cannot locate pinned SPEC literal'
    spec_runtime='''# Runtime unique W&B run ID / actual Kaggle reference, no stale shared account IDs.
from uuid import uuid4
SPEC['kernel_ref']=NOTEBOOK_REF
SPEC['account']='kaggle_ui_'+NOTEBOOK_REF.split('/',1)[0]
SPEC['submitted_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
rid=uuid4().hex[:8]
SPEC['runs']={'matd3':{'id':rid,'name':f'MATD3-POSTERIORONLY-MAP800-TX08-E4096-U4-B8192-H256x256-S44-{rid}'}}
SPEC['source_bundle_sha256']='''+repr(src_sha)+'''
SPEC['full_flow_protocol']={'all_checkpoints':True,'validation_episodes':100,
    'validation_pool_per_gpu':8192,'checkpoint_assignment':'contiguous_adaptive',
    'test_episodes':1000,'test_per_gpu':500,'confirmation':'positive+posterior>=0.99',
    'no_lr_threshold':True,'tx_power_cap_w':.8}
print('NEW_RUN_ID',rid,'WandB',f'https://wandb.ai/{SPEC["entity"]}/{SPEC["project"]}/runs/{rid}',flush=True)
'''
    script=script.replace(old,old+spec_runtime,1)
    # All external logging switches must be active BEFORE remote train and
    # one_session evaluation subprocesses, and never leak credential values.
    prefix='''# Deep W&B/Hugging Face forensic profile; HF upload needs Kaggle Secrets.
for key,value in '''+repr(FORENSIC_PROFILE)+'''.items():
    os.environ[key]=str(value)
os.environ['UAV_NO_CBF_FULLFLOW']='true'
os.environ['UAV_SOURCE_SHA256']='''+repr(src_sha)+'''
print('FULLFLOW_SOURCE_SHA256',os.environ['UAV_SOURCE_SHA256'],flush=True)
'''
    marker='# Kaggle runs only this Python bootstrap; source is inside the payload.\n'
    assert script.count(marker)==1
    script=script.replace(marker,prefix+marker,1)
    # Critical separate check: config is loaded from the *embedded archive*,
    # not the original old Kaggle code or an external GitHub branch at runtime.
    marker='spec_file = Path("/kaggle/working/job.json")'
    assert script.count(marker)==1
    inspect='''# Verify actual new scientific config before spending any GPU time on 20k episodes.
subprocess.run([str(python),'-c',
    'from uav_network.configuration import load_config; from uav_network.environment.observations import observation_version; '
    'c=load_config('+repr(CONFIG)+'); '
    'assert c.sensor.confirmation_mode=="posterior_only" and c.sensor.confirmation_min_likelihood_ratio==0; '
    'assert c.sensor.confirmation_threshold==.99 and c.network.tx_power_max_w==.8; '
    'assert c.world.map_size_m==800 and c.sensor.grid_cell_m==20; '
    'assert c.matd3.hidden_sizes==[256,256] and c.runtime.compile_models; '
    'assert observation_version(c)==11; '
    'print("BAYES_POSTERIOR_ONLY_VERSION11_CONFIG_OK",flush=True)'],cwd=root,check=True)
'''
    script=script.replace(marker,inspect+marker,1)
    old_run='''        "--spec", str(spec_file), "--policies-per-batch", "2",
        "--eval-world-pool-size", "0",
        "--max-eval-worlds-per-gpu", "256",
        "--max-test-worlds-per-gpu", "256",
        "--test-batch-size", "0"], cwd=root, check=True)'''
    new_run='''        "--spec", str(spec_file), "--policies-per-batch", str(CHECKPOINTS_PER_GPU_BATCH),
        "--checkpoint-assignment", "contiguous_adaptive",
        "--eval-world-pool-size", str(MAX_WARP_EVAL_WORLDS_PER_GPU),
        "--max-eval-worlds-per-gpu", "0",
        "--max-test-worlds-per-gpu", str(MAX_WARP_EVAL_WORLDS_PER_GPU),
        "--test-batch-size", str(FINAL_TEST_EPISODES // 2)], cwd=root, check=True)'''
    assert script.count(old_run)==1,'Upstream full-flow call arguments changed; refuse to silently downscale'
    script=script.replace(old_run,new_run,1)
    script=script.replace('MATD3_NO_ORACLE_TRAIN_VALIDATION100_TEST4000_COMPLETE',
                          'MATD3_POSTERIORONLY_TRAIN_VAL100_TEST1000_COMPLETE')
    # Ensure users can distinguish incomplete Kaggle sessions from success.
    script+='''\nreport=Path('/kaggle/working/one-session-train-eval-test-report.json')
if not report.is_file():raise RuntimeError('Missing full-flow completion report')
actual=json.loads(report.read_text())
if actual.get('gpu_val_status')!='complete' or actual.get('gpu_test_status')!='complete':
    raise RuntimeError('GPU Validation/Test not both complete; do not claim success: '+repr(actual))
print('POSTERIORONLY_TX08_FULLFLOW_COMPLETE',json.dumps(actual,indent=2),flush=True)
'''
    ast.parse(CONFIG_CELL)
    ast.parse(script)
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(INTRO),
        nbformat.v4.new_code_cell(CONFIG_CELL,execution_count=None,outputs=[]),
        nbformat.v4.new_code_cell(script,execution_count=None,outputs=[])],metadata={
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'},
        'source_sha256':src_sha,
        'reference_commit':commit,
        'ref_notebook':SOURCE_REF,
        'science_config':CONFIG,
        'full_20000_training':True,
        'validated_checkpoint_bank':'all',
        'actor_critic_hidden_sizes':[256,256],
        'num_envs':4096,
        'updates_per_step':4,
        'batch_size':8192,
        'confirmation_mode':'posterior_only',
        'likelihood_ratio_gate':False,
        'posterior_threshold':.99,
        'tx_power_max_w':.8,
        'two_gpu_eval_pool_each':8192,
        'validation_episodes_per_checkpoint':100,
        'test_episodes':1000,
        'deep_hf_forensics':True})
    nbformat.validate(n)
    # Budget strictly below Kaggle 1 MB source restriction; leave >150 KB margin.
    if len(n.cells[2].source.encode('utf-8'))>=825_000 or len(nbformat.writes(n).encode('utf-8'))>=850_000:
        raise ValueError('Kaggle kernel source exceeds safe 850 KB notebook limit')
    return n


def main():
    n=build_notebook()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUT)
    print('KAGGLE_NOTEBOOK_READY',OUT,'BYTES',OUT.stat().st_size,
          'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest(),
          'ARCHIVE_SHA256',n.metadata['source_sha256'],flush=True)


if __name__=='__main__':main()
