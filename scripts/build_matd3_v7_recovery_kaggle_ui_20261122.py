"""Pinned V7 Kaggle Eval/Test-only notebook: recover failed run 504d1b89, no training."""
from __future__ import annotations
import ast
import base64
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path
import re
import subprocess

from uav_network.cloud.kernel import kernel_script
from uav_network.cloud.source_bundle import build_source_archive, source_sha256

V7_ROOT=Path('/tmp/matd3_v7_source_fbba543')
EVIDENCE_ROOT=Path('/tmp/kaggle_372394922_diagnostics/last_artifacts')
REPO='duy95/uav-search-diagnostics'
RUN='504d1b89'
HF_OBJECT=f'deferred-gpu-eval/{RUN}/deferred-gpu-eval.tar.gz'
EXPECTED_SHA='84ac7136f5b39254bb6a0ce377781bca3b4e235a38eecdc20bd4705901cdd8a0'
COMMIT='fbba543010557deeb665b4c95081fcbfafcbfdf5'


def make_recovery_code():
    original=json.loads((EVIDENCE_ROOT/'job.json').read_text())
    summary=json.loads((EVIDENCE_ROOT/'train/matd3/summary.json').read_text())
    assert original['commit']==COMMIT
    assert original['runs']['matd3']['id']==RUN
    assert summary['episodes']==20000 and summary['episode_budget_complete'] is True
    assert summary['tracking_status']=='failed' and summary['tracking_error_type']=='ValueError'
    original_manifest=json.loads((EVIDENCE_ROOT/'train/matd3/deferred-gpu-eval/manifest.json').read_text())
    assert original_manifest['count']==original_manifest['collected']==134
    assert original_manifest['pruned']==0 and original_manifest['keep_all_distinct_versions'] is True
    assert original_manifest['snapshots'][-1]['global_step']==summary['global_step']
    source=build_source_archive(V7_ROOT)
    xz=lzma.compress(gzip.decompress(source),format=lzma.FORMAT_XZ,preset=6)
    code=kernel_script(original,source_archive=xz,credential_source='kaggle_secrets')
    assert code.count('with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:')==1
    code=code.replace('with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:',
                      'with tarfile.open(fileobj=io.BytesIO(lzma.decompress(payload)), mode="r:") as archive:',1)
    code=code.replace('import tarfile\n','import tarfile\nimport lzma\n',1)
    assert code.count('# The remote runner uses CUDA_VISIBLE_DEVICES per independent learner.')==1
    code=code.split('# The remote runner uses CUDA_VISIBLE_DEVICES per independent learner.',1)[0]
    data_bytes=gzip.compress(json.dumps(summary,sort_keys=True,allow_nan=False).encode(),compresslevel=9)
    data_b85=base64.b85encode(data_bytes).decode()
    code+=f'''
# This is RECOVERY ONLY: execute neither cloud.remote nor training.train().
# Source is pinned to Git {COMMIT}; Actor observation_version MUST remain 7.
import tarfile
from huggingface_hub import hf_hub_download
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError("Recovery requires Kaggle GPU T4x2 accelerator")
source_root=root
sys.path.insert(0,str(root))
work=Path("/kaggle/working")
folder=work/"train"/"matd3"/"deferred-gpu-eval"
folder.mkdir(parents=True,exist_ok=True)
archive_path=Path(hf_hub_download(repo_id="{REPO}",repo_type="dataset",
                   filename="{HF_OBJECT}",token=False))
h=hashlib.sha256(archive_path.read_bytes()).hexdigest()
if h != "{EXPECTED_SHA}":
    raise RuntimeError("Recovered Actor archive SHA256 differs from original Kaggle summary")
with tarfile.open(archive_path,"r:gz") as archive:
    for entry in archive:
        relative=PurePosixPath(entry.name)
        if (not entry.isfile() or relative.is_absolute()
                or len(relative.parts)!=2 or relative.parts[0]!="deferred-gpu-eval"
                or any(x in {{"..",".",""}} for x in relative.parts)
                or entry.size > 2_000_000):
            raise RuntimeError("Unsafe HF Actor archive member")
        target=work/"train"/"matd3"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with archive.extractfile(entry) as inp, target.open("wb") as out:
            shutil.copyfileobj(inp,out)
from uav_network.experiments.deferred_gpu_evaluation import verify_manifest
verified=verify_manifest(folder)
if verified["count"] != 134 or verified["collected"] != 134 or verified["pruned"] != 0:
    raise RuntimeError("Recovered Actor checkpoint archive incomplete")
summary_data=json.loads(gzip.decompress(base64.b85decode("{data_b85}")))
if summary_data["global_step"] != 29368320 or summary_data["episodes"] != 20000:
    raise RuntimeError("V7 original Kaggle training summary mismatch")
(work/"train"/"matd3"/"summary.json").write_text(json.dumps(summary_data,indent=2))
from uav_network.configuration import Config,_apply
from uav_network.environment.observations import observation_version
import torch
last=torch.load(folder/verified["snapshots"][-1]["file"],weights_only=True,map_location="cpu")
cfg=Config();_apply(cfg,last["config"]);cfg.validate()
if observation_version(cfg)!=7 or last["observation_version"]!=7:
    raise RuntimeError("Recovery cannot evaluate V7 checkpoint under V8 or V9 dynamics")
print("RECOVERY_VERIFIED",RUN,verified["count"],"actor snapshots, no training",flush=True)
# Val100 (prespecified real checkpoint grid) -> GPU best -> independent Test4000.
# Parent supervisor resumes original run ID {RUN} on W&B; NO new training data.
_run_command=[str(python),"-m","uav_network.experiments.one_session",
    "--spec",str(spec_file),"--policies-per-batch","2",
    "--eval-world-pool-size","0","--max-eval-worlds-per-gpu","256",
    "--max-test-worlds-per-gpu","256","--test-batch-size","0"]
_recovery_log=work/"recovery-fullflow-process.log"
try:
    subprocess.run(_run_command,cwd=root,check=True)
    print("RECOVERED_V7_VALIDATION100_TEST4000_COMPLETE",flush=True)
finally:
    # Append complete non-sensitive validation/test outputs to public HF.
    # This is separate from the original 504d1b89 diagnostic namespace.
    import json as _json
    from uav_network.experiments.public_diagnostic_store import PublicHubDiagnosticUploader
    from uav_network.configuration import load_config
    from uav_network.experiments.evidence import secret_values
    import tarfile as _tarfile
    _logs=work/"v7-recovery-forensic"
    _logs.mkdir(parents=True,exist_ok=True)
    _part=_logs/"recovery-eval-logs.tar.gz"
    _sources=[]
    for _path in (work/"gpu_batch_eval").rglob("*"):
        if _path.is_file() and _path.suffix in (".json",".jsonl",".log") and _path.stat().st_size < 100000000:
            _sources.append(_path)
    for _path in (work/"one-session-train-eval-test-report.json",):
        if _path.is_file():_sources.append(_path)
    with _tarfile.open(_part,"w:gz") as _tar:
        for _path in sorted(set(_sources)):
            _tar.add(_path,arcname=_path.relative_to(work).as_posix(),recursive=False)
    try:
        _cfg=load_config(str(root/"configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml"))
        _cfg.tracking.credentials_file="key.txt"
        _uploader=PublicHubDiagnosticUploader(_cfg,_logs,"{RUN}-recovery",verify_download=True)
        _uploader.save(str(_part),base_path=str(_logs),policy="now")
        print("V7_RECOVERY_HF_LOGS",_uploader.url,flush=True)
    except Exception as _upload_error:
        print("V7_RECOVERY_HF_LOG_UPLOAD_FAILED",type(_upload_error).__name__,str(_upload_error)[:200],flush=True)
'''
    ast.parse(code)
    return code,source,xz,summary


def build(output):
    import nbformat as nb
    code,src,xz,summary=make_recovery_code()
    markdown=f'''# MATD3 V7 Recovery-only · 504d1b89 · 134 Actor snapshots

**No retraining.** The original account-1 Kaggle notebook completed all **20,000 episodes** and then failed at premature best-checkpoint publication. Its Actor snapshots survive on Hugging Face. This notebook retrieves and SHA-verifies **134 immutable V7 Actors** and runs **Validation100 → best Actor → independent Test4000**, resuming the original W&B run if W&B permits. Never use V8/V9 source for these V7 checkpoints.

Requires Kaggle GPU **T4 ×2**, Internet, Kaggle Secrets **WANDB_API_KEY** and **HF_TOKEN**, and the original user-owned Hugging Face dataset. W&B resume is attempted; if the platform rejects it, the notebook fails without fabricating metrics. The recovery report and diagnostic output remain under `/kaggle/working`. Public log upload is attempted independently and reports its outcome.
'''
    preflight='''# RECOVERY ONLY — checkpoint compatibility must remain V7.
RECOVERY_RUN_ID = "504d1b89"
VAL_EPISODES = 100
FINAL_TEST_EPISODES = 4000
GPU_BATCH_CHECKPOINTS = 2
print("No training, recovered pinned source V7, validation 100 → best → test 4000")
'''
    doc=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(markdown),nb.v4.new_code_cell(preflight),nb.v4.new_code_cell(code)],
        metadata={'source_sha256':source_sha256(xz),'v7_gzip_sha256':hashlib.sha256(src).hexdigest(),
                  'v7_commit':COMMIT,'recover_run':RUN,'source_version':7,'workflow':'eval_test_only_no_train',
                  'reference_kaggle_script_version':'372394922'})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    nb.write(doc,output)
    size=output.stat().st_size
    source_size=sum(len(c.source.encode()) for c in doc.cells if c.cell_type=='code')
    if max(size,source_size)>=1_000_000:raise ValueError(f'Recovery notebook too large: {size}, {source_size}')
    return {'file':str(output),'bytes':size,'python_source_bytes':source_size,
            'source_sha256':doc.metadata['source_sha256'],'v7_commit':COMMIT}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(build(p.parse_args().output),indent=2))
