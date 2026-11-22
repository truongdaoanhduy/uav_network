"""Build a self-contained, eval-only Kaggle notebook from the exact V9 source.

No training, no credentials embedded, no mutation of source W&B/HF checkpoint.
The original V9 source bytes are SHA256-verified before embedding; the only
additional execution code is the isolated benchmark runner.
"""
from __future__ import annotations

import argparse
import ast
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'notebooks/MATD3-EVAL100-ACTORBN-PLAINCRITIC-NOORACLE-V9-E4096-U4-B8192-H256x512-S44-KAGGLE-UI.ipynb'
RUNNER = ROOT / 'scripts/evaltest_baseline_batch_from_hf.py'
OUTPUT = ROOT / 'notebooks/MATD3-NOORACLE-V9-e079eace-REVAL100-TEST4000-BASELINE-BATCH81-T4x2-KAGGLE.ipynb'

MARKDOWN = '''# V9 `e079eace` — Eval100/Test4000 độc lập, batch giống `18d756ef`

**Eval/Test ONLY. Không train lại MATD3. Không resume W&B run `e079eace`.**

Source khoa học: **đúng archive Python của notebook NOORACLE-V9 gốc**, không dùng phiên bản sparsefast. Snapshot Actor được tải từ Hugging Face dataset:
`duy95/uav-search-diagnostics/deferred-gpu-eval/e079eace/deferred-gpu-eval.tar.gz`.

**Baseline-style:** Mỗi checkpoint có 100 fixed validation seeds; GPU0 nhận checkpoint thứ 1,3,5... và GPU1 nhận 2,4,6...; gộp tối đa 81 checkpoint/GPU/batch, mỗi checkpoint đúng 100 Warp worlds. Nếu nguồn có 74 checkpoint thì 37×100=3.700 worlds/GPU trong một batch. **8192 là giới hạn, không phải 8192 GPU worlds được cấp phát cố định.**

**Test4000:** chỉ Best Actor được chọn từ Val100, test seeds độc lập. 2 GPU nhận 2.000 worlds/GPU, một batch/GPU nếu còn đủ bộ nhớ. Không nạp Critic/replay/optimizer, không thực hiện backward.

**Hướng dẫn:** Chọn Kaggle Accelerator **GPU T4×2**, Internet **On**, cấp quyền Secrets `WANDB_API_KEY` và `HF_TOKEN`, rồi Save & Run All. Notebook tự tải đúng archive snapshot, kiểm chứng SHA256 mọi Actor và chạy toàn bộ flow. Không cần key.txt trên Kaggle.

**Đo hiệu năng:** Saved Kaggle outputs chứa `benchmark_summary.json`, `benchmark_status.json`, `gpu_batch_eval/validation_results.csv`, `gpu_batch_eval/final_test.json`, per-checkpoint JSONs. Tạo W&B **run ID mới**, ghi Eval convergence và Test summary riêng, không sửa W&B `e079eace`; cũng có thể upload log hoàn tất vào prefix độc lập trên HF dataset.

**Giới hạn thực tế:** 3.700 V9 Warp worlds/GPU chưa được benchmark VRAM trên T4. Nếu CUDA OOM, chỉnh `CHECKPOINTS_PER_GPU_BATCH` thành 32 / 16 / 8 / 2, chạy một notebook mới để so throughput thực tế; không tự hạ batch âm thầm. Không khẳng định nhanh hơn trước khi có kết quả GPU.
'''

PREFLIGHT = '''# Kaggle UI knobs: independent Eval/Test of the already-trained V9 run.
SOURCE_RUN_ID = "e079eace"
HF_DATASET_REPO = "duy95/uav-search-diagnostics"
CHECKPOINTS_PER_GPU_BATCH = 81
VALIDATION_EPISODES = 100
MAX_WARP_EVAL_WORLDS_PER_GPU = 8192
MAX_WARP_TEST_WORLDS_PER_GPU = 4096
FINAL_TEST_EPISODES = 4000
UPLOAD_BENCHMARK_TO_HF = True

import torch
from kaggle_secrets import UserSecretsClient
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Chọn Kaggle GPU T4 x2 để chạy Eval/Test độc lập")
for index in (0,1):
    print("CUDA GPU",index,torch.cuda.get_device_name(index))
if not (1 <= CHECKPOINTS_PER_GPU_BATCH <= 81 and
        CHECKPOINTS_PER_GPU_BATCH * VALIDATION_EPISODES <= MAX_WARP_EVAL_WORLDS_PER_GPU):
    raise ValueError("Validation worlds vượt quá giới hạn mỗi GPU")
if VALIDATION_EPISODES != 100 or FINAL_TEST_EPISODES != 4000:
    raise ValueError("Giữ nguyên Val100 và Test4000 để so sánh với V9")
for secret_name in ("WANDB_API_KEY","HF_TOKEN"):
    try:
        value = UserSecretsClient().get_secret(secret_name)
    except Exception:
        raise RuntimeError(f"Chưa cấp Kaggle Secret: {secret_name}") from None
    if not isinstance(value,str) or not value.strip():
        raise RuntimeError(f"Kaggle Secret rỗng: {secret_name}")
    del value
print("V9 Eval/Test-only preflight PASS; max Eval",CHECKPOINTS_PER_GPU_BATCH * 100,
      "worlds/GPU per batch; Test max",MAX_WARP_TEST_WORLDS_PER_GPU)
'''


def read_source_bundle(reference: Path):
    original = nbformat.read(reference, as_version=4)
    source = original.cells[2].source
    matched = re.search(r'payload\s*=\s*base64\.b85decode\(([\'"])(.*?)\1\)',source,re.S)
    digest = re.search(r'if hashlib\.sha256\(payload\)\.hexdigest\(\) != "([0-9a-f]{64})":',source)
    if not matched or not digest:
        raise RuntimeError("Original V9 notebook does not contain a verifiable source archive")
    encoded = matched.group(2)
    archive = base64.b85decode(encoded)
    if hashlib.sha256(archive).hexdigest() != digest.group(1):
        raise ValueError("Original V9 source archive changed SHA256")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as tf:
        files={x.name for x in tf if x.isfile()}
    expected = {'uav_network/experiments/deferred_gpu_evaluation.py',
                'uav_network/experiments/deferred_gpu_batched.py',
                'uav_network/experiments/one_session_wandb.py',
                'configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml'}
    if not expected.issubset(files):
        raise ValueError("Source package missing V9 GPU evaluator or original config")
    return original, encoded, digest.group(1), len(files)


def build(reference: Path = REFERENCE):
    original, encoded, checksum, file_count = read_source_bundle(reference)
    benchmark_script = RUNNER.read_text()
    ast.parse(benchmark_script)
    body = '''import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

# Exact source archive embedded in the ORIGINAL V9 training notebook.
payload = base64.b85decode(''' + repr(encoded) + ''')
if hashlib.sha256(payload).hexdigest() != ''' + repr(checksum) + ''':
    raise RuntimeError("V9 source archive SHA256 mismatch")
root = Path("/tmp/uav_network_v9_evalonly")
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as archive:
    total = 0
    for entry in archive:
        relative = PurePosixPath(entry.name)
        if (not entry.isfile() or relative.is_absolute() or
                any(part in {"..", ".", ""} for part in relative.parts) or
                entry.size > 2_000_000):
            raise RuntimeError("Unsafe bundled V9 source entry")
        total += entry.size
        if total > 40_000_000:
            raise RuntimeError("Bundled source exceeds code budget")
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(entry) as stream, destination.open("wb") as target:
            shutil.copyfileobj(stream, target)

# Match V9 notebook dependency versions, no training command.
python = root / ".venv/bin/python"
subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                "--without-pip", str(root / ".venv")],check=True)
subprocess.run([str(python), "-m", "pip", "install", "-e", ".[cloud]",
                "-r", "scripts/requirements-network.txt"],cwd=root,check=True)

# Secrets are never embedded in this notebook or recorded in W&B metrics.
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
for key in ("WANDB_API_KEY", "HF_TOKEN"):
    value=secrets.get_secret(key)
    if not isinstance(value,str) or not value.strip():
        raise RuntimeError("Missing Kaggle Secret " + key)
    os.environ[key]=value.strip()
    del value
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ["WANDB_ENTITY"]="truongdaoanhduy-uit"
os.environ["WANDB_PROJECT"]="uav_search_target"
os.environ["WANDB_MODE"]="online"
os.environ["WANDB_SILENT"]="true"

# Save runner as an independent script alongside the frozen original source.
runner_path = root / "scripts/evaltest_baseline_batch_from_hf.py"
runner_path.write_text(''' + repr(benchmark_script) + ''')
work = Path("/kaggle/working/v9-e079eace-baseline-batch81-evaltest")
cmd = [str(python),str(runner_path),
       "--source-run-id",SOURCE_RUN_ID,
       "--hf-dataset-repo",HF_DATASET_REPO,
       "--output",str(work),
       "--validation-episodes",str(VALIDATION_EPISODES),
       "--test-episodes",str(FINAL_TEST_EPISODES),
       "--policies-per-batch",str(CHECKPOINTS_PER_GPU_BATCH),
       "--max-eval-worlds-per-gpu",str(MAX_WARP_EVAL_WORLDS_PER_GPU),
       "--max-test-worlds-per-gpu",str(MAX_WARP_TEST_WORLDS_PER_GPU)]
if UPLOAD_BENCHMARK_TO_HF:
    cmd.append("--hf-upload")
print("EVALTEST_BENCHMARK_SOURCE_RUN",SOURCE_RUN_ID,flush=True)
print("EVALTEST_BENCHMARK_ORIGINAL_CODE_SHA256",hashlib.sha256(payload).hexdigest(),flush=True)
subprocess.run(cmd,cwd=root,check=True)
print("V9_EVALTEST_BENCHMARK_COMPLETE",work / "benchmark_summary.json",flush=True)
'''
    ast.parse(PREFLIGHT)
    ast.parse(body)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(MARKDOWN),
        nbformat.v4.new_code_cell(PREFLIGHT),
        nbformat.v4.new_code_cell(body),
    ], metadata={
        'kernelspec': deepcopy(original.metadata.get('kernelspec',{'name':'python3','display_name':'Python 3'})),
        'language_info': deepcopy(original.metadata.get('language_info', {'name':'python'})),
        'benchmark_of_wandb_run': 'e079eace',
        'baseline_comparison_wandb_run': '18d756ef',
        'original_v9_embedded_source_sha256': checksum,
        'original_v9_archive_file_count': file_count,
        'no_training': True,
        'does_not_modify_source_wandb_run': True,
        'validation_episodes_per_checkpoint': 100,
        'final_test_episodes': 4000,
    })
    nbformat.validate(notebook)
    rendered = nbformat.writes(notebook).encode()
    if len(rendered) >= 1_000_000 or len(body.encode()) >= 1_000_000:
        raise ValueError("Kaggle notebook or Python cell exceeds 1 MB")
    return notebook


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,default=REFERENCE)
    ap.add_argument('--output',type=Path,default=OUTPUT)
    args=ap.parse_args()
    notebook=build(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook,args.output)
    print('BUILT',args.output,'bytes',args.output.stat().st_size,
          'source_sha256',notebook.metadata['original_v9_embedded_source_sha256'],flush=True)

if __name__=='__main__':
    main()
