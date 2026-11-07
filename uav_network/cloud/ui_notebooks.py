"""Self-contained manual Kaggle notebooks for the declared independent baseline."""
from hashlib import sha256
import json

import nbformat

from .kernel import kernel_script

PROFILE = "configs/comparison-independent-baseline-xyz-sensing.yaml"
STAGE_START = 'if SPEC["stage"] in {"visualize", "archive"}:'

PREFLIGHT = r'''# Dùng URL thật của notebook Kaggle bạn vừa tạo.
NOTEBOOK_REF = "YOUR_KAGGLE_USERNAME/YOUR_NOTEBOOK_SLUG"
TRAIN_SEED = __SEED__  # Giữ cùng seed cho cặp MASAC/MATD3; đổi seed cho lần train độc lập.

import re
if not re.fullmatch(r"[\w-]+/[\w-]+", NOTEBOOK_REF):
    raise ValueError("NOTEBOOK_REF phải có dạng username/notebook-slug")
owner, slug = NOTEBOOK_REF.split("/", 1)
if owner == "YOUR_KAGGLE_USERNAME" or slug == "YOUR_NOTEBOOK_SLUG":
    raise ValueError("Hãy sửa NOTEBOOK_REF theo URL thật của notebook Kaggle")
if type(TRAIN_SEED) is not int or not 0 <= TRAIN_SEED < 2**31:
    raise ValueError("TRAIN_SEED phải là số nguyên trong [0, 2**31)")

import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Chọn Accelerator GPU T4 x2 trên Kaggle")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
from kaggle_secrets import UserSecretsClient
client = UserSecretsClient()
for name in ("WANDB_API_KEY", "HF_TOKEN"):
    try:
        value = client.get_secret(name)
    except Exception:
        raise RuntimeError(f"Thiếu hoặc chưa cấp quyền Secret {name}") from None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Secret {name} rỗng")
    del value
print("Preflight OK; seed =", TRAIN_SEED)
print("Actor/critic độc lập; plain MLP; NOCHUNK; BASE; profile 1500 bước")
'''


def build_independent_notebook(algorithm, archive, source_commit, *, seed=44):
    """Build a notebook without contacting Kaggle/W&B or reading credentials."""
    if algorithm not in {"masac", "matd3"}:
        raise ValueError("algorithm must be masac or matd3")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    tags = [
        "uav-network", "independent-baseline", "plain-mlp", "independent-actors",
        "independent-critics", "no-critic-chunks", "scenario-v3", "steps1500",
        "env4096", "batch8192", "replay1m", "cpu75-gpu25", "fp32-eager",
        "training-only", "hf-best-only", "xyz", "bayesian-sensing",
    ]
    tags += (["alpha-dual-loss", "mask-aware-discrete-entropy"] if algorithm == "masac"
             else ["matd3-init-default"])
    spec = {
        "submitted_at_utc": "assigned-at-runtime", "commit": source_commit,
        "kernel_ref": "YOUR_KAGGLE_USERNAME/YOUR_NOTEBOOK_SLUG", "account": "kaggle_ui",
        "timeout_seconds": 39600, "warm_start_artifacts": {},
        "entity": "truongdaoanhduy-uit", "project": "uav_search_target", "stage": "train",
        "config": PROFILE, "overrides": [f"runtime.algorithm={algorithm}", "pql.enabled=false"],
        "runs": {algorithm: {"id": "assigned-at-runtime", "name": "assigned-at-runtime"}},
        "training_mode": "base", "accelerator": "NvidiaTeslaT4", "gpus_per_run": 2,
        "requested_episodes": 20000, "training_only": True, "auto_continue": False,
        "checkpoint_repo": "duy95/uav-search-checkpoints", "cpu_visualize": False,
    }
    suffix = "ALPHA-MASKAWARE" if algorithm == "masac" else "DEFAULT"
    identity = f'''# Choose a fresh identity for this Kaggle execution.
from uuid import uuid4
if type(TRAIN_SEED) is not int or not 0 <= TRAIN_SEED < 2**31:
    raise ValueError("TRAIN_SEED must be an integer in [0, 2**31)")
SPEC["kernel_ref"] = NOTEBOOK_REF
SPEC["account"] = "ui_" + NOTEBOOK_REF.split("/", 1)[0]
SPEC["submitted_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
new_run_id = uuid4().hex[:8]
SPEC["runs"]["{algorithm}"] = {{
    "id": new_run_id,
    "name": f"{algorithm.upper()}-BASE-INDEPENDENT-NOCHUNK-{suffix}-S{{TRAIN_SEED}}-{{new_run_id}}",
}}
tags = {tags!r} + [f"seed{{TRAIN_SEED}}"]
SPEC["overrides"] += [
    f"runtime.seed={{TRAIN_SEED}}",
    f"runtime.output_dir=outputs/independent-{algorithm}-s{{TRAIN_SEED}}-{{new_run_id}}",
    "tracking.tags=" + json.dumps(tags, separators=(",", ":")),
]
print("Kaggle notebook:", "https://www.kaggle.com/code/" + NOTEBOOK_REF, flush=True)
print("W&B run:", f"https://wandb.ai/{{SPEC['entity']}}/{{SPEC['project']}}/runs/{{new_run_id}}", flush=True)
print("{algorithm.upper()} BASE independent MLP / NOCHUNK / seed", TRAIN_SEED, flush=True)

'''
    launch = kernel_script(spec, source_archive=archive, credential_source="kaggle_secrets")
    launch = launch.replace(STAGE_START, identity + STAGE_START, 1)
    digest = sha256(archive).hexdigest()
    intro = f"""# {algorithm.upper()} BASE — INDEPENDENT MLP — NOCHUNK

Cấu hình baseline mới từ lần kiểm tra cặp 971f7cc2 / f6a196a7.
Actor và twin critic độc lập cho từng UAV; plain MLP, không LayerNorm.
Đây là lõi thuật toán theo nguồn gốc với adapter hành động lai UAV, chưa phải
bản sao toàn bộ thí nghiệm của một paper.

- **Profile:** `{PROFILE}`.
- **MASAC:** alpha tự học, `temperature_loss=alpha`, target entropy theo mask.
- **MATD3:** khởi tạo mặc định, không chuẩn hóa output; noise/delay giữ nguyên.
- **Chung:** 6 UAV, 20 target, 15 obstacle, XYZ/CBF/Bayesian, tối đa 1500 bước;
  hết pin toàn bộ UAV sẽ kết thúc episode. Reward/CBF giữ nguyên.
- **Train:** BASE, seed mặc định {seed}, 20000 episode, 4096 env, batch 8192,
  replay 1M (75% CPU/25% GPU), FP32, NOCHUNK; AMP/compile/fused tắt.
- **Eval:** cùng 5 layout validation định kỳ mỗi 3,6864M transition.
  So sánh tại cùng transition budget; 20000 episode có thể tiêu tốn số bước khác nhau.
- **Lưu:** W&B `truongdaoanhduy-uit/uav_search_target`; policy best trên HF
  `duy95/uav-search-checkpoints`. Fresh run, không warm-start hay tự chạy nối tiếp.
- **Source commit:** `{source_commit}`.
- **Archive SHA-256:** `{digest}`.

## Chạy trên Kaggle UI

1. Tạo notebook **Private** riêng cho thuật toán này, import file, bật Internet
   và **GPU T4 x2**.
2. Cấp quyền Add-ons → Secrets: `WANDB_API_KEY` và `HF_TOKEN`.
3. Sửa `NOTEBOOK_REF` ở ô code đầu theo URL thật. Giữ `TRAIN_SEED={seed}`
   cho cặp này; dùng thêm seed khác cho các lần train độc lập.
4. Chạy preflight, rồi **Save & Run All**. Source đã nhúng, không clone GitHub.

Mỗi lần train tiếp theo, dùng **phiên Kaggle mới** và Save & Run All. Chạy lại
riêng ô launch trong phiên đã train có thể dừng vì thư mục của lần trước còn tồn tại.

Notebook chưa được submit tự động. Kiểm tra CPU/schema không chứng minh đủ
VRAM T4 hoặc hội tụ; chưa có kết quả huấn luyện cho profile mới này.
"""
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(intro),
               nbformat.v4.new_code_cell(PREFLIGHT.replace("__SEED__", str(seed))),
               nbformat.v4.new_code_cell(launch)],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "uav_baseline": {"algorithm": algorithm, "profile": PROFILE,
                             "source_commit": source_commit, "archive_sha256": digest,
                             "default_training_seed": seed},
        },
    )
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<independent baseline notebook>", "exec")
    return notebook
