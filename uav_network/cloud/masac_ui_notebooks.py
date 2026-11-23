"""Five BASE MASAC manual Kaggle UI notebooks mirroring the MATD3 ablation cases.

Notebook bootstrap mirrors the existing MATD3 builder; only `masac` config
fields change. Scientific reward, observation, environment, and learning
algorithms are not modified by this module.
"""
from hashlib import sha256
import json

import nbformat

from .kernel import kernel_script
from .matd3_ui_notebooks import CASES
from .ui_notebooks import PREFLIGHT, SCALED_PROFILE, STAGE_START

GROUP = "uav-base-matd3-masac-architecture-35t25o-20261108"


def build_masac_ablation_notebook(case_id, archive, source_commit, *, seed=44):
    """Build one user-selected MASAC BASE case for UI import, without credentials."""
    if case_id not in CASES:
        raise ValueError("case_id must be one of 01,02,03,04,05")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    shared, width, updates, layernorm = CASES[case_id]
    mode, norm = ("SH" if shared else "IND"), ("ALN" if layernorm else "NOLN")
    layout = "shared" if shared else "independent"
    slug = f"{case_id}-{mode.lower()}-h{width}-u{updates}-{norm.lower()}-35t25o"
    filename = f"MASAC_{case_id}_{mode}_H{width}_U{updates}_{norm}_35T25O_S{seed}_Kaggle_UI.ipynb"
    tags = [
        "uav-network", "masac-ui-ablation", f"case{case_id}", "base", "no-critic-chunks",
        "scenario-v3", "targets35", "obstacles25", "steps1500", "env4096", "batch8192",
        "replay1m", "cpu75-gpu25", "fp32-eager", "training-only", "hf-best-only",
        "xyz", "bayesian-sensing", "report1kb", "power-learned",
        "alpha-dual-loss", "mask-aware-discrete-entropy",
        f"{layout}-actors", f"{layout}-critics", f"hidden{width}", f"updates{updates}",
        "actor-layernorm" if layernorm else "actor-no-layernorm", "critic-no-layernorm",
    ]
    overrides = [
        "runtime.algorithm=masac", "pql.enabled=false",
        f"masac.share_actor_parameters={str(shared).lower()}",
        f"masac.share_critic_parameters={str(shared).lower()}",
        f"masac.actor_layer_norm={str(layernorm).lower()}",
        f"masac.hidden_sizes=[{width},{width}]",
        f"masac.updates_per_step={updates}",
        f"tracking.group={GROUP}",
    ]
    spec = {
        "submitted_at_utc": "assigned-at-runtime", "commit": source_commit,
        "kernel_ref": "YOUR_KAGGLE_USERNAME/YOUR_NOTEBOOK_SLUG", "account": "kaggle_ui",
        "entity": "truongdaoanhduy-uit", "project": "uav_search_target", "stage": "train",
        "timeout_seconds": 39600, "config": SCALED_PROFILE, "overrides": overrides,
        "warm_start_artifacts": {}, "runs": {"masac": {"id": "assigned-at-runtime", "name": "assigned-at-runtime"}},
        "training_mode": "base", "accelerator": "NvidiaTeslaT4", "gpus_per_run": 2,
        "requested_episodes": 20000, "training_only": True, "auto_continue": False,
        "checkpoint_repo": "duy95/uav-search-checkpoints", "cpu_visualize": False,
        "ablation_case": case_id,
    }
    identity = f'''# New identity on each fresh Kaggle execution.
from uuid import uuid4
if type(TRAIN_SEED) is not int or not 0 <= TRAIN_SEED < 2**31:
    raise ValueError("TRAIN_SEED must be an integer in [0, 2**31)")
SPEC["kernel_ref"] = NOTEBOOK_REF
SPEC["account"] = "ui_" + NOTEBOOK_REF.split("/", 1)[0]
SPEC["submitted_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
new_run_id = uuid4().hex[:8]
SPEC["runs"]["masac"] = {{
    "id": new_run_id,
    "name": f"MASAC-BASE-{mode}-NOCHUNK-MASKAWARE-{norm}-H{width}-U{updates}-T35-O25-S{{TRAIN_SEED}}-{{new_run_id}}",
}}
tags = {tags!r} + [f"seed{{TRAIN_SEED}}"]
SPEC["overrides"] += [
    f"runtime.seed={{TRAIN_SEED}}",
    f"runtime.output_dir=outputs/masac-ablation-{slug}-s{{TRAIN_SEED}}-{{new_run_id}}",
    "tracking.tags=" + json.dumps(tags, separators=(",", ":")),
]
print("Kaggle notebook:", "https://www.kaggle.com/code/" + NOTEBOOK_REF, flush=True)
print("W&B run:", f"https://wandb.ai/{{SPEC['entity']}}/{{SPEC['project']}}/runs/{{new_run_id}}", flush=True)
print("MASAC case{case_id}: {layout} actors+critics, hidden{width}, U{updates}, actor LayerNorm={layernorm}; seed", TRAIN_SEED, flush=True)

'''
    launch = kernel_script(spec, source_archive=archive, credential_source="kaggle_secrets")
    if launch.count(STAGE_START) != 1:
        raise RuntimeError("The Kaggle bootstrap entrypoint changed")
    launch = launch.replace(STAGE_START, identity + STAGE_START, 1)
    digest = sha256(archive).hexdigest()
    actor_dim = 234 if shared else 228
    intro = f"""# MASAC {case_id} — {mode} — H{width} — U{updates} — {norm}

Notebook **MASAC BASE**, case{case_id} tương ứng case MATD3 cùng số thứ tự.
Chạy thủ công bằng Kaggle UI, không tự submit job.

| Tham số | Giá trị |
|---|---|
| Actor và twin critic | {layout}: cùng chia sẻ hoặc cùng independent theo các UAV |
| Mạng Actor/Critic | {width} → {width} |
| **Actor LayerNorm** | {layernorm} |
| **Critic LayerNorm** | Không có (giữ kiến trúc gốc) |
| **Critic updates / vòng thu dữ liệu** | {updates} |
| **Actor updates / vòng thu dữ liệu** | {updates} (MASAC, không policy_delay MATD3) |
| Input Actor / Critic | {actor_dim} / 896 |
| Thuật toán | MASAC BASE, entropy theo mask; alpha tự học, temperature_loss=alpha |
| Kịch bản | 6 UAV, 35 targets, 25 obstacles, bản đồ 1km, XYZ/Bayesian/CBF |
| Horizon | max 1.500 steps/episode; pin cạn có thể kết thúc sớm |
| Seed | {seed} |

Chung cả 10 notebook MATD3/MASAC: 20.000 episodes, 4.096 envs, batch 8.192,
replay 1M (75% CPU/25% GPU), warmup 131.072 transitions, NOCHUNK,
FP32, AMP/compile/fused tắt, deterministic=true, output-norm MATD3=false.
**Reward/CBF/env/observation và optimizer không bị sửa**. Mỗi vòng vector
collector thu 4.096 transitions, không phải {updates} updates cho mỗi transition.
Checkpoint policy định kỳ mỗi **256 completed episodes**; giữ 2 local files;
Validation 5 seed mỗi 1.843.200 transitions, giữ best policy trên HF;
final.pt được lưu khi training kết thúc bình thường.

Training-only (không tự Final Test), W&B online logging và best checkpoint
Hugging Face `duy95/uav-search-checkpoints`. Không PQL/warmstart/auto continue.
Kaggle runner chỉ có 1 job trên tài khoản với GPU T4 x2. Source code được
nhúng trong notebook và có thể được xem khi có quyền truy cập notebook;
muốn hạn chế lộ source, hãy để notebook **Private**.
Giới hạn kernel: 11 giờ; runner trừ thời gian khởi động và khoảng 10 phút
cho save/cleanup, nên training thực tế có thể dừng trước khi đủ 20k episodes.
Khả năng đủ VRAM và thời gian chưa được bảo đảm.

## Dùng trên Kaggle UI

1. Import file này vào một notebook Kaggle **Private**, bật Internet và GPU T4 x2.
2. Add-ons → Secrets: cấp quyền `WANDB_API_KEY` và `HF_TOKEN`.
3. Sửa `NOTEBOOK_REF` trong cell preflight thành `username/notebook-slug` thật.
4. Giữ `TRAIN_SEED={seed}` khi so sánh cùng seed, chạy **Save & Run All** ở phiên mới.
5. Đừng chạy lại cell launch riêng lẻ trên kernel đã train.

Profile: `{SCALED_PROFILE}`.
Case ID: `{case_id}`. Source commit: `{source_commit}`.
Source SHA256: `{digest}`.
"""
    preflight = PREFLIGHT.replace("__SEED__", str(seed))
    preflight += f'\nprint("MASAC case{case_id}: {layout} actors+critics, hidden[{width},{width}], U{updates}, actor LayerNorm={layernorm}, critic LayerNorm=False")\n'
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(preflight),
               nbformat.v4.new_code_cell(launch)],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "uav_ablation": {
                "algorithm": "masac", "case_id": case_id, "filename": filename,
                "profile": SCALED_PROFILE, "source_commit": source_commit,
                "archive_sha256": digest, "default_training_seed": seed,
                "shared_actor_critic": shared, "hidden_sizes": [width, width],
                "updates_per_collection": updates, "actor_layer_norm": layernorm,
                "critic_layer_norm": False, "mask_aware_discrete_entropy_target": True,
            },
        },
    )
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<MASAC ablation notebook>", "exec")
    return notebook
