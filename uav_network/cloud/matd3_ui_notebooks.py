"""The five requested MATD3 architecture/update ablations for manual Kaggle UI."""
from hashlib import sha256
import json

import nbformat

from .kernel import kernel_script
from .ui_notebooks import PREFLIGHT, SCALED_PROFILE, STAGE_START

# shared actor+critic, hidden width, updates/collection, actor LayerNorm
CASES = {
    "01": (True, 512, 8, True),
    "02": (True, 512, 16, True),
    "03": (False, 512, 16, True),
    "04": (False, 512, 16, False),
    "05": (True, 256, 8, True),
}
GROUP = "uav-matd3-ablation-35t25o-architecture-updates-20261107"


def build_matd3_ablation_notebook(case_id, archive, source_commit, *, seed=44):
    """Build one declared case without accessing credentials or online services."""
    if case_id not in CASES:
        raise ValueError("case_id must be one of01,02,03,04,05")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    shared, width, updates, layernorm = CASES[case_id]
    mode, norm = ("SH" if shared else "IND"), ("ALN" if layernorm else "NOLN")
    layout = "shared" if shared else "independent"
    slug = f"{case_id}-{mode.lower()}-h{width}-u{updates}-{norm.lower()}-35t25o"
    filename = f"MATD3_{case_id}_{mode}_H{width}_U{updates}_{norm}_35T25O_S{seed}_Kaggle_UI.ipynb"
    tags = [
        "uav-network", "matd3-ui-ablation", f"case{case_id}", "base", "no-critic-chunks",
        "scenario-v3", "targets35", "obstacles25", "steps1500", "env4096", "batch8192",
        "replay1m", "cpu75-gpu25", "fp32-eager", "training-only", "hf-best-only",
        "xyz", "bayesian-sensing", "report1kb", "power-learned", "matd3-init-default",
        f"{layout}-actors", f"{layout}-critics", f"hidden{width}", f"updates{updates}",
        "actor-layernorm" if layernorm else "actor-no-layernorm", "critic-no-layernorm",
    ]
    overrides = [
        "runtime.algorithm=matd3", "pql.enabled=false",
        f"matd3.share_actor_parameters={str(shared).lower()}",
        f"matd3.share_critic_parameters={str(shared).lower()}",
        f"matd3.actor_layer_norm={str(layernorm).lower()}",
        f"matd3.hidden_sizes=[{width},{width}]",
        f"matd3.updates_per_step={updates}",
        f"tracking.group={GROUP}",
    ]
    spec = {
        "submitted_at_utc": "assigned-at-runtime", "commit": source_commit,
        "kernel_ref": "YOUR_KAGGLE_USERNAME/YOUR_NOTEBOOK_SLUG", "account": "kaggle_ui",
        "entity": "truongdaoanhduy-uit", "project": "uav_search_target", "stage": "train",
        "timeout_seconds": 39600, "config": SCALED_PROFILE, "overrides": overrides,
        "warm_start_artifacts": {}, "runs": {"matd3": {"id": "assigned-at-runtime", "name": "assigned-at-runtime"}},
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
SPEC["runs"]["matd3"] = {{
    "id": new_run_id,
    "name": f"MATD3-BASE-{mode}-NOCHUNK-DEFAULT-{norm}-H{width}-U{updates}-T35-O25-S{{TRAIN_SEED}}-{{new_run_id}}",
}}
tags = {tags!r} + [f"seed{{TRAIN_SEED}}"]
SPEC["overrides"] += [
    f"runtime.seed={{TRAIN_SEED}}",
    f"runtime.output_dir=outputs/matd3-ablation-{slug}-s{{TRAIN_SEED}}-{{new_run_id}}",
    "tracking.tags=" + json.dumps(tags, separators=(",", ":")),
]
print("Kaggle notebook:", "https://www.kaggle.com/code/" + NOTEBOOK_REF, flush=True)
print("W&B run:", f"https://wandb.ai/{{SPEC['entity']}}/{{SPEC['project']}}/runs/{{new_run_id}}", flush=True)
print("MATD3 case{case_id}: {layout} actors+critics, hidden{width}, U{updates}, actor LayerNorm={layernorm}; seed", TRAIN_SEED, flush=True)

'''
    launch = kernel_script(spec, source_archive=archive, credential_source="kaggle_secrets")
    launch = launch.replace(STAGE_START, identity + STAGE_START, 1)
    digest = sha256(archive).hexdigest()
    actor_dim = 234 if shared else 228
    intro = f"""# MATD3 {case_id} — {mode} — H{width} — U{updates} — {norm}

Notebook BASE riêng cho một cấu hình trong ma trận5 thí nghiệm.
Không khởi chạy MASAC.

| Cài đặt | Giá trị |
|---|---|
| Actor và twin critic | {layout}, cùng chế độ chia sẻ |
| Hidden actor và critic | {width} → {width} |
| Actor LayerNorm | {layernorm} |
| Critic LayerNorm | Không |
| Critic updates / vòng thu dữ liệu | {updates} |
| Actor updates / vòng thu dữ liệu | {updates // 2}, policy_delay=2 |
| Input actor / input critic | {actor_dim} / 896 |
| Kịch bản | 6 UAV,35 mục tiêu,25 vật cản,bản đồ1km |
| Episode | Tối đa1500 bước; tất cả UAV hết pin thì kết thúc |
| Seed mặc định | {seed} |

LayerNorm ở đây là tùy chọn actor_layer_norm có sẵn: các lớp encoder actor,
không đặt lên đầu ra action hay critic. Shared actor được thêm6 chiều agent-ID;
independent actor dùng observation228 chiều. Hidden width không thay đổi
quy tắc quan sát. Actor cục bộ; CBF là tầng lọc riêng dùng hình học toàn cục.

Chung cho cả5 file: batch8192,num_envs4096,replay1M,75%CPU/25%GPU,
warmup131072,20000 episode,FP32,NOCHUNK,vectorize/AMP/compile/fused tắt.
MATD3 khởi tạo mặc định,output_normalization=false,noise và reward giữ nguyên.
Mỗi vòng collector tạo4096 joint transitions rồi mới học{updates} lượt;
đây không phải{updates} update trên từng transition riêng lẻ.

Cùng5 kịch bản validation mỗi1.8432M transition, cùng eval seed.
So sánh ở cùng transition budget;20000 episode có thể cần số bước khác nhau.
Lưu metrics W&B và validation-best policy lên HF duy95/uav-search-checkpoints.
Training-only; không visualize,không warm-start,không tự chạy tiếp.
Giới hạn train10 giờ; chưa chứng minh đủVRAM T4 batch8192 hay hội tụ.

## Chạy trên Kaggle UI

1. Tạo notebook Private riêng cho file này,import notebook,bật Internet vàT4x2.
2. Cấp quyền Secrets WANDB_API_KEY vàHF_TOKEN.
3. Sửa NOTEBOOK_REF ở ô code đầu theo URL thật của notebook.
   Giữ TRAIN_SEED={seed} cho5 phép thử.
4. Chạy preflight rồi Save & Run All trong phiên Kaggle mới.
   Source đã nhúng; không clone GitHub. Không chạy lại riêng ô launch trong
   phiên đã train vì thư mục lần trước vẫn tồn tại.

Profile gốc: {SCALED_PROFILE}.
Case ID: {case_id}. Override MATD3 được ghi rõ trong SPEC và config W&B.
Source commit: {source_commit}.
Archive SHA256: {digest}.
"""
    preflight = PREFLIGHT.replace("__SEED__", str(seed))
    preflight += f'\nprint("Case{case_id}: {layout} actors+critics; hidden[{width},{width}]; U{updates}; actor LayerNorm={layernorm}; critic LayerNorm=False")\n'
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(preflight),
               nbformat.v4.new_code_cell(launch)],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "uav_ablation": {
                "algorithm": "matd3", "case_id": case_id, "filename": filename,
                "profile": SCALED_PROFILE, "source_commit": source_commit,
                "archive_sha256": digest, "default_training_seed": seed,
                "shared_actor_critic": shared, "hidden_sizes": [width, width],
                "updates_per_collection": updates, "actor_layer_norm": layernorm,
                "critic_layer_norm": False,
            },
        },
    )
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<MATD3 ablation notebook>", "exec")
    return notebook
