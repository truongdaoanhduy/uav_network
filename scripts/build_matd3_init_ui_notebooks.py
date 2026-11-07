"""Create two pinned, self-contained MATD3 BASE Kaggle UI notebooks.

Usage: python scripts/build_matd3_init_ui_notebooks.py
Only writes ~/Downloads and outputs/kaggle; never submits a Kaggle job.
"""
import argparse
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_network.cloud.source_bundle import build_source_archive
TEMPLATE = Path.home() / "Downloads/MASAC_BASE_NOCHUNK_Kaggle_UI.ipynb"
HEADER = "# The notebook's owner/slug is set by the UI user in the preflight cell."
START = 'if SPEC["stage"] in {"visualize", "archive"}:'


def cell_text(cell):
    return "".join(cell["source"])


def replace_cell(cell, source):
    cell["source"] = source.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def build(mode, archive, source_commit, template):
    assert mode in {"default", "orthogonal", "kaiming"}
    book = copy.deepcopy(template)
    code = cell_text(book["cells"][2])
    spec_line = next(line for line in code.splitlines() if line.startswith("SPEC = "))
    spec = ast.literal_eval(spec_line[len("SPEC = "):])
    spec["commit"] = source_commit
    spec["kernel_ref"] = "YOUR_KAGGLE_USERNAME/YOUR_NOTEBOOK_SLUG"
    spec["account"] = "kaggle_ui"
    spec["runs"] = {"matd3": {"id": "assigned-at-runtime", "name": "assigned-at-runtime"}}
    spec["training_mode"] = "base"
    spec["gpus_per_run"] = 2
    spec["cpu_visualize"] = False
    tags = [
        "uav-network", "matd3-init-" + mode, "initialization-ablation",
        "source-backed-algorithm-core", "fp32-eager", "scenario-v3",
        "controlled-20t15o", "targets20-obstacles15", "power-learned",
        "full20k", "replay1m", "batch8192", "env4096", "steps1500",
        "seed44", "report1kb", "radio-energy", "xyz", "bayesian-sensing",
        "grid25m", "belief-prior-0.5", "amp-off", "compile-off", "fused-off",
        "actor-shared", "critic-shared", "actor-layernorm", "replay-hybrid-c75-g25",
        "generation3", "training-only", "hf-best-only", "no-critic-chunks",
    ]
    spec["overrides"] = [
        "runtime.algorithm=matd3",
        "pql.enabled=false",
        "runtime.critic_chunk_size=0",
        f"matd3.initialization={mode}",
        "tracking.tags=" + json.dumps(tags, separators=(",", ":")),
    ]
    assert spec["stage"] == "train" and spec["training_only"]
    assert spec["requested_episodes"] == 20000
    assert spec["config"] == "configs/comparison-reference-masac-matd3-xyz-sensing.yaml"
    assert spec["checkpoint_repo"] == "duy95/uav-search-checkpoints"
    assert code.count(spec_line) == 1
    code = code.replace(spec_line, "SPEC = " + repr(spec), 1)

    label = mode.upper()
    a = code.index(HEADER)
    b = code.index(START, a)
    new_header = f'''# Kaggle UI chooses an independent notebook identity and W&B run each time.
from uuid import uuid4
SPEC["kernel_ref"] = NOTEBOOK_REF
SPEC["account"] = "ui_" + NOTEBOOK_REF.split("/", 1)[0]
SPEC["submitted_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
new_run_id = uuid4().hex[:8]
SPEC["runs"]["matd3"] = {{
    "id": new_run_id,
    "name": f"MATD3-BASE-SH-NOCHUNK-{label}-S44-{{new_run_id}}",
}}
assert SPEC["training_mode"] == "base" and SPEC["gpus_per_run"] == 2
assert SPEC["runs"]["matd3"]["id"] == new_run_id
assert "runtime.algorithm=matd3" in SPEC["overrides"]
assert "pql.enabled=false" in SPEC["overrides"]
assert "runtime.critic_chunk_size=0" in SPEC["overrides"]
assert "matd3.initialization={mode}" in SPEC["overrides"]
print("Kaggle notebook:", "https://www.kaggle.com/code/" + NOTEBOOK_REF, flush=True)
print("W&B run:", f"https://wandb.ai/{{SPEC['entity']}}/{{SPEC['project']}}/runs/{{new_run_id}}", flush=True)
print("MATD3 BASE {label}: no chunk, 2x T4, 20000 episodes, seed44", flush=True)

'''
    code = code[:a] + new_header + code[b:]
    old_payload_line = next(line for line in code.splitlines() if line.startswith("payload = base64.b85decode("))
    new_payload_line = "payload = base64.b85decode(" + repr(base64.b85encode(archive).decode()) + ")"
    code = code.replace(old_payload_line, new_payload_line, 1)
    old_checksum_line = next(line for line in code.splitlines() if line.startswith("if hashlib.sha256(payload).hexdigest() != "))
    new_checksum_line = f'if hashlib.sha256(payload).hexdigest() != "{hashlib.sha256(archive).hexdigest()}":'
    code = code.replace(old_checksum_line, new_checksum_line, 1)

    assert "masac" not in str(spec["overrides"])
    assert code.count("payload = base64.b85decode(") == 1
    assert old_checksum_line != new_checksum_line
    assert re.search(r"SPEC\[.runs.\]\[.matd3.\]", code)
    compile(code, "<matd3 notebook launch>", "exec")
    replace_cell(book["cells"][2], code)

    preflight = cell_text(book["cells"][1]).replace(
        'Ví dụ: NOTEBOOK_REF = "my_other_kaggle_user/masac-base-nochunk-ui"',
        f'Ví dụ: NOTEBOOK_REF = "myusername/matd3-base-{mode}-ui"',
    ).replace(
        "Ready for MASAC BASE, critic_chunk_size = 0 (no chunking)",
        f"Ready for MATD3 BASE {label}, critic_chunk_size=0 (no chunking)",
    )
    replace_cell(book["cells"][1], preflight)
    intro = f"""# MATD3 BASE — {label} — NOCHUNK — Kaggle UI

Notebook này sử dụng cùng profile MATD3 BASE và chỉ thay đổi `matd3.initialization`.
Mã nguồn đã nhúng trực tiếp, SHA-256 được kiểm tra khi chạy; không cần clone GitHub.

- **Nhánh:** MATD3 BASE, `pql.enabled=false`, actor/critic shared, actor LayerNorm.
- **Khởi tạo:** `matd3.initialization={mode}`.
- **Chung:** 6 UAV, 1 km, 20 target, 15 obstacle, Warp XYZ/CBF/Bayesian, 1500 steps.
- **Training:** 20000 episodes, seed 44, deterministic, 4096 envs, batch 8192, 1M replay (75% CPU/25% GPU), NOCHUNK (`runtime.critic_chunk_size=0`).
- **Runtime:** Kaggle T4 ×2, FP32, AMP/compile/fused off, training-only.
- **Tracking/Checkpoint:** W&B `truongdaoanhduy-uit/uav_search_target`; best policy Hugging Face `duy95/uav-search-checkpoints`.
- **Source commit:** `{source_commit}`
- **Bundled source SHA-256:** `{hashlib.sha256(archive).hexdigest()}`

## Chạy bằng Kaggle UI

1. Tạo **notebook Private** mới trên Kaggle, import `.ipynb` này, bật Accelerator **GPU T4 x2** và Internet.
2. Trong Add-ons > Secrets, cấp quyền truy cập cho `WANDB_API_KEY` và `HF_TOKEN` trong notebook.
3. Ở ô code đầu tiên, thay đúng một dòng `NOTEBOOK_REF` theo URL thật `username/notebook-slug`.
4. Chạy ô preflight trước; chỉ sau khi pass mới chọn **Save & Run All**.
5. Dùng **hai notebook riêng** (nên tách theo account được phép) để tránh trùng lịch sử. Không cần chạy job bằng API hay GitHub Actions.

**Chú ý:** Khởi tạo tham số là một thí nghiệm mới, chưa chứng minh cải thiện delivery/search. Đánh giá bằng cùng seed/layout và cùng transition budget, không chỉ episode index. File này chưa được submit lên Kaggle.
"""
    replace_cell(book["cells"][0], intro)
    return book


def main():
    parser = argparse.ArgumentParser(description="Build manual MATD3 BASE Kaggle UI notebooks")
    parser.add_argument("--mode", choices=("default", "orthogonal", "kaiming"),
                        action="append", help="Only build selected initialization; may repeat")
    args = parser.parse_args()
    modes = args.mode or ["default", "orthogonal"]
    import nbformat

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip() == "", "Commit changes before bundling"
    archive = build_source_archive(ROOT)
    template = json.loads(TEMPLATE.read_text())
    assert len(template["cells"]) == 3
    outs = []
    for mode in modes:
        notebook = build(mode, archive, source_commit, template)
        nbformat.validate(nbformat.from_dict(notebook))
        filename = f"MATD3_BASE_NOCHUNK_{mode.upper()}_Kaggle_UI.ipynb"
        body = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
        for folder in (Path.home() / "Downloads", ROOT / "outputs/kaggle"):
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / filename
            target.write_text(body)
        outs.append(filename)
    print(json.dumps({
        "source_commit": source_commit,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_bytes": len(archive),
        "notebooks": outs,
        "destination": str(Path.home() / "Downloads"),
    }, indent=2))


if __name__ == "__main__":
    main()
