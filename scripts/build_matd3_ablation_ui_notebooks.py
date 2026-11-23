"""Export the five requested MATD3 notebooks for manual Kaggle UI import."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat

from uav_network.cloud.matd3_ui_notebooks import CASES, build_matd3_ablation_notebook
from uav_network.cloud.source_bundle import build_source_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-dir", type=Path, action="append",
                        help="Repeat for multiple copies; defaults to Downloads and outputs/kaggle.")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        parser.error("Commit source changes before building a pinned notebook")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = build_source_archive(ROOT)
    folders = args.output_dir or [Path.home() / "Downloads", ROOT / "outputs/kaggle"]
    files = []
    for case_id in CASES:
        notebook = build_matd3_ablation_notebook(case_id, archive, commit, seed=args.seed)
        body = nbformat.writes(notebook) + "\n"
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / notebook.metadata.uav_ablation.filename
            target.write_text(body)
            files.append(str(target))
    print(json.dumps({
        "source_commit": commit, "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_bytes": len(archive), "training_seed": args.seed,
        "notebooks": len(CASES), "files": files, "submitted": False,
    }, indent=2))


if __name__ == "__main__":
    main()
