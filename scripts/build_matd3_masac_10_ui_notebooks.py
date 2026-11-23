"""Export 5 MATD3 BASE + 5 MASAC BASE self-contained manual Kaggle UI notebooks.

Build does NOT submit training, contact Kaggle/W&B or read credentials.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat

from uav_network.cloud.matd3_ui_notebooks import CASES, build_matd3_ablation_notebook
from uav_network.cloud.masac_ui_notebooks import build_masac_ablation_notebook
from uav_network.cloud.source_bundle import build_source_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-dir", type=Path, action="append",
                        help="Repeat for copies; default Downloads and outputs/kaggle.")
    parser.add_argument("--zip-name", default="UAV_MATD3_MASAC_10_KAGGLE_UI_35T25O_S44.zip")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    if '/' in args.zip_name or '\\' in args.zip_name or not args.zip_name.endswith('.zip'):
        parser.error("--zip-name must be a local .zip filename")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        parser.error("Commit source changes before building pinned UI notebooks")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = build_source_archive(ROOT)
    folders = args.output_dir or [Path.home() / "Downloads", ROOT / "outputs/kaggle"]
    files = []
    for algorithm, build in (("matd3", build_matd3_ablation_notebook),
                             ("masac", build_masac_ablation_notebook)):
        for case_id in CASES:
            notebook = build(case_id, archive, commit, seed=args.seed)
            body = (nbformat.writes(notebook) + "\n").encode("utf-8")
            basename = notebook.metadata.uav_ablation.filename
            for folder in folders:
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / basename
                target.write_bytes(body)
                files.append({"algorithm":algorithm,"case_id":case_id,"path":str(target),
                              "bytes":len(body),"sha256":hashlib.sha256(body).hexdigest()})
    archives = []
    for folder in folders:
        zpath = folder / args.zip_name
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for item in files:
                if Path(item["path"]).parent == folder:
                    zf.write(item["path"], arcname=Path(item["path"]).name)
        with zipfile.ZipFile(zpath) as check:
            if len(check.namelist()) != 10 or check.testzip() is not None:
                raise RuntimeError("Invalid 10-file archive")
        archives.append(str(zpath))
    print(json.dumps({"source_commit":commit,"archive_sha256":hashlib.sha256(archive).hexdigest(),
                      "archive_bytes":len(archive),"training_seed":args.seed,"notebooks":10,
                      "files":files,"zip_archives":archives,"submitted":False},indent=2))


if __name__ == "__main__":
    main()
