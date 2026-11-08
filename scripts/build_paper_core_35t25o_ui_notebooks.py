"""Create exactly two new source-pinned paper-core 35T25O Kaggle UI notebooks.

No training, Kaggle API calls, W&B/HF mutation or credential reads.
These are *not* the existing Shared+LayerNorm ten-case results; use
distinct private Kaggle notebooks and a unique W&B run ID on launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat
from uav_network.cloud.source_bundle import build_source_archive
from uav_network.cloud.ui_notebooks import build_independent_notebook
from scripts.verify_paper_core_baseline import verify_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--output-dir",type=Path,action="append",
                        help="Repeat for copies; default ~/Downloads and outputs/kaggle.")
    args = parser.parse_args()
    if type(args.seed) is not int or args.seed < 0 or args.seed >= 2**31:
        parser.error("seed outside accepted range")
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        parser.error("Commit reviewed source changes before freezing pinned notebook payload")
    verify_config()
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    bundle = build_source_archive(ROOT)
    folders = args.output_dir or [Path.home()/"Downloads",ROOT/"outputs/kaggle"]
    files = []
    for algorithm in ("matd3","masac"):
        notebook = build_independent_notebook(
            algorithm,bundle,commit,seed=args.seed,scaled=True,paper_core=True)
        assert notebook.metadata.uav_baseline.baseline_class=="PAPER_CORE_CTDE_HYBRID"
        assert notebook.metadata.uav_baseline.profile=="configs/comparison-paper-core-35t25o-xyz-sensing.yaml"
        body = (nbformat.writes(notebook)+"\n").encode("utf-8")
        filename=f"UAV_{algorithm.upper()}_PAPER_CORE_IND_H512_U8_NOLN_35T25O_S{args.seed}_Kaggle_UI.ipynb"
        for folder in folders:
            folder.mkdir(exist_ok=True,parents=True)
            path=folder/filename
            path.write_bytes(body)
            files.append({"algorithm":algorithm,"file":str(path),
                          "sha256":hashlib.sha256(body).hexdigest(),"size":len(body)})
    archives=[]
    for folder in folders:
        target=folder/f"UAV_MASAC_MATD3_PAPER_CORE_35T25O_S{args.seed}_KAGGLE_UI.zip"
        with ZipFile(target,"w",ZIP_DEFLATED,compresslevel=6) as zipf:
            for entry in files:
                if Path(entry["file"]).parent==folder:
                    zipf.write(entry["file"],arcname=Path(entry["file"]).name)
        with ZipFile(target) as check:
            assert len(check.namelist())==2 and check.testzip() is None
        archives.append(str(target))
    print(json.dumps({"source_commit":commit,"source_bundle_sha256":hashlib.sha256(bundle).hexdigest(),
                      "submitted":False,"training_started":False,"notebooks":files,
                      "archives":archives},indent=2))


if __name__=="__main__":
    main()
