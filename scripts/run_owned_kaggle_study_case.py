#!/usr/bin/env python3
"""Owner-initiated Kaggle launch for ONE matched seed-44 experiment.

This reuses the cloud launcher, checkpoint storage and W&B tracking. It
never cycles through Kaggle accounts or embeds an API token in notebook code.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from scripts.prepare_matched_architecture_study import CASES, REFERENCE, ROOT
from uav_network.configuration import load_config

STUDY = "matched-architecture-seed44-20261104"


def build_case(number):
    if type(number) is not int or not 1 <= number <= len(CASES):
        raise ValueError("Study case must be 1..10")
    algorithm, pql, shared, vectorized = CASES[number - 1]
    name = f"{number:02d}-{algorithm}-{'pql' if pql else 'base'}-{'shared' if shared else 'independent'}"
    if vectorized:
        name += "-vmap"
    if number == 10:
        name += "-replica"
    base = load_config(REFERENCE)
    tags = list(base.tracking.tags) + ["matched-architecture-study"]
    overrides = [
        f"runtime.algorithm={algorithm}",
        "runtime.collector_overlap=false",
        f"runtime.vectorize_critics={str(vectorized).lower()}",
        f"runtime.output_dir=outputs/{STUDY}/training/{name}",
        f"pql.enabled={str(pql).lower()}",
        f"{algorithm}.share_actor_parameters={str(shared).lower()}",
        f"{algorithm}.share_critic_parameters={str(shared).lower()}",
        f"tracking.group={STUDY}",
        "tracking.tags=" + json.dumps(tags),
    ]
    config = "configs/comparison-reference-masac-matd3-xyz-sensing.yaml"
    resolved = load_config(REFERENCE, overrides)
    if (resolved.runtime.seed != 44 or not resolved.runtime.deterministic
            or resolved.runtime.critic_chunk_size != 0 or resolved.runtime.amp_mode != "off"
            or resolved.runtime.compile_models or resolved.runtime.fused_optimizers):
        raise RuntimeError("Reference profile changed; do not launch a non-matched run")
    return {"id": number, "name": name, "algorithm": algorithm,
            "config": config, "overrides": overrides}


def single_owner_credentials(shared, api_token):
    """Keep team W&B/HF metadata; isolate one authenticated Kaggle owner."""
    if not isinstance(api_token, str) or not api_token.strip():
        raise ValueError("KAGGLE_API_TOKEN is required for submission")
    return {"kaggle": {"accounts": [{"name": "owner", "token": api_token.strip(),
                                      "enabled": True}],
                       "selection": shared.get("kaggle", {}).get("selection", {})},
            "wandb": shared["wandb"],
            "huggingface": shared.get("huggingface", {})}


def verify_published():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit the experiment code before submitting Kaggle jobs")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if not branch:
        raise RuntimeError("Use a named, published experiment branch")
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=ROOT, text=True
    ).strip().split()
    if not remote or remote[0] != sha:
        raise RuntimeError("The current commit must be pushed to origin before Kaggle can clone it")
    return sha


def submit_case(case, key_file, api_token):
    from uav_network.cloud.cli import main as kaggle_main
    from uav_network.cloud.client import resolve_username
    from uav_network.cloud.credentials import Account

    receipt = ROOT / "outputs" / STUDY / "kaggle-submissions" / f"case-{case['id']:02d}.json"
    if receipt.exists():
        raise FileExistsError(f"This case was submitted already; inspect {receipt} before retrying")
    sha = verify_published()
    shared = json.loads(Path(key_file).expanduser().read_text())
    isolated = single_owner_credentials(shared, api_token)
    owner = resolve_username(Account(name="owner", token=api_token))
    with TemporaryDirectory(prefix="uav-one-owner-") as folder:
        private = Path(folder) / "key.json"
        private.write_text(json.dumps(isolated))
        private.chmod(0o600)
        options = ["--key-file", str(private), "launch", "--stage", "train",
                   "--account", "owner", "--algorithm", case["algorithm"],
                   "--config", case["config"],
                   "--credential-dataset", f"{owner}/uav-cpu-wandb-credential",
                   "--accelerator", "NvidiaTeslaT4", "--gpus-per-run", "2"]
        for override in case["overrides"]:
            options += ["--set", override]
        result = kaggle_main(options)
    record = {"case": case["id"], "case_name": case["name"], "commit": sha,
              "kernel_url": result["kernel_url"], "wandb_urls": result["wandb_urls"],
              "manifest": result["manifest"]}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x") as f:
        json.dump(record, f, indent=2)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Submit one Kaggle study case using its owner's API token")
    parser.add_argument("--case", type=int, required=True, choices=range(1, 11), metavar="1..10")
    parser.add_argument("--key-file", default=str(ROOT / "key.txt"),
                        help="Local private team W&B/HF credentials file; never commit it")
    parser.add_argument("--submit", action="store_true", help="Actually submit one GPU kernel; otherwise dry-run")
    args = parser.parse_args(argv)
    case = build_case(args.case)
    if not args.submit:
        print(json.dumps({**case, "submitted": False, "mode": "dry-run"}, indent=2))
        return 0
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token.strip():
        parser.error("Set KAGGLE_API_TOKEN in this owner's terminal (never paste it into the code)")
    try:
        record = submit_case(case, args.key_file, token)
    except Exception as error:
        # Kaggle/API exception text may contain a credential; do not log it here.
        print(f"Submission failed ({type(error).__name__}); check credentials, GPU quota and the local receipt.",
              file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
