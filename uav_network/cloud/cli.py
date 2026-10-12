"""Quota-aware private Kaggle launches and public job status manifests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from uuid import uuid4

from ..configuration import load_config
from .client import account_quotas, kernel_status, pushed_kernel_ref, resolve_username, run_kaggle, verify_credential_dataset
from .credentials import load_credentials, select_account
from .kernel import kernel_metadata, kernel_script

ROOT = Path(__file__).resolve().parents[2]


def launch(args, credentials):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit and publish the project changes before launching a pinned cloud job")
    account = select_account(credentials, account_quotas(credentials), args.required_gpu_hours, args.account)
    username = resolve_username(account)
    verify_credential_dataset(account, credentials, args.credential_dataset, username)
    config_file = args.config or ("configs/kaggle-smoke.yaml" if args.stage == "validate" else "configs/kaggle.yaml")
    config = load_config(ROOT / config_file, args.overrides)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = uuid4().hex[:6]
    slug = f"uavnet-{args.stage}-{stamp}-{token}"
    algorithms = ("masac", "matd3") if args.algorithm == "both" else (args.algorithm,)
    runs = {a: {"id": uuid4().hex[:8], "name": f"{a.upper()}-uav-network-2d-seed{config.runtime.seed}-{stamp}-{token}"}
            for a in algorithms}
    spec = {"commit": commit, "kernel_ref": f"{username}/{slug}", "account": account.name,
            "credential_dataset": args.credential_dataset,
            "entity": credentials.entity if config.tracking.entity == "auto" else config.tracking.entity,
            "project": credentials.project if config.tracking.project == "auto" else config.tracking.project,
            "stage": args.stage, "config": config_file, "overrides": args.overrides, "runs": runs}
    serialized = json.dumps(spec)
    if any(secret in serialized for secret in [credentials.wandb_key, *(a.token for a in credentials.accounts)]):
        raise RuntimeError("Refusing a job manifest containing credentials")
    folder = ROOT / "outputs/kaggle" / slug
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "kernel.py").write_text(kernel_script(spec))
    (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(spec), indent=2))
    # The timeout and accelerator are explicit CLI inputs, not assumptions about RTX naming.
    response = run_kaggle(["kernels", "push", "-p", str(folder), "--accelerator", args.accelerator,
                           "--timeout", str(args.timeout_seconds)], account, timeout=50)
    actual_ref = pushed_kernel_ref(response)
    manifest = {**spec, "kernel_ref": actual_ref, "kernel_url": f"https://www.kaggle.com/code/{actual_ref}",
                "wandb_urls": {a: f"https://wandb.ai/{spec['entity']}/{spec['project']}/runs/{r['id']}" for a, r in runs.items()},
                "manifest": str(folder / "launch.json")}
    (folder / "launch.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Private Kaggle CUDA + online W&B")
    parser.add_argument("--key-file", default=str(ROOT / "key.txt"))
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("accounts", help="Show verified quota without credentials")
    submit = commands.add_parser("launch")
    submit.add_argument("--stage", choices=["validate", "train"], default="train")
    submit.add_argument("--account")
    submit.add_argument("--required-gpu-hours", type=float)
    submit.add_argument("--credential-dataset", required=True)
    submit.add_argument("--algorithm", choices=["both", "masac", "matd3"], default="both")
    submit.add_argument("--config")
    submit.add_argument("--set", dest="overrides", action="append", default=[])
    submit.add_argument("--accelerator", default="NvidiaTeslaT4")
    submit.add_argument("--timeout-seconds", type=int, default=43200)
    status = commands.add_parser("status")
    status.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    credentials = load_credentials(args.key_file)
    if args.action == "accounts":
        return {"accounts": account_quotas(credentials)}
    if args.action == "launch":
        if args.timeout_seconds < 1:
            raise ValueError("Kernel timeout must be positive")
        return launch(args, credentials)
    manifest = json.loads(Path(args.manifest).read_text())
    account = next((a for a in credentials.accounts if a.name == manifest["account"]), None)
    if account is None:
        raise RuntimeError("Manifest account is missing from key.txt")
    return {"kernel_url": manifest["kernel_url"], "state": kernel_status(manifest["kernel_ref"], account),
            "wandb_urls": manifest["wandb_urls"]}
