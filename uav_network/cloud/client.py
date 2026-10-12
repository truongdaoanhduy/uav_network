"""Official Kaggle CLI calls with per-process tokens and redacted diagnostics."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from .credentials import parse_quota


def run_kaggle(args, account, timeout=45):
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = account.token
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        env.pop(name, None)
    process = subprocess.run(["kaggle", *args], env=env, text=True,
                             capture_output=True, timeout=timeout, check=False)
    stdout = process.stdout.replace(account.token, "[REDACTED]")
    stderr = process.stderr.replace(account.token, "[REDACTED]")
    if process.returncode:
        raise RuntimeError(f"Kaggle command failed ({process.returncode}): {(stderr or stdout)[-1000:]}")
    return stdout


def account_quotas(credentials):
    def quota(account):
        if not account.enabled:
            return {"name": account.name, "ok": False, "error": "disabled"}
        try:
            value = json.loads(run_kaggle(["quota", "--format", "json"], account, timeout=25))
            return {"name": account.name, "ok": True, **parse_quota(value)}
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            return {"name": account.name, "ok": False, "error": "quota unavailable"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(quota, credentials.accounts))


def resolve_username(account):
    for kind in ("kernels", "datasets"):
        rows = json.loads(run_kaggle([kind, "list", "--mine", "--page-size", "1", "--format", "json"], account))
        if rows and "/" in rows[0].get("ref", ""):
            return rows[0]["ref"].split("/", 1)[0]
    raise RuntimeError("Cannot resolve the selected account username from owned Kaggle resources")


def verify_credential_dataset(account, credentials, ref, username):
    if ref.split("/", 1)[0] != username:
        raise ValueError("Credential dataset must belong to the authenticated account")
    saved = os.environ.get("KAGGLE_API_TOKEN")
    os.environ["KAGGLE_API_TOKEN"] = account.token
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        rows = api.dataset_list(mine=True, search=ref.split("/", 1)[1])
        dataset = next((r for r in rows if r.ref == ref), None)
        if dataset is None or dataset.is_private is not True:
            raise RuntimeError("W&B credential source must be a verified private owned dataset")
    finally:
        if saved is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = saved
    with TemporaryDirectory(prefix="uav-wandb-check-") as temp:
        Path(temp).chmod(0o700)
        run_kaggle(["datasets", "download", "-d", ref, "-p", temp, "--unzip", "-q"], account)
        candidates = list(Path(temp).rglob("wandb_api_key.txt")) + list(Path(temp).rglob("wandb_secret.json"))
        if len(candidates) != 1:
            raise RuntimeError("Credential dataset needs exactly one W&B key file")
        path = candidates[0]
        key = path.read_text().strip() if path.suffix == ".txt" else json.loads(path.read_text())["api_key"]
        if key != credentials.wandb_key:
            raise RuntimeError("Private dataset W&B key does not match local key.txt")


def kernel_status(ref, account):
    import re
    text = run_kaggle(["kernels", "status", ref], account)
    match = re.search(r"has status [\"'](?:KernelWorkerStatus\.)?([a-z_]+)[\"']", text, re.I)
    state = match.group(1).lower() if match else "unknown"
    return {"cancel_acknowledged": "cancelled", "cancel_requested": "running",
            "new_script": "pending"}.get(state, state)
