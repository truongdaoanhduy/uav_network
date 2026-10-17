"""Official Kaggle CLI calls with per-process tokens and redacted diagnostics."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

from .credentials import parse_quota


class CredentialFileSetError(RuntimeError):
    """A private credential dataset contains files the CPU job must not mount."""


def run_kaggle(args, account, timeout=45):
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = account.token
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        env.pop(name, None)
    process = subprocess.run([sys.executable, "-m", "kaggle", *args], env=env, text=True,
                             capture_output=True, timeout=timeout, check=False)
    stdout = process.stdout.replace(account.token, "[REDACTED]")
    stderr = process.stderr.replace(account.token, "[REDACTED]")
    if process.returncode:
        raise RuntimeError(f"Kaggle command failed ({process.returncode}): {(stderr or stdout)[-1000:]}")
    return stdout


def account_quotas(credentials, requested=None):
    def quota(account):
        if not account.enabled:
            return {"name": account.name, "ok": False, "error": "disabled"}
        try:
            value = json.loads(run_kaggle(["quota", "--format", "json"], account, timeout=25))
            return {"name": account.name, "ok": True, **parse_quota(value)}
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            return {"name": account.name, "ok": False, "error": "quota unavailable"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(quota, [a for a in credentials.accounts if not requested or a.name == requested]))


def resolve_username(account):
    # Token introspection works for new accounts with no owned kernels/datasets.
    names = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
    saved = {name: os.environ.get(name) for name in names}
    os.environ["KAGGLE_API_TOKEN"] = account.token
    for name in names[1:]:
        os.environ.pop(name, None)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        username = api.get_config_value(api.CONFIG_NAME_USER)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if not isinstance(username, str) or not re.fullmatch(r"[\w-]+", username):
        raise RuntimeError("Cannot resolve the selected account's authenticated username")
    return username


def private_dataset_info(account, ref, username):
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
        if dataset is not None and dataset.is_private is not True:
            raise RuntimeError("Credential source must be a verified private owned dataset")
    finally:
        if saved is None:
            os.environ.pop("KAGGLE_API_TOKEN", None)
        else:
            os.environ["KAGGLE_API_TOKEN"] = saved
    return dataset


def private_dataset_file(account, ref, username, filenames, timeout=120, only_selected=False):
    if private_dataset_info(account, ref, username) is None:
        raise RuntimeError("Credential source must be a verified private owned dataset")
    files, page = [], []
    while True:
        lines = run_kaggle(["datasets", "files", ref, "--format", "json", *page], account).splitlines()
        token = lines.pop(0).removeprefix("Next Page Token = ").strip() if lines and lines[0].startswith("Next Page Token = ") else None
        files.extend(json.loads("\n".join(lines)))
        if not token:
            break
        page = ["--page-token", token]
    selected = [f["name"] for f in files if Path(f["name"]).name in filenames]
    if only_selected and len(files) != 1:
        raise CredentialFileSetError("CPU-flow credential dataset must contain only its one required credential file")
    if len(selected) != 1:
        raise RuntimeError("Credential dataset needs exactly one selected credential file")
    with TemporaryDirectory(prefix="uav-wandb-check-") as temp:
        Path(temp).chmod(0o700)
        single_file_archive = len(files) == 1
        selection = ["-f", selected[0]]
        run_kaggle(["datasets", "download", "-d", ref, *selection, "-p", temp, "-q"],
                   account, timeout=timeout)
        if single_file_archive:
            extracted = [p for p in Path(temp).rglob("*") if p.is_file()]
            if len(extracted) != 1 or extracted[0].name != Path(selected[0]).name:
                raise RuntimeError("Single credential archive must contain exactly its verified file")
        candidates = [p for p in Path(temp).rglob("*") if p.is_file() and p.name in filenames]
        if len(candidates) != 1:
            raise RuntimeError("Credential dataset needs exactly one selected credential file")
        return candidates[0].name, candidates[0].read_text().strip()


def verify_credential_dataset(account, credentials, ref, username, timeout=120, only_selected=False):
    name, value = private_dataset_file(account, ref, username,
                                       {"wandb_api_key.txt", "wandb_secret.json"}, timeout, only_selected)
    if name.endswith(".txt"):
        key = value
    else:
        credential = json.loads(value)
        if only_selected and (not isinstance(credential, dict) or set(credential) != {"api_key"}):
            raise CredentialFileSetError("CPU W&B credential JSON must contain only api_key")
        key = credential["api_key"]
    if key != credentials.wandb_key:
        raise RuntimeError("Private dataset W&B key does not match local key.txt")


def _ensure_single_credential(account, username, slug, title, filename, value, timeout):
    ref = username + "/" + slug
    deadline = time.monotonic() + timeout
    if private_dataset_info(account, ref, username) is None:
        with TemporaryDirectory(prefix="uav-cpu-launch-") as temp:
            folder = Path(temp)
            folder.chmod(0o700)
            (folder / filename).write_text(value)
            (folder / "dataset-metadata.json").write_text(json.dumps({
                "id": ref, "title": title, "licenses": [{"name": "CC0-1.0"}]}))
            run_kaggle(["datasets", "create", "-p", temp, "-q"], account, timeout=timeout)
        while private_dataset_info(account, ref, username) is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Private CPU credential dataset was not ready")
            time.sleep(2)
    # Kaggle can expose owned private metadata before DownloadDataset is ready.
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Private credential dataset download was not ready")
        try:
            _, token = private_dataset_file(account, ref, username, {filename},
                                            max(1, int(remaining)), only_selected=True)
            break
        except RuntimeError as error:
            if '404 Client Error' not in str(error):
                raise
            time.sleep(min(2, remaining))
    if token != value:
        raise RuntimeError("Private CPU-flow credential does not match the selected local credential")
    return ref


def ensure_pipeline_dataset(account, username, timeout=120):
    """Only this selected account's token; never the full local key.txt."""
    return _ensure_single_credential(account, username, "uav-cpu-launch-credential",
                                      "UAV CPU Launch Credential", "uav_cpu_launch_token.txt", account.token, timeout)


def ensure_cpu_wandb_dataset(account, credentials, username, timeout=120):
    return _ensure_single_credential(account, username, "uav-cpu-wandb-credential",
                                      "UAV CPU W&B Credential", "wandb_api_key.txt", credentials.wandb_key, timeout)


def ensure_storage_dataset(account, username, token, timeout=120):
    """One HF token in its own owned private dataset, never in a job manifest."""
    return _ensure_single_credential(account, username, "uav-hf-checkpoint-credential",
        "UAV Private Checkpoint Credential", "hf_token.txt", token, timeout)


def kernel_status(ref, account):
    text = run_kaggle(["kernels", "status", ref], account)
    match = re.search(r"has status [\"'](?:KernelWorkerStatus\.)?([a-z_]+)[\"']", text, re.I)
    state = match.group(1).lower() if match else "unknown"
    return {"cancel_acknowledged": "cancelled", "cancel_requested": "running",
            "new_script": "pending"}.get(state, state)


def pushed_kernel_ref(response):
    # Kaggle derives new slugs from titles and its CLI may return exit 0 on errors.
    match = re.search(r"https://(?:www\.)?kaggle\.com/code/([\w-]+/[\w-]+)", response)
    if "Kernel push error" in response or re.search(r"not valid .* sources", response) or not match:
        raise RuntimeError(f"Kaggle push was not confirmed: {response[-1000:]}")
    return match.group(1)
