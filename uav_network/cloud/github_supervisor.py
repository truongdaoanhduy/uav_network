"""Short-lived GitHub Actions reconciler for Kaggle training lineages."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

from .client import kernel_status, run_kaggle
from .continuation import (_owned_kernel_exists, kernel_ref_for_segment,
                           next_stage, submit_continuation)
from .credentials import Account, Credentials
from .visualize import submit_visualization, submit_visualization_spec

ACTIVE = {"running", "pending", "queued"}
TERMINAL = {"complete", "error", "failed", "cancelled"}
CONTROL_NAMES = ("job.json", "job-result.json", "continuation.json",
                 "cpu-visualization.json", "visualization-result.json",
                 "evaluation-progress.json", "job-failures.json")
ACCOUNT_ENV_RE = re.compile(r"^KAGGLE_API_TOKEN_ACCOUNT_(\d{2})$")


def credentials_from_environment(env=None):
    env = os.environ if env is None else env
    accounts = []
    for index in range(1, 25):
        token = str(env.get(f"KAGGLE_API_TOKEN_ACCOUNT_{index:02d}", "")).strip()
        if token:
            accounts.append(Account(f"account_{index:02d}", token))
    key = str(env.get("WANDB_API_KEY", "")).strip()
    entity = str(env.get("WANDB_ENTITY", "")).strip()
    project = str(env.get("WANDB_PROJECT", "")).strip()
    if not accounts:
        raise RuntimeError("No GitHub Kaggle account secret is configured")
    if not key or not entity or not project:
        raise RuntimeError("GitHub supervisor requires W&B key/entity/project")
    if not str(env.get("HF_TOKEN", "")).strip() or not str(env.get("HF_REPO_ID", "")).strip():
        raise RuntimeError("GitHub supervisor requires HF token/repository")
    return Credentials(accounts, key, entity, project)


def read_controls(ref, account):
    """Fetch only bounded control JSON from a terminal Kaggle job."""
    with TemporaryDirectory(prefix="uav-gh-controls-") as directory:
        folder = Path(directory)
        pattern = r"(^|.*/)(" + "|".join(re.escape(name) for name in CONTROL_NAMES) + r")$"
        run_kaggle(["kernels", "output", ref, "-p", str(folder),
                    "--file-pattern", pattern, "-q"], account, timeout=120)
        result = {}
        for name in CONTROL_NAMES:
            matches = list(folder.rglob(name))
            if len(matches) > 1:
                raise ValueError("Ambiguous cloud control receipt")
            if matches:
                if matches[0].stat().st_size > 32 * 1024 * 1024:
                    raise ValueError("Cloud control receipt exceeds its size limit")
                value = json.loads(matches[0].read_text())
                if not isinstance(value, dict):
                    raise ValueError("Cloud control receipt must be a JSON object")
                result[name] = value
        return result


def _validate_job(ref, job, account):
    root = job.get("root_kernel_ref", ref)
    if (job.get("kernel_ref") != ref or job.get("account") != account.name
            or ref.split("/", 1)[0] != root.split("/", 1)[0]):
        raise ValueError("Cloud receipt differs from the selected account/lineage")


def _visualization_complete(job, results):
    if not isinstance(results, dict) or set(results) != set(job.get("artifacts", {})):
        return False
    settings = job.get("cpu_eval", {})
    for algorithm, result in results.items():
        publication = result.get("publication", {})
        hf = publication.get("huggingface", {}) if isinstance(publication, dict) else {}
        if (not isinstance(result, dict) or result.get("device") != "cpu"
                or result.get("episodes") != settings.get("episodes")
                or result.get("checkpoint_artifact") != job["artifacts"][algorithm]
                or publication.get("wandb", {}).get("status") != "ok"
                or hf.get("status") != "ok"):
            return False
    return True


def cpu_retry_ref(root_ref, attempt):
    owner, separator, _ = root_ref.partition("/")
    if not separator or not re.fullmatch(r"[\w-]+", owner) or type(attempt) is not int or attempt < 1:
        raise ValueError("CPU retry requires a valid root lineage and positive attempt")
    digest = hashlib.sha256(root_ref.encode()).hexdigest()[:14]
    ref = f"{owner}/uav-media-{digest}-a{attempt:02d}"
    if len(ref) > 50:
        raise ValueError("CPU retry reference exceeds Kaggle's limit")
    return ref


def _cpu_retry_spec(job, controls):
    attempt = job.get("cpu_attempt", 0) + 1
    if type(attempt) is not int or not 1 <= attempt <= 20:
        raise ValueError("CPU retry attempt is invalid")
    recovery = deepcopy(job)
    recovery.update(kernel_ref=cpu_retry_ref(job["root_kernel_ref"], attempt),
                    parent_kernel_ref=job["kernel_ref"], cpu_attempt=attempt,
                    timeout_seconds=39600, session_resumable=True)
    recovery["runs"] = {algorithm: {"id": uuid4().hex[:8],
        "name": f"{identity.get('name', algorithm.upper() + '-EVALCPU')}-R{attempt:02d}"}
        for algorithm, identity in job.get("runs", {}).items()}
    progress = controls.get("evaluation-progress.json", {})
    if progress:
        if (progress.get("artifacts") != job.get("artifacts")
                or progress.get("cpu_eval") != job.get("cpu_eval")
                or not isinstance(progress.get("records"), dict)):
            raise ValueError("CPU progress differs from its immutable evaluation protocol")
        for records in progress["records"].values():
            if not isinstance(records, list):
                raise ValueError("CPU progress records must be lists")
        recovery["completed_records"] = progress["records"]
    results = controls.get("visualization-result.json", {})
    if (results and set(results) == set(job.get("artifacts", {}))
            and all(isinstance(value, dict) and value.get("episodes") == job.get("cpu_eval", {}).get("episodes")
                    and value.get("render_path") and value.get("animation_path") for value in results.values())):
        recovery.pop("completed_records", None)
        recovery["publication_source_ref"] = job["kernel_ref"]
        recovery["requires_kaggle_token"] = True
    return recovery


def _submit_output(action):
    return TemporaryDirectory(prefix=f"uav-gh-{action}-")


def reconcile_ref(ref, account, credentials, *, dry_run=False):
    status = kernel_status(ref, account)
    base = {"ref": ref, "status": status}
    if status in ACTIVE:
        return {**base, "action": "active"}
    if status not in TERMINAL:
        return {**base, "action": "service_unavailable"}

    controls = read_controls(ref, account)
    job = controls.get("job.json")
    if not isinstance(job, dict) or job.get("orchestration") != "github_actions":
        return {**base, "action": "ignored"}
    _validate_job(ref, job, account)

    if job.get("stage") == "visualize":
        if status == "complete" and _visualization_complete(job, controls.get("visualization-result.json", {})):
            return {**base, "action": "cpu_complete"}
        recovery = _cpu_retry_spec(job, controls)
        target = recovery["kernel_ref"]
        if _owned_kernel_exists(target, account):
            return {**base, "action": "existing_cpu_retry", "child_ref": target}
        if dry_run:
            return {**base, "action": "would_submit_cpu_retry", "child_ref": target}
        with _submit_output("cpu-retry") as directory:
            manifest = submit_visualization_spec(recovery, Path(directory), account, credentials)
        return {**base, "action": "submitted_cpu_retry", "child_ref": manifest["kernel_ref"]}

    if job.get("stage") not in {"train", "validate"}:
        return {**base, "action": "ignored"}
    if status != "complete":
        return {**base, "action": "recovery_required"}

    results = controls.get("job-result.json", {})
    if not isinstance(results, dict) or not results:
        return {**base, "action": "recovery_required"}
    action = next_stage(job, results)
    root = job.get("root_kernel_ref", ref)
    if action == "gpu":
        target = kernel_ref_for_segment(root, job.get("continuation_index", 0) + 1)
        if _owned_kernel_exists(target, account):
            return {**base, "action": "existing_gpu", "child_ref": target}
        if dry_run:
            return {**base, "action": "would_submit_gpu", "child_ref": target}
        with _submit_output("gpu") as directory:
            manifest = submit_continuation(job, results, Path(directory),
                                           account=account, credentials=credentials)
        submitted = manifest.get("kernel_ref", target)
        if manifest.get("status") == "waiting_for_quota":
            return {**base, "action": "waiting_for_quota", "child_ref": submitted,
                    "wait_reason": manifest.get("wait_reason")}
        return {**base, "action": "submitted_gpu", "child_ref": submitted}

    target = kernel_ref_for_segment(root, 0, cpu=True)
    if _owned_kernel_exists(target, account):
        return {**base, "action": "existing_cpu", "child_ref": target}
    if dry_run:
        return {**base, "action": "would_submit_cpu", "child_ref": target}
    with _submit_output("cpu") as directory:
        manifest = submit_visualization(job, results, Path(directory),
                                        account=account, credentials=credentials)
    return {**base, "action": "submitted_cpu", "child_ref": manifest["kernel_ref"]}


def _owned_refs(account, limit):
    text = run_kaggle(["kernels", "list", "--mine", "--format", "json",
                       "--page-size", str(limit), "--sort-by", "dateRun"], account)
    if not text.strip() or text.strip().lower().startswith("not found"):
        return []
    rows = json.loads(text)
    if not isinstance(rows, list):
        raise RuntimeError("Kaggle kernel list returned invalid JSON")
    return [row["ref"] for row in rows
            if isinstance(row, dict) and isinstance(row.get("ref"), str)]


def reconcile_account(account, credentials, *, limit=100, dry_run=False):
    outcomes = []
    for ref in _owned_refs(account, limit):
        try:
            outcomes.append(reconcile_ref(ref, account, credentials, dry_run=dry_run))
        except Exception as error:
            outcomes.append({"ref": ref, "action": "reconcile_error",
                             "error_type": type(error).__name__})
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Short-lived GitHub UAV cloud reconciler")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.limit > 500:
        raise ValueError("--limit must be 1..500")
    credentials = credentials_from_environment()
    report = {}
    for account in credentials.accounts:
        report[account.name] = reconcile_account(account, credentials, limit=args.limit,
                                                  dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if any(item.get("action") == "reconcile_error"
                    for rows in report.values() for item in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
