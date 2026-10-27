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
IMMUTABLE_ARTIFACT_RE = re.compile(r"^[\w.-]+/[\w.-]+/[\w.-]+:v[0-9]+$")


class RecoveryUnavailable(RuntimeError):
    """A failed lineage has no proven exact full-state recovery point."""


def credentials_from_environment(env=None):
    env = os.environ if env is None else env
    raw_enabled = str(env.get("KAGGLE_ENABLED_ACCOUNTS", "")).strip()
    enabled = [name.strip() for name in raw_enabled.split(",") if name.strip()]
    if not enabled:
        raise RuntimeError("GitHub supervisor requires KAGGLE_ENABLED_ACCOUNTS")
    if len(enabled) != len(set(enabled)) or any(not re.fullmatch(r"account_\d{2}", name) for name in enabled):
        raise RuntimeError("KAGGLE_ENABLED_ACCOUNTS is invalid")
    accounts = []
    for name in enabled:
        index = int(name.rsplit("_", 1)[1])
        if not 1 <= index <= 24:
            raise RuntimeError("KAGGLE_ENABLED_ACCOUNTS is invalid")
        token = str(env.get(f"KAGGLE_API_TOKEN_ACCOUNT_{index:02d}", "")).strip()
        if not token:
            raise RuntimeError(f"Missing GitHub Kaggle secret for enabled {name}")
        accounts.append(Account(name, token))
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


def verified_recovery_result(job, credentials, *, api=None):
    runs = job.get("runs", {})
    if len(runs) != 1:
        raise RecoveryUnavailable("Recovery requires exactly one learner lineage")
    algorithm, identity = next(iter(runs.items()))
    run_id = identity.get("id") if isinstance(identity, dict) else None
    if not isinstance(run_id, str) or not run_id:
        raise RecoveryUnavailable("Recovery source run is missing")
    try:
        if api is None:
            import wandb
            api = wandb.Api(api_key=credentials.wandb_key, timeout=30)
        run = api.run(f"{job['entity']}/{job['project']}/{run_id}")
        state = str(getattr(run, "state", "")).lower()
        if state not in {"crashed", "failed"}:
            raise RecoveryUnavailable("Recovery is only valid for a failed/crashed source run")
        summary = dict(run.summary)
        reference = summary.get("recovery/checkpoint_artifact")
        episodes = summary.get("recovery/last_verified_episodes")
        global_step = summary.get("recovery/last_verified_global_step")
        if (not isinstance(reference, str) or not IMMUTABLE_ARTIFACT_RE.fullmatch(reference)
                or type(episodes) is not int or episodes < 0
                or type(global_step) is not int or global_step < 0):
            raise RecoveryUnavailable("No immutable confirmed recovery pointer")
        artifact = api.artifact(reference, type="model")
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        entries = getattr(getattr(artifact, "manifest", None), "entries", {})
        if "checkpoint.json" not in entries:
            raise RecoveryUnavailable("Recovery artifact has no immutable HF checkpoint manifest")
        if (metadata.get("resumable") is not True
                or metadata.get("source_run_id") != run_id
                or metadata.get("source_git_commit") != job.get("commit")
                or metadata.get("completed_episodes") != episodes
                or metadata.get("global_step") != global_step):
            raise RecoveryUnavailable("Recovery artifact provenance differs from the failed lineage")
        initial_episodes = job.get("continuation_expected_episodes", {}).get(algorithm, 0)
        initial_step = job.get("continuation_expected_global_step", {}).get(algorithm, 0)
        target = job.get("requested_episodes")
        if (type(initial_episodes) is not int or type(initial_step) is not int
                or type(target) is not int or target < 1
                or episodes < initial_episodes or episodes > target
                or global_step < initial_step
                or (episodes == initial_episodes and global_step == initial_step)):
            raise RecoveryUnavailable("Recovery progress does not advance the immutable segment prefix")
        complete = metadata.get("episode_budget_complete") is True
        if complete and episodes < target:
            raise RecoveryUnavailable("Recovery metadata claims completion before the episode budget")
        return {algorithm: {
            "algorithm": algorithm, "requested_episodes": target,
            "episodes": episodes, "initial_episodes": initial_episodes,
            "segment_episodes": episodes - initial_episodes, "global_step": global_step,
            "updates": 0, "tracking_status": "finished",
            "episode_budget_complete": complete,
            "stop_reason": "episode_budget" if complete else "time_budget",
            "checkpoint_artifact": reference, "wandb_run_id": run_id,
            "recovery_source_state": state, "exact_recovery": True}}
    except RecoveryUnavailable:
        raise
    except Exception as error:
        raise RecoveryUnavailable(type(error).__name__) from error


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
        try:
            recovered = verified_recovery_result(job, credentials)
        except RecoveryUnavailable as error:
            return {**base, "action": "recovery_blocked", "error_type": type(error).__name__}
        recovery_stage = next_stage(job, recovered)
        root = job.get("root_kernel_ref", ref)
        if recovery_stage == "gpu":
            target = kernel_ref_for_segment(root, job.get("continuation_index", 0) + 1)
            if _owned_kernel_exists(target, account):
                return {**base, "action": "existing_recovery_gpu", "child_ref": target}
            if dry_run:
                return {**base, "action": "would_submit_recovery_gpu", "child_ref": target}
            with _submit_output("recovery-gpu") as directory:
                manifest = submit_continuation(job, recovered, Path(directory),
                                               account=account, credentials=credentials)
            return {**base, "action": "submitted_recovery_gpu",
                    "child_ref": manifest.get("kernel_ref", target)}
        target = kernel_ref_for_segment(root, 0, cpu=True)
        if _owned_kernel_exists(target, account):
            return {**base, "action": "existing_recovery_cpu", "child_ref": target}
        if dry_run:
            return {**base, "action": "would_submit_recovery_cpu", "child_ref": target}
        with _submit_output("recovery-cpu") as directory:
            manifest = submit_visualization(job, recovered, Path(directory),
                                            account=account, credentials=credentials)
        return {**base, "action": "submitted_recovery_cpu", "child_ref": manifest["kernel_ref"]}

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
