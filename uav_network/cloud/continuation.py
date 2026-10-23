"""Submit the next private GPU segment from Kaggle after verified real progress."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from ..configuration import load_config
from .client import account_quotas, private_dataset_file, pushed_kernel_ref, run_kaggle, verify_credential_dataset
from .credentials import Account, Credentials
from .kernel import kernel_metadata, kernel_script


def kernel_ref_for_segment(root_ref, index, cpu=False):
    """Leave room for the CPU suffix even when a continuation index grows."""
    owner, separator, slug = root_ref.partition("/")
    if (not separator or not re.fullmatch(r"[\w-]+", owner)
            or not re.fullmatch(r"[\w-]+", slug) or type(index) is not int or index < 0):
        raise ValueError("Continuation needs a valid root kernel ref and nonnegative index")
    suffix = f"-c{index:02d}" if index else ""
    available = 50 - len(owner) - 1 - len("-cpu") - len(suffix)
    if available < 1:
        raise ValueError("Continuation kernel ref cannot fit its required suffix")
    prefix = slug[:available].rstrip("-") or "u"
    return owner + "/" + prefix + suffix + ("-cpu" if cpu else "")


def next_stage(parent, results):
    """Validate a whole segment before allowing either GPU or CPU submission."""
    if (not parent.get("auto_continue") or not parent.get("cpu_visualize")
            or parent.get("stage") not in {"train", "validate"} or len(parent.get("runs", {})) != 1):
        raise ValueError("Automatic continuation requires one train/validate learner and CPU visualization")
    algorithm = next(iter(parent["runs"]))
    if algorithm not in {"masac", "matd3"} or set(results) != {algorithm}:
        raise ValueError("Continuation requires the selected learner's successful summary")
    config = load_config(parent["config"], parent["overrides"])
    target = parent.get("requested_episodes", config.runtime.episodes)
    if (type(target) is not int or target < 1 or target != config.runtime.episodes
            or not re.fullmatch(r"[0-9a-f]{40}", parent.get("commit", ""))):
        raise ValueError("Continuation requires the unchanged total episode budget and pinned commit")
    result = results[algorithm]
    completed, initial = result.get("episodes"), result.get("initial_episodes")
    if (result.get("algorithm") != algorithm or result.get("requested_episodes") != target
            or result.get("tracking_status") != "finished"
            or type(completed) is not int or type(initial) is not int
            or not 0 <= initial <= completed <= target
            or result.get("segment_episodes", completed - initial) != completed - initial):
        raise ValueError("Continuation summary must match the target and verified episode prefix")
    initial_step, current_step = result.get("initial_global_step"), result.get("global_step")
    expected_step = parent.get("continuation_expected_global_step", {}).get(algorithm)
    if type(current_step) is not int or current_step < 0:
        raise ValueError("Continuation global-step counter is invalid")
    # Legacy positive-episode summaries did not expose initial_global_step.
    # Every new child records it, and zero-episode segments require exact worlds.
    step_metadata = initial_step is not None or expected_step is not None or initial == completed
    if step_metadata and (type(initial_step) is not int or type(current_step) is not int
                          or not 0 <= initial_step < current_step
                          or (expected_step is not None and initial_step != expected_step)):
        raise ValueError("Continuation requires verified positive global-step progress and input prefix")
    if initial == completed and result.get("environment_state_saved") is not True:
        raise ValueError("Zero-episode continuation requires exact saved environment state")
    expected = parent.get("continuation_expected_episodes", {}).get(algorithm)
    index = parent.get("continuation_index", 0)
    if (type(index) is not int or index < 0 or index > current_step
            or (index and expected is None) or (expected is not None and initial != expected)
            or (index and initial == completed and expected_step is None)):
        raise ValueError("Continuation initial episodes differ from its verified input prefix")
    artifact = result.get("checkpoint_artifact")
    prefix = re.escape(parent["entity"] + "/" + parent["project"] + "/")
    if (not isinstance(artifact, str) or not re.fullmatch(prefix + r"[\w.-]+:v[0-9]+", artifact)
            or artifact in {parent.get("warm_start_artifacts", {}).get(algorithm),
                            parent.get("resolved_warm_start_artifacts", {}).get(algorithm)}):
        raise ValueError("Continuation requires a newly uploaded immutable checkpoint artifact")
    complete = result.get("episode_budget_complete")
    if type(complete) is not bool or complete != (completed == target):
        raise ValueError("Continuation episode completion flag disagrees with its exact budget")
    if result.get("stop_reason") != ("episode_budget" if complete else "time_budget"):
        raise ValueError("Continuation requires an episode-budget or successful time-budget stop")
    return "cpu" if complete else "gpu"


def build_continuation_spec(parent, results):
    if next_stage(parent, results) != "gpu":
        raise ValueError("Completed episode budget needs CPU visualization, not GPU continuation")
    algorithm = next(iter(parent["runs"]))
    index = parent.get("continuation_index", 0) + 1
    root = parent.get("root_kernel_ref", parent["kernel_ref"])
    if root.split("/", 1)[0] != parent["kernel_ref"].split("/", 1)[0]:
        raise ValueError("Continuation must keep the original selected account")
    fields = ("commit", "account", "entity", "project", "stage", "config", "overrides",
              "credential_dataset", "pipeline_credential_dataset", "cpu_credential_dataset",
              "cpu_eval", "storage_credential_dataset", "checkpoint_repo", "timeout_seconds", "accelerator", "gpus_per_run", "requested_episodes", "training_mode")
    child = {key: deepcopy(parent[key]) for key in fields if key in parent}
    if not child.get("pipeline_credential_dataset") or not child.get("cpu_credential_dataset"):
        raise ValueError("Continuation needs its existing private GPU and CPU credential inputs")
    config = load_config(parent["config"], parent["overrides"])
    mode = parent.get("training_mode", "pql" if config.pql.enabled else "base")
    root_name = parent.get("root_run_name", parent["runs"][algorithm]["name"])
    identity = algorithm.upper() + "-" + mode
    if identity.lower() not in root_name.lower():
        root_name = identity + "-" + root_name
    run_id = uuid4().hex[:8]
    child.update(submitted_at_utc=datetime.now(timezone.utc).isoformat(),
                 kernel_ref=kernel_ref_for_segment(root, index), root_kernel_ref=root,
                 parent_kernel_ref=parent["kernel_ref"], continuation_index=index,
                 root_run_name=root_name, auto_continue=True, cpu_visualize=True,
                 requested_episodes=config.runtime.episodes, training_mode=mode,
                 warm_start_artifacts={algorithm: results[algorithm]["checkpoint_artifact"]},
                 continuation_expected_episodes={algorithm: results[algorithm]["episodes"]},
                 continuation_expected_global_step={algorithm: results[algorithm]["global_step"]},
                 continuation_requires_environment_state=(
                     results[algorithm]["episodes"] == results[algorithm]["initial_episodes"]),
                 runs={algorithm: {"id": run_id, "name": f"{root_name}-c{index:02d}-{run_id}"}})
    return child


def check_submission_deadline(parent):
    started = os.environ.get("UAV_JOB_STARTED_AT")
    if started is not None:
        remaining = min(parent.get("timeout_seconds", 39600), 39600) - (time.monotonic() - float(started))
        if remaining <= 60:
            raise TimeoutError("No safe time remains to submit a continuation before the kernel deadline")
        return min(120, max(1, int(remaining - 60)))
    return 120


def write_failure(output, error, manifest=None):
    """Retain a credential-free scheduling record alongside the parent results."""
    output = Path(output)
    failure = {"error_type": type(error).__name__}
    path = output / "continuation.json"
    if manifest is None and path.is_file():
        manifest = json.loads(path.read_text())
    path.write_text(json.dumps({**(manifest or {}), "status": "failed", **failure}, indent=2))
    (output / "continuation-failure.json").write_text(json.dumps(failure, indent=2))


def submit_continuation(parent, results, output, account=None, credentials=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"parent_kernel_ref": parent["kernel_ref"],
                "root_kernel_ref": parent.get("root_kernel_ref", parent["kernel_ref"]),
                "continuation_index": parent.get("continuation_index", 0) + 1, "status": "pending"}
    try:
        spec = build_continuation_spec(parent, results)
        if account is None:
            token = os.environ.get("KAGGLE_API_TOKEN")
            if not token:
                raise RuntimeError("Autonomous GPU continuation requires its private Kaggle credential")
            account = Account(parent["account"], token)
        if credentials is None:
            key = os.environ.get("WANDB_API_KEY")
            if not key:
                raise RuntimeError("Autonomous GPU continuation requires its private W&B credential")
            credentials = Credentials([account], key, parent["entity"], parent["project"])
        if account.name != parent["account"] or not account.enabled:
            raise ValueError("Continuation must use the same enabled selected account")
        serialized = json.dumps(spec)
        if any(value and value in serialized for value in
               [credentials.wandb_key, *(a.token for a in credentials.accounts), account.token]):
            raise ValueError("Refusing a continuation manifest containing credentials")
        manifest = {**spec, "status": "pending"}
        (output / "continuation.json").write_text(json.dumps(manifest, indent=2))
        check_submission_deadline(parent)
        quotas = account_quotas(credentials, requested=account.name)
        quota = next((q for q in quotas if q.get("name") == account.name and q.get("ok")), None)
        remaining = quota.get("remaining_hours") if quota else None
        manifest["failure_reason"] = "quota_unavailable"
        if (type(remaining) not in (float, int) or not math.isfinite(remaining)
                or remaining < max(.25, credentials.min_gpu_hours)):
            raise RuntimeError("Selected account GPU quota is unavailable or insufficient for continuation")
        manifest["remaining_gpu_hours"] = remaining
        manifest["failure_reason"] = "insufficient_quota"
        # Keep a quarter hour of verified quota as a scheduling/teardown cushion.
        seconds = min(spec.get("timeout_seconds", 39600), 39600, math.floor((remaining - .25) * 3600))
        config = load_config(spec["config"], spec["overrides"])
        reserve = max(600, config.tracking.artifact_timeout_seconds + 120)
        if seconds <= reserve:
            raise RuntimeError("Selected account GPU quota cannot cover a safe training and upload segment")
        spec["timeout_seconds"] = seconds
        manifest = {**spec, "status": "pending", "remaining_gpu_hours": remaining,
                    "failure_reason": "credential_verification_failed"}
        (output / "continuation.json").write_text(json.dumps(manifest, indent=2))
        username = spec["kernel_ref"].split("/", 1)[0]
        verify_credential_dataset(account, credentials, spec["credential_dataset"], username,
                                  timeout=check_submission_deadline(parent))
        verify_credential_dataset(account, credentials, spec["cpu_credential_dataset"], username,
                                  timeout=check_submission_deadline(parent), only_selected=True)
        _, token = private_dataset_file(account, spec["pipeline_credential_dataset"], username,
                                        {"uav_cpu_launch_token.txt"},
                                        timeout=check_submission_deadline(parent), only_selected=True)
        if token != account.token:
            raise RuntimeError("Private continuation credential differs from the selected account")
        folder = output / "continuation-kernel"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "kernel.py").write_text(kernel_script(spec))
        (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(spec), indent=2))
        check_submission_deadline(parent)
        manifest["failure_reason"] = "submission_failed"
        response = run_kaggle(["kernels", "push", "-p", str(folder), "--accelerator",
                               spec.get("accelerator", "NvidiaTeslaT4"), "--timeout", str(seconds)],
                              account, timeout=50)
        ref = pushed_kernel_ref(response)
        if ref.split("/", 1)[0] != username:
            raise RuntimeError("Kaggle continuation submission returned a different account")
        manifest.update(status="submitted", kernel_ref=ref,
                        kernel_url="https://www.kaggle.com/code/" + ref,
                        wandb_urls={a: f"https://wandb.ai/{spec['entity']}/{spec['project']}/runs/{r['id']}"
                                    for a, r in spec["runs"].items()})
        manifest.pop("failure_reason", None)
        (output / "continuation.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"continuation_url": manifest["kernel_url"]}), flush=True)
        return manifest
    except Exception as error:
        write_failure(output, error, manifest)
        raise
