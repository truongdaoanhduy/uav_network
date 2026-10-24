"""CPU-only Kaggle supervisor for one immutable training lineage.

GPU jobs own their normal handoff. This worker reconciles only terminal jobs,
waits for the same account's quota, and observes final CPU media delivery.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from .client import kernel_status, private_dataset_file, pushed_kernel_ref, run_kaggle, verify_credential_dataset
from .continuation import _owned_kernel_exists, kernel_ref_for_segment, next_stage, submit_continuation
from .credentials import Account, Credentials
from .kernel import kernel_metadata, kernel_script
from .visualize import build_visualization_spec, submit_visualization

ACTIVE = {"running", "pending", "queued"}
TERMINAL = {"complete", "error", "failed", "cancelled"}
CONTROL_NAMES = ("job.json", "job-result.json", "continuation.json", "cpu-visualization.json",
                 "visualization-result.json", "evaluation-progress.json")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def controller_ref(root_ref, generation):
    owner, separator, slug = root_ref.partition("/")
    if (not separator or not re.fullmatch(r"[\w-]+", owner)
            or not re.fullmatch(r"[\w-]+", slug) or type(generation) is not int or generation < 0):
        raise ValueError("Controller requires a valid lineage and generation")
    digest = hashlib.sha256(root_ref.encode()).hexdigest()[:14]
    ref = f"{owner}/uav-flow-{digest}-{generation:04d}"
    if len(ref) > 50:
        raise ValueError("Controller reference exceeds Kaggle's limit")
    return ref


def build_controller_spec(parent, commit, *, gpu_commit=None, state=None, generation=0):
    root = parent.get("root_kernel_ref", parent["kernel_ref"])
    if (not parent.get("auto_continue") or not parent.get("cpu_visualize")
            or len(parent.get("runs", {})) != 1
            or root.split("/")[0] != parent["kernel_ref"].split("/")[0]):
        raise ValueError("Controller requires one existing automatic same-account lineage")
    gpu_commit = gpu_commit or commit
    if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in (commit, gpu_commit)):
        raise ValueError("Controller and recovery sources must be immutable commits")
    state = deepcopy(state) if state is not None else {
        "root_kernel_ref": root, "current_gpu_ref": parent["kernel_ref"], "phase": "training"}
    if state.get("root_kernel_ref") != root:
        raise ValueError("Controller state differs from its lineage")
    identity = uuid4().hex[:8]
    return {"stage": "continuation-watch", "commit": commit, "gpu_commit": gpu_commit,
            "account": parent["account"], "entity": parent["entity"], "project": parent["project"],
            "kernel_ref": controller_ref(root, generation), "root_kernel_ref": root,
            "credential_dataset": parent["cpu_credential_dataset"],
            "pipeline_credential_dataset": parent["pipeline_credential_dataset"],
            "requires_kaggle_token": True, "timeout_seconds": 39600, "generation": generation,
            "poll_seconds": 900, "parent": deepcopy(parent), "state": state,
            "run": {"id": identity, "name": f"FLOW-{parent.get('training_mode', 'train')}-{root.split('/')[-1]}-g{generation}"},
            "submitted_at_utc": datetime.now(timezone.utc).isoformat()}


def _check_identity(spec, account, credentials):
    owner = spec["root_kernel_ref"].split("/")[0]
    if (account.name != spec["account"] or not account.enabled
            or spec["kernel_ref"].split("/")[0] != owner
            or (credentials.entity, credentials.project) != (spec["entity"], spec["project"])):
        raise ValueError("Cloud controller must keep the enabled account and main W&B project")
    serialized = json.dumps(spec)
    if any(secret and secret in serialized for secret in
           [credentials.wandb_key, *(a.token for a in credentials.accounts)]):
        raise ValueError("Refusing a controller manifest containing credentials")
    return owner


def submit_controller(spec, output, account, credentials):
    owner = _check_identity(spec, account, credentials)
    if _owned_kernel_exists(spec["kernel_ref"], account):
        return {**spec, "status": "adopted", "kernel_url": "https://www.kaggle.com/code/" + spec["kernel_ref"]}
    verify_credential_dataset(account, credentials, spec["credential_dataset"], owner, only_selected=True)
    _, token = private_dataset_file(account, spec["pipeline_credential_dataset"], owner,
                                    {"uav_cpu_launch_token.txt"}, only_selected=True)
    if token != account.token:
        raise ValueError("Controller credential differs from the selected account")
    folder = Path(output) / f"controller-{spec['generation']:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "job.json", spec)
    (folder / "kernel.py").write_text(kernel_script(spec))
    atomic_json(folder / "kernel-metadata.json", kernel_metadata(spec))
    response = run_kaggle(["kernels", "push", "-p", str(folder), "--timeout",
                           str(spec["timeout_seconds"])], account, timeout=50)
    ref = pushed_kernel_ref(response)
    if ref != spec["kernel_ref"]:
        raise RuntimeError("Controller submission returned an unexpected reference")
    manifest = {**spec, "status": "submitted", "kernel_url": "https://www.kaggle.com/code/" + ref}
    atomic_json(folder / "launch.json", manifest)
    return manifest


def read_controls(ref, account):
    """Fetch only small pipeline receipts, never a full replay checkpoint."""
    with TemporaryDirectory(prefix="uav-flow-controls-") as directory:
        folder = Path(directory)
        pattern = r"(^|.*/)(" + "|".join(re.escape(name) for name in CONTROL_NAMES) + r")$"
        run_kaggle(["kernels", "output", ref, "-p", str(folder),
                    "--file-pattern", pattern, "-q"], account, timeout=120)
        result = {}
        for name in CONTROL_NAMES:
            paths = list(folder.rglob(name))
            if len(paths) > 1:
                raise ValueError("Ambiguous cloud control receipt")
            if paths:
                if paths[0].stat().st_size > 32 * 1024 * 1024:
                    raise ValueError("Cloud control receipt exceeds its size limit")
                result[name] = json.loads(paths[0].read_text())
        return result


def _validate_job(job, ref, spec, account):
    if (job.get("kernel_ref") != ref or job.get("account") != account.name
            or job.get("root_kernel_ref", ref) != spec["root_kernel_ref"]
            or ref.split("/")[0] != spec["root_kernel_ref"].split("/")[0]):
        raise ValueError("Downloaded job differs from the selected account/lineage")
    if job.get("stage") != "visualize" and any(
            job.get(key) != spec["parent"].get(key) for key in
            ("config", "overrides", "requested_episodes", "training_mode", "entity", "project")):
        raise ValueError("Downloaded GPU job changes the original scientific configuration")


def submit_cpu_attempt(cpu, output, account, credentials):
    """Submit a new CPU attempt; callers first reconcile its unique reference."""
    from .visualize import submit_visualization_spec
    return submit_visualization_spec(cpu, output, account, credentials)


def _cpu_complete(cpu, results):
    if set(results) != set(cpu.get("artifacts", {})):
        return False
    settings = cpu["cpu_eval"]
    for algorithm, result in results.items():
        records = result.get("records", [])
        publication = result.get("publication", {})
        hf = publication.get("huggingface", {})
        if (result.get("device") != "cpu" or result.get("episodes") != settings["episodes"]
                or result.get("checkpoint_artifact") != cpu["artifacts"][algorithm]
                or len(records) != settings["episodes"]
                or any(record.get("seed") != settings["seed"] + i or record.get("episode") != i+1
                       for i, record in enumerate(records))
                or publication.get("wandb", {}).get("status") != "ok" or hf.get("status") != "ok"
                or not re.fullmatch(r"[0-9a-f]{40}", str(hf.get("revision", "")))
                or any(not str(hf.get("files", {}).get(role, {}).get("url", "")).startswith("https://huggingface.co/")
                       for role in ("video", "image", "metrics"))):
            return False
    return True


def _cpu_step(spec, state, account, credentials, output):
    ref = state["current_cpu_ref"]
    status = kernel_status(ref, account)
    if status in ACTIVE:
        state.update(phase="visualizing", remote_status=status)
        return state
    if status not in TERMINAL:
        state.update(phase="service_unavailable", remote_status=status)
        return state
    controls = read_controls(ref, account)
    cpu = controls.get("job.json", {})
    _validate_job(cpu, ref, spec, account)
    if cpu.get("stage") != "visualize" or not cpu.get("artifacts"):
        raise ValueError("CPU receipt is not a visualization")
    if state.get("expected_artifacts") and state["expected_artifacts"] != cpu["artifacts"]:
        raise ValueError("CPU visualization differs from the completed training artifact")
    if cpu.get("cpu_eval") != spec["parent"]["cpu_eval"]:
        raise ValueError("CPU evaluation settings differ from the configured protocol")
    results = controls.get("visualization-result.json", {})
    if _cpu_complete(cpu, results):
        state.update(phase="complete", remote_status=status,
                     visualization_runs={a: r.get("wandb_url") for a, r in results.items()},
                     publication={a: r["publication"] for a, r in results.items()})
        return state
    attempt = state.get("cpu_attempt", 0) + 1
    recovery = deepcopy(cpu)
    digest = hashlib.sha256(spec["root_kernel_ref"].encode()).hexdigest()[:14]
    recovery.update(commit=spec["commit"], kernel_ref=f"{ref.split('/')[0]}/uav-media-{digest}-a{attempt:02d}",
                    timeout_seconds=39600, session_resumable=True, requires_kaggle_token=True,
                    pipeline_credential_dataset=spec["pipeline_credential_dataset"])
    recovery["runs"] = {a: {"id": uuid4().hex[:8], "name": f"{a.upper()}-CPU-visualize-recovery-{attempt}"}
                        for a in cpu["artifacts"]}
    # Completed evaluation needs upload recovery, not another evaluation.
    if set(results) == set(cpu["artifacts"]) and all(
            r.get("episodes") == cpu["cpu_eval"]["episodes"] and r.get("render_path") and r.get("animation_path")
            for r in results.values()):
        recovery["publication_source_ref"] = ref
        recovery.pop("completed_records", None)
    else:
        progress = controls.get("evaluation-progress.json", {})
        if progress:
            if progress.get("artifacts") != cpu["artifacts"] or progress.get("cpu_eval") != cpu["cpu_eval"]:
                raise ValueError("CPU progress differs from its immutable policy/protocol")
            recovery["completed_records"] = progress.get("records", {})
        # A failed publication-only attempt retains its original asset source.
    completed = 0
    for records in recovery.get("completed_records", {}).values():
        if (not isinstance(records, list) or len(records) > cpu["cpu_eval"]["episodes"]
                or any(not isinstance(r, dict) or r.get("episode") != i+1
                       or r.get("seed") != cpu["cpu_eval"]["seed"]+i for i, r in enumerate(records))):
            raise ValueError("CPU recovery records differ from the exact episode/seed prefix")
        completed += len(records)
    advanced = completed > state.get("evaluation_records_completed", 0)
    failures = 0 if advanced else state.get("cpu_failures", 0) + 1
    if failures > 3:
        state.update(phase="blocked", error_type="CPURecoveryAttemptsExhausted")
        return state
    state.update(cpu_failures=failures, evaluation_records_completed=max(
        completed, state.get("evaluation_records_completed", 0)))
    target = recovery["kernel_ref"]
    if _owned_kernel_exists(target, account):
        state.update(current_cpu_ref=target, cpu_attempt=attempt, phase="visualizing")
        return state
    state["submission_intent"] = {"ref": target, "kind": "cpu", "attempt": attempt}
    atomic_json(output / "controller-state.json", state)
    manifest = submit_cpu_attempt(recovery, output, account, credentials)
    state.pop("submission_intent", None)
    state.update(current_cpu_ref=manifest["kernel_ref"], cpu_attempt=attempt, phase="visualizing")
    return state


def controller_step(spec, state, account, credentials, output):
    """Perform one reconciled decision, with all progress in a portable state."""
    _check_identity(spec, account, credentials)
    output = Path(output)
    try:
        intent = state.get("submission_intent")
        if intent:
            if not _owned_kernel_exists(intent["ref"], account):
                state.update(phase="submission_uncertain", error_type="UnconfirmedSubmission")
                return state
            field = "current_cpu_ref" if intent["kind"] == "cpu" else "current_gpu_ref"
            state[field] = intent["ref"]
            if intent.get("attempt"):
                state["cpu_attempt"] = intent["attempt"]
            state.pop("submission_intent", None)
        if state.get("current_cpu_ref"):
            return _cpu_step(spec, state, account, credentials, output)
        ref = state["current_gpu_ref"]
        status = kernel_status(ref, account)
        if status in ACTIVE:
            state.update(phase="training", remote_status=status)
            return state
        if status not in TERMINAL:
            state.update(phase="service_unavailable", remote_status=status)
            return state
        controls = read_controls(ref, account)
        parent = controls.get("job.json", {})
        _validate_job(parent, ref, spec, account)
        results = controls.get("job-result.json", {})
        stage = next_stage(parent, results)
        state["verified_prefix"] = {a: {k: r[k] for k in ("episodes", "global_step")} for a, r in results.items()}
        recovery = {**parent, "timeout_seconds": 39600}
        if stage == "gpu":
            target = kernel_ref_for_segment(spec["root_kernel_ref"], parent.get("continuation_index", 0) + 1)
            if _owned_kernel_exists(target, account):
                state.update(current_gpu_ref=target, phase="training")
                return state
            recovery["commit"] = spec["gpu_commit"]
            state["submission_intent"] = {"ref": target, "kind": "gpu"}
            atomic_json(output / "controller-state.json", state)
            manifest = submit_continuation(recovery, results, output, account=account, credentials=credentials)
            state.pop("submission_intent", None)
            if manifest.get("status") == "waiting_for_quota":
                state.update(phase="waiting_for_quota", wait_reason=manifest.get("wait_reason"))
            elif manifest.get("status") == "submitted":
                state.update(current_gpu_ref=manifest["kernel_ref"], phase="training")
            else:
                raise ValueError("Unrecognized continuation submission receipt")
        else:
            target = kernel_ref_for_segment(spec["root_kernel_ref"], 0, cpu=True)
            state["expected_artifacts"] = {a: r["checkpoint_artifact"] for a, r in results.items()}
            if _owned_kernel_exists(target, account):
                state.update(current_cpu_ref=target, phase="visualizing")
                return state
            recovery["commit"] = spec["commit"]
            state["submission_intent"] = {"ref": target, "kind": "cpu"}
            atomic_json(output / "controller-state.json", state)
            manifest = submit_visualization(recovery, results, output, account=account, credentials=credentials)
            state.pop("submission_intent", None)
            state.update(current_cpu_ref=manifest["kernel_ref"], phase="visualizing")
        return state
    except (ValueError, KeyError, TypeError) as error:
        state.update(phase="blocked", error_type=type(error).__name__)
        return state


def run_controller(spec, output=Path("/kaggle/working")):
    import wandb
    account = Account(spec["account"], os.environ["KAGGLE_API_TOKEN"])
    credentials = Credentials([account], os.environ["WANDB_API_KEY"], spec["entity"], spec["project"])
    _check_identity(spec, account, credentials)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = deepcopy(spec["state"])
    started = float(os.environ.get("UAV_JOB_STARTED_AT", time.monotonic()))
    deadline = started + min(spec["timeout_seconds"], 39600) - 600
    run = wandb.init(entity=spec["entity"], project=spec["project"], id=spec["run"]["id"],
                    name=spec["run"]["name"], job_type="cloud-controller", mode="online",
                    config={k: spec[k] for k in ("stage", "commit", "gpu_commit", "kernel_ref",
                                                "account", "root_kernel_ref", "generation")},
                    settings=wandb.Settings(init_timeout=120))
    failed = True
    try:
        while True:
            if time.monotonic() >= deadline:
                successor = build_controller_spec(spec["parent"], spec["commit"], gpu_commit=spec["gpu_commit"],
                    state=state, generation=spec["generation"] + 1)
                try:
                    manifest = submit_controller(successor, output, account, credentials)
                except Exception as error:
                    state.update(controller_status="handoff_pending", error_type=type(error).__name__)
                    atomic_json(output / "controller-state.json", state)
                    run.summary.update(state)
                    if time.monotonic() >= started + min(spec["timeout_seconds"], 39600) - 120:
                        raise RuntimeError("Controller successor could not be confirmed before session deadline") from None
                    time.sleep(60)
                    continue
                state["next_controller_ref"] = manifest["kernel_ref"]
                state["controller_status"] = "handed_off"
                atomic_json(output / "controller-state.json", state)
                run.summary.update(state)
                failed = False
                return state
            try:
                state = controller_step(spec, state, account, credentials, output)
            except Exception as error:
                # Service failures are retried without leaking command output or credentials.
                state.update(phase="service_unavailable", error_type=type(error).__name__)
            state["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / "controller-state.json", state)
            run.summary.update(state)
            run.log({"controller/heartbeat": 1, "controller/phase": state["phase"]})
            print(json.dumps({k: state[k] for k in ("phase", "current_gpu_ref", "heartbeat_utc")
                              if k in state}), flush=True)
            if state["phase"] in {"complete", "blocked"}:
                failed = state["phase"] != "complete"
                return state
            time.sleep(min(spec.get("poll_seconds", 900), max(1, deadline - time.monotonic())))
    finally:
        run.finish(exit_code=1 if failed else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    run_controller(json.loads(Path(args.spec).read_text()))


if __name__ == "__main__":
    main()
