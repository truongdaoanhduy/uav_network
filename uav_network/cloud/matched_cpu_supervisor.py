"""GitHub Actions watcher: completed matched-seed44 GPU runs -> one CPU Kaggle visualization.

Never starts, retries or resumes a GPU learner. Refuses partial/failed runs.
All source IDs are pinned in the public study manifest; secrets stay in Actions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .credentials import Account, Credentials
from .continuation import _owned_kernel_exists
from .visualize import build_visualization_spec, submit_visualization
from scripts.run_owned_kaggle_study_case import build_case

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "configs/matched-seed44-cpu-automation.json"
VALID_ARTIFACT = re.compile(r"^[\w.-]+/[\w.-]+/[\w.-]+:v([0-9]+)$")


def validate_study(study):
    assert len(study["cases"]) == 10
    assert [row["case"] for row in study["cases"]] == list(range(1, 11))
    assert len({row["run_id"] for row in study["cases"]}) == 10
    assert len({row["kernel_ref"] for row in study["cases"]}) == 10
    assert study["requested_episodes"] == 20000
    assert study["source_commit"].isalnum() and len(study["source_commit"]) == 40
    for row in study["cases"]:
        assert row["account"] == f"account_{row['case']:02d}"
        assert re.fullmatch(r"[\w-]+/[\w-]+", row["kernel_ref"])
        assert re.fullmatch(r"[0-9a-f]{8}", row["run_id"])


def validate_identity(run, row, study):
    """Fail closed on accidental matches from different source/config/seed."""
    c = run.config
    case = build_case(row["case"])
    algorithm = case["algorithm"]
    code = c.get(algorithm, {})
    runtime = c.get("runtime", {})
    expected = dict(
        source_git_commit=study["source_commit"],
        kaggle_kernel=row["kernel_ref"],
        training_mode="pql" if "pql" in case["name"] else "base",
    )
    if (run.id != row["run_id"] or run.group != study["group"]
            or any(c.get(key) != val for key, val in expected.items())
            or runtime.get("algorithm") != algorithm or runtime.get("seed") != 44
            or runtime.get("episodes") != study["requested_episodes"]
            or runtime.get("deterministic") is not True
            or c.get("pql", {}).get("enabled") is not ("pql" in case["name"])
            or code.get("share_actor_parameters") is not ("shared" in case["name"])
            or code.get("share_critic_parameters") is not ("shared" in case["name"])
            or runtime.get("vectorize_critics") is not ("vmap" in case["name"])):
        raise ValueError(f"Case {row['case']:02d}: W&B run/config identity mismatch")
    return case


def final_artifact(run, study):
    """Return only immutable, provenance-matched, fully completed final artifacts."""
    matches = []
    for artifact in run.logged_artifacts():
        if getattr(artifact, "type", None) != "model":
            continue
        meta = dict(getattr(artifact, "metadata", {}) or {})
        qualified = getattr(artifact, "qualified_name", "") or (
            f"{study['entity']}/{study['project']}/{artifact.name}"
        )
        match = VALID_ARTIFACT.fullmatch(qualified)
        if (not match or meta.get("source_run_id") != run.id
                or meta.get("source_git_commit") != study["source_commit"]
                or meta.get("episode_budget_complete") is not True
                or type(meta.get("completed_episodes")) is not int
                or meta["completed_episodes"] != study["requested_episodes"]):
            continue
        if "checkpoint.json" not in artifact.manifest.entries:
            continue
        matches.append((int(match.group(1)), qualified))
    return max(matches)[1] if matches else None


def completed_result(run, study):
    s = dict(run.summary)
    if (run.state != "finished"
            or s.get("episode_budget_complete") is not True
            or s.get("stop_reason") != "episode_budget"
            or type(s.get("episodes_completed")) is not int
            or s["episodes_completed"] != study["requested_episodes"]
            or type(s.get("global_step")) is not int):
        return None
    artifact = final_artifact(run, study)
    if not artifact:
        return None
    return {
        "episodes": study["requested_episodes"],
        "global_step": s["global_step"],
        "episode_budget_complete": True,
        "stop_reason": "episode_budget",
        "tracking_status": "finished",
        "checkpoint_artifact": artifact,
    }


def cpu_parent(row, study, run, case):
    algorithm = case["algorithm"]
    owner = row["kernel_ref"].split("/")[0]
    return {
        "entity": study["entity"], "project": study["project"],
        "account": row["account"],
        "stage": "train", "kernel_ref": row["kernel_ref"],
        "root_kernel_ref": row["kernel_ref"],
        "commit": study["source_commit"],
        "config": case["config"], "overrides": case["overrides"],
        "runs": {algorithm: {"id":run.id, "name":run.name}},
        "requested_episodes": study["requested_episodes"],
        "training_mode": "pql" if "pql" in case["name"] else "base",
        "checkpoint_repo": os.environ.get("HF_REPO_ID", "duy95/uav-search-checkpoints"),
        "credential_dataset": f"{owner}/uav-cpu-wandb-credential",
        "storage_credential_dataset": f"{owner}/uav-hf-checkpoint-credential",
        "cpu_eval": {
            "episodes": study["eval_episodes"],
            "seed": study["eval_seed"],
            "frame_stride": 5, "fps": 10,
        },
        "orchestration": "github_actions",
        "external_supervisor_required": False,
    }


def reconcile_case(api, row, study, *, dry_run, env):
    run = api.run(f"{study['entity']}/{study['project']}/{row['run_id']}")
    case = validate_identity(run, row, study)
    result = {"case": row["case"], "run_id": row["run_id"], "state": run.state}
    if run.state != "finished":
        result["action"] = "blocked_failed" if run.state in {"crashed", "failed", "killed"} else "not_finished"
        result["last_step"] = run.summary.get("global_step")
        result["learner_updates"] = run.summary.get("train/updates")
        return result
    completed = completed_result(run, study)
    if completed is None:
        return {**result, "action": "completion_or_final_checkpoint_unverified"}
    parent = cpu_parent(row, study, run, case)
    cpu = build_visualization_spec(parent, {case["algorithm"]: completed})
    cpu_ref = cpu["kernel_ref"]
    result["cpu_ref"] = cpu_ref
    if dry_run:
        return {**result, "action": "would_submit_cpu", "checkpoint_artifact": completed["checkpoint_artifact"]}

    token = env.get(f"KAGGLE_API_TOKEN_ACCOUNT_{row['case']:02d}", "").strip()
    wkey = env.get("WANDB_API_KEY", "").strip()
    if not token or not wkey or not env.get("HF_TOKEN", "").strip():
        return {**result, "action": "missing_actions_secret"}
    account = Account(row["account"], token)
    if _owned_kernel_exists(cpu_ref, account):
        return {**result, "action": "cpu_kernel_already_exists"}
    creds = Credentials([account], wkey, study["entity"], study["project"])
    with TemporaryDirectory(prefix=f"matched-cpu-{row['case']:02d}-") as d:
        submitted = submit_visualization(
            parent, {case["algorithm"]: completed}, Path(d),
            account=account, credentials=creds,
        )
    if submitted["kernel_ref"] != cpu_ref:
        raise RuntimeError("CPU kernel ref mismatched the deterministic intended ref")
    return {**result, "action": "submitted_cpu", "cpu_url": submitted["kernel_url"]}


def reconcile(study, api, *, dry_run=True, env=None):
    validate_study(study)
    env = os.environ if env is None else env
    report = []
    for row in study["cases"]:
        try:
            report.append(reconcile_case(api, row, study, dry_run=dry_run, env=env))
        except Exception as e:
            # Never expose third-party exception strings: may contain credentials.
            report.append({"case": row["case"], "run_id": row["run_id"],
                           "action": "service_or_validation_error", "error_type": type(e).__name__})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Success-only W&B -> Kaggle CPU visualization supervisor")
    parser.add_argument("--study", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--submit", action="store_true", help="Enable CPU-only child submission")
    args = parser.parse_args(argv)
    study = json.loads(args.study.read_text())
    import wandb
    api = wandb.Api(timeout=35)
    rows = reconcile(study, api, dry_run=not args.submit)
    result = {"group": study["group"], "dry_run": not args.submit, "cases": rows}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(any(row["action"] in {"service_or_validation_error", "missing_actions_secret"} for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
