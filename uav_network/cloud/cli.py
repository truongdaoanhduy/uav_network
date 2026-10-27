"""Quota-aware private Kaggle launches and public job status manifests."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from uuid import uuid4

from ..configuration import load_config
from .client import CredentialFileSetError, account_quotas, ensure_cpu_wandb_dataset, ensure_pipeline_dataset, kernel_status, pushed_kernel_ref, resolve_username, run_kaggle, verify_credential_dataset
from .credentials import load_credentials, select_account
from .kernel import kernel_metadata, kernel_script
from .identities import authoritative_tags, kernel_slug, run_name
from .acceleration import apply_measured_acceleration_policy

ROOT = Path(__file__).resolve().parents[2]


def parse_warm_start_artifacts(values, algorithms):
    result = {}
    for value in values:
        algorithm, separator, ref = value.partition("=")
        if not separator or algorithm not in algorithms or not ref or algorithm in result:
            raise ValueError("warm-start-artifact needs one ALGORITHM=ENTITY/PROJECT/ARTIFACT:VERSION per selected algorithm")
        result[algorithm] = ref
    return result


def launch(args, credentials):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit and publish the project changes before launching a pinned cloud job")
    quotas = account_quotas(credentials, requested=args.account)
    account = select_account(credentials, quotas, args.required_gpu_hours, args.account)
    username = resolve_username(account)
    canonical_wandb = f"{username}/uav-cpu-wandb-credential"
    if args.credential_dataset == canonical_wandb:
        args.credential_dataset = ensure_cpu_wandb_dataset(
            account, credentials, username, args.credential_timeout_seconds)
    verify_credential_dataset(account, credentials, args.credential_dataset, username, args.credential_timeout_seconds)
    config_file = args.config or ("configs/kaggle-smoke.yaml" if args.stage == "validate" else "configs/kaggle.yaml")
    config = load_config(ROOT / config_file, args.overrides)
    auto_continue = getattr(args, "auto_continue", False)
    cpu_visualize = getattr(args, "cpu_visualize", False) or auto_continue
    timeout_seconds = args.timeout_seconds
    if auto_continue:
        quota = next(row for row in quotas if row.get("name") == account.name and row.get("ok"))
        remaining = quota.get("remaining_hours")
        if type(remaining) not in (int, float) or not math.isfinite(remaining):
            raise RuntimeError("Selected account GPU quota is unavailable")
        timeout_seconds = min(timeout_seconds, math.floor((remaining - .25) * 3600))
        reserve = max(600, config.tracking.artifact_timeout_seconds + 120)
        if timeout_seconds <= reserve:
            raise RuntimeError("Selected account GPU quota cannot cover a safe training and upload segment")
    compare_legacy = getattr(args, "compare_legacy", False)
    diagnostic = getattr(args, "diagnostic", None)
    if diagnostic:
        expected = ("configs/comparison-20k-hybrid75-1m.yaml" if diagnostic == "hybrid75-4096-1m"
                    else "configs/comparison-20k-steps1800-scenario-v3.yaml")
        if (diagnostic not in {"cpu-replay-1m", "gpu-replay-1m", "hybrid-replay-1m", "hybrid75-replay-1m", "hybrid75-4096-1m"} or args.stage != "benchmark" or args.algorithm != "masac"
                or compare_legacy or args.warm_start_artifact or args.overrides
                or auto_continue or cpu_visualize or (args.config not in {None, expected})):
            raise ValueError("CPU replay diagnostic requires benchmark/MASAC, the pinned V3 config, and no extra overrides")
        config_file = expected
        config = load_config(ROOT / config_file)
    effective_overrides = apply_measured_acceleration_policy(
        ROOT / config_file, args.stage, args.algorithm, config, list(args.overrides),
        ROOT / "configs/comparison-20k-hybrid75-1kb.yaml")
    if effective_overrides != list(args.overrides):
        config = load_config(ROOT / config_file, effective_overrides)
    if compare_legacy and (args.stage != "benchmark" or not config.pql.enabled):
        raise ValueError("Matched comparison requires --stage benchmark and pql.enabled=true")
    if compare_legacy and (config.runtime.replay_cpu_fraction or config.runtime.replay_device == "cpu"):
        raise ValueError("Legacy cloud comparison requires its original same-device replay layout")
    if auto_continue and (args.algorithm == "both" or args.stage == "benchmark"):
        raise ValueError("Automatic continuation requires one algorithm in a training/validate stage")
    if auto_continue and config.pql.enabled and not config.pql.checkpoint_replay:
        raise ValueError("Automatic PQL continuation requires pql.checkpoint_replay=true")
    if cpu_visualize and args.stage == "benchmark":
        raise ValueError("CPU visualization requires a training/validate stage that creates checkpoints")
    if cpu_visualize and not config.tracking.log_checkpoint:
        raise ValueError("CPU visualization requires tracking.log_checkpoint=true for checkpoint transfer")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = uuid4().hex[:6]
    algorithms = ("masac", "matd3") if args.algorithm == "both" else (args.algorithm,)
    inputs = parse_warm_start_artifacts(args.warm_start_artifact, algorithms)
    training_mode = "pql" if diagnostic or config.pql.enabled else "base"
    slug = kernel_slug(username, config, args.algorithm, training_mode, args.stage, stamp, token,
                       external_supervisor=auto_continue)
    runs = {a: {"id": uuid4().hex[:8],
                "name": run_name(config, a, training_mode, args.stage, stamp + "-" + token)}
            for a in algorithms}
    overrides = list(effective_overrides)
    if not diagnostic:
        config.tracking.tags = authoritative_tags(config)
        overrides = [item for item in overrides if not item.startswith("tracking.tags=")]
        overrides.append("tracking.tags=" + json.dumps(config.tracking.tags))
    spec = {"submitted_at_utc": datetime.now(timezone.utc).isoformat(), "commit": commit, "kernel_ref": f"{username}/{slug}", "account": account.name,
            "timeout_seconds": timeout_seconds, "warm_start_artifacts": inputs,
            "credential_dataset": args.credential_dataset,
            "entity": credentials.entity if config.tracking.entity == "auto" else config.tracking.entity,
            "project": credentials.project if config.tracking.project == "auto" else config.tracking.project,
            "stage": args.stage, "config": config_file, "overrides": overrides, "runs": runs}
    spec.update({"training_mode": training_mode,
                 "accelerator": args.accelerator, "gpus_per_run": args.gpus_per_run})
    if args.stage == "benchmark":
        benchmark_steps = getattr(args, "benchmark_steps", 2)
        benchmark_learner_updates = getattr(args, "benchmark_learner_updates", 2)
        if benchmark_steps < 1 or benchmark_learner_updates < 1:
            raise ValueError("Benchmark steps and learner updates must be positive")
        spec.update(benchmark_steps=benchmark_steps,
                    benchmark_learner_updates=benchmark_learner_updates)
    if diagnostic:
        spec["diagnostic"] = diagnostic
    else:
        spec["requested_episodes"] = config.runtime.episodes
    if auto_continue:
        spec.update({"auto_continue": True, "root_kernel_ref": spec["kernel_ref"],
                     "continuation_index": 0, "orchestration": "github_actions",
                     "external_supervisor_required": True})
    if compare_legacy:
        spec["compare_legacy"] = True
    if cpu_visualize:
        try:
            verify_credential_dataset(account, credentials, args.credential_dataset, username,
                                      args.credential_timeout_seconds, only_selected=True)
            cpu_dataset = args.credential_dataset
        except CredentialFileSetError:
            cpu_dataset = ensure_cpu_wandb_dataset(account, credentials, username, args.credential_timeout_seconds)
        spec.update({"cpu_visualize": True,
                     "pipeline_credential_dataset": ensure_pipeline_dataset(account, username, args.credential_timeout_seconds),
                     "cpu_credential_dataset": cpu_dataset,
                     "cpu_eval": {"episodes": config.runtime.eval_episodes, "seed": config.runtime.eval_seed,
                                  "frame_stride": 5, "fps": 10}})
    if (spec["entity"], spec["project"]) != (credentials.entity, credentials.project):
        raise ValueError("Cloud job must use the main W&B project in key.txt")
    storage_token = None
    if not diagnostic and config.tracking.checkpoint_store == "huggingface":
        from ..experiments.checkpoint_store import ensure_private_store, storage_credentials
        from .client import ensure_storage_dataset
        config.tracking.credentials_file = args.key_file
        _, repository = ensure_private_store(config)
        storage_token, _ = storage_credentials(config)
        spec["checkpoint_repo"] = repository
        spec["storage_credential_dataset"] = ensure_storage_dataset(account, username, storage_token, args.credential_timeout_seconds)
    serialized = json.dumps(spec)
    if any(secret in serialized for secret in [credentials.wandb_key, *(a.token for a in credentials.accounts), *([storage_token] if storage_token else [])]):
        raise RuntimeError("Refusing a job manifest containing credentials")
    if diagnostic:
        recent = json.loads(run_kaggle(["kernels", "list", "--mine", "--format", "json",
                                       "--page-size", "20", "--sort-by", "dateRun"], account))
        active = [{"ref": row["ref"], "status": kernel_status(row["ref"], account)} for row in recent]
        active = [row for row in active if row["status"] in {"running", "pending", "queued"}]
        if active:
            raise RuntimeError(f"Diagnostic account has active kernels: {active}")
    folder = ROOT / "outputs/kaggle" / slug
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "kernel.py").write_text(kernel_script(spec))
    (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(spec), indent=2))
    # The timeout and accelerator are explicit CLI inputs, not assumptions about RTX naming.
    response = run_kaggle(["kernels", "push", "-p", str(folder), "--accelerator", args.accelerator,
                           "--timeout", str(timeout_seconds)], account, timeout=50)
    actual_ref = pushed_kernel_ref(response)
    manifest = {**spec, "kernel_ref": actual_ref, "kernel_url": f"https://www.kaggle.com/code/{actual_ref}",
                "wandb_urls": {a: f"https://wandb.ai/{spec['entity']}/{spec['project']}/runs/{r['id']}" for a, r in runs.items()},
                "manifest": str(folder / "launch.json")}
    if auto_continue:
        manifest["root_kernel_ref"] = actual_ref
    (folder / "launch.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Private Kaggle CUDA + online W&B")
    parser.add_argument("--key-file", default=str(ROOT / "key.txt"))
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("accounts", help="Show verified quota without credentials")
    submit = commands.add_parser("launch")
    submit.add_argument("--stage", choices=["validate", "train", "benchmark"], default="train")
    submit.add_argument("--compare-legacy", action="store_true",
                        help="Benchmark matched legacy, single-GPU PQL and dual-GPU PQL; no production training")
    submit.add_argument("--diagnostic", choices=["cpu-replay-1m", "gpu-replay-1m", "hybrid-replay-1m", "hybrid75-replay-1m", "hybrid75-4096-1m"],
                        help="Run one bounded reviewed diagnostic instead of a training/benchmark entrypoint")
    submit.add_argument("--account")
    submit.add_argument("--required-gpu-hours", type=float)
    submit.add_argument("--credential-dataset", required=True)
    submit.add_argument("--credential-timeout-seconds", type=int, default=120)
    submit.add_argument("--algorithm", choices=["both", "masac", "matd3"], default="both")
    submit.add_argument("--warm-start-artifact", action="append", default=[], metavar="ALGORITHM=ARTIFACT_REF")
    submit.add_argument("--config")
    submit.add_argument("--cpu-visualize", action="store_true",
                        help="After uploaded training checkpoints, submit a separate CPU-only visualization job")
    submit.add_argument("--auto-continue", action="store_true",
                        help="Seed a cloud CPU supervisor, resume GPU sessions to the episode target, then CPU visualization")
    submit.add_argument("--set", dest="overrides", action="append", default=[])
    submit.add_argument("--gpus-per-run", type=int, choices=[1, 2], default=1,
                        help="Visible GPUs per BASE learner; PQL continues to own all visible GPUs")
    submit.add_argument("--benchmark-steps", type=int, default=2,
                        help="BASE benchmark collector steps; benchmark stage only")
    submit.add_argument("--benchmark-learner-updates", type=int, default=2,
                        help="BASE benchmark measured learner updates; benchmark stage only")
    submit.add_argument("--accelerator", default="NvidiaTeslaT4")
    submit.add_argument("--timeout-seconds", type=int, default=39600)
    resume = commands.add_parser("resume", help="Resume a quota-paused continuation from a completed kernel")
    resume.add_argument("--account", required=True)
    resume.add_argument("--kernel-ref", required=True)
    resume.add_argument("--output", required=True)
    status = commands.add_parser("status")
    status.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    credentials = load_credentials(args.key_file)
    if args.action == "accounts":
        return {"accounts": account_quotas(credentials)}
    if args.action == "launch":
        if not 1 <= args.timeout_seconds <= 39600 or args.credential_timeout_seconds < 1:
            raise ValueError("Kernel timeout must be 1..39600 seconds; credential timeout must be positive")
        return launch(args, credentials)
    if args.action == "resume":
        account = next((a for a in credentials.accounts if a.name == args.account and a.enabled), None)
        if account is None:
            raise RuntimeError("Resume account is missing or disabled in key.txt")
        from .continuation import resume_waiting_continuation
        return resume_waiting_continuation(args.kernel_ref, account, credentials, Path(args.output))
    manifest = json.loads(Path(args.manifest).read_text())
    account = next((a for a in credentials.accounts if a.name == manifest["account"]), None)
    if account is None:
        raise RuntimeError("Manifest account is missing from key.txt")
    return {"kernel_url": manifest["kernel_url"], "state": kernel_status(manifest["kernel_ref"], account),
            "wandb_urls": manifest["wandb_urls"]}
