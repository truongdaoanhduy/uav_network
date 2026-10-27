"""A private CPU-only job evaluates uploaded immutable policy checkpoints."""
import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import re
import time
from tempfile import TemporaryDirectory
from urllib.parse import quote
from uuid import uuid4

from .identities import related_run_name

from .client import record_submission_phase, pushed_kernel_ref, run_kaggle, verify_credential_dataset
from .credentials import Account, Credentials
from .kernel import kernel_metadata, kernel_script

# Lazy, patchable hooks keep GitHub orchestration free of the torch dependency.
class _LazyTorch:
    def __getattr__(self, name):
        import importlib
        module = importlib.import_module("torch")
        globals()["torch"] = module
        return getattr(module, name)


torch = _LazyTorch()
evaluate = None
hardware_info = None


def _visualization_asset_files(result):
    image = Path(result.get("render_path", ""))
    video = Path(result.get("animation_path", ""))
    metrics = image.parent / "metrics.json"
    if (not image.is_file() or image.suffix.lower() != ".png"
            or not video.is_file() or video.suffix.lower() not in {".mp4", ".gif", ".webm", ".ogg"}
            or not metrics.is_file()):
        raise ValueError("Visualization publication requires rendered image, video and metrics files")
    return {"video": video, "image": image, "metrics": metrics}


def publish_visualization_assets(config, spec, algorithm, run_id, result):
    """Upload public evaluation media to an immutable Hub commit and verify fresh downloads."""
    if config.tracking.checkpoint_repo_private:
        raise RuntimeError("Visualization Hub publication requires an explicitly public repository")
    if algorithm not in {"masac", "matd3"} or not re.fullmatch(r"[\w.-]+", str(run_id)):
        raise ValueError("Unsafe visualization publication identity")
    mode = str(spec.get("training_mode", "unknown")).lower()
    if not re.fullmatch(r"[a-z0-9_-]+", mode):
        raise ValueError("Unsafe visualization training mode")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from ..experiments.checkpoint_store import file_sha256, storage_credentials
    token, repo = storage_credentials(config)
    expected = spec.get("checkpoint_repo")
    if expected and expected != repo:
        raise ValueError("Visualization repository differs from the pinned checkpoint repository")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
    if api.repo_info(repo, repo_type="model").private is not False:
        raise RuntimeError("Visualization repository must be public")
    prefix = f"visualizations/{mode}/{algorithm}/{run_id}"
    local = _visualization_asset_files(result)
    items = {}
    for role, path in local.items():
        remote_path = f"{prefix}/{path.name}"
        items[role] = {"path": remote_path, "size": path.stat().st_size, "sha256": file_sha256(path)}
    commit = api.create_commit(repo_id=repo, repo_type="model",
        operations=[CommitOperationAdd(path_in_repo=items[role]["path"], path_or_fileobj=str(local[role]))
                    for role in ("video", "image", "metrics")],
        commit_message=f"Publish {mode} {algorithm.upper()} visualization {run_id}")
    revision = commit.oid
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise RuntimeError("Visualization Hub commit is not immutable")
    with TemporaryDirectory(prefix="uav-visualization-verify-") as directory:
        for role, item in items.items():
            downloaded = Path(hf_hub_download(repo_id=repo, repo_type="model", filename=item["path"],
                revision=revision, token=token, local_dir=directory, force_download=True))
            if downloaded.stat().st_size != item["size"] or file_sha256(downloaded) != item["sha256"]:
                raise RuntimeError("Fresh Hub visualization verification failed")
    for item in items.values():
        item["url"] = (f"https://huggingface.co/{repo}/resolve/{revision}/"
                       + quote(item["path"], safe="/"))
    return {"repo_id": repo, "revision": revision, "files": items}


def log_visualization_media(run, result, published=None, fps=10):
    """Attach the rendered episode directly to the active W&B CPU-visualize run."""
    import wandb
    files = _visualization_asset_files(result)
    video_format = files["video"].suffix.lower().lstrip(".")
    run.log({"global_step": int(result["episodes"]),
             "visualization/video": wandb.Video(str(files["video"]), fps=fps, format=video_format),
             "visualization/final_episode": wandb.Image(str(files["image"]),
                 caption=f"{str(result.get('algorithm', 'policy')).upper()} final evaluation episode")})
    summary = {"visualization/wandb_media_uploaded": True}
    if published:
        summary.update({"visualization/hf_repo_id": published["repo_id"],
                        "visualization/hf_revision": published["revision"],
                        "visualization/hf_video_url": published["files"]["video"]["url"],
                        "visualization/hf_image_url": published["files"]["image"]["url"],
                        "visualization/hf_metrics_url": published["files"]["metrics"]["url"]})
    run.summary.update(summary)


def publish_visualization_media(config, spec, algorithm, run_id, result, run, fps=10):
    """Publish HF and W&B copies independently so one service cannot erase the other."""
    publication, published = {}, None
    try:
        published = publish_visualization_assets(config, spec, algorithm, run_id, result)
        publication["huggingface"] = {"status": "ok", **published}
    except Exception as error:
        publication["huggingface"] = {"status": "failed", "error_type": type(error).__name__}
    try:
        log_visualization_media(run, result, published, fps=fps)
        publication["wandb"] = {"status": "ok"}
    except Exception as error:
        publication["wandb"] = {"status": "failed", "error_type": type(error).__name__}
    try:
        run.summary["visualization/publication"] = publication
    except Exception:
        pass
    return publication


def build_visualization_spec(parent, results):
    target = parent.get("requested_episodes")
    if target is not None and (type(target) is not int or target < 1):
        raise ValueError("CPU visualization requires a valid requested episode budget")
    artifacts, statuses = {}, {}
    for algorithm, result in results.items():
        artifact = result.get("checkpoint_artifact")
        prefix = re.escape(parent["entity"] + "/" + parent["project"] + "/")
        if (algorithm not in {"masac", "matd3"} or algorithm not in parent["runs"]
                or result.get("tracking_status") != "finished" or not isinstance(artifact, str)
                or not re.fullmatch(prefix + r"[\w.-]+:v[0-9]+", artifact)):
            raise ValueError("CPU visualization requires each successful learner's uploaded immutable checkpoint version")
        if target is not None and (type(result.get("episodes")) is not int
                or result["episodes"] < target or result.get("episode_budget_complete") is not True
                or result.get("stop_reason") != "episode_budget"):
            raise ValueError("Final CPU visualization requires the completed episode budget")
        artifacts[algorithm] = artifact
        statuses[algorithm] = {key: result.get(key) for key in
                               ("episodes", "global_step", "episode_budget_complete", "stop_reason")}
    if not artifacts:
        raise ValueError("CPU visualization needs at least one completed learner checkpoint")
    from .continuation import kernel_ref_for_segment
    cpu_ref = kernel_ref_for_segment(parent.get("root_kernel_ref", parent["kernel_ref"]), 0, cpu=True)
    runs = {a: {"id": uuid4().hex[:8],
                "name": related_run_name(parent["runs"][a]["name"], "EVALCPU", uuid4().hex[:6])}
            for a in artifacts}
    return {"stage": "visualize", "runs": runs,
            "config": parent.get("config"), "overrides": parent.get("overrides", []),
            **{k: parent[k] for k in ("storage_credential_dataset", "checkpoint_repo") if k in parent}, "commit": parent["commit"], "entity": parent["entity"],
            "project": parent["project"],
            "credential_dataset": parent.get("cpu_credential_dataset", parent["credential_dataset"]),
            "kernel_ref": cpu_ref, "account": parent["account"],
            "parent_kernel_ref": parent["kernel_ref"],
            "root_kernel_ref": parent.get("root_kernel_ref", parent["kernel_ref"]),
            "continuation_index": parent.get("continuation_index", 0),
            "requested_episodes": parent.get("requested_episodes"),
            "training_mode": parent.get("training_mode", "unknown"), "artifacts": artifacts,
            "training_status": statuses, "cpu_eval": parent.get("cpu_eval", {
                "episodes": 1, "seed": 10044, "frame_stride": 5, "fps": 10}),
            "session_resumable": True,
            **{k: parent[k] for k in ("orchestration", "external_supervisor_required") if k in parent},
            "requires_kaggle_token": bool(parent.get("pipeline_credential_dataset")),
            **{k: parent[k] for k in ("pipeline_credential_dataset",) if k in parent},
            "timeout_seconds": 39600}


def submit_visualization(parent, results, output, account=None, credentials=None):
    spec = build_visualization_spec(parent, results)
    if account is None:
        token = os.environ.get("KAGGLE_API_TOKEN")
        if not token:
            raise RuntimeError("Autonomous CPU launch requires its private Kaggle credential")
        account = Account(parent["account"], token)
    if credentials is None:
        credentials = Credentials([account], os.environ["WANDB_API_KEY"], parent["entity"], parent["project"])
    return submit_visualization_spec(spec, output, account, credentials)


def submit_visualization_spec(spec, output, account, credentials):
    record_submission_phase(output, "cpu", spec["kernel_ref"], "preparing")
    if account.name != spec["account"] or not account.enabled:
        raise ValueError("CPU visualization must keep the selected enabled account")
    if any(secret and secret in json.dumps(spec) for secret in
           [credentials.wandb_key, *(a.token for a in credentials.accounts)]):
        raise ValueError("Refusing a CPU manifest containing credentials")
    username = spec["kernel_ref"].split("/", 1)[0]
    verify_credential_dataset(account, credentials, spec["credential_dataset"], username, only_selected=True)
    folder = Path(output) / "cpu-kernel"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(kernel_script(spec))
    (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(spec), indent=2))
    record_submission_phase(output, "cpu", spec["kernel_ref"], "pushing")
    response = run_kaggle(["kernels", "push", "-p", str(folder), "--timeout", str(spec["timeout_seconds"])], account, timeout=50)
    ref = pushed_kernel_ref(response)
    manifest = {**spec, "kernel_ref": ref, "kernel_url": "https://www.kaggle.com/code/" + ref}
    (Path(output) / "cpu-visualization.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"cpu_visualization_url": manifest["kernel_url"]}), flush=True)
    return manifest


def load_publication_assets(spec, algorithm, output):
    """Recover only final evaluation media from an owned terminal CPU job."""
    source = spec["publication_source_ref"]
    if source.split("/")[0] != spec["kernel_ref"].split("/")[0]:
        raise ValueError("Publication recovery must keep the original account")
    account = Account(spec["account"], os.environ["KAGGLE_API_TOKEN"])
    folder = Path(output) / "visualize" / algorithm
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="uav-media-recovery-") as directory:
        pattern = (r"(^|.*/)visualize/" + re.escape(algorithm)
                   + r"/(metrics\.json|final_episode\.(png|mp4|gif|webm|ogg))$")
        run_kaggle(["kernels", "output", source, "-p", directory,
                    "--file-pattern", pattern, "-q"], account, timeout=180)
        files = list(Path(directory).rglob("metrics.json"))
        if len(files) != 1:
            raise ValueError("Publication recovery requires one saved evaluation metrics file")
        result = json.loads(files[0].read_text())
        if (result.get("algorithm") != algorithm or result.get("episodes") != spec["cpu_eval"]["episodes"]
                or result.get("checkpoint_artifact") != spec["artifacts"][algorithm]):
            raise ValueError("Publication recovery policy/evaluation identity differs")
        import shutil
        for asset in files[0].parent.iterdir():
            if asset.is_file() and asset.name in {"metrics.json", "final_episode.png",
                                                  "final_episode.mp4", "final_episode.gif",
                                                  "final_episode.webm", "final_episode.ogg"}:
                shutil.copyfile(asset, folder / asset.name)
    result["render_path"] = str(folder / "final_episode.png")
    videos = [p for p in folder.glob("final_episode.*") if p.suffix in {".mp4", ".gif", ".webm", ".ogg"}]
    if len(videos) != 1:
        raise ValueError("Publication recovery requires one saved animation")
    result["animation_path"] = str(videos[0])
    _visualization_asset_files(result)
    return result


def run_visualization(spec, output=Path("/kaggle/working")):
    global evaluate, hardware_info
    if evaluate is None:
        from ..experiments.evaluation import evaluate as _evaluate
        evaluate = _evaluate
    if hardware_info is None:
        from ..experiments.hardware import hardware_info as _hardware_info
        hardware_info = _hardware_info
    if torch.cuda.is_available():
        raise RuntimeError("Visualization must run in a separate CPU-only session")
    from .remote import prepare_checkpoints
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "hardware.json").write_text(json.dumps(hardware_info(), indent=2))
    settings = spec["cpu_eval"]
    from ..experiments.evaluation import EvaluationPaused
    from ..experiments.progress_watchdog import ProgressWatchdog
    from .controller import atomic_json
    results, failures = {}, {}
    progress = {"artifacts": spec["artifacts"], "cpu_eval": settings,
                "records": dict(spec.get("completed_records", {}))}
    started = float(os.environ.get("UAV_JOB_STARTED_AT", time.monotonic()))
    deadline = started + min(spec.get("timeout_seconds", 39600), 39600)
    episode_started, last_episode_seconds = time.monotonic(), 0.0
    def checkpoint_progress(algorithm, records):
        nonlocal episode_started, last_episode_seconds
        now = time.monotonic()
        last_episode_seconds = now - episode_started
        episode_started = now
        progress["records"][algorithm] = list(records)
        progress["last_episode_seconds"] = last_episode_seconds
        atomic_json(output / "evaluation-progress.json", progress)
    def should_pause():
        return deadline - time.monotonic() < max(900, last_episode_seconds * 1.5 + 300)
    for algorithm, artifact in spec["artifacts"].items():
        tracker = None
        config = None
        tracking_output = output / "tracking" / algorithm
        tracking_output.mkdir(parents=True, exist_ok=True)
        watchdog = ProgressWatchdog(tracking_output)
        last_cpu_point, cpu_progress_serial, last_progress_log = None, 0, None
        failed = True
        completed_eval_episodes = (len(progress["records"].get(algorithm, []))
                                   if spec.get("session_resumable") else 0)
        def record_failure(error, phase="cpu_evaluation"):
            failures.setdefault(algorithm, {"error_type": type(error).__name__})
            try:
                from ..experiments.diagnostics import failure_report
                completed = max(completed_eval_episodes,
                                len(progress["records"].get(algorithm, [])))
                path = failure_report(config, tracking_output, error, {
                    "algorithm":algorithm, "stage":"visualize", "phase":phase,
                    "checkpoint_artifact":artifact, "source_commit":spec.get("commit"),
                    "completed_eval_episodes":completed, "requested_eval_episodes":settings["episodes"],
                    "seed_first":settings["seed"]})
                if tracker is not None:
                    tracker.failure(path, completed)
            except Exception as reporting_error:
                with suppress(OSError, ValueError):
                    print("CPU failure reporting also failed: " + type(reporting_error).__name__, flush=True)
        try:
            from ..configuration import load_config
            from ..experiments.tracking import start_tracking
            config = load_config(spec.get("config"), spec.get("overrides", []))
            config.runtime.algorithm = algorithm
            config.runtime.device = config.runtime.replay_device = "cpu"
            config.runtime.num_envs = 1
            config.runtime.episodes = settings["episodes"]
            config.pql.enabled = False
            config.tracking.enabled = True
            config.tracking.entity, config.tracking.project = spec["entity"], spec["project"]
            identity = spec.get("runs", {}).get(algorithm, {"id": uuid4().hex[:8],
                "name": algorithm.upper() + "-CPU-visualize"})
            config.tracking.run_name = identity["name"]
            tracking_output = output / "tracking" / algorithm
            tracking_output.mkdir(parents=True, exist_ok=True)
            previous_id = os.environ.get("WANDB_RUN_ID")
            previous_stage = os.environ.get("UAV_STAGE")
            os.environ["WANDB_RUN_ID"] = identity["id"]
            os.environ["UAV_STAGE"] = "visualize"
            try:
                tracker = start_tracking(config, tracking_output,
                                         {"observation_version": None, "generation_version": None})
            finally:
                if previous_stage is None:
                    os.environ.pop("UAV_STAGE", None)
                else:
                    os.environ["UAV_STAGE"] = previous_stage
                if previous_id is None:
                    os.environ.pop("WANDB_RUN_ID", None)
                else:
                    os.environ["WANDB_RUN_ID"] = previous_id
            tracker.run.summary.update({"stage": "visualize", "execution_device": "cpu",
                "parent_kernel_ref": spec.get("parent_kernel_ref"),
                "checkpoint_artifact": artifact,
                "visualization_output_url": "https://www.kaggle.com/code/" + spec["kernel_ref"] + "/output"})
            checkpoint_step = spec.get("training_status", {}).get(algorithm, {}).get("global_step")
            initial_metrics = {"global_step": completed_eval_episodes,
                               "evaluation/episodes_completed": completed_eval_episodes}
            if type(checkpoint_step) is int and checkpoint_step >= 0:
                tracker.run.summary["evaluation/checkpoint_global_step"] = checkpoint_step
                initial_metrics["evaluation/checkpoint_global_step"] = checkpoint_step
            tracker.diagnostic(initial_metrics, completed_eval_episodes)
            def record_progress(point):
                nonlocal last_cpu_point, cpu_progress_serial, last_progress_log
                current = (point.get("seed"), point.get("step"))
                if current != last_cpu_point:
                    cpu_progress_serial += 1
                    watchdog.progress(cpu_progress_serial)
                    last_cpu_point = current
                now = time.monotonic()
                if (point["status"] != "running" or last_progress_log is None
                        or now - last_progress_log >= 30):
                    tracker.diagnostic({"evaluation_progress/" + k: v for k, v in point.items()},
                                       completed_eval_episodes)
                    last_progress_log = now
            def record_episode(record):
                nonlocal completed_eval_episodes
                completed_eval_episodes = record["episode"]
                tracker.diagnostic({"evaluation/episodes_completed": record["episode"],
                    "evaluation/return": record["return"],
                    "evaluation/search_rate_percent": record["search_rate_percent"],
                    "evaluation/delivery_rate_percent": record["delivery_rate_percent"],
                    "evaluation/success": int(record["success"])}, record["episode"])
            resolved = artifact
            if spec.get("publication_source_ref"):
                results[algorithm] = load_publication_assets(spec, algorithm, output)
            else:
                inputs = {"warm_start_artifacts": {algorithm: artifact}}
                checkpoint = prepare_checkpoints(inputs, output)[algorithm]
                resolved = inputs.get("resolved_warm_start_artifacts", {}).get(algorithm, artifact)
                tracker.run.use_artifact(resolved, type="model")
                options = {}
                if spec.get("session_resumable"):
                    options = {"completed_records": progress["records"].get(algorithm, []),
                               "should_pause": should_pause,
                               "on_checkpoint": lambda records: checkpoint_progress(algorithm, records)}
                results[algorithm] = evaluate(checkpoint, episodes=settings["episodes"], seed=settings["seed"],
                                              output_dir=output / "visualize" / algorithm, animate=True,
                                              frame_stride=settings["frame_stride"], fps=settings["fps"],
                                              on_episode=record_episode, on_progress=record_progress, **options)
            completed_eval_episodes = results[algorithm]["episodes"]
            # Publication recovery preserves the saved policy protocol, not this runtime's semantics.
            protocol = results[algorithm].get("protocol", {})
            source_version = protocol.get("scenario", {}).get("observation_version")
            tracker.run.summary["evaluation/observation_version"] = (source_version
                if type(source_version) is int else None)
            generation = protocol.get("scenario", {}).get("world", {}).get("generation_version")
            tracker.run.summary["evaluation/generation_version"] = (generation
                if type(generation) is int else None)
            source_step = protocol.get("training_global_step", checkpoint_step)
            if type(source_step) is int and source_step >= 0:
                tracker.run.summary["evaluation/checkpoint_global_step"] = source_step
            results[algorithm]["checkpoint_artifact"] = artifact
            results[algorithm]["training_status"] = spec["training_status"][algorithm]
            results[algorithm].update(tracker.identity())
            publication = publish_visualization_media(config, spec, algorithm, identity["id"],
                                                      results[algorithm], tracker.run, fps=settings["fps"])
            results[algorithm]["publication"] = publication
            metrics_path = output / "visualize" / algorithm / "metrics.json"
            metrics_path.write_text(json.dumps(results[algorithm], indent=2, allow_nan=False))
            aggregates = {"eval/" + k: results[algorithm][k] for k in
                ("mean_return", "success_rate", "mean_search_rate_percent", "mean_delivery_rate_percent")}
            seeds = protocol.get("seeds", [])
            aggregates.update({"evaluation/episodes": results[algorithm]["episodes"],
                "evaluation/checkpoint_artifact": resolved,
                "evaluation/kind": "final_cpu_evaluation"})
            if seeds:
                aggregates.update({"evaluation/seed_first": seeds[0], "evaluation/seed_last": seeds[-1]})
            for key in ("scenario_hash", "deterministic_policy", "training_seed_overlap", "validation_seed_overlap"):
                if key in protocol:
                    aggregates["evaluation/" + key] = protocol[key]
            if type(source_step) is int and source_step >= 0:
                aggregates["evaluation/checkpoint_global_step"] = source_step
            tracker.run.summary.update(aggregates)
            tracker.diagnostic(aggregates, results[algorithm]["episodes"])
            tracker.run.summary["evaluation/episodes_completed"] = results[algorithm]["episodes"]
            tracker.run.save(str(output / "visualize" / algorithm / "metrics.json"),
                             base_path=str(output), policy="now")
            failed = False
        except EvaluationPaused as paused:
            completed_eval_episodes = len(paused.records)
            tracker.run.summary["evaluation/session_status"] = "paused"
            tracker.diagnostic({"evaluation_status/status":"paused",
                                "evaluation/episodes_completed":completed_eval_episodes},
                               completed_eval_episodes)
            failed = False
        except Exception as error:
            record_failure(error)
        finally:
            try:
                if tracker is not None:
                    try:
                        tracker.close(failed=failed)
                    except Exception as error:
                        if algorithm not in failures:
                            record_failure(error, "tracker_close")
                        failures[algorithm]["close_error_type"] = type(error).__name__
            finally:
                watchdog.close()
        (output / "visualization-result.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    if "masac" in results and "matd3" in results:
        from ..experiments.comparison import compare_evaluations, save_comparison
        try:
            save_comparison(compare_evaluations(results["masac"], results["matd3"]), output / "visualize")
        except Exception as error:
            failures["comparison"] = {"error_type": type(error).__name__}
    if failures:
        (output / "visualization-failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError("CPU visualization failures: " + ", ".join(failures))
    with suppress(OSError, ValueError):
        print(json.dumps({a: {k: v[k] for k in ("device", "render_path", "animation_path")} for a, v in results.items()}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    run_visualization(json.loads(Path(args.spec).read_text()))


if __name__ == "__main__":
    main()
