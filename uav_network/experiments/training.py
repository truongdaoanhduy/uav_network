"""Off-policy training with spawned CPU simulators and one batched learner."""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from ..algorithms.optimization import compile_models
from ..algorithms.replay import resolve_replay_device
from .pql_replay import MeasuredReplay, make_replay
from ..environment.observations import state_dim
from ..environment.vector import make_vector, random_actions, unpack_info
from .tracking import start_tracking, final_metrics
from .lifecycle import graceful_termination
from .continuation import restore_learner
from .fixed_checkpoints import save_fixed_checkpoint, restore_replay, restore_runtime_rng
from .reproducibility import reproducibility_result
from .environment_checkpoints import snapshot_environment, initialize_environment
from .diagnostics import TrainingDiagnostics, failure_report
from .evidence import print_evidence, learner_telemetry
from .progress_watchdog import ProgressWatchdog
from .overflow_capture import attach_overflow_capture
from .async_evaluation import AsyncCheckpointEvaluator
from .recovery_checkpoints import PeriodicRecovery
from .local_checkpoints import prune_episode_checkpoints, disk_metrics, release_staged_checkpoint
from .pql_workers import cloned_state, inference_copy, synchronize as synchronize_worker


def seed_runtime(config):
    seed = config.runtime.seed
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Native numerical libraries in spawned workers must not create large pools.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.runtime.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(config.runtime.torch_threads)
    torch.use_deterministic_algorithms(config.runtime.deterministic)


def choose_device(value):
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value


def resolve_base_devices(config):
    """Place BASE collection on GPU0 and its learner/replay on GPU1 when visible."""
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    requested = config.runtime.device
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if requested == "cpu" or (requested == "auto" and not available):
        collector = learner = torch.device("cpu")
    else:
        collector = torch.device("cuda:0")
        learner = torch.device("cuda:1" if available >= 2 else "cuda:0")
    replay = resolve_replay_device(config.runtime.replay_device, learner)
    return collector, learner, replay


def refresh_base_collector(collector, learner):
    if collector is learner:
        return
    snapshot = cloned_state(learner.actors)
    synchronize_worker(learner.device)
    collector.actors.load_state_dict(snapshot)
    synchronize_worker(collector.device)


def train(config):
    with graceful_termination():
        if config.pql.enabled:
            from .pql_training import train_pql
            return train_pql(config)
        return _train(config)


def _train(config):
    config.validate()
    seed_runtime(config)
    device = choose_device(config.runtime.device)  # legacy summary field
    legacy_replay_device = device if config.runtime.replay_device == "auto" else choose_device(config.runtime.replay_device)
    collector_device, learner_device, replay_device = resolve_base_devices(config)
    settings = config.algorithm_config
    output = Path(config.runtime.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}; choose a new runtime.output_dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    start = time.perf_counter()
    diagnostics = TrainingDiagnostics(config, output)
    watchdog = ProgressWatchdog(output)
    count = config.runtime.num_envs
    completed = global_step = collection_steps = 0
    latest = {}
    update_s = collection_wait_s = 0.0
    next_metrics_at = 0
    episode_seeds = np.arange(count) + config.runtime.seed
    episode_returns, episode_lengths = np.zeros(count), np.zeros(count, dtype=int)
    tracker = vector = learner = evaluator = None
    completed_ok = False
    result = None
    summary = output / "summary.json"
    try:
        learner, initial = restore_learner(config, learner_device)
        completed, global_step, collection_steps = (initial[k] for k in ("episodes", "global_step", "collection_steps"))
        diagnostics.start_windows(global_step)
        initial_updates = learner.updates
        restored_training = initial.pop("_training", None)
        resumed = restored_training is not None
        pending_learning = restored_training["pending_learning"] if resumed else False
        warmup_origin = restored_training["warmup_origin"] if restored_training else initial["global_step"]
        random_warmup = restored_training["random_warmup"] if restored_training else config.runtime.warm_start_checkpoint == "none"
        if restored_training:
            next_metrics_at, latest = restored_training["next_metrics_at"], restored_training["latest"]
        history = [{**r, "learner": learner_telemetry(r.get("learner", {}))}
                   for r in initial["records"]]
        collector = learner if collector_device == learner_device else inference_copy(learner, collector_device)
        collector_version = learner.actor_updates
        vector = make_vector(config, device=collector_device)
        environment = restored_training.get("environment") if restored_training else None
        obs, episode_returns, episode_lengths, episode_seeds, next_seed = initialize_environment(
            vector, initial, environment)
        initial["state_restore"] = ("training_state_exact_environment" if environment is not None
                                    else initial.get("state_restore", "fresh"))
        environment = None  # Release the large CPU snapshot after restoring worlds.
        device_env = config.runtime.environment_backend == "warp_packet"
        if config.runtime.compile_models:
            compile_models(learner)
        replay = make_replay(config, obs["observation"].shape[1:], state_dim(config), learner_device)
        rng = np.random.default_rng(config.runtime.seed + 1)
        if restored_training:
            restore_replay(replay, restored_training["replay"])
        initial["optimizer_replay_restored"] = resumed
        tracker = start_tracking(config, output, initial_progress=initial)
        evaluator = AsyncCheckpointEvaluator(config, output, tracker, diagnostics, global_step, completed,
            selection_state=restored_training.get("best_checkpoint") if restored_training else None)
        if restored_training:
            restore_runtime_rng(restored_training["rng"], rng, action_generator=getattr(vector, "action_generator", None))
        restored_training = None  # Release the CPU checkpoint copy after restoring replay.
        warm_start_sha256 = release_staged_checkpoint(config.runtime.warm_start_checkpoint)
        attach_overflow_capture(learner, output, "base")
        startup_s = time.perf_counter() - start
        recovery = PeriodicRecovery(config, output, started=start)

        def learn(force_metrics=False):
            nonlocal latest, update_s, next_metrics_at, pending_learning
            if not pending_learning:
                return
            pending_learning = False
            if global_step - warmup_origin < settings.learning_starts or len(replay) < settings.batch_size:
                return
            checked = force_metrics or global_step >= next_metrics_at
            begin = time.perf_counter()
            for update in range(settings.updates_per_step):
                batch = replay.sample(settings.batch_size)
                try:
                    metrics = learner.update(batch,
                                             collect_metrics=checked and update == settings.updates_per_step - 1)
                except (FloatingPointError, RuntimeError):
                    torch.save({k: v.detach().cpu() for k, v in batch.items()}, output / "failed_batch.pt")
                    raise
                if metrics:
                    latest = metrics
                    if tracker:
                        tracker.learner(metrics, global_step)
            update_s += time.perf_counter() - begin
            if checked:
                if tracker:
                    tracker.diagnostic({**replay.stats(), **disk_metrics(output),
                                        "timing/collector_wait_seconds": collection_wait_s,
                                        "timing/learner_seconds": update_s,
                                        "train/sample_uses_per_new_transition":
                                            (learner.updates - initial_updates) * settings.batch_size /
                                            max(1, global_step - initial["global_step"])}, global_step)
                next_metrics_at = global_step + config.runtime.log_interval_steps

        def save(path, final=False, full_state=True, resume_path=None):
            progress = {"episodes": completed, "global_step": global_step,
                        "collection_steps": collection_steps, "next_seed": next_seed,
                        "records": history}
            if final:
                progress["episode_budget_complete"] = completed >= config.runtime.episodes
                progress["stop_reason"] = "episode_budget" if completed >= config.runtime.episodes else "time_budget"
            if tracker:
                progress["source_run_id"] = tracker.identity()["wandb_run_id"]
            environment = (snapshot_environment(vector, episode_returns, episode_lengths,
                                                episode_seeds, next_seed) if full_state else None)
            progress["environment_state_saved"] = environment is not None
            learner.save(path, progress=progress)
            if full_state:
                save_fixed_checkpoint(learner, resume_path or output / "resume_latest.pt", progress, replay, rng,
                                      warmup_origin=warmup_origin, random_warmup=random_warmup,
                                      next_metrics_at=next_metrics_at, latest=latest, pending_learning=pending_learning,
                                      action_generator=getattr(vector, "action_generator", None),
                                      environment=environment, selection_state=evaluator.best.state_dict())
            return progress

        time_limit = False
        with (output / "episodes.jsonl").open("w") as stream:
            for old_record in history:
                stream.write(json.dumps(old_record, allow_nan=False) + "\n")
            while completed < config.runtime.episodes:
                if global_step < settings.learning_starts and random_warmup:
                    action = (vector.random_actions(obs["observation"]) if device_env else
                              random_actions(obs["observation"],rng,config.world.num_uavs+1,config.continuous_dim))
                else:
                    action = collector.act_batch(obs["observation"],step=global_step,return_tensors=device_env)
                actions = {"continuous": action.continuous, "destination": action.destination}
                overlap = count > 1 and config.runtime.collector_overlap
                if overlap:
                    vector.step_async(actions)
                    learn()  # Use previously stored transitions while CPU workers advance.
                begin = time.perf_counter()
                next_obs, reward, terminated, truncated, infos = (vector.step_wait() if overlap else vector.step(actions))
                collection_wait_s += time.perf_counter() - begin
                # DISABLED autoreset guarantees these are terminal observations/state.
                replay.add_batch(obs["observation"], obs["state"], action, reward,
                                 next_obs["observation"], next_obs["state"], terminated)
                pending_learning = True
                global_step += count
                watchdog.progress(global_step)
                collection_steps += 1
                episode_returns += reward
                episode_lengths += 1
                done = terminated | truncated
                done_indices = (list(infos["device_episode_infos"]) if device_env else
                                np.flatnonzero(done).tolist())
                time_limit = (config.runtime.max_train_seconds > 0
                              and time.perf_counter() - start >= config.runtime.max_train_seconds)
                final_batch = completed + len(done_indices) >= config.runtime.episodes or time_limit
                if not overlap or final_batch:
                    # The last collected batch has no following CPU step in
                    # which to execute its deferred updates; drain them here.
                    learn(force_metrics=final_batch)
                batch_records = []
                before_completed = completed
                for i in done_indices[:config.runtime.episodes - completed]:
                    completed += 1
                    info = unpack_info(infos, i)
                    record = {"episode": completed, "worker": int(i), "seed": int(episode_seeds[i]),
                              "global_step": global_step, "return": float(episode_returns[i]),
                              "episode_length": int(episode_lengths[i]), "terminated": bool(terminated[i]),
                              "truncated": bool(truncated[i]), **info, "learner": learner_telemetry(latest)}
                    record["num_uavs"] = config.world.num_uavs
                    record["motion_dim"] = config.motion_dim
                    diagnostic_values = diagnostics.episode(record)
                    encoded = json.dumps(record, allow_nan=False)
                    stream.write(encoded + "\n")
                    stream.flush()
                    history.append(json.loads(encoded))
                    batch_records.append(record)
                    if tracker:
                        tracker.episode(record)
                        tracker.diagnostic(diagnostic_values, global_step)
                    print_evidence({"algorithm": learner.algorithm, "episode": completed,
                                      "global_step": global_step, "return": round(record["return"], 3),
                                      "search_rate_percent": info["search_rate_percent"],
                                      "delivery_rate_percent": info["delivery_rate_percent"],
                                      "updates": learner.updates,
                                      "termination_reason": info["termination_reason"],
                                      "minimum_battery_fraction": min(info["energy"]["battery_fraction"]),
                                      "active_uavs": info["energy"]["active_uavs"],
                                      "expired_reports": info["expired_reports"],
                                      "dropped_reports": info["dropped_reports"],
                                      "regenerated_reports": info["regenerated_reports"],
                                      "motion": info["motion_counts"],
                                      "routing": info["network"].get("selection_counts", {})})
                values = diagnostics.batch(batch_records, global_step, final=final_batch,
                    partial=len(batch_records) < len(done_indices))
                if tracker and values:
                    tracker.diagnostic(values, global_step)
                evaluator.tick(global_step, completed,
                    lambda: (learner.actors.state_dict(), learner.actor_updates),
                    {"episodes": completed, "global_step": global_step,
                     "collection_steps": collection_steps, "next_seed": next_seed},
                    allow_submit=not time_limit and (not config.runtime.max_train_seconds or
                        time.perf_counter() - start + 10 < config.runtime.max_train_seconds))
                # One policy per completed collection; every worker has identical weights here.
                if completed // config.runtime.checkpoint_interval > before_completed // config.runtime.checkpoint_interval:
                    path = output / f"episode_{completed:06d}.pt"
                    progress = save(path, full_state=False)
                    save(output / "latest.pt", final=final_batch, full_state=False)
                    if tracker:
                        tracker.checkpoint(path, progress)
                    prune_episode_checkpoints(output, config.tracking.local_checkpoints_to_keep)
                if collector is not learner and learner.actor_updates != collector_version:
                    refresh_base_collector(collector, learner)
                    collector_version = learner.actor_updates
                obs = next_obs
                if tracker:
                    tracker.progress(obs, global_step, completed, learner, len(replay), episode_lengths=episode_lengths, force=final_batch)
                if final_batch:
                    break
                if done_indices:
                    seeds = [None] * count
                    for i in done_indices:
                        seeds[i] = next_seed
                        episode_seeds[i] = next_seed
                        next_seed += 1
                        episode_returns[i] = episode_lengths[i] = 0
                    reset_mask = done.cpu().numpy() if device_env else done
                    obs, _ = vector.reset(seed=seeds, options={"reset_mask": reset_mask})
                recovery.maybe_save(save, tracker, step=global_step)
        checkpoint = output / "final.pt"
        drain = min(10., max(0., config.runtime.max_train_seconds - (time.perf_counter() - start))) if config.runtime.max_train_seconds else 10.
        evaluator.close(global_step, drain_seconds=drain)
        save(checkpoint, final=True, full_state=True)
        elapsed = time.perf_counter() - start
        result = {"checkpoint": str(checkpoint), "resume_checkpoint": str(output / "resume_latest.pt"),
                  "algorithm": learner.algorithm,
                  "requested_episodes": config.runtime.episodes, "episodes": completed,
                  "initial_episodes": initial["episodes"], "segment_episodes": completed - initial["episodes"],
                  "initial_global_step": initial["global_step"], "new_updates": learner.updates - initial_updates,
                  "episode_budget_complete": completed >= config.runtime.episodes,
                  "stop_reason": "episode_budget" if completed >= config.runtime.episodes else "time_budget",
                  "state_restore": initial.get("state_restore", "fresh"),
                  "optimizer_replay_restored": resumed,
                  "environment_state_saved": device_env,
                  "global_step": global_step, "collection_steps": collection_steps,
                  "num_envs": count, "updates": learner.updates, "device": device,
                  "devices": {"collector": str(collector_device), "learner": str(learner_device),
                              "replay": str(replay_device)},
                  "environment_backend": config.runtime.environment_backend,
                  "num_cpu_env_workers": vector.num_cpu_workers if device_env else (count if count>1 else 0),
                  "replay_device": legacy_replay_device, "replay_bytes": replay.allocated_bytes,
                  "sample_updates_per_transition": learner.updates * settings.batch_size / global_step,
                  "collector_overlap": count > 1 and config.runtime.collector_overlap,
                  "startup_s": startup_s, "collection_wait_s": collection_wait_s, "learner_s": update_s,
                  "elapsed_s": elapsed, "joint_transitions_per_s": (global_step - initial["global_step"]) / elapsed,
                  "final_episode": history[-1] if history else None}
        if "policy_warm_start" in initial:
            result["policy_warm_start"] = initial["policy_warm_start"]
        result.update(evaluator.best.result(checkpoint))
        result["periodic_evaluation"] = dict(evaluator.counts)
        result["storage"] = disk_metrics(output)
        result["replay"] = replay.stats()
        result["reproducibility"] = reproducibility_result(learner, history, warm_start_sha256=warm_start_sha256)
        if history:
            result.update(final_metrics(history))
        if tracker:
            result.update(tracker.identity())
            result["tracking_status"] = "pending"
        summary.write_text(json.dumps(result, indent=2, allow_nan=False))
        if tracker:
            try:
                result.update(tracker.complete(result))
            except Exception as error:
                result["tracking_status"] = "failed"
                result["tracking_error_type"] = type(error).__name__
                summary.write_text(json.dumps(result, indent=2, allow_nan=False))
                raise
        summary.write_text(json.dumps(result, indent=2, allow_nan=False))
        completed_ok = True
        return result
    except BaseException as error:
        try:
            bad_parameters = []
            if learner is not None:
                state = {name: tensor.detach().cpu() for name, tensor in learner.state_dict().items()}
                bad_parameters = [name for name, tensor in state.items() if not torch.isfinite(tensor).all()]
            path = failure_report(config, output, error, {
                "algorithm": config.runtime.algorithm, "global_step": global_step,
                "episodes": completed, "updates": learner.updates if learner is not None else None,
                "episode_seeds": episode_seeds.tolist(),
                "last_learner_metrics": latest, "device": str(learner_device),
                "nonfinite_parameter_names": bad_parameters})
            if learner is not None:
                torch.save({"model": state, "config": config.to_dict(), "updates": learner.updates,
                            "snapshot_stage": "after_failure; optimizer may have partially updated"},
                           output / "failure_state.pt")
            if tracker:
                tracker.failure(path, global_step)
        except Exception as reporting_error:
            print_evidence({"failure_reporting_error": {"error_type": type(reporting_error).__name__}})
        raise
    finally:
        try:
            cleanup_ok = False
            try:
                try:
                    if evaluator is not None:
                        evaluator.close(global_step, drain_seconds=0)
                finally:
                    if vector is not None:
                        vector.close()
                cleanup_ok = True
            finally:
                if tracker:
                    failed = not completed_ok or not cleanup_ok
                    try:
                        tracker.close(failed=failed)
                    except Exception as error:
                        if result is not None:
                            result["tracking_status"] = "failed"
                            result["tracking_error_type"] = type(error).__name__
                            summary.write_text(json.dumps(result, indent=2, allow_nan=False))
                        raise
                    if result is not None:
                        result["tracking_status"] = "failed" if failed else "finished"
                        summary.write_text(json.dumps(result, indent=2, allow_nan=False))
        finally:
            watchdog.close()
