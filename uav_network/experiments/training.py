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
from ..algorithms.replay import TensorReplayBuffer
from ..environment.observations import state_dim
from ..environment.vector import make_vector, random_actions, unpack_info
from .tracking import start_tracking, final_metrics
from .lifecycle import graceful_termination
from .continuation import restore_learner
from .reproducibility import reproducibility_result
from .diagnostics import TrainingDiagnostics, failure_report, evaluate_checkpoint


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


def train(config):
    with graceful_termination():
        return _train(config)


def _train(config):
    config.validate()
    seed_runtime(config)
    device = choose_device(config.runtime.device)
    replay_device = device if config.runtime.replay_device == "auto" else choose_device(config.runtime.replay_device)
    settings = config.algorithm_config
    output = Path(config.runtime.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}; choose a new runtime.output_dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    start = time.perf_counter()
    diagnostics = TrainingDiagnostics(config, output)
    count = config.runtime.num_envs
    completed = global_step = collection_steps = 0
    latest = {}
    update_s = collection_wait_s = 0.0
    next_metrics_at = 0
    episode_seeds = np.arange(count) + config.runtime.seed
    episode_returns, episode_lengths = np.zeros(count), np.zeros(count, dtype=int)
    tracker = vector = learner = None
    completed_ok = False
    try:
        learner, initial = restore_learner(config, device)
        completed, global_step, collection_steps = (initial[k] for k in ("episodes", "global_step", "collection_steps"))
        initial_updates = learner.updates
        history = list(initial["records"])
        episode_seeds = np.arange(count) + initial["next_seed"]
        next_seed = initial["next_seed"] + count
        vector = make_vector(config)
        tracker = start_tracking(config, output, initial_progress=initial)
        obs, _ = vector.reset(seed=episode_seeds.tolist())
        if config.runtime.compile_models:
            compile_models(learner)
        replay = TensorReplayBuffer(settings.replay_capacity, obs["observation"].shape[1:],
                                    state_dim(config), config.runtime.seed, replay_device,
                                    pin_memory=device == "cuda" and replay_device == "cpu")
        rng = np.random.default_rng(config.runtime.seed + 1)
        startup_s = time.perf_counter() - start

        def learn(force_metrics=False):
            nonlocal latest, update_s, next_metrics_at
            if global_step - initial["global_step"] < settings.learning_starts or len(replay) < settings.batch_size:
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
                    tracker.diagnostic({"timing/collector_wait_seconds": collection_wait_s,
                                        "timing/learner_seconds": update_s,
                                        "train/sample_uses_per_new_transition":
                                            (learner.updates - initial_updates) * settings.batch_size /
                                            max(1, global_step - initial["global_step"])}, global_step)
                next_metrics_at = global_step + config.runtime.log_interval_steps

        def save(path, final=False):
            progress = {"episodes": completed, "global_step": global_step,
                        "collection_steps": collection_steps, "next_seed": next_seed,
                        "records": history}
            if final:
                progress["episode_budget_complete"] = completed >= config.runtime.episodes
                progress["stop_reason"] = "episode_budget" if completed >= config.runtime.episodes else "time_budget"
            if tracker:
                progress["source_run_id"] = tracker.identity()["wandb_run_id"]
            learner.save(path, progress=progress)
            return progress

        time_limit = False
        with (output / "episodes.jsonl").open("w") as stream:
            for old_record in history:
                stream.write(json.dumps(old_record, allow_nan=False) + "\n")
            while completed < config.runtime.episodes:
                action = (random_actions(obs["observation"], rng, config.world.num_uavs + 1)
                          if global_step < settings.learning_starts and config.runtime.warm_start_checkpoint == "none" else
                          learner.act_batch(obs["observation"], step=global_step))
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
                global_step += count
                collection_steps += 1
                episode_returns += reward
                episode_lengths += 1
                done = terminated | truncated
                time_limit = (config.runtime.max_train_seconds > 0
                              and time.perf_counter() - start >= config.runtime.max_train_seconds)
                final_batch = completed + int(done.sum()) >= config.runtime.episodes or time_limit
                if not overlap or final_batch:
                    # The last collected batch has no following CPU step in
                    # which to execute its deferred updates; drain them here.
                    learn(force_metrics=final_batch)
                for i in np.flatnonzero(done):
                    completed += 1
                    info = unpack_info(infos, i)
                    record = {"episode": completed, "worker": int(i), "seed": int(episode_seeds[i]),
                              "global_step": global_step, "return": float(episode_returns[i]),
                              "episode_length": int(episode_lengths[i]), "terminated": bool(terminated[i]),
                              "truncated": bool(truncated[i]), **info, "learner": latest}
                    record["num_uavs"] = config.world.num_uavs
                    diagnostic_values = diagnostics.episode(record)
                    encoded = json.dumps(record, allow_nan=False)
                    stream.write(encoded + "\n")
                    stream.flush()
                    history.append(json.loads(encoded))
                    if tracker:
                        tracker.episode(record)
                        tracker.diagnostic(diagnostic_values, global_step)
                    print(json.dumps({"algorithm": learner.algorithm, "episode": completed,
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
                                      "routing": info["network"].get("selection_counts", {})}), flush=True)
                    if completed % config.runtime.checkpoint_interval == 0:
                        path = output / f"episode_{completed:06d}.pt"
                        progress = save(path)
                        if tracker:
                            tracker.checkpoint(path, progress)
                    interval = config.tracking.evaluation_interval
                    if tracker and interval and completed % interval == 0:
                        path = output / f"episode_{completed:06d}.pt"
                        if completed % config.runtime.checkpoint_interval != 0:
                            save(path)
                        evaluate_checkpoint(config, path, completed, global_step, tracker, diagnostics)
                obs = next_obs
                if tracker:
                    tracker.progress(obs, global_step, completed, learner, len(replay), force=final_batch)
                if final_batch:
                    break
                if done.any():
                    seeds = [None] * count
                    for i in np.flatnonzero(done):
                        seeds[i] = next_seed
                        episode_seeds[i] = next_seed
                        next_seed += 1
                        episode_returns[i] = episode_lengths[i] = 0
                    obs, _ = vector.reset(seed=seeds, options={"reset_mask": done})
        checkpoint = output / "final.pt"
        save(checkpoint, final=True)
        elapsed = time.perf_counter() - start
        result = {"checkpoint": str(checkpoint), "algorithm": learner.algorithm,
                  "requested_episodes": config.runtime.episodes, "episodes": completed,
                  "initial_episodes": initial["episodes"], "segment_episodes": completed - initial["episodes"],
                  "initial_global_step": initial["global_step"], "new_updates": learner.updates - initial_updates,
                  "episode_budget_complete": completed >= config.runtime.episodes,
                  "stop_reason": "episode_budget" if completed >= config.runtime.episodes else "time_budget",
                  "state_restore": "policy_warm_start" if config.runtime.warm_start_checkpoint != "none" else "fresh",
                  "global_step": global_step, "collection_steps": collection_steps,
                  "num_envs": count, "updates": learner.updates, "device": device,
                  "replay_device": replay_device, "replay_bytes": replay.allocated_bytes,
                  "sample_updates_per_transition": learner.updates * settings.batch_size / global_step,
                  "collector_overlap": count > 1 and config.runtime.collector_overlap,
                  "startup_s": startup_s, "collection_wait_s": collection_wait_s, "learner_s": update_s,
                  "elapsed_s": elapsed, "joint_transitions_per_s": (global_step - initial["global_step"]) / elapsed,
                  "final_episode": history[-1] if history else None}
        result["reproducibility"] = reproducibility_result(learner, history)
        if history:
            result.update(final_metrics(history))
        if tracker:
            result.update(tracker.identity())
            result["tracking_status"] = "pending"
        summary = output / "summary.json"
        summary.write_text(json.dumps(result, indent=2, allow_nan=False))
        if tracker:
            try:
                result.update(tracker.complete(result))
                tracker.close()
                result["tracking_status"] = "finished"
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
                torch.save({"model": state, "config": config.to_dict(), "updates": learner.updates,
                            "snapshot_stage": "after_failure; optimizer may have partially updated"},
                           output / "failure_state.pt")
            path = failure_report(config, output, error, {
                "algorithm": config.runtime.algorithm, "global_step": global_step,
                "episodes": completed, "updates": learner.updates if learner is not None else None,
                "episode_seeds": episode_seeds.tolist(),
                "last_learner_metrics": latest, "device": device,
                "nonfinite_parameter_names": bad_parameters})
            if tracker:
                tracker.failure(path, global_step)
        except Exception as reporting_error:
            print(f"Failure reporting also failed: {type(reporting_error).__name__}", flush=True)
        raise
    finally:
        try:
            if vector is not None:
                vector.close()
        finally:
            if tracker:
                tracker.close(failed=not completed_ok)
