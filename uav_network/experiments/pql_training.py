"""Terminal-safe Warp/native collection with separate rate-controlled learners."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from ..environment.vector import make_vector, random_actions, unpack_info
from .diagnostics import TrainingDiagnostics, evaluate_checkpoint, failure_report
from .pql_system import PQLSystem
from .reproducibility import reproducibility_result
from .tracking import final_metrics, progress_metrics, start_tracking
from .training import seed_runtime


def train_pql(config):
    config.validate()
    seed_runtime(config)
    output = Path(config.runtime.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}; choose a new runtime.output_dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    started = time.perf_counter()
    engine = vector = tracker = None
    completed_ok = False
    completed = global_step = collection_steps = 0
    history = []
    diagnostics = TrainingDiagnostics(config, output)
    try:
        checkpoint = config.runtime.warm_start_checkpoint
        engine = PQLSystem(config, checkpoint=None if checkpoint == "none" else checkpoint)
        initial = deepcopy(engine.initial_progress)
        completed, global_step, collection_steps = (initial[key] for key in ("episodes", "global_step", "collection_steps"))
        if completed >= config.runtime.episodes:
            raise ValueError("Requested episode budget must exceed completed checkpoint episodes")
        history = list(initial["records"])
        resumed = checkpoint != "none"
        initial["optimizer_replay_restored"] = resumed and len(engine.replay) > 0
        initial["state_restore"] = "training_state_environment_reset" if resumed else "fresh"
        tracker = start_tracking(config, output, initial_progress=initial)
        vector = make_vector(config, device=engine.collector_device)
        count = config.runtime.num_envs
        seeds = np.arange(count) + initial["next_seed"]
        next_seed = initial["next_seed"] + count
        obs, _ = vector.reset(seed=seeds.tolist())
        device_env = config.runtime.environment_backend == "warp_packet"
        if device_env:
            returns = torch.zeros(count, device=engine.collector_device, dtype=torch.float64)
            lengths = torch.zeros(count, device=engine.collector_device, dtype=torch.int64)
        else:
            returns, lengths = np.zeros(count), np.zeros(count, dtype=np.int64)
        rng = np.random.default_rng(config.runtime.seed + 1)
        next_log = global_step
        next_checkpoint = (completed // config.runtime.checkpoint_interval + 1) * config.runtime.checkpoint_interval
        latest = {}
        startup_s = time.perf_counter() - started
        time_limit = False

        def progress(final=False):
            result = {"episodes": completed, "global_step": global_step,
                      "collection_steps": collection_steps, "next_seed": next_seed, "records": history}
            if tracker:
                result["source_run_id"] = tracker.identity()["wandb_run_id"]
            if final:
                result.update(episode_budget_complete=completed >= config.runtime.episodes,
                              stop_reason="episode_budget" if completed >= config.runtime.episodes else "time_budget")
            return result

        def save(policy_path, final=False):
            state = progress(final)
            # Resume state stays separate from the small model used for W&B/evaluation.
            engine.save(output / "resume_latest.pt", state)
            engine.critic.save(policy_path, progress=state)
            if tracker and not final:
                tracker.checkpoint(policy_path, state)
            return state

        with (output / "episodes.jsonl").open("w") as stream:
            for record in history:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            while completed < config.runtime.episodes:
                begin = time.perf_counter()
                warmup = engine.collections - engine.warmup_origin < config.pql.warmup_collection_steps
                if warmup:
                    action = (vector.random_actions(obs["observation"]) if device_env else
                              random_actions(obs["observation"], rng, config.world.num_uavs + 1))
                else:
                    action = engine.act_batch(obs["observation"], step=global_step, return_tensors=device_env)
                following, reward, terminated, truncated, infos = vector.step({
                    "continuous": action.continuous, "destination": action.destination})
                engine.record_collection_time(time.perf_counter() - begin)
                # Cloning before reset/next step makes double-buffer and reward storage safe.
                engine.submit((obs["observation"], obs["state"], action, reward,
                               following["observation"], following["state"], terminated))
                global_step += count
                collection_steps += 1
                returns += reward
                lengths += 1
                done = terminated | truncated
                indices = list(infos["device_episode_infos"]) if device_env else np.flatnonzero(done).tolist()
                time_limit = (config.runtime.max_train_seconds > 0 and
                              time.perf_counter() - started >= config.runtime.max_train_seconds)
                final_batch = completed + len(indices) >= config.runtime.episodes or time_limit
                if final_batch:
                    engine.flush()
                with engine.condition:
                    latest = dict(engine.latest)
                for i in indices:
                    completed += 1
                    info = unpack_info(infos, i)
                    record = {"episode": completed, "worker": int(i), "seed": int(seeds[i]),
                              "global_step": global_step, "return": float(returns[i]),
                              "episode_length": int(lengths[i]), "terminated": bool(terminated[i]),
                              "truncated": bool(truncated[i]), **info, "learner": latest,
                              "num_uavs": config.world.num_uavs}
                    encoded = json.dumps(record, allow_nan=False)
                    stream.write(encoded + "\n")
                    history.append(json.loads(encoded))
                    values = diagnostics.episode(record)
                    if tracker:
                        tracker.episode(record)
                        tracker.diagnostic(values, global_step)
                stream.flush()
                obs = following
                if global_step >= next_log or indices or final_batch:
                    values = engine.metrics()
                    if tracker:
                        tracker.learner(latest, global_step)
                        tracker.diagnostic(values, global_step)
                        tracker.progress(obs, global_step, completed, engine.critic, len(engine.replay), force=True)
                    print(json.dumps({"algorithm": config.runtime.algorithm, "architecture": "pql-inspired",
                                      "episodes": completed, "global_step": global_step,
                                      "critic_updates": engine.critic.updates,
                                      "actor_updates": engine.policy.actor_updates,
                                      **progress_metrics(obs, config),
                                      **{key: values[key] for key in (
                                          "pql/transitions_per_s", "pql/critic_updates_per_s",
                                          "pql/actor_updates_per_s", "pql/critic_collector_ratio",
                                          "pql/sample_uses_per_new_transition", "pql/queued_collections",
                                          "pql/collector_wait_seconds", "replay/evicted_unsampled_fraction")}},
                                     allow_nan=False), flush=True)
                    next_log = global_step + config.runtime.log_interval_steps
                if not final_batch and completed >= next_checkpoint:
                    path = output / f"episode_{completed:06d}.pt"
                    save(path)
                    next_checkpoint = (completed // config.runtime.checkpoint_interval + 1) * config.runtime.checkpoint_interval
                    interval = config.tracking.evaluation_interval
                    if tracker and interval and completed % interval == 0:
                        evaluate_checkpoint(config, path, completed, global_step, tracker, diagnostics)
                if final_batch:
                    break
                if indices:
                    reset_seeds = [None] * count
                    for i in indices:
                        reset_seeds[i] = next_seed
                        seeds[i] = next_seed
                        next_seed += 1
                        returns[i] = lengths[i] = 0
                    reset_mask = done.cpu().numpy() if device_env else done
                    obs, _ = vector.reset(seed=reset_seeds, options={"reset_mask": reset_mask})
        final = output / "final.pt"
        save(final, final=True)
        elapsed = time.perf_counter() - started
        result = {"checkpoint": str(final), "resume_checkpoint": str(output / "resume_latest.pt"),
                  "training_architecture": "pql-inspired", "pql_mode": config.pql.mode,
                  "algorithm": config.runtime.algorithm, "requested_episodes": config.runtime.episodes,
                  "episodes": completed, "initial_episodes": initial["episodes"],
                  "segment_episodes": completed - initial["episodes"],
                  "initial_global_step": initial["global_step"], "global_step": global_step,
                  "collection_steps": collection_steps, "num_envs": count,
                  "updates": engine.critic.updates, "new_updates": engine.critic.updates - engine.initial_updates,
                  "episode_budget_complete": completed >= config.runtime.episodes,
                  "stop_reason": "episode_budget" if completed >= config.runtime.episodes else "time_budget",
                  "state_restore": "training_state_environment_reset" if resumed else "fresh",
                  "optimizer_replay_restored": initial["optimizer_replay_restored"],
                  "devices": {"collector": str(engine.collector_device), "critic": str(engine.critic_device),
                              "policy": str(engine.policy_device)},
                  "device": str(engine.critic_device), "replay_device": str(engine.replay.device),
                  "environment_backend": config.runtime.environment_backend,
                  "num_cpu_env_workers": vector.num_cpu_workers if device_env else (count if count > 1 else 0),
                  "replay_bytes": engine.replay.allocated_bytes, "collector_overlap": engine.asynchronous,
                  "sample_updates_per_transition": engine.metrics()["pql/sample_uses_per_new_transition"],
                  "startup_s": startup_s, "elapsed_s": elapsed,
                  "joint_transitions_per_s": (global_step - initial["global_step"]) / max(elapsed, 1e-9),
                  "pql": engine.metrics(), "final_episode": history[-1] if history else None}
        result["reproducibility"] = reproducibility_result(engine.critic, history)
        result["reproducibility"].update(scheduling_deterministic=not engine.asynchronous,
            scope=("Synchronous schedule requires identical source/settings/platform; resume resets simulators"
                   if not engine.asynchronous else "Seeded algorithms/environment; asynchronous scheduling is not bitwise deterministic"))
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
                result["tracking_status"], result["tracking_error_type"] = "failed", type(error).__name__
                summary.write_text(json.dumps(result, indent=2, allow_nan=False))
                raise
        summary.write_text(json.dumps(result, indent=2, allow_nan=False))
        completed_ok = True
        return result
    except BaseException as error:
        if engine and hasattr(engine, "failed_batch"):
            torch.save(engine.failed_batch, output / "failed_batch.pt")
        path = failure_report(config, output, error, {"global_step": global_step, "episodes": completed,
                              "updates": engine.critic.updates if engine else 0,
                              "training_architecture": "pql-inspired"})
        if tracker:
            tracker.failure(path, global_step)
        raise
    finally:
        try:
            if engine:
                engine.close()
        finally:
            try:
                if vector:
                    vector.close()
            finally:
                if tracker:
                    tracker.close(failed=not completed_ok)
