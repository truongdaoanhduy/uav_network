"""Terminal-safe Warp/native collection with separate rate-controlled learners."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from ..environment.vector import make_vector, random_actions, unpack_info
from .diagnostics import TrainingDiagnostics, failure_report
from .evidence import print_evidence, learner_telemetry
from .progress_watchdog import ProgressWatchdog
from .overflow_capture import attach_overflow_capture
from .async_evaluation import AsyncCheckpointEvaluator
from .recovery_checkpoints import PeriodicRecovery
from .local_checkpoints import prune_episode_checkpoints, disk_metrics, release_staged_checkpoint
from .pql_system import PQLSystem
from .reproducibility import reproducibility_result
from .environment_checkpoints import snapshot_environment, initialize_environment
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
    engine = vector = tracker = evaluator = None
    completed_ok = False
    result = None
    summary = output / "summary.json"
    completed = global_step = collection_steps = 0
    history = []
    latest, last_runtime_metrics, seeds = {}, {}, None
    diagnostics = TrainingDiagnostics(config, output)
    watchdog = ProgressWatchdog(output)
    try:
        checkpoint = config.runtime.warm_start_checkpoint
        engine = PQLSystem(config, checkpoint=None if checkpoint == "none" else checkpoint)
        attach_overflow_capture(engine.critic, output, "critic")
        if engine.policy is not engine.critic:
            attach_overflow_capture(engine.policy, output, "policy")
        initial = deepcopy(engine.initial_progress)
        completed, global_step, collection_steps = (initial[key] for key in ("episodes", "global_step", "collection_steps"))
        if completed >= config.runtime.episodes:
            raise ValueError("Requested episode budget must exceed completed checkpoint episodes")
        diagnostics.start_windows(global_step)
        history = [{**r, "learner": learner_telemetry(r.get("learner", {}))}
                   for r in initial["records"]]
        resumed = checkpoint != "none"
        initial["optimizer_replay_restored"] = resumed and len(engine.replay) > 0
        initial["state_restore"] = ("training_state_exact_environment" if engine.initial_environment is not None
                                    else "training_state_environment_reset" if resumed else "fresh")
        tracker = start_tracking(config, output, initial_progress=initial)
        evaluator = AsyncCheckpointEvaluator(config, output, tracker, diagnostics, global_step, completed,
            selection_state=engine.initial_selection)
        engine.initial_selection = None
        vector = make_vector(config, device=engine.collector_device)
        count = config.runtime.num_envs
        obs, returns, lengths, seeds, next_seed = initialize_environment(
            vector, initial, engine.initial_environment)
        engine.initial_environment = None
        device_env = config.runtime.environment_backend == "warp_packet"
        rng = np.random.default_rng(config.runtime.seed + 1)
        if engine.initial_collector_rng is not None:
            rng.bit_generator.state = engine.initial_collector_rng["numpy"]
            generator = getattr(vector, "action_generator", None)
            if generator is not None and engine.initial_collector_rng["torch"] is not None:
                generator.set_state(engine.initial_collector_rng["torch"].cpu())
        engine.initial_collector_rng = None
        warm_start_sha256 = release_staged_checkpoint(checkpoint)
        next_log = global_step
        next_checkpoint = (completed // config.runtime.checkpoint_interval + 1) * config.runtime.checkpoint_interval
        latest = {}
        startup_s = time.perf_counter() - started
        recovery = PeriodicRecovery(config, output, started=started)
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

        def save(policy_path, final=False, full_state=True, resume_path=None, publish=True):
            state = progress(final)
            # Resume state stays separate from the small model used for W&B/evaluation.
            if full_state:
                generator = getattr(vector, "action_generator", None)
                collector_rng = {"numpy": rng.bit_generator.state,
                                 "torch": generator.get_state().cpu() if generator is not None else None}
                engine.flush()
                environment = snapshot_environment(vector, returns, lengths, seeds, next_seed)
                state["environment_state_saved"] = environment is not None
                engine.save(resume_path or output / "resume_latest.pt", state, collector_rng=collector_rng,
                            environment=environment, selection_state=evaluator.best.state_dict())
            else:
                # vmap temporarily swaps live critic parameters; light snapshots
                # need the same drained boundary as a full resume checkpoint.
                engine.flush()
            engine.critic.save(policy_path, progress=state)
            if tracker and not final and publish:
                tracker.checkpoint(policy_path, state)
            prune_episode_checkpoints(output, config.tracking.local_checkpoints_to_keep)
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
                watchdog.progress(global_step)
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
                batch_records = []
                for i in indices:
                    if completed >= config.runtime.episodes:
                        break
                    completed += 1
                    info = unpack_info(infos, i)
                    record = {"episode": completed, "worker": int(i), "seed": int(seeds[i]),
                              "global_step": global_step, "return": float(returns[i]),
                              "episode_length": int(lengths[i]), "terminated": bool(terminated[i]),
                              "truncated": bool(truncated[i]), **info, "learner": learner_telemetry(latest),
                              "num_uavs": config.world.num_uavs}
                    encoded = json.dumps(record, allow_nan=False)
                    stream.write(encoded + "\n")
                    history.append(json.loads(encoded))
                    batch_records.append(record)
                    values = diagnostics.episode(record)
                    if tracker:
                        tracker.episode(record)
                        tracker.diagnostic(values, global_step)
                stream.flush()
                values = diagnostics.batch(batch_records, global_step, final=final_batch,
                    partial=len(batch_records) < len(indices))
                if tracker and values:
                    tracker.diagnostic(values, global_step)

                def evaluation_source():
                    snapshot, version = engine.policy_mailbox.read()
                    return snapshot["actors"], version

                evaluator.tick(global_step, completed, evaluation_source,
                    {"episodes": completed, "global_step": global_step,
                     "collection_steps": collection_steps, "next_seed": next_seed},
                    allow_submit=not time_limit and (not config.runtime.max_train_seconds or
                        time.perf_counter() - started + 10 < config.runtime.max_train_seconds))
                obs = following
                if global_step >= next_log or indices or final_batch:
                    values = engine.metrics()
                    last_runtime_metrics = values.copy()
                    if tracker:
                        tracker.learner(latest, global_step)
                        tracker.diagnostic({**values, **disk_metrics(output)}, global_step)
                        tracker.progress(obs, global_step, completed, engine.critic, len(engine.replay), episode_lengths=lengths, force=True)
                    print_evidence({"algorithm": config.runtime.algorithm, "architecture": "pql-inspired",
                                      "episodes": completed, "global_step": global_step,
                                      "critic_updates": engine.critic.updates,
                                      "actor_updates": engine.policy.actor_updates,
                                      **progress_metrics(obs, config),
                                      **{key: values[key] for key in (
                                          "pql/transitions_per_s", "pql/critic_updates_per_s",
                                          "pql/actor_updates_per_s", "pql/critic_collector_ratio",
                                          "pql/sample_uses_per_new_transition", "pql/queued_collections",
                                          "pql/collector_wait_seconds", "replay/evicted_unsampled_fraction")}},
                                     )
                    next_log = global_step + config.runtime.log_interval_steps
                if not final_batch and completed >= next_checkpoint:
                    path = output / f"episode_{completed:06d}.pt"
                    save(path, full_state=False)
                    next_checkpoint = (completed // config.runtime.checkpoint_interval + 1) * config.runtime.checkpoint_interval
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
                recovery.maybe_save(
                    lambda path, resume_path: save(path, resume_path=resume_path, publish=False),
                    tracker, step=global_step)
        final = output / "final.pt"
        drain = min(10., max(0., config.runtime.max_train_seconds - (time.perf_counter() - started))) if config.runtime.max_train_seconds else 10.
        evaluator.close(global_step, drain_seconds=drain)
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
                  "state_restore": initial["state_restore"],
                  "environment_state_saved": device_env,
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
        result.update(evaluator.best.result(final))
        result["periodic_evaluation"] = dict(evaluator.counts)
        result["storage"] = disk_metrics(output)
        result["replay"] = engine.replay.stats()
        result["reproducibility"] = reproducibility_result(engine.critic, history, warm_start_sha256=warm_start_sha256)
        result["reproducibility"].update(scheduling_deterministic=not engine.asynchronous,
            scope=("Synchronous schedule requires identical source/settings/platform; exact Warp state when available"
                   if not engine.asynchronous else "Seeded algorithms/environment; asynchronous scheduling is not bitwise deterministic"))
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
                result["tracking_status"], result["tracking_error_type"] = "failed", type(error).__name__
                summary.write_text(json.dumps(result, indent=2, allow_nan=False))
                raise
        summary.write_text(json.dumps(result, indent=2, allow_nan=False))
        completed_ok = True
        return result
    except BaseException as error:
        try:
            path = failure_report(config, output, error, {"global_step": global_step, "episodes": completed,
                                  "updates": engine.critic.updates if engine else 0,
                                  "training_architecture": "pql-inspired", "stage": "train",
                                  "algorithm": config.runtime.algorithm, "seed": config.runtime.seed,
                                  "collection_steps": collection_steps, "episode_seeds": seeds,
                                  "last_learner_metrics": latest, "last_runtime_metrics": last_runtime_metrics})
            if engine and hasattr(engine, "failed_batch"):
                torch.save(engine.failed_batch, output / "failed_batch.pt")
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
                    try:
                        if engine:
                            engine.close()
                    finally:
                        if vector:
                            vector.close()
                cleanup_ok = True
            finally:
                if tracker:
                    failed = not completed_ok or not cleanup_ok
                    try:
                        tracker.close(failed=failed)
                    except Exception as error:
                        if result is not None:
                            result["tracking_status"], result["tracking_error_type"] = "failed", type(error).__name__
                            summary.write_text(json.dumps(result, indent=2, allow_nan=False))
                        raise
                    if result is not None:
                        result["tracking_status"] = "failed" if failed else "finished"
                        summary.write_text(json.dumps(result, indent=2, allow_nan=False))
        finally:
            watchdog.close()
