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
from .convergence_logging import PerVectorStepRewardJournal
from .vector_step_mission_progress import (
    PerVectorStepMissionProgressJournal, pair_reward_and_progress_events,
)
from .vector_step_optimizer_loss import (
    PerVectorStepOptimizerLossJournal, UnifiedTrainingStepJoiner,
)
from .vector_step_completed_outcomes import completed_episode_step_metrics
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
from .deferred_eval_snapshots import DeferredSnapshotBank, publish_bundle
from .training_metric_candidates import TrainingMetricCandidates
from .online_gpu_evaluation import LiveGPUResultLogger
from .recovery_checkpoints import PeriodicRecovery
from .local_checkpoints import prune_episode_checkpoints, disk_metrics, release_staged_checkpoint
from .resource_memory import training_memory_metrics, reset_driver_watermarks
from .action_cbf_monitor import ActionCBFMonitor
from .training_health import TrainingHealthMonitor
from .mission_cause_monitor import MissionCauseMonitor
from .forensic_transitions import ForensicTransitionRecorder, write_run_provenance
from .layerwise_gradient_journal import LayerwiseGradientJournal
from .root_cause_audit import RootCauseAudit
from .forensic_minibatches import ReplayMinibatchForensics
from .forensic_integrity import ForensicIntegrityJournal
from ..algorithms.layerwise_gradient_probe import LayerwiseGradientProbe
from .storage_retention import finalize_local_diagnostics, verify_archive_parts
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
    count = config.runtime.num_envs
    env_step_limit = int(os.environ.get("UAV_TRAIN_ENV_STEP_LIMIT", "0"))
    if env_step_limit < 0 or env_step_limit % count:
        raise ValueError("UAV_TRAIN_ENV_STEP_LIMIT must be a nonnegative multiple of num_envs")
    seed_runtime(config)
    fast_training = os.environ.get("UAV_FAST_TRAINING", "false").lower() in ("1", "true")
    reset_driver_watermarks()
    device = choose_device(config.runtime.device)  # legacy summary field
    legacy_replay_device = device if config.runtime.replay_device == "auto" else choose_device(config.runtime.replay_device)
    collector_device, learner_device, replay_device = resolve_base_devices(config)
    settings = config.algorithm_config
    output = Path(config.runtime.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}; choose a new runtime.output_dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    write_run_provenance(config, output)
    start = time.perf_counter()
    diagnostics = TrainingDiagnostics(config, output)
    action_probe = ActionCBFMonitor(config, output)
    mission_probe = MissionCauseMonitor(config, output)
    forensic = ForensicTransitionRecorder(config, output)
    layer_journal = LayerwiseGradientJournal(output, config.runtime.algorithm)
    root_audit = RootCauseAudit(output, config.runtime.algorithm,
        envs=config.runtime.num_envs, learning_starts=config.algorithm_config.learning_starts,
        normal_interval=config.runtime.log_interval_steps)
    health_probe = TrainingHealthMonitor(config, output)
    minibatch_probe = ReplayMinibatchForensics(output,
        enabled=root_audit.enabled and config.runtime.algorithm == "matd3")
    integrity_probe = ForensicIntegrityJournal(output,
        enabled=root_audit.enabled and config.runtime.environment_backend == "warp_packet")
    watchdog = ProgressWatchdog(output)
    completed = global_step = collection_steps = 0
    latest = {}
    update_s = collection_wait_s = 0.0
    replay_sample_s = optimizer_update_s = 0.0
    next_metrics_at = 0
    episode_seeds = np.arange(count) + config.runtime.seed
    episode_returns, episode_lengths = np.zeros(count), np.zeros(count, dtype=int)
    tracker = vector = learner = evaluator = None
    reward_window = None
    progress_window = None
    loss_window = None
    step_joiner = None
    deferred_bank = None
    live_gpu_logger = None
    completed_ok = False
    result = None
    summary = output / "summary.json"
    try:
        learner, initial = restore_learner(config, learner_device)
        completed, global_step, collection_steps = (initial[k] for k in ("episodes", "global_step", "collection_steps"))
        if env_step_limit and (global_step >= env_step_limit or global_step % count):
            raise ValueError("Restored training step cannot reach the requested environment-step limit")
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
        deferred_bank = (DeferredSnapshotBank(config, output)
                         if config.tracking.deferred_gpu_eval_enabled else None)
        # Training metrics only nominate Actor snapshots for later fixed-seed Validation.
        # Rolling 256 completed episodes, same metric priority as the evaluator;
        # no training-derived checkpoint can directly become the final best.pt.
        training_candidate_selector = (
            TrainingMetricCandidates(window=256)
            if deferred_bank is not None and config.tracking.deferred_gpu_eval_keep_all
            and not env_step_limit else None)
        collector = learner if collector_device == learner_device else inference_copy(learner, collector_device)
        collector_version = learner.actor_updates
        vector = make_vector(config, device=collector_device)
        environment = restored_training.get("environment") if restored_training else None
        obs, episode_returns, episode_lengths, episode_seeds, next_seed = initialize_environment(
            vector, initial, environment)
        if (deferred_bank is not None and deferred_bank.keep_all
                and random_warmup and global_step == 0 and learner.actor_updates == 0):
            # A real policy checkpoint at the first point of the evaluation grid.
            deferred_bank.capture(
                learner.actors.state_dict(), 0, episodes=0, global_step=0,
                collection_steps=0, next_seed=next_seed, force=True, reason="periodic")
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
        reward_window = (PerVectorStepRewardJournal(
            count, output, host_sync_every=4, plot_windows=(1,2,4,8))
            if tracker is not None else None)
        # Same GPU post-step, pre-reset state. Warp -> Torch is zero-copy.
        progress_visited = None
        if tracker is not None and device_env:
            import warp as wp
            progress_visited = wp.to_torch(vector.s.visited)
            progress_window = PerVectorStepMissionProgressJournal(
                count, output, config.world.num_uavs,
                config.world.num_targets, config.world.num_obstacles,
                host_sync_every=4, plot_windows=(1,2,4,8))
        if tracker is not None:
            loss_window = PerVectorStepOptimizerLossJournal(
                count, output, host_sync_every=4, plot_windows=(1,2,4,8))
            step_joiner = UnifiedTrainingStepJoiner()
            # No change to learning schedule or optimizer math: algorithms only
            # expose their already-computed detached scalar loss/grad values.
            learner._dense_loss_sink = loss_window
        # CUDA_VISIBLE_DEVICES remaps physical GPU1 to logical cuda:0 in the
        # strict split, so label the hardware indices explicitly in W&B.
        physical_training = os.environ.get("UAV_TRAINING_PHYSICAL_GPU_INDEX")
        if tracker is not None and physical_training is not None:
            tracker.run.summary.update({
                "hardware/train_physical_gpu_index": int(physical_training),
                "hardware/eval_physical_gpu_index": int(
                    os.environ["UAV_EVAL_PHYSICAL_GPU_INDEX"]),
                "hardware/training_collector_and_learner_co_located": True,
                "hardware/cuda0_metrics_are_physical_gpu_index": int(physical_training),
            })
        if config.tracking.online_gpu_eval_enabled:
            live_gpu_logger = LiveGPUResultLogger(output / "online-gpu-eval-results", tracker)
        evaluator = AsyncCheckpointEvaluator(config, output, tracker, diagnostics, global_step, completed,
            selection_state=restored_training.get("best_checkpoint") if restored_training else None)
        if restored_training:
            restore_runtime_rng(restored_training["rng"], rng, action_generator=getattr(vector, "action_generator", None))
        restored_training = None  # Release the CPU checkpoint copy after restoring replay.
        warm_start_sha256 = release_staged_checkpoint(config.runtime.warm_start_checkpoint)
        attach_overflow_capture(learner, output, "base")
        startup_s = time.perf_counter() - start
        recovery = PeriodicRecovery(config, output, started=start)

        cuda_update_profile_ms = None

        def learn(force_metrics=False):
            nonlocal latest, update_s, replay_sample_s, optimizer_update_s, next_metrics_at, pending_learning, cuda_update_profile_ms
            if not pending_learning:
                return
            pending_learning = False
            if loss_window is not None:
                loss_window.begin(global_step)
            if global_step - warmup_origin < settings.learning_starts or len(replay) < settings.batch_size:
                if loss_window is not None:
                    for raw_event in loss_window.end(global_step):
                        for aligned in step_joiner.add_loss([raw_event]):
                            tracker.step_reward(aligned)
                return
            checked = force_metrics or global_step >= next_metrics_at
            audit_checked = (not fast_training and root_audit.due(global_step))
            early_all_updates = (not fast_training and root_audit.all_updates_due(global_step))
            begin = time.perf_counter()
            for update in range(settings.updates_per_step):
                # Host-side phase timing, deliberately without CUDA synchronize:
                # synchronizing each update would alter the throughput being measured.
                t_sample = time.perf_counter()
                batch = replay.sample(settings.batch_size)
                replay_sample_s += time.perf_counter() - t_sample
                if root_audit.enabled and root_audit.reference_observations is None:
                    # Only transfer the frozen prefix to learner GPU, not
                    # the entire B8192*6*local-observation batch for a 32-row audit.
                    root_audit.remember_reference(learner.actors[0],
                        torch.as_tensor(batch["observation"][:root_audit.max_observations],
                                        device=learner.device),
                        shared=learner.share_actor_parameters,num_uavs=config.world.num_uavs)
                try:
                    t_learner = time.perf_counter()
                    sample_gpu_update = (checked and update == settings.updates_per_step - 1
                        and learner.device.type == "cuda"
                        and os.environ.get("UAV_CUDA_EVENT_PROFILE", "0").strip().lower() in {"1", "true", "yes"})
                    if sample_gpu_update:
                        cuda_start = torch.cuda.Event(enable_timing=True)
                        cuda_end = torch.cuda.Event(enable_timing=True)
                        cuda_start.record(torch.cuda.current_stream(learner.device))
                    sample_metrics = (checked or audit_checked) and (
                        update == settings.updates_per_step - 1 or early_all_updates)
                    pre_update_head = None
                    if early_all_updates and audit_checked and sample_metrics:
                        # Same actual sampled minibatch before optimizer.step;
                        # no RNG/backward/weight mutation from this observer.
                        pre_update_head = root_audit.actor_snapshot(
                            learner.actors[0],
                            torch.as_tensor(batch["observation"][:root_audit.max_observations],
                                            device=learner.device),
                            shared=learner.share_actor_parameters,
                            num_uavs=config.world.num_uavs)
                    if audit_checked and sample_metrics and learner.algorithm == "masac":
                        # One external observer for each MASAC tower; no new backward.
                        with LayerwiseGradientProbe(learner.actors,"actor",enabled=True) as pa, \
                             LayerwiseGradientProbe(learner.critics,"critic",enabled=True) as pc:
                            metrics = learner.update(batch,collect_metrics=True)
                            metrics.update({k: float(v.detach().item()) if torch.is_tensor(v) else v
                                            for k,v in pa.metrics().items()})
                            metrics.update({k: float(v.detach().item()) if torch.is_tensor(v) else v
                                            for k,v in pc.metrics().items()})
                    else:
                        metrics = learner.update(batch,collect_metrics=sample_metrics)
                    if sample_gpu_update:
                        cuda_end.record(torch.cuda.current_stream(learner.device))
                        cuda_end.synchronize()  # Only one diagnostic update, not every update.
                        cuda_update_profile_ms = cuda_start.elapsed_time(cuda_end)
                    optimizer_update_s += time.perf_counter() - t_learner
                except (FloatingPointError, RuntimeError):
                    torch.save({k: v.detach().cpu() for k, v in batch.items()}, output / "failed_batch.pt")
                    raise
                if metrics:
                    latest = metrics
                    if not fast_training:
                        layer_journal.capture(metrics,global_step=global_step,
                            critic_updates=learner.updates,
                            actor_updates=learner.actor_updates,source="base")
                    if audit_checked and sample_metrics:
                        root_audit.capture(global_step, learner.actors[0],
                            torch.as_tensor(batch["observation"][:root_audit.max_observations],
                                            device=learner.device),
                            metrics,actor_updates=learner.actor_updates,
                            critic_updates=learner.updates,
                            num_uavs=config.world.num_uavs,shared=learner.share_actor_parameters,
                            replay=replay.stats(),update_index=update,
                            pre_update_head=pre_update_head)
                        if minibatch_probe.enabled:
                            try:
                                minibatch_probe.capture(learner,batch,
                                    global_step=global_step,update_index=update,
                                    critic_updates=learner.updates,
                                    actor_updates=learner.actor_updates)
                            except Exception as error:
                                diagnostics.event("replay_forensic_logging_failed", global_step,
                                    {"update_index":update,"error_type":type(error).__name__,
                                     "error_message":str(error)[:300]},severity="warning")
                    if tracker and checked:
                        tracker.learner(metrics, global_step)
            update_s += time.perf_counter() - begin
            if checked:
                if tracker:
                    tracker.diagnostic({**replay.stats(), **disk_metrics(output),
                                        **training_memory_metrics(),
                                        "timing/collector_wait_seconds": collection_wait_s,
                                        "timing/learner_seconds": update_s,
                                        "timing/replay_sample_host_seconds": replay_sample_s,
                                        "timing/optimizer_update_host_seconds": optimizer_update_s,
                                        **({"timing/probe_update_cuda_ms": cuda_update_profile_ms}
                                           if cuda_update_profile_ms is not None else {}),
                                        "train/sample_uses_per_new_transition":
                                            (learner.updates - initial_updates) * settings.batch_size /
                                            max(1, global_step - initial["global_step"])}, global_step)
                next_metrics_at = global_step + config.runtime.log_interval_steps
            if loss_window is not None:
                # Includes ALL 8 actual optimizer minibatch updates per vector
                # step. GPU->CPU transfers are batched every 4 vector steps.
                for raw_event in loss_window.end(global_step):
                    for aligned in step_joiner.add_loss([raw_event]):
                        tracker.step_reward(aligned)

        def save(path, final=False, full_state=True, resume_path=None):
            progress = {"episodes": completed, "global_step": global_step,
                        "collection_steps": collection_steps, "next_seed": next_seed,
                        "records": history}
            if final:
                progress["episode_budget_complete"] = completed >= config.runtime.episodes
                progress["stop_reason"] = stop_reason
                if env_step_limit:
                    progress["requested_env_steps"] = env_step_limit
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
        stop_reason = "episode_budget"
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
                trace = None
                if device_env and not fast_training:
                    try:
                        trace = forensic.begin(vector, obs, action, collector,
                            collection_step=collection_steps, global_step=global_step,
                            phase="random_warmup" if global_step < settings.learning_starts and random_warmup else "policy",
                            policy_version=collector_version)
                    except Exception as error:
                        diagnostics.event("forensic_capture_begin_failed", global_step,
                            {"error_type":type(error).__name__, "error_message":str(error)[:200]})
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
                if reward_window is not None:
                    # Same worlds and same global_step, one W&B numeric row.
                    reward_events = reward_window.observe(reward, global_step)
                    if progress_window is not None:
                        progress_events = progress_window.observe(
                            next_obs, global_step, visited=progress_visited)
                        joined = pair_reward_and_progress_events(
                            reward_events, progress_events)
                    else:
                        joined = reward_events
                    for event in step_joiner.add_training(joined) if step_joiner else joined:
                        tracker.step_reward(event)
                episode_returns += reward
                episode_lengths += 1
                if trace is not None:
                    try:
                        forensic.finish(trace, vector, next_obs, reward, terminated,
                            global_step=global_step, episode_returns=episode_returns,
                            learner_metrics=latest, replay_metrics=replay.stats(),
                            policy_version=collector_version)
                    except Exception as error:
                        diagnostics.event("forensic_capture_end_failed", global_step,
                            {"error_type":type(error).__name__, "error_message":str(error)[:200]})
                done = terminated | truncated
                done_indices = (list(infos["device_episode_infos"]) if device_env else
                                np.flatnonzero(done).tolist())
                time_limit = (config.runtime.max_train_seconds > 0
                              and time.perf_counter() - start >= config.runtime.max_train_seconds)
                episode_limit = completed + len(done_indices) >= config.runtime.episodes
                step_limit = bool(env_step_limit and global_step >= env_step_limit)
                final_batch = episode_limit or time_limit or step_limit
                if final_batch:
                    stop_reason = ("env_step_budget" if step_limit else
                                   "episode_budget" if episode_limit else "time_budget")
                if not overlap or final_batch:
                    # The last collected batch has no following CPU step in
                    # which to execute its deferred updates; drain them here.
                    learn(force_metrics=final_batch)

                # Diagnostics read a completed Warp step BEFORE any affected
                # world is reset. Limit instrumentation to log cadence: do not
                # slow every 4096-env/1024-env collection with host transfers.
                if not fast_training and root_audit.action_due(global_step):
                    action_probe.next_step = min(action_probe.next_step,global_step)
                if not fast_training and action_probe.due(global_step) and device_env:
                    action_values,mission_values,health_values = {},{},{}
                    try:
                        phase = ("random_warmup" if global_step - warmup_origin
                                 <= settings.learning_starts and random_warmup else "policy")
                        action_values = action_probe.sample(
                            vector, obs["observation"], action, global_step, phase=phase)
                        if action_values:
                            mission_values = mission_probe.sample(vector, global_step, phase=phase, episode_returns=episode_returns)
                            mem = training_memory_metrics()
                            replay_values = replay.stats()
                            health_values, alerts = health_probe.capture(
                                global_step, action_values, latest, replay_values, mem,
                                mission=mission_values, updates=learner.updates, phase=phase)
                            evidence_integrity = integrity_probe.capture(
                                step=global_step,action=action_values,mission=mission_values,
                                health=health_values,root_records=root_audit.records,
                                minibatch_records=minibatch_probe.records,
                                reason="pre_reset_warp_sample")
                            if tracker:
                                tracker.diagnostic({**action_values, **mission_values, **mem,
                                                    **health_values, **evidence_integrity}, global_step)
                            if alerts:
                                diagnostics.event("training_health_warning", global_step,
                                    {"alerts":alerts,
                                     "interpretation":"Threshold-based symptoms; verify CBF, replay and gradient evidence before changing code/config"})
                    except Exception as error:
                        # A missing log is explicitly marked missing, NOT zero.
                        # Preserve this report even when mission/safety probe fails.
                        if integrity_probe.enabled:
                            integrity_probe.capture(step=global_step,
                                action=action_values,mission=mission_values,
                                health=health_values,root_records=root_audit.records,
                                minibatch_records=minibatch_probe.records,
                                reason="observer_failure",errors=[{
                                    "type":type(error).__name__,
                                    "message":str(error)[:300]}])
                        diagnostics.event("action_safety_logging_failed", global_step,
                            {"error_type":type(error).__name__, "error_message":str(error)[:200]},
                            severity="warning")
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
                    record["episode_horizon_s"] = config.world.max_steps * config.world.dt_s
                    record["episode_dt_s"] = config.world.dt_s
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
                if tracker and batch_records:
                    # One truthful point at this vector step only when actual
                    # worlds finished. The full per-world raw data are in
                    # HF episodes.jsonl. Never fabricate an AUC between ends.
                    tracker.diagnostic(
                        completed_episode_step_metrics(batch_records,global_step),
                        global_step)
                values = diagnostics.batch(batch_records, global_step, final=final_batch,
                    partial=len(batch_records) < len(done_indices))
                if tracker and values:
                    # Use the exact completed-episode count of this batch as
                    # the X coordinate for pooled cohort metrics.
                    tracker.diagnostic({**values, "train/episodes_completed": completed}, global_step)
                evaluator.tick(global_step, completed,
                    lambda: (learner.actors.state_dict(), learner.actor_updates),
                    {"episodes": completed, "global_step": global_step,
                     "collection_steps": collection_steps, "next_seed": next_seed},
                    allow_submit=not time_limit and (not config.runtime.max_train_seconds or
                        time.perf_counter() - start + 10 < config.runtime.max_train_seconds))
                if deferred_bank is not None:
                    deferred_bank.capture(learner.actors.state_dict(), learner.actor_updates,
                        episodes=completed, global_step=global_step,
                        collection_steps=collection_steps, next_seed=next_seed,
                        reason="periodic")
                    if training_candidate_selector is not None:
                        training_candidate = training_candidate_selector.update(
                            batch_records, global_step=global_step, episodes=completed)
                        if training_candidate is not None:
                            deferred_bank.capture(
                                learner.actors.state_dict(), learner.actor_updates,
                                episodes=completed, global_step=global_step,
                                collection_steps=collection_steps, next_seed=next_seed,
                                force=True, reason="training_metric_best_candidate",
                                training_metric_candidate=training_candidate)
                            deferred_bank.write_training_best_alias(training_candidate)
                if live_gpu_logger is not None:
                    live_gpu_logger.poll()  # Only the main Training process writes to W&B.
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
        # Flush the final genuine shorter window when the run ends.
        if reward_window is not None:
            reward_tail = reward_window.close()
            progress_tail = (progress_window.close()
                             if progress_window is not None else None)
            training_tail = (pair_reward_and_progress_events(reward_tail,progress_tail)
                             if progress_tail is not None else reward_tail)
            for row in step_joiner.add_training(training_tail) if step_joiner else training_tail:
                tracker.step_reward(row)
            if loss_window is not None:
                for row in step_joiner.add_loss(loss_window.close()):
                    tracker.step_reward(row)
                for row in step_joiner.close():
                    tracker.step_reward(row)
        checkpoint = output / "final.pt"
        drain = min(10., max(0., config.runtime.max_train_seconds - (time.perf_counter() - start))) if config.runtime.max_train_seconds else 10.
        evaluator.close(global_step, drain_seconds=drain)
        save(checkpoint, final=True, full_state=os.environ.get("UAV_TRAINING_ONLY") != "true")
        deferred_report = None
        if deferred_bank is not None:
            deferred_bank.capture(learner.actors.state_dict(), learner.actor_updates,
                episodes=completed, global_step=global_step,
                collection_steps=collection_steps, next_seed=next_seed,
                force=True, reason="final")
            # A failed HF export never deletes the local Actor archive or final policy.
            archive, manifest = deferred_bank.bundle()
            deferred_report = {"snapshot_count": manifest["count"],
                               "training_metric_best_candidates": (
                                   training_candidate_selector.improvements
                                   if training_candidate_selector is not None else 0),
                               "training_metric_window_episodes": (
                                   training_candidate_selector.window
                                   if training_candidate_selector is not None else None),
                               "target": manifest["target"],
                               "archive": str(archive), "manifest": str(deferred_bank.folder / "manifest.json")}
            try:
                run_id = tracker.identity()["wandb_run_id"] if tracker else ""
                published = publish_bundle(archive, run_id=run_id)
                if published is not None:
                    deferred_report["huggingface"] = published
            except Exception as error:
                deferred_report["export_error_type"] = type(error).__name__
                diagnostics.event("deferred_eval_hf_export_failed", global_step,
                    {"error_type": type(error).__name__}, severity="warning")
        if live_gpu_logger is not None:
            live_gpu_logger.poll(force=True)
        elapsed = time.perf_counter() - start
        result = {"checkpoint": str(checkpoint), "resume_checkpoint": str(output / "resume_latest.pt"),
                  "algorithm": learner.algorithm,
                  "requested_episodes": config.runtime.episodes, "episodes": completed,
                  "initial_episodes": initial["episodes"], "segment_episodes": completed - initial["episodes"],
                  "initial_global_step": initial["global_step"], "new_updates": learner.updates - initial_updates,
                  "episode_budget_complete": completed >= config.runtime.episodes,
                  "stop_reason": stop_reason,
                  **({"requested_env_steps": env_step_limit} if env_step_limit else {}),
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
                  "replay_sample_host_s": replay_sample_s,
                  "optimizer_update_host_s": optimizer_update_s,
                  "elapsed_s": elapsed, "joint_transitions_per_s": (global_step - initial["global_step"]) / elapsed,
                  "final_episode": history[-1] if history else None}
        if "policy_warm_start" in initial:
            result["policy_warm_start"] = initial["policy_warm_start"]
        result.update(evaluator.best.result(checkpoint))
        result["periodic_evaluation"] = dict(evaluator.counts)
        if deferred_report is not None:
            result["deferred_gpu_evaluation"] = deferred_report
        if live_gpu_logger is not None:
            result["online_gpu_evaluation"] = {
                **live_gpu_logger.summary(),
                "checkpoint_snapshots": len(deferred_bank.rows) if deferred_bank else 0,
                "pending_at_training_end": max(0, len(deferred_bank.rows) - live_gpu_logger.reported)
                if deferred_bank else 0,
            }
        result["storage"] = disk_metrics(output)
        result["replay"] = replay.stats()
        # Persist CUDA allocator high-water marks and cgroup peaks even if
        # periodic online system samples missed short backward/checkpoint spikes.
        result["resource_memory"] = training_memory_metrics()
        result["diagnostic_files"] = {
            "action_cbf_examples": str(action_probe.path),
            "mission_cause_snapshots": str(mission_probe.path),
            "training_health_snapshots": str(health_probe.path),
            "episode_outcomes": str(output / "episodes.jsonl"),
            "warnings": str(diagnostics.path),
        }
        result["last_action_cbf_metrics"] = action_probe.last_values
        result["last_mission_cause_metrics"] = mission_probe.last_values
        result["forensic_trace"] = forensic.summary()
        result["gradient_layer_journal"] = layer_journal.summary()
        result["root_cause_audit"] = {"enabled":root_audit.enabled,
            "records":root_audit.records,"path":str(root_audit.path)}
        result["last_training_health_flags"] = health_probe.latest
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
                "nonfinite_parameter_names": bad_parameters,
                "resource_memory": training_memory_metrics(),
                "last_action_cbf_metrics": action_probe.last_values,
                "last_health_flags": health_probe.latest})
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
        # Preserve trailing 1-3 real vector-step samples on interruption.
        # This precedes tracker.close(), which finalizes the HF archive.
        if reward_window is not None:
            try:
                reward_tail = reward_window.close()
                progress_tail = (progress_window.close()
                                 if progress_window is not None else None)
                training_tail = (pair_reward_and_progress_events(reward_tail,progress_tail)
                                 if progress_tail is not None else reward_tail)
                for row in step_joiner.add_training(training_tail) if step_joiner else training_tail:
                    if tracker is not None:
                        tracker.step_reward(row)
                if loss_window is not None:
                    for row in step_joiner.add_loss(loss_window.close()):
                        if tracker is not None:
                            tracker.step_reward(row)
                    for row in step_joiner.close():
                        if tracker is not None:
                            tracker.step_reward(row)
            except Exception as step_journal_error:
                print_evidence({"step_observability_journal_close_failed":
                                type(step_journal_error).__name__})
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
            # Offline/Kaggle UI smoke tests also need compressed, reconstructable
            # evidence. Online runs already publish immutable .tar.gz via
            # OnlineTracker and can verify the local parts without duplicating.
            try:
                archival = (finalize_local_diagnostics(output) if tracker is None
                            else {"reconstruction_verified":
                                  verify_archive_parts(output)["full_reconstruction_verified"]})
                if result is not None:
                    result["forensic_archive"] = archival
                    summary.write_text(json.dumps(result, indent=2, allow_nan=False))
            except Exception as archival_error:
                # Never claim evidence is guaranteed if compression or
                # reconstruction failed; preserve the original local JSONL.
                if result is not None:
                    result["forensic_archive"] = {"reconstruction_verified":False,
                        "error_type":type(archival_error).__name__,
                        "error_message":str(archival_error)[:240]}
                    summary.write_text(json.dumps(result,indent=2,allow_nan=False))
            watchdog.close()
