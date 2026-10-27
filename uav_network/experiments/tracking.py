"""Online W&B mission metrics and policy/full-state checkpoint artifacts."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from .hardware import hardware_info
from .telemetry import episode_telemetry
from .run_journal import RunJournal, EVIDENCE_METRICS
from ..environment.observations import OBSERVATION_VERSION

CPU_EPISODE_METRICS = ("evaluation/return", "evaluation/search_rate_percent",
                       "evaluation/delivery_rate_percent", "evaluation/success",
                       "evaluation/episodes_completed")


def progress_metrics(observations, config):
    local, state = observations["observation"], observations["state"]
    n, t, k = config.world.num_uavs, config.world.num_targets, config.world.num_obstacles
    delivered_start = 4 * n + 2 * t + 3 * k + 2 * n * t
    if hasattr(local,"detach"):
        import torch
        found = (local[:,:,10:10+t] > .5).any(1).double().mean()*100
        delivered = (state[:,delivered_start:delivered_start+t] > .5).double().mean()*100
        reduced = torch.stack([found,delivered,local[:,:,8].mean(),local[:,:,9].sum(1).mean()])
        values = reduced.detach().cpu().tolist()
        return dict(zip(("progress/search_rate_percent","progress/delivery_rate_percent",
                         "progress/mean_battery_fraction","progress/mean_active_uavs"),values))
    known = local[:, :, 10:10 + t] > .5
    delivered = state[:, delivered_start:delivered_start + t] > .5
    return {"progress/search_rate_percent": float(known.any(1).mean() * 100),
            "progress/delivery_rate_percent": float(delivered.mean() * 100),
            "progress/mean_battery_fraction": float(local[:, :, 8].mean()),
            "progress/mean_active_uavs": float(local[:, :, 9].sum(1).mean())}


def episode_metrics(record, num_uavs):
    energy, network = record["energy"], record["network"]
    opportunities = max(1, record["episode_length"] * num_uavs)
    result = {"global_step": record["global_step"], "episode/index": record["episode"],
              "episode/return": record["return"], "episode/length": record["episode_length"],
              "outcome/search_rate_percent": record["search_rate_percent"],
              "outcome/delivery_rate_percent": record["delivery_rate_percent"],
              "outcome/success": int(record["success"]),
              "safety/blocked_motion_rate": record["blocked_moves"] / opportunities,
              "safety/boundary_clip_rate": record["boundary_clips"] / opportunities,
              "reports/expired": record["expired_reports"], "reports/dropped": record["dropped_reports"],
              "reports/regenerated": record.get("regenerated_reports", 0),
              "energy/propulsion_energy_j": energy["propulsion_energy_j"],
              "energy/uav_rf_energy_j": network.get("uav_communication_energy_j", 0),
              "energy/gcs_rf_energy_j": network.get("gcs_communication_energy_j", 0),
              "energy/uav_tx_radio_energy_j": network.get("uav_tx_energy_j", 0),
              "energy/uav_rx_radio_energy_j": network.get("uav_rx_energy_j", 0),
              "energy/gcs_tx_radio_energy_j": network.get("gcs_tx_energy_j", 0),
              "energy/gcs_rx_radio_energy_j": network.get("gcs_rx_energy_j", 0),
              "energy/mean_battery_fraction": float(np.mean(energy["battery_fraction"])),
              "energy/exhausted_uavs": num_uavs - energy["active_uavs"]}
    result["energy/total_uav_energy_j"] = result["energy/propulsion_energy_j"] + result["energy/uav_rf_energy_j"]
    mapping = {"packet_pdr_percent": "pdr_percent", "packet_e2e_delay_ms": "e2e_delay_ms",
               "payload_throughput_kbps": "payload_throughput_kbps", "phy_success_percent": "phy_success_percent",
               "collisions": "collisions", "report_e2e_delay_s": "report_e2e_delay_s",
               "report_average_hops": "report_average_hops", "energy_blocked_transmissions": "energy_blocked"}
    result.update({f"network/{target}": network[source] for source, target in mapping.items() if source in network})
    result.update({f"learner/{key}": value for key, value in record.get("learner", {}).items()})
    motion = record.get("motion_counts", {})
    result.update({f"motion/{k}": v for k, v in motion.items()})
    result.update({f"safety/{name}_rate": motion.get(key, 0) / opportunities for name, key in (
        ("shield_intervention", "shield_interventions"),
        ("shield_infeasible", "shield_infeasible"), ("commanded_stall", "commanded_stalls"))})
    result.update({f"routing/{k}": v for k, v in network.get("selection_counts", {}).items()})
    result.update({f"reward/{k}": v for k, v in record.get("reward_components", {}).items()})
    counts = record.get("action_counts", {})
    result.update({f"action/{k}_fraction": counts[k] / opportunities for k in ("silent", "peer", "gcs") if k in counts})
    if "saturated" in counts:
        result["action/movement_saturation_fraction"] = counts["saturated"] / (2 * opportunities)
    result["reports/buffer_bytes"] = record.get("buffer_bytes", 0)
    result["reports/buffer_count"] = record.get("buffer_reports", 0)
    result["motion/final_mean_speed_mps"] = record.get("mean_speed_mps", 0)
    result["episode/termination_reason"] = record.get("termination_reason") or "unknown"
    result.update(episode_telemetry(record))
    return result


def final_metrics(records):
    if not records:
        raise ValueError("Final episode aggregates need at least one completed episode")
    return {"final_mean_return": float(np.mean([r["return"] for r in records])),
            "final_search_rate_percent": float(np.mean([r["search_rate_percent"] for r in records])),
            "final_delivery_rate_percent": float(np.mean([r["delivery_rate_percent"] for r in records])),
            "final_success_rate": float(np.mean([r["success"] for r in records]))}


def _authentication(config):
    path = Path(config.tracking.credentials_file).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    data = json.loads(path.read_text()).get("wandb", {}) if path.is_file() else {}
    key = data.get("api_key") or os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("Online W&B requires key.txt or WANDB_API_KEY; no offline fallback")
    entity = config.tracking.entity
    project = config.tracking.project
    if entity == "auto":
        entity = data.get("entity") or os.environ.get("WANDB_ENTITY")
    if project == "auto":
        project = data.get("project") or os.environ.get("WANDB_PROJECT")
    if not entity or not project:
        raise RuntimeError("W&B entity/project must be configured in key.txt, env or config")
    expected = (data.get("entity") or os.environ.get("WANDB_ENTITY"),
                data.get("project") or os.environ.get("WANDB_PROJECT"))
    if all(expected) and (entity, project) != expected:
        raise ValueError("W&B destination differs from the main project in credentials/environment")
    os.environ["WANDB_API_KEY"] = str(key)
    return entity, project


def default_workspace_url(entity, project):
    return f"https://wandb.ai/{entity}/{project}/workspace"


class OnlineTracker:
    def __init__(self, config, output, initial_progress=None):
        import wandb
        entity, project = _authentication(config)
        self.config, self.output, self.wandb = config, Path(output), wandb
        if config.tracking.enabled:
            from .dashboard import ensure_project_workspace
            ensure_project_workspace(config)
        self.started = time.perf_counter()
        initial = initial_progress or {}
        self.initial_step = initial.get("global_step", 0)
        self.next_progress = 0
        self.records = list(initial.get("records", []))
        self.closed = False
        self.last_checkpoint_upload = None
        self.last_recovery_confirmed_at = None
        self.latest_evidence = {}
        self.journal = None
        self.learning_phase = "initializing"
        run_id = os.environ.get("WANDB_RUN_ID") or uuid4().hex[:8]
        name = config.tracking.run_name
        if name == "auto":
            from ..cloud.identities import run_name
            name = os.environ.get("WANDB_NAME") or run_name(
                config, config.runtime.algorithm, "PQL" if config.pql.enabled else "BASE",
                os.environ.get("UAV_STAGE", "train"), run_id)
        metadata = {"observation_version": initial.get("observation_version", OBSERVATION_VERSION), "source_git_commit": os.environ.get("UAV_GIT_COMMIT", "local"),
                    "kaggle_account": os.environ.get("UAV_KAGGLE_ACCOUNT", "local"),
                    "kaggle_kernel": os.environ.get("UAV_KAGGLE_KERNEL", "local"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                    "hardware": hardware_info(),
                    "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "kaggle_account_alias": os.environ.get("UAV_KAGGLE_ACCOUNT_ALIAS", "local"),
                    "job_submitted_at_utc": os.environ.get("UAV_JOB_SUBMITTED_AT_UTC", "unknown"),
                    "job_started_at_utc": os.environ.get("UAV_JOB_STARTED_AT_UTC", "unknown"),
                    "kaggle_url": "https://www.kaggle.com/code/" + os.environ.get("UAV_KAGGLE_KERNEL", "local")}
        metadata["stage"] = os.environ.get("UAV_STAGE", "train")
        metadata["training_mode"] = os.environ.get("UAV_TRAINING_MODE", "PQL" if config.pql.enabled else "BASE")
        metadata["pipeline"] = {
            "auto_continue": os.environ.get("UAV_AUTO_CONTINUE") == "true",
            "root_kernel_ref": os.environ.get("UAV_ROOT_KAGGLE_KERNEL") or metadata["kaggle_kernel"],
            "continuation_index": int(os.environ.get("UAV_CONTINUATION_INDEX") or 0),
            "target_episodes": int(os.environ.get("UAV_EPISODE_BUDGET") or config.runtime.episodes),
        }
        metadata["continuation"] = {
            "kind": initial.get("state_restore", "policy_warm_start" if config.runtime.warm_start_checkpoint != "none" else "fresh"),
            "initial_episodes": initial.get("episodes", 0), "initial_global_step": self.initial_step,
            "source_run_id": initial.get("source_run_id"), "optimizer_replay_restored": initial.get("optimizer_replay_restored", False),
            "state_restore": initial.get("state_restore", "policy_warm_start" if config.runtime.warm_start_checkpoint != "none" else "fresh"),
            "input_artifact": os.environ.get("UAV_INPUT_ARTIFACT"),
        }
        benchmark_file = os.environ.get("UAV_BENCHMARK_FILE")
        if benchmark_file:
            metadata["benchmark"] = json.loads(Path(benchmark_file).read_text())["results"]
        tracked_config = config.to_dict()
        if "generation_version" in initial:
            tracked_config["world"]["generation_version"] = initial["generation_version"]
        self.run = wandb.init(entity=entity, project=project, id=run_id, name=name,
                              group=config.tracking.group, tags=list(config.tracking.tags),
                              config={**tracked_config, **metadata}, dir=str(output), mode="online",
                              resume="never", reinit="finish_previous", save_code=False,
                              settings=wandb.Settings(x_disable_stats=not config.tracking.system_metrics,
                                                     disable_code=True, console="wrap"))
        try:
            if self.run.offline:
                raise RuntimeError("W&B did not start online")
            if (self.run.entity, self.run.project) != (entity, project):
                raise RuntimeError("W&B run destination differs from the configured main project")
            self.journal = RunJournal(self.output, {
                "run_id": self.run.id, "algorithm": config.runtime.algorithm,
                "mode": metadata["training_mode"], "seed": config.runtime.seed,
                "source_commit": metadata["source_git_commit"], "stage": metadata["stage"],
                "account": metadata["kaggle_account"], "num_envs": config.runtime.num_envs,
                "batch_size": config.algorithm_config.batch_size,
                "replay_cpu_fraction": config.runtime.replay_cpu_fraction,
                "root_kernel_ref": metadata["pipeline"]["root_kernel_ref"]})
            periodic_eval = metadata["stage"] in {"train", "validate"} and bool(
                config.tracking.evaluation_interval or config.tracking.evaluation_interval_steps)
            self.run.summary["evaluation_status/periodic_enabled"] = periodic_eval
            self.journal.record("started", {"global_step": self.initial_step,
                "completed_episodes": initial.get("episodes", 0) if metadata["stage"] in {"train", "validate"} else None,
                "phase": "initializing", "periodic_evaluation_enabled": periodic_eval})
            self._publish_observability()
            if os.environ.get("UAV_INPUT_ARTIFACT"):
                self.run.use_artifact(os.environ["UAV_INPUT_ARTIFACT"], type="model")
            self.run.save(str(self.output / "cohorts.jsonl"), base_path=str(self.output), policy="live")
            self.run.save(str(self.output / "resolved_config.yaml"), base_path=str(self.output), policy="now")
            self.run.summary.update({"run_started_at_utc": metadata["run_started_at_utc"],
                                     "training_mode": metadata["training_mode"],
                                     "root_kernel_ref": metadata["pipeline"]["root_kernel_ref"],
                                     "continuation_index": metadata["pipeline"]["continuation_index"],
                                     "kaggle_account": metadata["kaggle_account"],
                                     "kaggle_url": metadata["kaggle_url"],
                                     "workspace_url": default_workspace_url(entity, project)})
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step", step_sync=True)
            self.run.define_metric("evaluation/checkpoint_global_step")
            for prefix in ("evaluation/*", "eval/*"):
                self.run.define_metric(prefix, step_metric="evaluation/checkpoint_global_step", step_sync=True)
            for metric in CPU_EPISODE_METRICS:
                self.run.define_metric(metric, step_metric="evaluation/episodes_completed", step_sync=True)
            (self.output / "wandb_run.json").write_text(json.dumps(self.identity(), indent=2))
            print(f"W&B Personal Workspace: {self.identity()['wandb_workspace_url']}", flush=True)
        except BaseException:
            self.close(failed=True)
            raise

    def _publish_observability(self, final=False):
        policies = {"training-status.json": "live", "metrics-guide.md": "now",
                    "training-events.jsonl": "end", "diagnostics.jsonl": "end"}
        for name, policy in policies.items():
            path = self.output / name
            try:
                if path.is_file():
                    self.run.save(str(path), base_path=str(self.output), policy="now" if final else policy)
            except Exception as error:
                if self.journal is not None:
                    self.journal._warn(error)

    def identity(self):
        return {"wandb_run_id": self.run.id, "wandb_url": self.run.url,
                "wandb_entity": self.run.entity, "wandb_project": self.run.project,
                "wandb_workspace_url": default_workspace_url(self.run.entity, self.run.project)}

    def progress(self, observations, step, completed, learner, replay_size, episode_lengths=None, force=False):
        if not force and step < self.next_progress:
            return
        episode_progress = {}
        if episode_lengths is not None:
            lengths = (episode_lengths.detach().cpu().numpy() if hasattr(episode_lengths, "detach")
                       else np.asarray(episode_lengths))
            if lengths.size:
                mean_step = float(lengths.mean())
                episode_progress = {
                    "progress/episode_step_min": int(lengths.min()),
                    "progress/episode_step_mean": mean_step,
                    "progress/episode_step_max": int(lengths.max()),
                    "progress/episode_progress_percent_mean": mean_step / self.config.world.max_steps * 100,
                }
        values = {"global_step": step, "train/episodes_completed": completed,
                  "train/updates": learner.updates, "train/replay_size": replay_size,
                  "train/joint_transitions_per_s": (step - self.initial_step) / max(time.perf_counter() - self.started, 1e-9),
                  "run/elapsed_seconds": time.perf_counter() - self.started,
                  "run/elapsed_hours": (time.perf_counter() - self.started) / 3600,
                  **progress_metrics(observations, self.config), **episode_progress}
        values.update({"recovery/full_checkpoint_available": int(self.last_recovery_confirmed_at is not None),
                       "recovery/seconds_since_confirmed_upload": (time.perf_counter() - self.last_recovery_confirmed_at
                           if self.last_recovery_confirmed_at is not None else None)})
        if self.journal.identity.get("stage") in {"train", "validate"}:
            values["run/episode_budget_percent"] = completed / self.config.runtime.episodes * 100
        self.learning_phase = "learning" if learner.updates else "collecting_before_first_update"
        self.journal.record("progress", {**self.latest_evidence, **values,
            "completed_episodes": completed, "updates": learner.updates, "phase":self.learning_phase})
        self.run.log(values)
        self.next_progress = step + self.config.runtime.log_interval_steps

    def learner(self, metrics, step):
        values = {f"learner/{k}": v for k, v in metrics.items()}
        self.latest_evidence.update({k:v for k,v in values.items() if k in EVIDENCE_METRICS})
        self.run.log({"global_step": step, **values})

    def diagnostic(self, values, step):
        self.latest_evidence.update({k:v for k,v in values.items() if k in EVIDENCE_METRICS})
        event = None
        if "recovery/status" in values:
            event = "recovery"
            status = values["recovery/status"]
            phase = {"snapshot_started":"checkpoint_snapshot", "upload_started":"checkpoint_upload"}.get(status)
            self.latest_evidence["phase"] = phase or self.learning_phase
        elif any(k in values for k in ("eval/mean_return", "eval/mean_delivery_rate_percent", "eval/success_rate")):
            event = "evaluation_result"
        elif "evaluation/episodes_completed" in values or "evaluation_status/status" in values:
            event = "evaluation"
        if event and self.journal is not None:
            context = {**self.latest_evidence, "global_step":step}
            if self.journal.identity.get("stage") == "visualize":
                context["phase"] = "cpu_evaluation_completed" if event == "evaluation_result" else "cpu_evaluation"
            self.journal.record(event, context)
        self.run.log({"global_step": step, **values})

    def failure(self, path, step):
        error_type = "unknown"
        try:
            candidate = json.loads(Path(path).read_text()).get("error_type")
            if isinstance(candidate, str) and candidate.isidentifier():
                error_type = candidate
        except (OSError, ValueError, TypeError):
            pass
        if self.journal is not None:
            self.journal.record("failure", {**self.latest_evidence,"global_step":step,"phase":"failed", "error_type":error_type})
            self._publish_observability(final=True)
        self.run.summary.update({"diagnostics/fatal_error": True,"diagnostics/failure_type":error_type})
        self.run.log({"global_step": step, "diagnostics/fatal_error_count": 1})
        self.run.save(str(path), base_path=str(self.output), policy="now")
        for name in ("failed_batch.pt", "failure_state.pt"):
            file = self.output / name
            if file.is_file():
                self.run.save(str(file), base_path=str(self.output), policy="now")

    def episode(self, record):
        self.records.append(record)
        self.run.log(episode_metrics(record, self.config.world.num_uavs))

    def checkpoint(self, path, progress, final=False, resumable=False):
        if not self.config.tracking.log_checkpoint:
            return None
        started = time.perf_counter()
        full = bool(resumable or (final and (self.config.pql.enabled or progress.get("resume_checkpoint") is not None)))
        self.journal.record("checkpoint_requested", {"global_step":progress["global_step"],
            "checkpoint_kind":"full" if full else "policy", "phase":"checkpoint_upload"})
        try:
            reference = self._checkpoint(path, progress, final=final, resumable=resumable)
        except Exception as error:
            self.journal.record("checkpoint_failed", {"error_type":type(error).__name__,
                "checkpoint_elapsed_seconds":time.perf_counter()-started,"phase":self.learning_phase})
            raise
        if reference and full:
            self.last_recovery_confirmed_at = time.perf_counter()
            self.journal.record("full_checkpoint_confirmed", {"full_checkpoint_artifact":reference,
                "full_checkpoint_global_step":progress["global_step"]})
        self.journal.record("checkpoint_confirmed" if reference else "checkpoint_skipped", {
            "checkpoint_artifact":reference, "checkpoint_elapsed_seconds":time.perf_counter()-started,
            "phase":self.learning_phase})
        return reference

    def _checkpoint(self, path, progress, final=False, resumable=False):
        if not self.config.tracking.log_checkpoint:
            return None
        if (not final and not resumable and self.config.tracking.checkpoint_store == "huggingface"
                and self.last_checkpoint_upload is not None
                and time.monotonic() - self.last_checkpoint_upload
                    < self.config.tracking.checkpoint_upload_interval_seconds):
            self.run.summary["checkpoint_uploads_skipped"] = self.run.summary.get("checkpoint_uploads_skipped", 0) + 1
            return None
        full_state = resumable or (final and (self.config.pql.enabled or progress.get("resume_checkpoint") is not None))
        artifact = self.wandb.Artifact(f"uav-network-{self.config.runtime.algorithm}-{self.run.id}", type="model",
                                      metadata={"completed_episodes": progress["episodes"],
                                                "global_step": progress["global_step"],
                                                "episode_budget_complete": progress.get("episode_budget_complete", False),
                                                "resumable": bool(full_state),
                                                "environment_state_saved": bool(full_state and progress.get("environment_state_saved") is True),
                                                "source_git_commit": os.environ.get("UAV_GIT_COMMIT", "local"),
                                                "source_run_id": self.run.id})
        files = {"checkpoint.pt": Path(path)}
        if full_state:
            resume = progress.get("resume_checkpoint")
            if resume is None or not Path(resume).is_file():
                raise ValueError("Resumable artifact requires a complete resume checkpoint")
            files["resume.pt"] = Path(resume)
        if self.config.tracking.checkpoint_store == "huggingface":
            from .checkpoint_store import upload_checkpoint_files, MANIFEST
            descriptor = upload_checkpoint_files(self.config, files,
                f"uav-network/{self.run.id}/step-{progress['global_step']}-episodes-{progress['episodes']}")
            manifest = self.output / MANIFEST
            manifest.write_text(json.dumps(descriptor, indent=2))
            artifact.add_file(str(manifest), name=MANIFEST)
            self.run.summary["checkpoint_storage"] = {"backend": "huggingface",
                "repo_id": descriptor["repo_id"], "revision": descriptor["revision"],
                "bytes": sum(item["size"] for item in descriptor["files"].values())}
        else:
            for name, file in files.items():
                artifact.add_file(str(file), name=name)
        aliases = ["latest", "final"] if final else ["latest"]
        logged = self.run.log_artifact(artifact, aliases=aliases)
        logged.wait(timeout=self.config.tracking.artifact_timeout_seconds)
        if "resume.pt" in files:
            # A timed-out initial publication must not move the recovery alias.
            logged.aliases = list(dict.fromkeys([*logged.aliases, "resumable"]))
            logged.save()
        self.last_checkpoint_upload = time.monotonic()
        reference = logged.qualified_name
        if "resume.pt" in files:
            self.run.summary.update({"recovery/checkpoint_artifact": reference,
                                     "recovery/last_verified_global_step": progress["global_step"],
                                     "recovery/last_verified_episodes": progress["episodes"],
                                     "recovery/last_verified_utc": datetime.now(timezone.utc).isoformat()})
        return reference

    def complete(self, result):
        metrics = final_metrics(self.records) if self.records else {}
        completion = result.get("episode_budget_complete")
        session_status = ("episode_budget_complete" if completion is True else
                          "session_time_budget_reached" if completion is False else
                          "session_completion_unknown")
        self.run.summary.update({**metrics, "global_step": result["global_step"],
                                 "episodes_completed": result["episodes"], "learner_updates": result["updates"],
                                 "episode_budget_complete": completion is True,
                                 "stop_reason": result.get("stop_reason", "unknown"),
                                 "session_status": session_status,
                                 "initial_global_step": result.get("initial_global_step", self.initial_step),
                                 "environment_state_saved": result.get("environment_state_saved", False)})
        if "reproducibility" in result:
            self.run.summary["reproducibility"] = result["reproducibility"]
        artifact = self.checkpoint(result["checkpoint"], result, final=True)
        self.journal.record("training_segment_complete", {**self.latest_evidence,"global_step":result["global_step"],
            "completed_episodes":result["episodes"], "episode_budget_complete":completion is True,
            "stop_reason":result.get("stop_reason","unknown")})
        return {**metrics, **self.identity(), "checkpoint_artifact": artifact}

    def close(self, failed=False):
        if not self.closed:
            self.closed = True
            if self.journal is not None:
                self.journal.record("closed", {**self.latest_evidence,"phase":"failed" if failed else "closed",
                    "elapsed_seconds":time.perf_counter()-self.started})
                self._publish_observability(final=True)
            self.run.summary.update({"run_ended_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "run_elapsed_seconds": time.perf_counter() - self.started})
            self.run.finish(exit_code=1 if failed else 0)


def start_tracking(config, output, initial_progress=None):
    return OnlineTracker(config, output, initial_progress) if config.tracking.enabled else None
