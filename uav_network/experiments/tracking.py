"""Online W&B in the learner process; mission metrics and final policy only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from .hardware import hardware_info


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
    os.environ["WANDB_API_KEY"] = str(key)
    return entity, project


def default_workspace_url(entity, project):
    return f"https://wandb.ai/{entity}/{project}/workspace"


class OnlineTracker:
    def __init__(self, config, output, initial_progress=None):
        import wandb
        entity, project = _authentication(config)
        self.config, self.output, self.wandb = config, Path(output), wandb
        self.started = time.perf_counter()
        initial = initial_progress or {}
        self.initial_step = initial.get("global_step", 0)
        self.next_progress = 0
        self.records = list(initial.get("records", []))
        self.closed = False
        run_id = os.environ.get("WANDB_RUN_ID") or uuid4().hex[:8]
        name = config.tracking.run_name
        if name == "auto":
            name = os.environ.get("WANDB_NAME") or f"{config.runtime.algorithm.upper()}-uav-network-2d-seed{config.runtime.seed}-{run_id}"
        metadata = {"source_git_commit": os.environ.get("UAV_GIT_COMMIT", "local"),
                    "kaggle_account": os.environ.get("UAV_KAGGLE_ACCOUNT", "local"),
                    "kaggle_kernel": os.environ.get("UAV_KAGGLE_KERNEL", "local"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                    "hardware": hardware_info(),
                    "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "kaggle_account_alias": os.environ.get("UAV_KAGGLE_ACCOUNT_ALIAS", "local"),
                    "job_submitted_at_utc": os.environ.get("UAV_JOB_SUBMITTED_AT_UTC", "unknown"),
                    "job_started_at_utc": os.environ.get("UAV_JOB_STARTED_AT_UTC", "unknown"),
                    "kaggle_url": "https://www.kaggle.com/code/" + os.environ.get("UAV_KAGGLE_KERNEL", "local")}
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
        self.run = wandb.init(entity=entity, project=project, id=run_id, name=name,
                              group=config.tracking.group, tags=list(config.tracking.tags),
                              config={**config.to_dict(), **metadata}, dir=str(output), mode="online",
                              resume="never", reinit="finish_previous", save_code=False,
                              settings=wandb.Settings(x_disable_stats=not config.tracking.system_metrics,
                                                     disable_code=True, console="wrap"))
        try:
            if self.run.offline:
                raise RuntimeError("W&B did not start online")
            if os.environ.get("UAV_INPUT_ARTIFACT"):
                self.run.use_artifact(os.environ["UAV_INPUT_ARTIFACT"], type="model")
            self.run.save(str(self.output / "diagnostics.jsonl"), base_path=str(self.output), policy="live")
            self.run.save(str(self.output / "resolved_config.yaml"), base_path=str(self.output), policy="now")
            self.run.summary.update({"run_started_at_utc": metadata["run_started_at_utc"],
                                     "kaggle_account": metadata["kaggle_account"],
                                     "kaggle_url": metadata["kaggle_url"],
                                     "workspace_url": default_workspace_url(entity, project)})
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step", step_sync=True)
            (self.output / "wandb_run.json").write_text(json.dumps(self.identity(), indent=2))
            print(f"W&B Personal Workspace: {self.identity()['wandb_workspace_url']}", flush=True)
        except BaseException:
            self.close(failed=True)
            raise

    def identity(self):
        return {"wandb_run_id": self.run.id, "wandb_url": self.run.url,
                "wandb_entity": self.run.entity, "wandb_project": self.run.project,
                "wandb_workspace_url": default_workspace_url(self.run.entity, self.run.project)}

    def progress(self, observations, step, completed, learner, replay_size, force=False):
        if not force and step < self.next_progress:
            return
        values = {"global_step": step, "train/episodes_completed": completed,
                  "train/updates": learner.updates, "train/replay_size": replay_size,
                  "train/joint_transitions_per_s": (step - self.initial_step) / max(time.perf_counter() - self.started, 1e-9),
                  "run/elapsed_seconds": time.perf_counter() - self.started,
                  "run/elapsed_hours": (time.perf_counter() - self.started) / 3600,
                  **progress_metrics(observations, self.config)}
        self.run.log(values)
        self.next_progress = step + self.config.runtime.log_interval_steps

    def learner(self, metrics, step):
        self.run.log({"global_step": step, **{f"learner/{k}": v for k, v in metrics.items()}})

    def diagnostic(self, values, step):
        self.run.log({"global_step": step, **values})

    def failure(self, path, step):
        self.run.summary.update({"diagnostics/fatal_error": True})
        self.run.log({"global_step": step, "diagnostics/fatal_error_count": 1})
        self.run.save(str(path), base_path=str(self.output), policy="now")
        for name in ("failed_batch.pt", "failure_state.pt"):
            file = self.output / name
            if file.is_file():
                self.run.save(str(file), base_path=str(self.output), policy="now")

    def episode(self, record):
        self.records.append(record)
        self.run.log(episode_metrics(record, self.config.world.num_uavs))

    def checkpoint(self, path, progress, final=False):
        if not self.config.tracking.log_checkpoint:
            return None
        self.run.save(str(self.output / "episodes.jsonl"), base_path=str(self.output), policy="now")
        artifact = self.wandb.Artifact(f"uav-network-{self.config.runtime.algorithm}-{self.run.id}", type="model",
                                      metadata={"completed_episodes": progress["episodes"],
                                                "global_step": progress["global_step"],
                                                "episode_budget_complete": progress.get("episode_budget_complete", False)})
        artifact.add_file(str(path), name="checkpoint.pt")
        if final and self.config.pql.enabled:
            resume = progress.get("resume_checkpoint")
            if resume is None or not Path(resume).is_file():
                raise ValueError("Final PQL artifact requires a complete resume checkpoint")
            artifact.add_file(str(resume), name="resume.pt")
        logged = self.run.log_artifact(artifact, aliases=["latest", "final"] if final else ["latest"])
        logged.wait(timeout=self.config.tracking.artifact_timeout_seconds)
        return logged.qualified_name

    def complete(self, result):
        metrics = final_metrics(self.records) if self.records else {}
        self.run.summary.update({**metrics, "global_step": result["global_step"],
                                 "episodes_completed": result["episodes"], "learner_updates": result["updates"],
                                 "episode_budget_complete": result.get("episode_budget_complete", True),
                                 "stop_reason": result.get("stop_reason", "episode_budget")})
        if "reproducibility" in result:
            self.run.summary["reproducibility"] = result["reproducibility"]
        artifact = self.checkpoint(result["checkpoint"], result, final=True)
        return {**metrics, **self.identity(), "checkpoint_artifact": artifact}

    def close(self, failed=False):
        if not self.closed:
            self.closed = True
            self.run.summary.update({"run_ended_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "run_elapsed_seconds": time.perf_counter() - self.started})
            self.run.finish(exit_code=1 if failed else 0)


def start_tracking(config, output, initial_progress=None):
    return OnlineTracker(config, output, initial_progress) if config.tracking.enabled else None
