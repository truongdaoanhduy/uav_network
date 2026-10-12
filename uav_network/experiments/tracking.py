"""Online W&B in the learner process; mission metrics and final policy only."""
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


class OnlineTracker:
    def __init__(self, config, output):
        import wandb
        entity, project = _authentication(config)
        self.config, self.output, self.wandb = config, Path(output), wandb
        self.started = time.perf_counter()
        self.next_progress = 0
        self.records = []
        self.closed = False
        run_id = os.environ.get("WANDB_RUN_ID") or uuid4().hex[:8]
        name = config.tracking.run_name
        if name == "auto":
            name = os.environ.get("WANDB_NAME") or f"{config.runtime.algorithm.upper()}-uav-network-2d-seed{config.runtime.seed}-{run_id}"
        metadata = {"source_git_commit": os.environ.get("UAV_GIT_COMMIT", "local"),
                    "kaggle_account": os.environ.get("UAV_KAGGLE_ACCOUNT", "local"),
                    "kaggle_kernel": os.environ.get("UAV_KAGGLE_KERNEL", "local"),
                    "hardware": hardware_info()}
        self.run = wandb.init(entity=entity, project=project, id=run_id, name=name,
                              group=config.tracking.group, tags=list(config.tracking.tags),
                              config={**config.to_dict(), **metadata}, dir=str(output), mode="online",
                              resume="never", reinit="finish_previous", save_code=False,
                              settings=wandb.Settings(x_disable_stats=not config.tracking.system_metrics,
                                                     disable_code=True, console="off"))
        try:
            if self.run.offline:
                raise RuntimeError("W&B did not start online")
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step", step_sync=True)
            (self.output / "wandb_run.json").write_text(json.dumps(self.identity(), indent=2))
        except BaseException:
            self.close(failed=True)
            raise

    def identity(self):
        return {"wandb_run_id": self.run.id, "wandb_url": self.run.url,
                "wandb_entity": self.run.entity, "wandb_project": self.run.project}

    def progress(self, observations, step, completed, learner, replay_size, force=False):
        if not force and step < self.next_progress:
            return
        values = {"global_step": step, "train/episodes_completed": completed,
                  "train/updates": learner.updates, "train/replay_size": replay_size,
                  "train/joint_transitions_per_s": step / max(time.perf_counter() - self.started, 1e-9),
                  **progress_metrics(observations, self.config)}
        self.run.log(values)
        self.next_progress = step + self.config.runtime.log_interval_steps

    def episode(self, record):
        self.records.append(record)
        self.run.log(episode_metrics(record, self.config.world.num_uavs))

    def complete(self, result):
        metrics = final_metrics(self.records)
        self.run.summary.update({**metrics, "global_step": result["global_step"],
                                 "episodes_completed": result["episodes"], "learner_updates": result["updates"]})
        if self.config.tracking.log_checkpoint:
            artifact = self.wandb.Artifact(f"uav-network-{self.config.runtime.algorithm}-{self.run.id}", type="model")
            artifact.add_file(result["checkpoint"], name="final.pt")
            self.run.log_artifact(artifact)
        return {**metrics, **self.identity()}

    def close(self, failed=False):
        if not self.closed:
            self.closed = True
            self.run.finish(exit_code=1 if failed else 0)


def start_tracking(config, output):
    return OnlineTracker(config, output) if config.tracking.enabled else None
