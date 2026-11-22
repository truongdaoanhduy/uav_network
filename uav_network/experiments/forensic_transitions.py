"""Bounded read-only causal transition traces for selected Warp worlds.

A trace is an exact observation at one collection boundary, NOT proof that the
agent's neural network had a particular 'reason'. Neural activations/logits,
policy versions, stochastic execution, safety results and reward deltas are
recorded separately. Recording every world at every timestep is intentionally
not the default (potentially TB of data). Raw JSONL stays local and is archived
only as incremental .tar.gz parts by storage_retention.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from .mission_cause_monitor import REWARD_NAMES


class ForensicTransitionRecorder:
    """Selected-world pre/post trace, with byte caps and deterministic sampling."""

    def __init__(self, config, output, *, stride=None, world_ids=None,
                 max_raw_bytes=None, include_belief=None):
        self.enabled = (config.runtime.environment_backend == "warp_packet"
                        and os.getenv("UAV_FORENSIC_ENABLED", "true").lower() not in ("false","0"))
        self.stride = max(1, int(stride if stride is not None
                                 else os.getenv("UAV_FORENSIC_STRIDE", "64")))
        raw_world_ids = (world_ids if world_ids is not None else
                         os.getenv("UAV_FORENSIC_WORLD_IDS", "0"))
        if isinstance(raw_world_ids, str):
            raw_world_ids = [int(x.strip()) for x in raw_world_ids.split(",") if x.strip()]
        # Global forensic preference may list worlds 0..3 for E4096, while
        # the required CUDA preflight deliberately creates E1/E2 simulators.
        # Preserve all requested worlds in production, but never fail the
        # SCIENTIFIC test because an optional read-only observer requests a
        # world index that does not exist in a smaller test environment.
        requested_world_ids = list(dict.fromkeys(int(x) for x in raw_world_ids))
        if len(requested_world_ids) > 8 or any(i < 0 for i in requested_world_ids):
            raise ValueError("Forensic world indices must be nonnegative and at most 8 worlds")
        self.requested_world_ids = requested_world_ids
        self.world_ids = [i for i in requested_world_ids if i < int(config.runtime.num_envs)]
        self.out_of_range_world_ids = [i for i in requested_world_ids if i >= int(config.runtime.num_envs)]
        self.max_raw_bytes = int(max_raw_bytes if max_raw_bytes is not None
                                 else float(os.getenv("UAV_FORENSIC_MAX_RAW_MIB", "128")) * 2**20)
        if self.max_raw_bytes < 4096:
            raise ValueError("Forensic max_raw_bytes is too small")
        self.include_belief = (bool(include_belief) if include_belief is not None
                               else os.getenv("UAV_FORENSIC_FULL_BELIEF", "true").lower() == "true")
        self.path = Path(output) / "forensic-transitions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.num_uavs = int(config.world.num_uavs)
        self.num_envs = int(config.runtime.num_envs)
        self.learning_starts = int(config.algorithm_config.learning_starts)
        self.critical_dense_vector_steps = 16
        self.bytes_written = self.path.stat().st_size
        self.records_written = 0
        self.skipped_budget = 0
        self.skipped_cadence = 0
        self.write_errors = 0
        self.last_step = None
        self.last_warning = None

    @staticmethod
    def _row(values, index):
        if hasattr(values, "detach"):
            return values[index].detach().cpu().tolist()
        return torch.as_tensor(values[index]).detach().cpu().tolist()

    @staticmethod
    def _world(vector, index, *, deep=False):
        import warp as wp
        state = vector.s
        names = (
            "steps", "seeds", "positions", "altitudes", "velocities",
            "vertical_velocity", "battery", "known", "delivered",
            "report_present", "report_origin", "report_hops", "report_created",
            "sensor_counts", "report_counts", "packet_counts", "diagnostic_counts",
            "diagnostic_stats", "selection_counts", "flight_energy",
            "communication_energy", "tx_energy", "rx_energy", "power_histogram", "motion_counts",
            "action_counts", "motion_flags", "xyz_work", "reward",
            "reward_components", "gcs", "targets", "obstacles",
        )
        if deep:
            names += ("belief", "visited", "positive", "pkt_status",
                      "pkt_target", "pkt_attempt", "pkt_source", "pkt_destination",
                      "pkt_power", "pkt_cell", "pkt_fragment", "tx_power", "tx_source",
                      "tx_destination", "tx_receive")
        result = {}
        for name in names:
            field = getattr(state, name, None)
            if field is None:
                continue
            try:
                result[name] = ForensicTransitionRecorder._row(wp.to_torch(field), index)
            except (ValueError, TypeError, AttributeError, RuntimeError):
                # Field may be unavailable in a particular simulator variant.
                result[name + "_not_available"] = True
        return result

    @staticmethod
    @torch.no_grad()
    def _actor_heads(collector, observation, num_uavs):
        """Evaluate deterministic heads without sampling or changing the RNG.

        For MATD3 the recorded policy output may differ from the actually
        executed action due to exploration noise, random warmup, action masks,
        or execution/safety projections. These are distinct causal stages.
        """
        actors = getattr(collector, "actors", None)
        if actors is None or len(actors) not in (1, num_uavs):
            return {"status":"not_available"}
        result = []
        for agent_index in range(num_uavs):
            actor = actors[0 if len(actors) == 1 else agent_index]
            local = torch.as_tensor(observation[agent_index]).detach()
            actor_device = next(actor.parameters()).device
            local = local.to(actor_device, dtype=torch.float32).unsqueeze(0)
            # Shared MATD3/MASAC Actors concatenate the UAV identity to the
            # actual local observation. Reconstruct the exact inference input
            # WITHOUT altering the saved raw local observation or route mask.
            shared = len(actors) == 1
            identity = F.one_hot(torch.tensor([agent_index], device=actor_device),
                                 num_classes=num_uavs).to(local.dtype)
            actor_input = torch.cat((local, identity), dim=-1) if shared else local
            hidden = F.relu(actor.encoder(actor_input))
            logits = actor.destination(hidden).squeeze(0)
            valid = local[0, -(num_uavs+1):] > 0.5
            valid = valid.clone()
            valid[0] = True
            if hasattr(actor, "continuous"):
                raw = actor.continuous(hidden).squeeze(0)
                transformed = actor.normalize_pretanh(raw)
                result.append({
                    "actor": agent_index, "kind": "matd3",
                    "continuous_head_pretanh": raw.cpu().tolist(),
                    "continuous_head_after_output_normalization": transformed.cpu().tolist(),
                    "continuous_head_tanh": transformed.tanh().cpu().tolist(),
                    "destination_logits_raw": logits.cpu().tolist(),
                    "destination_valid_mask": valid.cpu().tolist(),
                    "greedy_masked_destination": int(logits.masked_fill(~valid, -1e30).argmax()),
                })
            elif hasattr(actor, "mean"):
                mean = actor.mean(hidden).squeeze(0)
                log_std = actor.log_std(hidden).squeeze(0)
                result.append({
                    "actor":agent_index,"kind":"masac",
                    "gaussian_mean":mean.cpu().tolist(),
                    "gaussian_log_std_raw":log_std.cpu().tolist(),
                    "gaussian_log_std_clamped":log_std.clamp(-20,2).cpu().tolist(),
                    "deterministic_tanh_mean":mean.tanh().cpu().tolist(),
                    "destination_logits_raw":logits.cpu().tolist(),
                    "destination_valid_mask":valid.cpu().tolist(),
                    "greedy_masked_destination":int(logits.masked_fill(~valid,-1e30).argmax()),
                })
        return result

    def begin(self, vector, observation, action, collector, *, collection_step,
              global_step, phase, policy_version=None):
        # The old stride=64 missed the first 10 MATD3 Actor updates where
        # the actual train run saturated. Capture early BEFORE/AFTER world
        # and neural evidence at every vector step in this critical window.
        critical=(self.learning_starts <= global_step <=
                  self.learning_starts+self.num_envs*self.critical_dense_vector_steps)
        if not self.enabled or not self.world_ids or (not critical and collection_step % self.stride != 0):
            self.skipped_cadence += 1
            return None
        if self.bytes_written >= self.max_raw_bytes:
            self.skipped_budget += 1
            return None
        token = []
        for index in self.world_ids:
            before_obs = self._row(observation["observation"], index)
            token.append({
                "schema":1,
                "timestamp_utc":datetime.now(timezone.utc).isoformat(),
                "global_step_before":int(global_step),
                "collection_step":int(collection_step),
                "world_index":int(index),
                "phase":str(phase),
                "policy_actor_update_version":policy_version,
                "actor_observation_before_exact":before_obs,
                "centralized_critic_state_before_exact":self._row(observation["state"], index),
                "actual_actor_action_continuous_xyz_rf":self._row(action.continuous, index),
                "actual_actor_action_destination":self._row(action.destination, index),
                "policy_head_evidence":self._actor_heads(collector,
                    torch.as_tensor(observation["observation"][index]), self.num_uavs),
                "world_before":self._world(vector,index,deep=self.include_belief),
            })
        return token

    def finish(self, token, vector, observation, reward, terminated, *, global_step,
               episode_returns=None, learner_metrics=None, replay_metrics=None,
               policy_version=None):
        if token is None:
            return 0
        saved = 0
        for document in token:
            index = document["world_index"]
            world_after = self._world(vector,index,deep=self.include_belief)
            old_parts = document["world_before"].get("reward_components")
            new_parts = world_after.get("reward_components")
            step_reward = float(self._row(reward,index))
            decomposition = {}
            if old_parts is not None and new_parts is not None:
                deltas = [float(a)-float(b) for a,b in zip(new_parts,old_parts)]
                delta_sum = sum(deltas)
                decomposition = {
                    "component_names":REWARD_NAMES,
                    "step_component_delta":deltas,
                    "component_cumulative_after":new_parts,
                    "component_cumulative_before":old_parts,
                    "step_component_delta_sum":delta_sum,
                    "step_reward":step_reward,
                    "step_delta_minus_reward":delta_sum-step_reward,
                    "step_abs_discrepancy":abs(delta_sum-step_reward),
                    "is_exactly_next_step":int(world_after["steps"]) ==
                        int(document["world_before"]["steps"])+1,
                }
                if episode_returns is not None:
                    cumulative = float(self._row(episode_returns,index))
                    decomposition["episode_return_cumulative"] = cumulative
                    decomposition["episode_component_sum_minus_episode_return"] = (
                        sum(float(v) for v in new_parts)-cumulative)
                if abs(delta_sum-step_reward)>1e-3:
                    self.last_warning = {"step":int(global_step),"world":index,
                                         "component_discrepancy":delta_sum-step_reward}
            before = document["world_before"]
            changes = {}
            for name in ("battery", "flight_energy", "communication_energy",
                         "sensor_counts", "report_counts", "packet_counts",
                         "diagnostic_counts", "selection_counts"):
                old_value, new_value = before.get(name), world_after.get(name)
                if old_value is None or new_value is None:
                    continue
                old_t, new_t = torch.as_tensor(old_value), torch.as_tensor(new_value)
                if old_t.shape == new_t.shape:
                    changes[name + "_delta"] = (new_t.double()-old_t.double()).tolist()
            for name in ("known", "delivered", "report_present"):
                old_value, new_value = before.get(name), world_after.get(name)
                if old_value is None or new_value is None:
                    continue
                old_t, new_t = torch.as_tensor(old_value), torch.as_tensor(new_value)
                changes[name + "_new_count"] = int(((new_t>0)&(old_t==0)).sum())
                changes[name + "_lost_count"] = int(((new_t==0)&(old_t>0)).sum())
            # The observations are exactly those fed to collector policy and
            # the next state produced by Warp before any autoreset.
            document.update({
                "environment_changes":changes,
                "global_step_after":int(global_step),
                "world_after":world_after,
                "actor_observation_after_exact":self._row(observation["observation"],index),
                "centralized_critic_state_after_exact":self._row(observation["state"],index),
                "actual_step_reward":step_reward,
                "terminated":bool(self._row(terminated,index)),
                "reward_accounting":decomposition,
                "policy_actor_update_version_after":policy_version,
                "learner_metrics_nearest_available":learner_metrics or {},
                "replay_metrics_nearest_available":replay_metrics or {},
                "interpretation":"Read-only selected-world trace; no hidden causal attribution. "
                    "Actor deterministic heads are pre-exploration; action is executed "
                    "after policy noise, canonicalization and safety. World snapshots "
                    "are from completed pre-reset collector boundaries.",
            })
            raw=(json.dumps(document,allow_nan=False,separators=(",",":"))+"\n").encode("utf-8")
            if len(raw)>2_000_000 or self.bytes_written+len(raw)>self.max_raw_bytes:
                self.skipped_budget+=1
                continue
            try:
                with self.path.open("ab") as stream:
                    stream.write(raw)
                self.bytes_written+=len(raw)
                self.records_written+=1
                self.last_step=int(global_step)
                saved+=1
            except OSError:
                self.write_errors+=1
                raise
        return saved

    def summary(self):
        return {
            "forensic/records_written":self.records_written,
            "forensic/raw_bytes_written":self.bytes_written,
            "forensic/records_skipped_budget":self.skipped_budget,
            "forensic/steps_skipped_cadence":self.skipped_cadence,
            "forensic/write_errors":self.write_errors,
            "forensic/last_step":self.last_step,
            "forensic/last_reward_discrepancy":self.last_warning,
            "forensic/stride":self.stride,
            "forensic/world_ids":self.world_ids,
            "forensic/requested_world_ids":self.requested_world_ids,
            "forensic/skipped_out_of_range_world_ids":self.out_of_range_world_ids,
            "forensic/full_belief_enabled":self.include_belief,
            "forensic/critical_dense_vector_steps":self.critical_dense_vector_steps,
            "forensic/completeness":"selected_worlds_at_cadence_only; not all transitions",
        }


def write_run_provenance(config, output):
    """Archive-ready provenance: sanitized resolved config and source hashes.

    Embedded Kaggle source may have no .git; hashes still allow after-the-fact
    verification of exactly which core modules participated in training.
    Never serializes key.txt, environment secrets or credentials themselves.
    """
    root = Path(__file__).resolve().parents[2]
    files = (
        "uav_network/algorithms/matd3.py",
        "uav_network/algorithms/layerwise_gradient_probe.py",
        "uav_network/algorithms/masac.py",
        "uav_network/algorithms/models.py",
        "uav_network/experiments/training.py",
        "uav_network/experiments/action_cbf_monitor.py",
        "uav_network/experiments/root_cause_audit.py",
        "uav_network/experiments/forensic_minibatches.py",
        "uav_network/experiments/forensic_integrity.py",
        "uav_network/experiments/layerwise_gradient_journal.py",
        "uav_network/experiments/training_health.py",
        "uav_network/experiments/pql_training.py",
        "uav_network/experiments/forensic_transitions.py",
        "uav_network/experiments/mission_cause_monitor.py",
        "uav_network/environment/gpu/vector.py",
        "uav_network/communication/gpu/reports.py",
        "uav_network/experiments/storage_retention.py",
    )
    hashed = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in files if (root/name).is_file()}
    tracked = config.to_dict()
    tracked.setdefault("tracking", {})["credentials_file"] = "[redacted]"
    document = {
        "schema": 1, "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "kind":"forensic_run_provenance",
        "resolved_configuration":tracked,
        "source_sha256":hashed,
        "source_git_commit":os.getenv("UAV_GIT_COMMIT","unknown"),
        "source_archive_sha256":os.getenv("UAV_SOURCE_SHA256","unknown"),
        "runtime_torch":torch.__version__,
        "cuda_torch_build":torch.version.cuda,
        "cuda_available":torch.cuda.is_available(),
        "visible_device_count":torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "training_seed":config.runtime.seed,
        "reproducibility_note":"Package and CUDA versions plus fixed seed do not prove bitwise determinism.",
        "forensic_environment_flags":{
            "enabled":os.getenv("UAV_FORENSIC_ENABLED","true"),
            "stride":os.getenv("UAV_FORENSIC_STRIDE","64"),
            "world_ids":os.getenv("UAV_FORENSIC_WORLD_IDS","0"),
            "full_belief":os.getenv("UAV_FORENSIC_FULL_BELIEF","true"),
            "max_raw_mib":os.getenv("UAV_FORENSIC_MAX_RAW_MIB","128"),
        },
    }
    dest=Path(output)/"run-provenance.jsonl"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(document,allow_nan=False,separators=(",",":"))+"\n")
    return dest
