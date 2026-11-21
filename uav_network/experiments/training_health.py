"""Bounded longitudinal training-health log; observed symptoms, not root causes.

Structured snapshots survive a later model failure and can be correlated
with W&B scalars by global_step. Heuristic alerts do not change optimization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path

from .evidence import json_safe


class TrainingHealthMonitor:
    def __init__(self, config, output):
        self.path = Path(output) / "training-health-snapshots.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.interval = max(1, int(config.runtime.log_interval_steps))
        self.last_alert = {}
        self.alert_count = 0
        self.latest = {}

    def capture(self, step, action, learner, replay, memory, *, mission=None, system=None, updates=0, phase="policy"):
        """Persist one bounded record per sampled completed collection step."""
        mission = mission or {}
        system = system or {}
        risks = {}
        def finite_num(value):
            return isinstance(value, (int, float)) and math.isfinite(float(value))
        def flag(name, source, key, condition):
            value = source.get(key)
            risks[name] = int(bool(finite_num(value) and condition(float(value))))
        flag("gpu1_low_physical_headroom", memory, "memory/cuda1_driver_free_now_gib", lambda v:v < 2)
        flag("gpu0_low_physical_headroom", memory, "memory/cuda0_driver_free_now_gib", lambda v:v < 2)
        flag("host_low_cgroup_headroom", memory, "memory/cgroup_headroom_now_gib", lambda v:v < 3)
        flag("actor_motion_tanh_saturated", learner,
             "actor_motion_tanh_derivative_lt_1e3_fraction", lambda v: updates > 0 and v > .5)
        flag("actor_first_encoder_near_zero_grad", learner,
             "actor_first_encoder_grad_postclip", lambda v: updates > 0 and abs(v) < 1e-9)
        flag("destination_no_route", action, "action_probe/mask_silent_only_fraction", lambda v:v>.95)
        flag("cbf_high_correction", action, "action_probe/cbf_projected_fraction", lambda v:v>.4)
        flag("command_saturated", action, "action_probe/motion_command_saturation_fraction",lambda v:v>.5)
        flag("critic_large_td_error", learner, "td_error_abs_mean", lambda v:v>100)
        flag("replay_eviction_unsampled", replay, "replay/evicted_unsampled_fraction",lambda v:v>.05)
        # Heuristic gates; retain raw metrics/denominators for interpretation.
        flag("found_but_delivery_stalled", mission,
             "mission_probe/delivery_given_found_fraction", lambda v:
             mission.get("mission_probe/found_target_fraction", 0) > .2 and v < .05)
        flag("confirmed_but_no_fifo_report", mission,
             "mission_probe/pending_not_in_fifo_fraction", lambda v:
             mission.get("mission_probe/pending_target_observation_count", 0) > 10 and v > .5)
        flag("confirmed_no_fifo_or_active_packet", mission,
             "mission_probe/pending_without_fifo_or_active_packet_fraction", lambda v:
             mission.get("mission_probe/pending_target_observation_count", 0) > 10 and v > .5)
        flag("report_geometry_disconnected", mission,
             "mission_probe/pending_with_report_but_no_optimistic_path_fraction", lambda v:
             mission.get("mission_probe/pending_target_observation_count", 0) > 10 and v > .5)
        flag("report_ttl_near_expiry", mission,
             "mission_probe/pending_report_ttl_gt_75pct_fraction", lambda v:
             mission.get("mission_probe/pending_report_observation_count", 0) > 10 and v > .5)
        flag("report_buffer_near_capacity", mission,
             "mission_probe/buffer_full_fraction", lambda v: v > .5)
        flag("battery_near_empty", mission,
             "mission_probe/battery_below_10pct_fraction", lambda v: v > .2)
        flag("reward_component_inconsistent", mission,
             "mission_probe/reward_accounting_abs_discrepancy_mean", lambda v: v > 1e-3)
        flag("large_cbf_velocity_correction", action,
             "action_probe/cbf_correction_gt_1mps_fraction", lambda v: v > .3)
        flag("outward_edge_commands", action,
             "action_probe/command_outward_at_boundary_fraction", lambda v: v > .3)
        flag("stalled_despite_command", action,
             "action_probe/stalled_with_strong_command_fraction", lambda v: v > .2)
        flag("pql_collector_policy_stale", system,
             "pql/collector_policy_lag", lambda v: v > 2)
        flag("pql_critic_policy_stale", system,
             "pql/critic_policy_lag", lambda v: v > 2)
        flag("pql_policy_critic_stale", system,
             "pql/policy_critic_lag", lambda v: v > 8)
        flag("pql_collection_backpressure", system,
             "pql/queued_collections", lambda v: v > 1)
        flag("masac_logstd_low_clipped", learner,
             "masac_probe/std_clamped_low_fraction", lambda v: v > .5)
        flag("masac_logstd_high_clipped", learner,
             "masac_probe/std_clamped_high_fraction", lambda v: v > .5)
        risks["nonfinite_learner_metric"] = int(any(
            isinstance(v,(float,int)) and not math.isfinite(float(v))
            for v in learner.values()))
        new_alerts = []
        for name, active in risks.items():
            if active and step - self.last_alert.get(name, -10**12) >= 10 * self.interval:
                self.last_alert[name] = step
                new_alerts.append(name)
        self.alert_count += len(new_alerts)
        indicators = {f"health/{name}":val for name,val in risks.items()}
        indicators["health/new_alert_count"] = len(new_alerts)
        indicators["health/total_alerts"] = self.alert_count
        snapshot = {
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "global_step":int(step), "training_phase":phase,
            "interpretation":"Heuristic symptom flags only; no root cause is confirmed by these thresholds.",
            "new_alerts":new_alerts,
            "health":indicators,
            "action_cbf":action,
            "learner":learner,
            "replay":replay,
            "resource":memory,
            "mission":mission,
            "system":system,
        }
        encoded=json_safe(snapshot)
        # json_safe() limits nested mappings to 128 fields as a general
        # defense. Full MATD3/Q1/Q2 layer probes can exceed 300 scalar keys;
        # retaining only the first 128 silently drops the root-cause fields.
        # The learner dictionary is produced by our own scalar_metrics() and
        # remains bounded and numeric, so preserve ALL values separately.
        encoded["learner"] = {
            name:json_safe(value)
            for name,value in learner.items()
            if isinstance(name,str) and len(name)<=256
            and isinstance(value,Real)
        }
        encoded["learner_metric_count"] = len(encoded["learner"])
        with self.path.open("a") as out:
            out.write(json.dumps(encoded,allow_nan=False)+"\n")
        self.latest=indicators
        return indicators, new_alerts
