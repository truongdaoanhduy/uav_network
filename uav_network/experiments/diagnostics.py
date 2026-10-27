"""Evidence-based warnings and crash context; no automatic root-cause claims."""
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

from .telemetry import OutcomePool


class TrainingDiagnostics:
    def __init__(self, config, output, initial_step=0):
        self.settings = config.tracking
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "diagnostics.jsonl"
        self.path.touch()
        self.cohorts_path = self.output / "cohorts.jsonl"
        self.cohorts_path.touch()
        self.window = deque(maxlen=2 * self.settings.diagnostic_window)
        self.warning_count = 0
        self.last_warning = {}
        self.cohort_interval = config.world.max_steps * config.runtime.num_envs
        self.start_windows(initial_step)

    def start_windows(self, step):
        self.cohort_start = step // self.cohort_interval * self.cohort_interval
        self.cohort_partial = step != self.cohort_start
        self.pool = OutcomePool()
        self.previous_cohort = None

    def batch(self, records, step, final=False, partial=False):
        """Call once after ALL completed worlds in the collection have been consumed."""
        values = {}
        while step > self.cohort_start + self.cohort_interval:
            values = self._finish_window(self.cohort_start + self.cohort_interval, False)
        self.pool.add(records)
        self.cohort_partial |= bool(partial)
        if step == self.cohort_start + self.cohort_interval or final:
            values = self._finish_window(step, final and step < self.cohort_start + self.cohort_interval)
        return values

    def _finish_window(self, step, partial):
        partial = bool(partial or self.cohort_partial)
        values = {f"cohort/{key}": value for key, value in self.pool.metrics().items()}
        values.update({"cohort/source_start_step": self.cohort_start,
                       "cohort/source_end_step": step, "cohort/partial": int(partial),
                       "cohort/source": "completed_outcomes_in_absolute_transition_window"})
        previous = self.previous_cohort
        mean, n = values["cohort/return_mean"], values["cohort/episodes"]
        ready = not partial and n >= self.settings.diagnostic_window
        warning = False
        if ready and previous is not None:
            values["cohort/return_change"] = mean - previous["cohort/return_mean"]
            warning = values["cohort/return_change"] < -self.settings.return_drop_threshold
        values["diagnostics/return_drop_warning"] = int(warning)
        if warning:
            self.warning_count += 1
            self.event("return_drop", step, {"previous_cohort": previous, "current_cohort": values.copy(),
                       "interpretation": "Completed outcomes in distinct transition windows; observed symptom"})
        self.previous_cohort = values.copy() if ready else None
        values["diagnostics/warning_count"] = self.warning_count
        with self.cohorts_path.open("a") as stream:
            stream.write(json.dumps(values, allow_nan=False) + "\n")
        self.pool = OutcomePool()
        self.cohort_start = step
        self.cohort_partial = False
        return values

    def event(self, code, step, evidence, severity="warning"):
        event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                 "code": code, "severity": severity, "global_step": int(step),
                 "evidence": evidence, "root_cause_confirmed": False}
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps({"diagnostic_event": event}, allow_nan=False), flush=True)
        return event

    def episode(self, record):
        self.window.append(record)
        n = self.settings.diagnostic_window
        recent = list(self.window)[-n:]
        values = {"rolling/return_mean": float(np.mean([r["return"] for r in recent])),
                  "rolling/return_std": float(np.std([r["return"] for r in recent])),
                  "rolling/search_rate_percent": float(np.mean([r["search_rate_percent"] for r in recent])),
                  "rolling/delivery_rate_percent": float(np.mean([r["delivery_rate_percent"] for r in recent])),
                  "rolling/success_rate": float(np.mean([r["success"] for r in recent])),
                  "diagnostics/window_ready": int(len(recent) >= n)}
        for name, group, count, components in (
                ("shield_intervention", "motion_counts", "shield_interventions", 1),
                ("commanded_stall", "motion_counts", "commanded_stalls", 1),
                ("movement_saturation", "action_counts", "saturated", 2)):
            values[f"rolling/{name}_rate"] = float(np.mean([
                r.get(group, {}).get(count, 0) / max(1, components * r.get("episode_length", 1)
                                                  * r.get("num_uavs", 6)) for r in recent]))
        if len(self.window) == 2 * n:
            previous = list(self.window)[:n]
            # Descriptive compatibility metric; regression warnings use closed cohorts.
            values["rolling/return_change"] = values["rolling/return_mean"] - float(
                np.mean([r["return"] for r in previous]))
        checks = {
            **{name: values[f"rolling/{name}_rate"] >= self.settings.blocked_rate_threshold
               for name in ("shield_intervention", "commanded_stall", "movement_saturation")},
            "delivery_gap": values["rolling/search_rate_percent"] - values["rolling/delivery_rate_percent"]
                            >= self.settings.delivery_gap_threshold_percent,
            "blocked_motion": float(np.mean([r.get("blocked_moves", 0) /
                               max(1, r.get("episode_length", 1) * r.get("num_uavs", 6)) for r in recent]))
                              >= self.settings.blocked_rate_threshold,
            "no_search": values["rolling/search_rate_percent"] == 0,
            "report_loss": any(r.get("expired_reports", 0) + r.get("dropped_reports", 0) > 0 for r in recent),
            "battery_exhausted": any(r.get("termination_reason") == "battery_exhausted" for r in recent),
            "reward_accounting": any("reward_components" in r and not math.isclose(
                sum(r["reward_components"].values()), r["return"], rel_tol=1e-6, abs_tol=1e-5) for r in recent),
        }
        for code, condition in checks.items():
            active = bool(condition and (len(recent) >= n or code == "reward_accounting"))
            values[f"diagnostics/{code}_warning"] = int(active)
            episode = record["episode"]
            if active and episode - self.last_warning.get(code, -n) >= n:
                self.warning_count += 1
                self.last_warning[code] = episode
                self.event(code, record["global_step"], {
                    "episode": episode, "window_episodes": len(recent), "rolling_metrics": values.copy(),
                    "latest_seed": record.get("seed"), "termination_reason": record.get("termination_reason"),
                    "network": record.get("network", {}), "reward_components": record.get("reward_components", {}),
                    "interpretation": "Observed symptom; inspect environment, policy and network evidence before attributing cause"})
        values["diagnostics/warning_count"] = self.warning_count
        return values


def failure_report(config, output, error, context):
    """Redact credentials before writing a traceback; retain the original exception."""
    secrets = [os.environ.get(k, "") for k in ("WANDB_API_KEY", "KAGGLE_API_TOKEN", "HF_TOKEN")]
    try:
        from ..cloud.credentials import load_credentials
        credentials = load_credentials(config.tracking.credentials_file)
        secrets += [credentials.wandb_key, *(a.token for a in credentials.accounts)]
    except (OSError, ValueError, KeyError, RuntimeError, AttributeError):
        pass
    detail = "".join(traceback.format_exception(error))
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[REDACTED]")
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "error_type": type(error).__name__, "traceback": detail,
              "context": context, "root_cause_confirmed": False}
    path = Path(output) / "failure.json"
    print(json.dumps({"training_failure": result}), flush=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return path
