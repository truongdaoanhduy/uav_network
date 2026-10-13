"""Evidence-based warnings and crash context; no automatic root-cause claims."""
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import traceback

import numpy as np


class TrainingDiagnostics:
    def __init__(self, config, output):
        self.settings = config.tracking
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "diagnostics.jsonl"
        self.path.touch()
        self.window = deque(maxlen=2 * self.settings.diagnostic_window)
        self.warning_count = 0
        self.last_warning = {}

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
        checks = {
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
        if len(self.window) == 2*n:
            previous = list(self.window)[:n]
            values["rolling/return_change"] = values["rolling/return_mean"] - float(
                np.mean([r["return"] for r in previous]))
            checks["return_drop"] = values["rolling/return_change"] < -self.settings.return_drop_threshold
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
    except (OSError, ValueError, KeyError, RuntimeError):
        pass
    detail = "".join(traceback.format_exception(error))
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "[REDACTED]")
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "error_type": type(error).__name__, "traceback": detail,
              "context": context, "root_cause_confirmed": False}
    path = Path(output) / "failure.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"training_failure": result}), flush=True)
    return path


def evaluate_checkpoint(config, checkpoint, episode, step, tracker, diagnostics):
    """Separate CPU process preserves learner RNG and uses identical evaluation seeds."""
    folder = Path(checkpoint).parent / "evaluations" / f"episode_{episode:06d}"
    log = Path(checkpoint).parent / f"evaluation_{episode:06d}.log"
    command = [sys.executable, "-m", "uav_network", "evaluate", "--checkpoint", str(checkpoint),
               "--output-dir", str(folder), "--episodes", str(config.runtime.eval_episodes),
               "--seed", str(config.runtime.eval_seed)]
    try:
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           timeout=config.tracking.evaluation_timeout_seconds)
        result = json.loads((folder / "metrics.json").read_text())
        metrics = {f"eval/{key}": result[key] for key in
                   ("mean_return", "success_rate", "mean_search_rate_percent", "mean_delivery_rate_percent")}
        metrics.update({"eval/episodes": result["episodes"], "eval/seed": config.runtime.eval_seed,
                        "eval/checkpoint_episode": episode})
        tracker.diagnostic(metrics, step)
        tracker.run.save(str(folder / "metrics.json"), base_path=str(diagnostics.output), policy="now")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        diagnostics.event("evaluation_failed", step, {"episode": episode, "error_type": type(error).__name__,
                                                     "log": str(log)}, severity="error")
        tracker.diagnostic({"diagnostics/evaluation_failed": 1}, step)
    finally:
        tracker.run.save(str(log), base_path=str(diagnostics.output), policy="now")
