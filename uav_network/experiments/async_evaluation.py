"""One exec'd CPU evaluator, no queue; busy thresholds are explicitly skipped."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .evaluation_snapshots import save_evaluation_snapshot
from .evidence import json_safe, secret_values, write_evidence


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AsyncCheckpointEvaluator:
    def __init__(self, config, output, tracker, diagnostics, initial_step=0, initial_episodes=0):
        self.config, self.output = config, Path(output)
        self.tracker, self.diagnostics = tracker, diagnostics
        self.secrets = secret_values(config)
        self.interval = config.tracking.evaluation_interval_steps
        self.next_step = (initial_step // self.interval + 1) * self.interval if self.interval else None
        self.legacy_interval = config.tracking.evaluation_interval if tracker is not None else 0
        self.next_episode = ((initial_episodes // self.legacy_interval + 1) * self.legacy_interval
                             if self.legacy_interval else None)
        self.folder = self.output / "evaluations"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.events_path = self.folder / "events.jsonl"
        self.active = self.last_result = None
        self.closed = False
        self.counts = dict.fromkeys(("submitted", "completed", "failed", "invalid_result", "timed_out",
                                    "cancelled", "reaped", "skipped_busy", "skipped_deadline", "skipped_thresholds"), 0)

    def command(self, checkpoint, folder, episodes, seed):
        return [sys.executable, "-m", "uav_network", "evaluate", "--checkpoint", str(checkpoint),
                "--output-dir", str(folder), "--episodes", str(episodes), "--seed", str(seed)]

    def _event(self, status, step, source=None, **detail):
        self.counts[status] += 1
        event = {"status": status, "arrival_global_step": int(step), "source": source, **detail}
        write_evidence(self.events_path, json_safe(event, self.secrets), append=True)
        if self.tracker:
            try:
                self.tracker.run.save(str(self.events_path), base_path=str(self.output), policy="live")
            except Exception as error:
                self.diagnostics.event("evaluation_upload_failed", step, {"error_type": type(error).__name__})
            try:
                self.tracker.diagnostic({"evaluation_status/status": status,
                                         **{f"evaluation_status/{key}": value for key, value in self.counts.items()},
                                         "evaluation_status/running": int(self.active is not None)}, step)
            except Exception as error:
                self.diagnostics.event("evaluation_status_logging_failed", step, {"error_type": type(error).__name__})
        if status in {"failed", "invalid_result", "timed_out"}:
            self.diagnostics.event(f"evaluation_{status}", step, event, severity="error")

    def tick(self, step, episodes, source, progress, allow_submit=True):
        self.poll(step)
        due = []
        if self.next_step is not None and step >= self.next_step:
            crossed = (step - self.next_step) // self.interval + 1
            threshold = self.next_step + (crossed - 1) * self.interval
            self.next_step += crossed * self.interval
            self.counts["skipped_thresholds"] += crossed - 1
            due.append(("periodic_validation", threshold,
                        self.config.tracking.periodic_evaluation_episodes,
                        self.config.tracking.periodic_evaluation_seed))
        if self.next_episode is not None and episodes >= self.next_episode:
            crossed = (episodes - self.next_episode) // self.legacy_interval + 1
            self.next_episode += crossed * self.legacy_interval
            self.counts["skipped_thresholds"] += crossed - 1
            due.append(("legacy_validation", None, self.config.runtime.eval_episodes, self.config.runtime.eval_seed))
        for kind, threshold, n, seed in due:
            meta = {"kind": kind, "scheduled_step": threshold, "global_step": int(step),
                    "episodes": int(episodes), "evaluation_episodes": n,
                    "seed_first": seed, "seed_last": seed + n - 1, "deterministic_policy": True}
            if self.closed or not allow_submit:
                self._event("skipped_deadline", step, meta)
                continue
            if self.active is not None:
                self._event("skipped_busy", step, meta)
                continue
            stream = None
            try:
                actors, version = source()
                checkpoint = save_evaluation_snapshot(self.config, self.folder, actors, version, progress)
                folder = self.folder / checkpoint.stem
                log = self.folder / f"{checkpoint.stem}.log"
                meta.update({"policy_version": int(version), "checkpoint": str(checkpoint),
                             "checkpoint_sha256": sha256(checkpoint)})
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = ""
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    environment[name] = "1"
                stream = log.open("w")
                process = subprocess.Popen(self.command(checkpoint, folder, n, seed),
                    stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
                self.active = {"process": process, "stream": stream, "started": time.monotonic(),
                               "folder": folder, "log": log, "source": meta}
                self._event("submitted", step, meta)
            except (OSError, ValueError, RuntimeError) as error:
                if stream is not None:
                    stream.close()
                self._event("failed", step, meta, error_type=type(error).__name__, error=str(error))

    @staticmethod
    def _stop(process):
        # start_new_session isolates this evaluator's descendants from other jobs.
        for sig, timeout in ((signal.SIGTERM, .25), (signal.SIGKILL, .5)):
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                continue
        return process.poll() is not None

    def poll(self, step):
        task = self.active
        if task is None:
            return
        process = task["process"]
        status = process.poll()
        if task.get("stopping"):
            if status is not None:
                task["stream"].close()
                self.active = None
                self._event("reaped", step, task["source"], reason=task["stopping"])
            return
        if status is None and time.monotonic() - task["started"] < self.config.tracking.evaluation_timeout_seconds:
            return
        result = None
        detail = {"log": str(task["log"])}
        if status is None:
            reaped = self._stop(process)
            outcome = "timed_out"
            detail["reaped"] = reaped
            if not reaped:
                # Keep the single-worker slot reserved until the OS confirms exit.
                task["stopping"] = outcome
                self._event(outcome, step, task["source"], **detail)
                return
        elif status:
            outcome = "failed"
            detail["returncode"] = status
        else:
            try:
                result = json.loads((task["folder"] / "metrics.json").read_text())
                required = ("mean_return", "success_rate", "mean_search_rate_percent",
                            "mean_delivery_rate_percent", "episodes", "protocol", "records")
                if not isinstance(result, dict) or any(key not in result for key in required):
                    raise ValueError("Incomplete evaluation result")
                expected = task["source"]["evaluation_episodes"]
                if type(result["episodes"]) is not int or result["episodes"] != expected:
                    raise ValueError("Evaluation episode count differs")
                if any(type(result[key]) not in (int, float) or not math.isfinite(result[key])
                       for key in required[:4]):
                    raise ValueError("Nonfinite or invalid evaluation metric")
                protocol, records = result["protocol"], result["records"]
                if (not isinstance(protocol, dict) or not isinstance(protocol.get("scenario_hash"), str)
                        or len(protocol["scenario_hash"]) != 64 or not isinstance(records, list)
                        or len(records) != expected or any(not isinstance(record, dict) or
                            record.get("seed") != task["source"]["seed_first"] + i
                            for i, record in enumerate(records))):
                    raise ValueError("Evaluation protocol or seed bank differs")
                summaries, pooled = result.get("metric_summary", {}), result.get("pooled", {})
                if (not isinstance(summaries, dict) or any(not isinstance(v, dict) for v in summaries.values())
                        or not isinstance(pooled, dict)):
                    raise ValueError("Invalid evaluation summary shape")
                json.dumps(result, allow_nan=False)
                outcome = "completed"
            except (OSError, ValueError, TypeError) as error:
                outcome = "invalid_result"
                detail["error_type"] = type(error).__name__
        task["stream"].close()
        self.active = None
        if outcome == "completed":
            result["source"] = task["source"]
            path = task["folder"] / "metrics.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False))
            self.last_result = result
            if self.tracker:
                values = {f"evaluation/{key}": result[key] for key in
                          ("mean_return", "success_rate", "mean_search_rate_percent", "mean_delivery_rate_percent", "episodes")}
                values.update({f"eval/{key}": result[key] for key in
                               ("mean_return", "success_rate", "mean_search_rate_percent", "mean_delivery_rate_percent")})
                for name, summary in result.get("metric_summary", {}).items():
                    values[f"evaluation/{name}"] = summary.get("mean", summary.get("value"))
                    values[f"evaluation/{name}_episodes"] = summary.get("episodes")
                    values[f"eval/{name}"] = summary.get("mean", summary.get("value"))
                values.update({f"evaluation/pooled/{key}": value for key, value in result.get("pooled", {}).items()})
                source = task["source"]
                values.update({"evaluation/checkpoint_global_step": source["global_step"],
                               "evaluation/checkpoint_episode": source["episodes"],
                               "evaluation/scheduled_global_step": source["scheduled_step"],
                               "evaluation/policy_version": source["policy_version"],
                               "evaluation/checkpoint_sha256": source["checkpoint_sha256"],
                               "evaluation/seed_first": source["seed_first"], "evaluation/seed_last": source["seed_last"],
                               "evaluation/scenario_hash": result["protocol"]["scenario_hash"],
                               "evaluation/deterministic_policy": True, "evaluation/kind": source["kind"],
                               "eval/episodes": result["episodes"], "eval/seed": source["seed_first"],
                               "eval/checkpoint_episode": source["episodes"]})
                self.tracker.diagnostic(values, step)
                self.tracker.run.save(str(path), base_path=str(self.output), policy="now")
        self._event(outcome, step, task["source"], **detail)
        if self.tracker:
            self.tracker.run.save(str(task["log"]), base_path=str(self.output), policy="now")

    def close(self, step, drain_seconds=10.):
        if self.closed:
            self.poll(step)
            return
        self.closed = True
        deadline = time.monotonic() + max(0., min(10., drain_seconds))
        while self.active is not None and time.monotonic() < deadline:
            self.poll(step)
            if self.active is not None:
                time.sleep(min(.02, max(0., deadline - time.monotonic())))
        if self.active is not None:
            task = self.active
            reaped = self._stop(task["process"])
            if reaped:
                task["stream"].close()
                self.active = None
            else:
                task["stopping"] = "cancelled"
            self._event("cancelled", step, task["source"], reason="bounded_shutdown", reaped=reaped)
            if self.tracker:
                self.tracker.run.save(str(task["log"]), base_path=str(self.output), policy="now")
