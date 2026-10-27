"""Bounded full-state backups; only owned staging files are ever released."""
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
import time


def _report(tracker, values, step):
    try:
        tracker.diagnostic(values, step)
    except Exception as error:
        with suppress(OSError, ValueError):
            print(f"Recovery status logging failed: {type(error).__name__}; backup status={values['recovery/status']}", flush=True)


class PeriodicRecovery:
    def __init__(self, config, output, *, started):
        settings = config.tracking
        self.enabled = (settings.enabled and settings.log_checkpoint
                        and settings.checkpoint_store == "huggingface"
                        and settings.resumable_checkpoint_interval_seconds > 0)
        self.output = Path(output)
        self.maximum = settings.resumable_checkpoint_max_uploads
        # Spread the finite storage budget over the session, rather than using
        # every backup slot in its first few minutes.
        self.interval = max(settings.resumable_checkpoint_interval_seconds,
                            config.runtime.max_train_seconds / self.maximum)
        self.next_due = started + self.interval
        self.attempts = 0

    def maybe_save(self, save, tracker, *, step=0):
        if (not self.enabled or tracker is None or self.attempts >= self.maximum
                or time.perf_counter() < self.next_due):
            return
        self.attempts += 1
        started = time.perf_counter()
        progress = {}
        values = {"recovery/attempts": self.attempts,
                  "recovery/max_attempts": self.maximum,
                  "recovery/effective_interval_seconds": self.interval}
        _report(tracker, {**values,"recovery/status":"snapshot_started"}, step)
        try:
            # A primary full checkpoint is ~15.8GB on a ~20GB Kaggle disk.
            # Never keep two stages or overwrite the final resume_latest file.
            with TemporaryDirectory(prefix=".recovery-", dir=self.output) as directory:
                policy, resume = Path(directory) / "policy.pt", Path(directory) / "resume.pt"
                progress = save(policy, resume_path=resume)
                if not policy.is_file() or not resume.is_file():
                    raise ValueError("Recovery callback did not produce full resume files")
                progress = {**progress, "resume_checkpoint": str(resume)}
                size = policy.stat().st_size + resume.stat().st_size
                _report(tracker, {**values,"recovery/status":"upload_started",
                                  "recovery/checkpoint_bytes":size}, progress.get("global_step",step))
                artifact = tracker.checkpoint(policy, progress, resumable=True)
                if not artifact:
                    raise RuntimeError("Recovery artifact was not confirmed")
                values.update({"recovery/status": "uploaded", "recovery/checkpoint_bytes": size})
        except Exception as error:
            # A transient backup failure must not erase the previous verified
            # artifact or terminate otherwise healthy training.
            values.update({"recovery/status": "failed", "recovery/error_type": type(error).__name__})
        finally:
            finished = time.perf_counter()
            self.next_due = finished + self.interval
        values["recovery/elapsed_seconds"] = finished - started
        _report(tracker, values, progress.get("global_step", step))
