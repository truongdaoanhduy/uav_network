"""Own one non-terminating native traceback timer for a learner process."""
from datetime import datetime, timezone
import faulthandler
import json
from pathlib import Path
import time

from .evidence import print_evidence


class ProgressWatchdog:
    # ponytail: one timer owner per process; independent sampling if concurrent learners share a process.
    _active = None

    def __init__(self, output, timeout=300):
        if timeout <= 0:
            raise ValueError("Traceback timeout must be positive")
        self.path = Path(output) / "thread-stacks.log"
        self.timeout, self.stream = timeout, None
        self.last_step, self.last_arm = -1, 0.
        self.warned = False
        previous = type(self)._active
        if previous is not None:
            previous.close()
            if type(self)._active is not None:
                return
        try:
            self.stream = self.path.open("a")
            type(self)._active = self
            self._arm()
        except Exception as error:
            self._warn(error)
            self.close()

    def _warn(self, error):
        if not self.warned:
            self.warned = True
            print_evidence({"traceback_watchdog_unavailable": {"error_type": type(error).__name__}})

    def _arm(self):
        faulthandler.cancel_dump_traceback_later()
        self.stream.write(json.dumps({"armed_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_progress_marker": self.last_step, "timeout_seconds": self.timeout,
            "interpretation": "No new progress marker; long calls may be legitimate. This does not terminate or diagnose a deadlock."}) + "\n")
        self.stream.flush()
        faulthandler.dump_traceback_later(self.timeout, repeat=True, file=self.stream, exit=False)
        self.last_arm = time.monotonic()

    def progress(self, step):
        if type(self)._active is not self or step <= self.last_step:
            return
        self.last_step = step
        if time.monotonic() - self.last_arm < min(30., self.timeout / 10):
            return
        try:
            self._arm()
        except Exception as error:
            self._warn(error)
            self.close()

    def close(self):
        if self.stream is None:
            return
        if type(self)._active is self:
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception as error:
                # Keep the descriptor alive if cancellation cannot be confirmed.
                self._warn(error)
                return
            type(self)._active = None
        try:
            self.stream.close()
        except OSError as error:
            self._warn(error)
        self.stream = None
