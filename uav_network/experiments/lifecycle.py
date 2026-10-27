"""Let scheduler SIGTERM run Python cleanup before the process-group fallback."""
from contextlib import contextmanager
import signal
import threading


@contextmanager
def graceful_termination():
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def terminate(signum, frame):
        error = KeyboardInterrupt("Training terminated by scheduler")
        error.signal_number = signum
        raise error

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
