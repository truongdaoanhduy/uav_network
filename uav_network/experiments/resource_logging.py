"""Durable phase journal; device-wide snapshots and Torch-only allocation peaks."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time
import traceback

import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def memory_snapshot(devices):
    host = {"rss_bytes": None, "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "available_bytes": None, "total_bytes": None}
    for filename, mappings in (
        ("/proc/self/status", {"VmRSS:": "rss_bytes"}),
        ("/proc/meminfo", {"MemAvailable:": "available_bytes", "MemTotal:": "total_bytes"}),
    ):
        try:
            for line in Path(filename).read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in mappings:
                    host[mappings[parts[0]]] = int(parts[1]) * 1024
        except OSError:
            pass
    cuda = {}
    for name in dict.fromkeys(devices):
        device = torch.device(name)
        if device.type != "cuda":
            continue
        if not torch.cuda.is_available():
            cuda[name] = {"available": False}
            continue
        try:
            free, total = torch.cuda.mem_get_info(device)
            cuda[name] = {"available": True, "name": torch.cuda.get_device_name(device),
                          "device_free_bytes": free, "device_total_bytes": total,
                          "device_used_snapshot_bytes": total - free,
                          "torch_allocated_bytes": torch.cuda.memory_allocated(device),
                          "torch_reserved_bytes": torch.cuda.memory_reserved(device),
                          "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        except (RuntimeError, ValueError) as error:
            cuda[name] = {"available": False, "error_type": type(error).__name__}
    return {"host": host, "cuda": cuda,
            "measurement_scope": "device_used is a snapshot; Torch peaks exclude Warp allocations"}


class ResourceJournal:
    def __init__(self, output_dir, *, devices):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.devices = list(devices)
        self.current_phase = None
        self.started = time.monotonic()

    def record(self, event, phase, **values):
        row = {"time_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_s": time.monotonic() - self.started,
               "event": event, "phase": phase, **values,
               "memory": memory_snapshot(self.devices)}
        # A flushed begin record is retained even if the OS kills the process.
        with (self.output_dir / "resources.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_json(self.output_dir / "progress.json", row)

    @contextmanager
    def phase(self, name, **values):
        self.current_phase = name
        self.record("phase_begin", name, **values)
        start = time.monotonic()
        try:
            yield
            # Include completed CUDA work, rather than just enqueue time.
            for device in dict.fromkeys(self.devices):
                if torch.device(device).type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
        except Exception as error:
            self.record("phase_error", name, duration_s=time.monotonic() - start,
                        error_type=type(error).__name__, error=str(error),
                        traceback=traceback.format_exc())
            raise
        else:
            self.record("phase_end", name, duration_s=time.monotonic() - start)
