"""Low-frequency training memory watermarks, not an OOM-safety guarantee.

PyTorch CUDA peaks include its allocator (including backward and optimizer
temporary tensors) but not Warp/external CUDA allocations. Driver free memory
covers all clients only at sampling time. Linux cgroup memory.peak (when
available) captures the process group high-water mark even between samples.
"""
from pathlib import Path
import resource

import torch

GIB = float(1024 ** 3)

# Cumulative minimum sampled physical free VRAM per logical CUDA device.
# Not a device high-water mark: it can miss short-lived external Warp kernels.
_DRIVER_MIN_FREE_GIB = {}


def reset_driver_watermarks():
    """Start a new run's sampled-driver history without resetting CUDA allocator."""
    _DRIVER_MIN_FREE_GIB.clear()



def _read_number(path):
    try:
        raw = Path(path).read_text().strip()
        return int(raw) if raw.isdigit() else None
    except (OSError, ValueError):
        return None


def host_memory_metrics():
    result = {}
    hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if hwm:
        # Linux ru_maxrss is KiB, unlike macOS bytes.
        result["memory/host_process_peak_rss_gib"] = hwm * 1024 / GIB
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["memory/host_process_rss_gib"] = int(line.split()[1]) * 1024 / GIB
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                result["memory/host_system_available_gib"] = int(line.split()[1]) * 1024 / GIB
                break
    except (OSError, ValueError, IndexError):
        pass
    # cgroup OOM event counters are system/cgroup-wide, not necessarily the
    # current training process; record them as evidence, not causal proof.
    for memory_events in ("/sys/fs/cgroup/memory.events",
                          "/sys/fs/cgroup/memory/memory.oom_control"):
        try:
            for line in Path(memory_events).read_text().splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[0] in {"oom", "oom_kill", "oom_group_kill"}:
                    result[f"memory/cgroup_{fields[0]}_events"] = int(fields[1])
            break
        except (OSError, ValueError):
            pass
    for prefix, limit_key, current_key, peak_key in (
        ("/sys/fs/cgroup", "memory.max", "memory.current", "memory.peak"),
        ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
    ):
        limit = _read_number(f"{prefix}/{limit_key}")
        current = _read_number(f"{prefix}/{current_key}")
        if limit is None or current is None or limit >= (1 << 60):
            continue
        result["memory/cgroup_limit_gib"] = limit / GIB
        result["memory/cgroup_current_gib"] = current / GIB
        result["memory/cgroup_headroom_now_gib"] = max(0, limit - current) / GIB
        peak = _read_number(f"{prefix}/{peak_key}")
        if peak is not None:
            result["memory/cgroup_peak_gib"] = peak / GIB
        break
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["cpu/process_user_seconds"] = float(usage.ru_utime)
        result["cpu/process_system_seconds"] = float(usage.ru_stime)
    except (OSError, ValueError):
        pass
    return result


def cuda_memory_metrics():
    result = {}
    if not torch.cuda.is_available():
        return result
    for i in range(torch.cuda.device_count()):
        try:
            prefix = f"memory/cuda{i}_"
            # Exact high-water marks for PyTorch allocator, including
            # transient grad/Adam buffers between W&B system polls.
            result[prefix + "torch_peak_allocated_gib"] = torch.cuda.max_memory_allocated(i) / GIB
            result[prefix + "torch_peak_reserved_gib"] = torch.cuda.max_memory_reserved(i) / GIB
            result[prefix + "torch_now_reserved_gib"] = torch.cuda.memory_reserved(i) / GIB
            # Driver reading includes Warp, CUDA contexts and non-Torch users;
            # this value is only an instantaneous sample, NOT a peak.
            free, total = torch.cuda.mem_get_info(i)
            result[prefix + "driver_free_now_gib"] = free / GIB
            result[prefix + "driver_total_gib"] = total / GIB
            _DRIVER_MIN_FREE_GIB[i] = min(_DRIVER_MIN_FREE_GIB.get(i, free / GIB), free / GIB)
            result[prefix + "driver_min_free_observed_gib"] = _DRIVER_MIN_FREE_GIB[i]
            result[prefix + "driver_peak_used_observed_gib"] = total / GIB - _DRIVER_MIN_FREE_GIB[i]
            result[prefix + "driver_peak_used_observed_fraction"] = (
                (total / GIB - _DRIVER_MIN_FREE_GIB[i]) / max(total / GIB, 1e-9))
            # Distinguish a warning from a hard memory limit. User decides.
            result[f"diagnostics/cuda{i}_physical_free_under_2gib"] = int(free / GIB < 2.0)
        except (RuntimeError, AssertionError):
            # Do not crash a running training job just for instrumentation.
            continue
    return result


def training_memory_metrics():
    host = host_memory_metrics()
    cuda = cuda_memory_metrics()
    headroom = host.get("memory/cgroup_headroom_now_gib",
                        host.get("memory/host_system_available_gib"))
    if headroom is not None:
        host["diagnostics/host_headroom_under_3gib"] = int(headroom < 3.0)
    return {**host, **cuda}
