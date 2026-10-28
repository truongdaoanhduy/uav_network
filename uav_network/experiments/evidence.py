"""Bounded, redacted evidence and best-effort channels; no learner dependencies."""
import errno
import json
from itertools import islice
import math
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
import sys


SENSITIVE_FIELDS = ("api_key", "password", "secret", "token", "authorization", "access_key", "private_key", "credential")


def secret_values(config=None):
    def sensitive_name(name):
        name = name.lower()
        return name == "key" or name.endswith("_key") or any(word in name for word in SENSITIVE_FIELDS)
    values = [value for name, value in os.environ.items() if value and sensitive_name(name)]
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and item and sensitive_name(key):
                    values.append(item)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    if config is not None:
        try:
            path = Path(config.tracking.credentials_file).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            collect(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return sorted(set(values), key=len, reverse=True)


def redact(text, secrets):
    for value in secrets:
        text = text.replace(value, "[REDACTED]")
    return text


def json_safe(value, secrets=(), depth=0):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else ("NaN" if math.isnan(number) else str(number))
    if isinstance(value, str):
        text = redact(value, secrets)
        return text if len(text) <= 8192 else text[:8192] + "<truncated>"
    if depth >= 6:
        return "<depth limit>"
    if isinstance(value, dict):
        result = {}
        key_secrets = [v for v in secrets if len(v) >= 8]
        for key, item in islice(value.items(), 128):
            name = str(key) if isinstance(key, (str, Integral, Real, bool)) else "<" + type(key).__name__ + ">"
            hidden = any(s in name.lower() for s in SENSITIVE_FIELDS)
            result[redact(name, key_secrets)[:256]] = "[REDACTED]" if hidden else json_safe(item, secrets, depth + 1)
        if len(value) > 128:
            result["_truncated_items"] = len(value) - 128
        return result
    if isinstance(value, (tuple, list)):
        result = [json_safe(v, secrets, depth + 1) for v in value[:8192]]
        if len(value) > 8192:
            result.append({"_truncated_items": len(value) - 8192})
        return result
    return "<" + type(value).__name__ + ">"


def print_evidence(record):
    try:
        print(json.dumps(record, allow_nan=False), flush=True)
    except (OSError, ValueError, TypeError):
        pass


def write_evidence(path, record, *, append=False):
    try:
        with Path(path).open("a" if append else "w") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError) as error:
        print_evidence({"evidence_write_error": {"file": Path(path).name,
                       "error_type": type(error).__name__, "errno": getattr(error, "errno", None)}})
        return False
    return True


def resource_snapshot(output):
    result = {"disk_free_bytes": None, "process_rss_bytes": None,
              "process_peak_rss_bytes": None, "host_available_bytes": None, "cuda": []}
    try:
        result["disk_free_bytes"] = shutil.disk_usage(output).free
    except OSError:
        pass
    try:
        import resource
        result["process_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    except (ImportError, OSError, ValueError):
        pass
    for path, prefix, key in (("/proc/self/status", "VmRSS:", "process_rss_bytes"),
                              ("/proc/meminfo", "MemAvailable:", "host_available_bytes")):
        try:
            for line in Path(path).read_text().splitlines():
                if line.startswith(prefix):
                    result[key] = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError):
            pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                try:
                    result["cuda"].append({"logical_index": index,
                        "allocated_bytes": torch.cuda.memory_allocated(index),
                        "reserved_bytes": torch.cuda.memory_reserved(index),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(index)})
                except Exception as error:
                    result["cuda"].append({"logical_index": index, "error_type": type(error).__name__})
    except Exception as error:
        result["cuda_sampling_error"] = type(error).__name__
    return result


def exception_chain(error):
    chain, seen = [], set()
    while error is not None and id(error) not in seen and len(chain) < 8:
        chain.append(error); seen.add(id(error))
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return chain


def exception_category(chain):
    for error in chain:
        if type(error).__name__ == "OutOfMemoryError" and type(error).__module__.startswith("torch"):
            return "torch_allocation"
        if isinstance(error, MemoryError):
            return "memory_allocation"
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            return "storage_exhausted"
        if getattr(error, "code", None) in (401, 403):
            return "service_access_denied"
        if isinstance(error, FloatingPointError):
            return "nonfinite_numeric"
        if isinstance(error, TimeoutError):
            return "timeout"
        if isinstance(error, ConnectionError):
            return "connection"
        if isinstance(error, FileNotFoundError):
            return "missing_file"
        if isinstance(error, PermissionError):
            return "permission"
        if isinstance(error, ImportError):
            return "dependency"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return "interrupted"
    return "unknown_exception"
