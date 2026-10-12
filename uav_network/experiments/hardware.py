"""Inspect usable hardware without assuming which RTX 6000 the host has."""
import os
import platform
import torch


def hardware_info():
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            devices.append({"index": i, "name": properties.name,
                            "memory_gib": properties.total_memory / 1024 ** 3,
                            "compute_capability": [properties.major, properties.minor]})
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"platform": platform.platform(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "available_cpus": affinity,
            "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "cuda_devices": devices}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
