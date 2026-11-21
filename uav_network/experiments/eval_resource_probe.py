"""Best-effort physical GPU and host CPU/RAM samples for post-training Eval/Test.

Physical GPU indices come from nvidia-smi (both CUDA_VISIBLE_DEVICES workers
are separate); none of these metrics are inferred from training memory.
Absent/failed sensors are omitted, never silently replaced with zero.
"""
from __future__ import annotations

import math
import subprocess


def parse_nvidia_smi(output: str) -> dict[str, float]:
    """Parse physical index, GPU%, mem-controller%, used MiB, total MiB."""
    values = {}
    for line in output.splitlines():
        fields=[p.strip() for p in line.split(",")]
        if len(fields)!=5:
            continue
        try:
            index=int(fields[0])
            util, memory_util, used_mib, total_mib=(float(p) for p in fields[1:])
        except (TypeError,ValueError):
            continue
        if index < 0 or index > 31 or not all(math.isfinite(x) and x >= 0
                for x in (util,memory_util,used_mib,total_mib)):
            continue
        base=f"eval_resources/gpu{index}_"
        values[base+"util_percent"]=util
        values[base+"memory_controller_util_percent"]=memory_util
        values[base+"vram_used_gib"]=used_mib/1024.0
        values[base+"vram_total_gib"]=total_mib/1024.0
    return values


def snapshot() -> dict[str,float]:
    values={}
    try:
        # Instantaneous device counters from NVML, NOT the training allocator.
        proc=subprocess.run([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],capture_output=True,text=True,check=True,timeout=7)
        values.update(parse_nvidia_smi(proc.stdout))
    except (FileNotFoundError,OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
        pass
    try:
        import psutil
        cpu=float(psutil.cpu_percent(interval=0.1))
        ram=psutil.virtual_memory()
        if math.isfinite(cpu):
            values["eval_resources/cpu_system_percent"]=cpu
        values["eval_resources/ram_system_used_gib"]=float(ram.used)/(1024**3)
        values["eval_resources/ram_system_total_gib"]=float(ram.total)/(1024**3)
        values["eval_resources/ram_system_percent"]=float(ram.percent)
    except (ImportError,OSError,AttributeError,ValueError):
        pass
    return values
