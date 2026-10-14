"""Sample all physical GPU boards; keep NVML observations separate from Torch counters."""
import csv
import io
import math
import subprocess
from threading import Event, Thread
import time


def parse_gpu_samples(csv_text, timestamp):
    """nvidia-smi noheader,nounits: UUID,index,utilization %,memory MiB,power W."""
    def number(value):
        try:
            result = float(value.strip())
            return result if math.isfinite(result) else None
        except ValueError:
            return None

    samples = []
    for row in csv.reader(io.StringIO(csv_text)):
        if len(row) != 5:
            continue
        uuid, index, utilization, memory, power = (value.strip() for value in row)
        try:
            physical_index = int(index)
        except ValueError:
            continue
        memory = number(memory)
        samples.append({"timestamp": timestamp, "uuid": uuid, "physical_index": physical_index,
                        "utilization_gpu_percent": number(utilization),
                        "memory_used_bytes": None if memory is None else memory * 1024 ** 2,
                        "power_w": number(power)})
    return samples


def summarize_gpu_samples(samples, window_start, window_end, max_gap_seconds=None):
    """Integrate clipped trapezoids between measured endpoints, with no extrapolation."""
    if window_end < window_start:
        raise ValueError("GPU sampling window end must follow its start")
    duration = window_end - window_start
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample["uuid"], []).append(sample)
    gpus = []
    for uuid, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["timestamp"])
        inside = [row for row in rows if window_start <= row["timestamp"] <= window_end]
        left = [row for row in rows if row["timestamp"] < window_start]
        right = [row for row in rows if row["timestamp"] > window_end]
        bounds = left[-1:] + inside + right[:1]
        memory = [row["memory_used_bytes"] for row in inside if row["memory_used_bytes"] is not None]
        utilization = [row["utilization_gpu_percent"] for row in inside
                       if row["utilization_gpu_percent"] is not None]
        energy = power_coverage = sample_coverage = 0.0
        missing_power = False
        skipped_gap_seconds = 0.
        for a, b in zip(rows, rows[1:]):
            begin, end = max(window_start, a["timestamp"]), min(window_end, b["timestamp"])
            if end <= begin:
                continue
            if max_gap_seconds is not None and b["timestamp"] - a["timestamp"] > max_gap_seconds:
                skipped_gap_seconds += end - begin
                missing_power = True
                continue
            sample_coverage += end - begin
            if a["power_w"] is None or b["power_w"] is None:
                missing_power = True
                continue
            interval = b["timestamp"] - a["timestamp"]
            slope = (b["power_w"] - a["power_w"]) / interval
            p_begin = a["power_w"] + slope * (begin - a["timestamp"])
            p_end = a["power_w"] + slope * (end - a["timestamp"])
            energy += (p_begin + p_end) * .5 * (end - begin)
            power_coverage += end - begin
        power_status = ("unavailable" if not power_coverage else
                        "partial" if missing_power else "available")
        gpus.append({"uuid": uuid, "physical_index": rows[0]["physical_index"],
                     "sample_count": len(inside), "samples": bounds,
                     "skipped_poll_gap_seconds": skipped_gap_seconds,
                     "coverage_seconds": sample_coverage,
                     "coverage_fraction": sample_coverage / duration if duration else 0.0,
                     "utilization_gpu_percent_mean": sum(utilization) / len(utilization) if utilization else None,
                     "physical_memory_used_bytes_mean": sum(memory) / len(memory) if memory else None,
                     "physical_memory_used_bytes_max": max(memory) if memory else None,
                     "power": {"status": power_status,
                               "energy_joules": energy if power_coverage else None,
                               "mean_power_w": energy / power_coverage if power_coverage else None,
                               "coverage_seconds": power_coverage,
                               "coverage_fraction": power_coverage / duration if duration else 0.0}})
    return {"status": "available" if gpus else "unavailable",
            "source": "nvidia-smi", "scope": "physical_whole_gpu",
            "measurement_note": "Physical board usage includes other processes and non-Torch allocations; "
                                "energy covers only sampled intervals within the benchmark window.",
            "max_integrated_poll_gap_seconds": max_gap_seconds,
            "window_start": window_start, "window_end": window_end, "window_seconds": duration,
            "gpus": sorted(gpus, key=lambda gpu: gpu["physical_index"])}


class GPUResourceSampler:
    """Small stdlib poller independent of CUDA_VISIBLE_DEVICES and PyTorch."""
    def __init__(self, interval_seconds=.25):
        if interval_seconds <= 0:
            raise ValueError("GPU sample interval must be positive")
        self.interval_seconds = interval_seconds
        self.samples = []
        self.error = None
        self.stop = Event()
        self.thread = None

    def _sample(self):
        begin = time.perf_counter()
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=uuid,index,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2, check=True)
            self.samples.extend(parse_gpu_samples(result.stdout, (begin + time.perf_counter()) * .5))
        except (OSError, subprocess.SubprocessError) as error:
            self.error = f"{type(error).__name__}: {error}"

    def _run(self):
        deadline = time.perf_counter() + self.interval_seconds
        while not self.stop.wait(max(0.0, deadline - time.perf_counter())):
            self._sample()
            deadline = max(deadline + self.interval_seconds, time.perf_counter())

    def start(self):
        self._sample()
        if self.samples:
            self.thread = Thread(target=self._run, name="physical-gpu-sampler", daemon=True)
            self.thread.start()

    def finish(self, window_start, window_end):
        self.stop.set()
        if self.thread:
            self.thread.join()
        # A final endpoint can cover the end of the window without extrapolating.
        self._sample()
        report = summarize_gpu_samples(self.samples, window_start, window_end,
                                       max_gap_seconds=max(1., 4 * self.interval_seconds))
        report["sample_interval_seconds"] = self.interval_seconds
        if self.error:
            report["error"] = self.error
        return report
