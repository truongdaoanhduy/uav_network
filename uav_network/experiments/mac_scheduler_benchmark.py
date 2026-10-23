"""Reproducible Warp MAC Scheduler Gen1/Gen2 throughput benchmark."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import warp as wp

from ..configuration import load_config
from ..environment.gpu.geometry import sense
from ..environment.gpu.vector import WarpVectorEnv
from ..communication.gpu.mac import packet_step, packet_step_gen2, packet_step_framefree, sense_reports


CORPORA = {"low_contention", "active_streaming"}
SCHEDULERS = {"gen1", "gen2", "framefree"}


def _commit():
    value = os.environ.get("UAV_GIT_COMMIT")
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
            cwd=Path(__file__).resolve().parents[2], timeout=2).strip()
    except Exception:
        return "local"


def _sync(vector):
    vector._bind_stream()
    wp.synchronize_device(vector.wp_device)


def _fixed_actions(vector, corpus):
    continuous = torch.zeros((vector.num_envs, vector.p.n, 3), dtype=torch.float32,
                             device=vector.device)
    destination = torch.zeros((vector.num_envs, vector.p.n), dtype=torch.int64,
                              device=vector.device)
    if corpus == "active_streaming":
        destination[:, 0] = vector.p.n
        continuous[:, 0, 2] = 1.0
    return {"continuous": continuous, "destination": destination}


def _prepare_corpus(vector, corpus, seed):
    vector.reset(seed=seed)
    if corpus == "active_streaming":
        # Force target 0 onto UAV 0 in every world, then use the real sensing/report
        # kernels. All other Scenario V3 parameters remain unchanged.
        positions = vector.s.positions.numpy()
        targets = vector.s.targets.numpy()
        targets[:, 0, :] = positions[:, 0, :]
        vector.s.targets.assign(targets)
        vector._launch(sense, (vector.num_envs, vector.p.n, vector.p.targets_count),
                       vector.s, vector.p)
        vector._launch(sense_reports, vector.num_envs, vector.s, vector.p)
        vector._publish(vector._buffers[vector._buffer])
        _sync(vector)


def _scientific_metrics(vector):
    counts = wp.to_torch(vector.s.packet_counts).detach().to(device="cpu").sum(0)
    delivered = wp.to_torch(vector.s.delivered).detach().to(device="cpu").sum()
    reports = wp.to_torch(vector.s.report_delivery_count).detach().to(device="cpu").sum()
    energy = wp.to_torch(vector.s.communication_energy).detach().to(device="cpu").sum()
    tx_energy = wp.to_torch(vector.s.tx_energy).detach().to(device="cpu").sum()
    rx_energy = wp.to_torch(vector.s.rx_energy).detach().to(device="cpu").sum()
    return {
        "generated_packets": int(counts[0]), "received_packets": int(counts[1]),
        "phy_successes": int(counts[2]), "phy_failures": int(counts[3]),
        "collisions": int(counts[4]), "energy_blocked_transmissions": int(counts[5]),
        "payload_bytes_received": int(counts[6]), "expired_packets": int(counts[7]),
        "reports_delivered": int(reports), "targets_delivered": int(delivered),
        "communication_energy_j": float(energy), "tx_energy_j": float(tx_energy),
        "rx_energy_j": float(rx_energy),
    }


def _memory(vector):
    derived = sum(a.size * wp.types.type_size_in_bytes(a.dtype)
                  for a in (getattr(vector, "_frame_free_words", None),
                            getattr(vector, "_frame_reclaim_words", None)) if a is not None)
    if vector.device.type != "cuda":
        return {"allocated_bytes": int(vector.allocated_bytes), "derived_bitmap_bytes": int(derived),
                "peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "device_total_bytes": 0}
    props = torch.cuda.get_device_properties(vector.device)
    return {
        "allocated_bytes": int(vector.allocated_bytes),
        "derived_bitmap_bytes": int(derived),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(vector.device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(vector.device)),
        "device_total_bytes": int(props.total_memory),
    }


def benchmark_mac_scheduler(config, *, num_envs: int, steps: int, scheduler: str, corpus: str) -> dict:
    if scheduler not in SCHEDULERS:
        raise ValueError("scheduler must be gen1, gen2, or framefree")
    if corpus not in CORPORA:
        raise ValueError("corpus must be low_contention or active_streaming")
    if int(num_envs) < 1:
        raise ValueError("num_envs must be positive")
    if int(steps) < 1:
        raise ValueError("steps must be positive")

    selected = deepcopy(config)
    selected.runtime.num_envs = int(num_envs)
    selected.tracking.enabled = False
    selected = selected.validate()
    device = selected.runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vector = WarpVectorEnv(selected, device=device, packet_scheduler=scheduler)
    packet_seconds = 0.0
    original_launch = vector._launch
    selected_kernel = (packet_step_gen2 if scheduler == "gen2" else
                       packet_step_framefree if scheduler == "framefree" else packet_step)

    def measured_launch(kernel, dim, *args):
        nonlocal packet_seconds
        if kernel is selected_kernel:
            _sync(vector)
            started = time.perf_counter()
            original_launch(kernel, dim, *args)
            _sync(vector)
            packet_seconds += time.perf_counter() - started
        else:
            original_launch(kernel, dim, *args)

    try:
        # Compile all kernels outside the measured region.
        _prepare_corpus(vector, corpus, selected.runtime.seed)
        warm_actions = _fixed_actions(vector, corpus)
        vector.step(warm_actions)
        _sync(vector)

        # Reset to the same deterministic mission after JIT warmup.
        _prepare_corpus(vector, corpus, selected.runtime.seed)
        actions = _fixed_actions(vector, corpus)
        vector._launch = measured_launch
        packet_seconds = 0.0
        if vector.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(vector.device)
        _sync(vector)
        started = time.perf_counter()
        executed = 0
        for _ in range(int(steps)):
            _, _, done, _, _ = vector.step(actions)
            executed += 1
            if bool(done.any()):
                # Keep the benchmark deterministic and bounded rather than auto-resetting
                # a subset into a different corpus.
                break
        _sync(vector)
        collector_seconds = time.perf_counter() - started
        errors = wp.to_torch(vector.s.errors).detach().cpu()
        terminated = wp.to_torch(vector.s.terminated).detach().cpu()
        joint = int(num_envs) * executed
        return {
            "scheduler": scheduler,
            "commit": _commit(),
            "device": vector.device.type,
            "num_envs": int(num_envs),
            "steps": int(executed),
            "requested_steps": int(steps),
            "corpus": corpus,
            "collector_seconds": float(collector_seconds),
            "packet_step_seconds": float(packet_seconds),
            "packet_step_fraction": float(packet_seconds / collector_seconds) if collector_seconds else 0.0,
            "joint_transitions_per_second": float(joint / collector_seconds),
            "scheduler_metrics": vector.scheduler_metrics(),
            "scientific_metrics": _scientific_metrics(vector),
            "gpu_memory": _memory(vector),
            "simulator_error_count": int((errors != 0).sum()),
            "terminated_worlds": int((terminated != 0).sum()),
        }
    finally:
        vector._launch = original_launch
        vector.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--scheduler", choices=sorted(SCHEDULERS), required=True)
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.runtime.device = config.runtime.replay_device = args.device
    result = benchmark_mac_scheduler(config, num_envs=args.num_envs, steps=args.steps,
                                     scheduler=args.scheduler, corpus=args.corpus)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
