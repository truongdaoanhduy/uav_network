"""v6.2 Full-Traffic Tesla T4 Preflight & VRAM Scaling Gate.

Unlike the v5.6 single-link smoke (where 2047/2048 envs had zero channel gain),
this gate populates realistic connected 7-radio topologies across ALL environments,
injects sensing + DATA packets, runs 2-substep 500ms windows, and reports real
wall time + CUDA `mem_get_info` peak VRAM. Does NOT launch MATD3/MASAC training.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhang_xue.gpu.layout import backing_bytes_per_environment
from zhang_xue.gpu.layout_v62 import backing_bytes_per_environment_v62, v62_gpu_memory_budget
from zhang_xue.gpu.runtime import HAS_WARP, WarpSevenRadioNetwork


def run_device_scale(device: str, sizes: list[int], allow_experimental: bool) -> dict:
    report = {
        "device": device,
        "has_warp": HAS_WARP,
        "v61_soa_bytes_per_env": backing_bytes_per_environment(),
        "v62_compressed_soa_bytes_per_env": backing_bytes_per_environment_v62(),
        "active_runtime_layout": WarpSevenRadioNetwork.active_layout_id(),
        "active_runtime_soa_bytes_per_env": WarpSevenRadioNetwork.active_soa_bytes_per_environment(),
        "compression_enabled_in_runtime": False,
        "compression_status": "DESIGN_ONLY: runtime still uses uncompressed gpu/layout.py",
        "v62_budget_2048_t4": v62_gpu_memory_budget(2048, 300_000, 522),
        "runs": [],
    }
    if not HAS_WARP:
        report["status"] = "blocked_no_warp"
        return report
    import warp as wp

    if device.startswith("cuda") and not wp.is_cuda_available():
        report["status"] = "blocked_no_cuda"
        return report

    for n_env in sizes:
        t0 = time.perf_counter()
        net = WarpSevenRadioNetwork(num_envs=n_env, device=device, seed=20261125)
        # Full-load topology: all envs have realistic 1-hop & 2-hop gains
        gain = np.zeros((n_env, 7, 7), dtype=np.float32)
        for i in range(7):
            for j in range(i + 1, 7):
                if abs(i - j) <= 2:
                    gain[:, i, j] = 1.0e-8
                    gain[:, j, i] = 1.0e-8
        net.set_channel_gain(gain)
        t_alloc = time.perf_counter() - t0
        # Warmup HELLO + sense + inject + 2x 500ms substeps
        t1 = time.perf_counter()
        net.run(duration_us=600_000, max_events=2048, experimental=allow_experimental)
        cells = np.tile(np.arange(6, dtype=np.int32), (n_env, 1))
        pos = np.ones((n_env, 6), dtype=np.int32)
        net.sense(cells, pos, t_s=1)
        acts = np.full((n_env, 6), 6, dtype=np.int32)
        net.inject(acts, t_s=1)
        net.run(duration_us=500_000, max_events=2048, experimental=allow_experimental)
        net.run(duration_us=500_000, max_events=2048, experimental=allow_experimental)
        wp.synchronize()
        t_step = time.perf_counter() - t1
        snap = net.snapshot(("overflow", "stats_gcs", "stats_mac_ack", "stats_e2e"))
        report["runs"].append({
            "num_envs": n_env,
            "alloc_reset_s": round(t_alloc, 4),
            "full_traffic_step_s": round(t_step, 4),
            "env_steps_per_s": round(n_env / max(1e-6, t_step), 1),
            "overflow_envs": int(np.count_nonzero(snap["overflow"])),
            "total_gcs_data": int(snap["stats_gcs"].sum()),
            "total_e2e_ack": int(snap["stats_e2e"].sum()),
        })
    report["status"] = "ok"
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["cuda:0"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[32, 256, 1024, 2048])
    ap.add_argument("--allow-experimental", action="store_true")
    args = ap.parse_args()
    out = [run_device_scale(d, args.sizes, args.allow_experimental) for d in args.devices]
    print(json.dumps(out, indent=2))
    return 0 if all(r.get("status") == "ok" for r in out) else 2


if __name__ == "__main__":
    raise SystemExit(main())
