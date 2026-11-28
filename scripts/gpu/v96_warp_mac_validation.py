#!/usr/bin/env python3
"""V96 Gate S12: Direct Warp GPU single-hop DCF saturation validation.

Compares WarpCompressedRadioNetworkV67 directly in BOTH cw_inclusive_upper=1
(IEEE 802.11 [0, CW]) and cw_inclusive_upper=0 (UavNetSim-v1 [0, CW-1]) modes
against exact finite-retry Bianchi & Tinnirello (2005) with total_attempts=5
(matching events_v67.py:322 ret >= max_retx=5).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src"), str(ROOT / "scripts/gpu")]

from v94_mac_analytic_validation import solve_finite_retry_bianchi


def run_warp_single_hop_saturation(
    n_uavs: int,
    *,
    cw_inclusive_upper: int,
    num_envs: int = 32,
    warmup_chunks: int = 2,
    measure_chunks: int = 40,
    chunk_us: int = 2_000,
    device: str = "cuda:0",
    seed: int = 20261127,
) -> dict[str, float]:
    import warp as wp
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    from zhang_xue.gpu.state_v67 import NEVER_US

    net = WarpCompressedRadioNetworkV67(
        num_envs=num_envs,
        num_uavs=n_uavs,
        grid_w=2,
        enable_data_mule=False,
        enable_peer=False,
        enable_peer_telemetry=False,
        q_cap=64,
        cw_inclusive_upper=cw_inclusive_upper,
        device=device,
        seed=seed,
    )
    gcs = n_uavs

    def _put(name: str, arr: np.ndarray) -> None:
        a = net.data[name]
        wp.copy(a, wp.array(np.asarray(arr), dtype=a.dtype, device=device))

    gain = np.full((num_envs, n_uavs + 1, n_uavs + 1), 1.0e-4, dtype=np.float32)
    gain[:, np.arange(n_uavs + 1), np.arange(n_uavs + 1)] = 0.0
    net.set_channel_gain(gain)
    _put("hello_due", np.full((num_envs, n_uavs + 1), NEVER_US, dtype=np.int32))
    _put("hello_pending", np.zeros((num_envs, n_uavs + 1), dtype=np.uint8))

    rh = np.full((num_envs, n_uavs + 1, n_uavs + 1), -1, dtype=np.int8)
    rd = np.full((num_envs, n_uavs + 1, n_uavs + 1), 127, dtype=np.int8)
    re = np.zeros((num_envs, n_uavs + 1, n_uavs + 1), dtype=np.int32)
    rh[:, :n_uavs, gcs] = gcs
    rd[:, :n_uavs, gcs] = 1
    re[:, :n_uavs, gcs] = NEVER_US
    # Bidirectional E2E acknowledgment routes: otherwise outstanding pending windows stall.
    rh[:, gcs, :n_uavs] = np.arange(n_uavs, dtype=np.int8)
    rd[:, gcs, :n_uavs] = 1
    re[:, gcs, :n_uavs] = NEVER_US

    def _maintain_saturation(step_idx: int) -> None:
        _put("route_hop", rh)
        _put("route_dist", rd)
        _put("route_expiry", re)
        # Preserve GCS E2E-ACK frames to clear per-destination pending windows.
        net.sense_batch(np.zeros((num_envs, n_uavs), dtype=np.int32), np.ones((num_envs, n_uavs), dtype=np.int32), step_idx)
        net.inject_compact3(np.ones((num_envs, n_uavs), dtype=np.int32), np.zeros((num_envs, n_uavs, 2), dtype=np.float32))

    for k in range(1, warmup_chunks + 1):
        _maintain_saturation(k)
        net.step(duration_us=chunk_us, max_events=8192)

    ack0 = net.data["stats_mac_ack"].numpy().astype(np.int64).copy()
    ret0 = net.data["stats_retry"].numpy().astype(np.int64).copy()
    drp0 = net.data["stats_drop"].numpy().astype(np.int64).copy()
    now0 = net.data["now"].numpy().astype(np.int64).copy()

    for k in range(warmup_chunks + 1, warmup_chunks + measure_chunks + 1):
        _maintain_saturation(k)
        net.step(duration_us=chunk_us, max_events=8192)

    ack1 = net.data["stats_mac_ack"].numpy().astype(np.int64)
    ret1 = net.data["stats_retry"].numpy().astype(np.int64)
    drp1 = net.data["stats_drop"].numpy().astype(np.int64)
    now1 = net.data["now"].numpy().astype(np.int64)

    d_ack = int(np.sum(ack1 - ack0))
    d_ret = int(np.sum(ret1 - ret0))
    d_drp = int(np.sum(drp1 - drp0))
    d_us = float(np.sum(now1 - now0))
    n_att = d_ack + d_ret
    p_warp = d_ret / max(1, n_att)
    s_warp = (d_ack * 312.0) / max(1.0, d_us)
    se_p = math.sqrt(p_warp * (1.0 - p_warp) / max(1, n_att))
    return {
        "n_uavs": n_uavs,
        "cw_inclusive_upper": cw_inclusive_upper,
        "p_warp": float(p_warp),
        "se_p": float(se_p),
        "throughput_airtime_frac_warp": float(s_warp),
        "attempts": n_att,
        "acks": d_ack,
        "retries": d_ret,
        "drops": d_drp,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default="2,4,6")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    directional_ok = True
    nonzero_traffic = True
    statistically_clear_direction = True
    for n in [int(x) for x in args.nodes.split(",") if x.strip()]:
        an_1 = solve_finite_retry_bianchi(n, inclusive=True, total_attempts=5)
        an_0 = solve_finite_retry_bianchi(n, inclusive=False, total_attempts=5)
        w_1 = run_warp_single_hop_saturation(n, cw_inclusive_upper=1, num_envs=args.envs, measure_chunks=args.chunks, device=args.device)
        w_0 = run_warp_single_hop_saturation(n, cw_inclusive_upper=0, num_envs=args.envs, measure_chunks=args.chunks, device=args.device)
        shift = w_0["p_warp"] - w_1["p_warp"]
        se_shift = math.sqrt(w_0["se_p"] ** 2 + w_1["se_p"] ** 2)
        if shift <= 0:
            directional_ok = False
        if shift <= 1.96 * se_shift:
            statistically_clear_direction = False
        if min(w_0["attempts"], w_1["attempts"]) <= 0:
            nonzero_traffic = False
        rows.append({
            "n_uavs": n,
            "bianchi_att5_ieee_0_CW": an_1,
            "bianchi_att5_uavnetsim_0_CWm1": an_0,
            "warp_ieee_0_CW": w_1,
            "warp_uavnetsim_0_CWm1": w_0,
            "warp_cw_shift_p": float(shift),
            "warp_cw_shift_approx_ci95": [float(shift - 1.96 * se_shift), float(shift + 1.96 * se_shift)],
            "bianchi_cw_shift_p": float(an_0["p_collision"] - an_1["p_collision"]),
        })
    out = {
        "status": ("TRAFFIC_VALID_PARITY_UNVERIFIED" if nonzero_traffic else "INVALID_NO_MAC_TRAFFIC"),
        "s12_freeze_certified": False,
        "nonzero_traffic": nonzero_traffic,
        "directional_point_estimate_all_nodes": directional_ok,
        "direction_statistically_clear_at_95pct_all_nodes": statistically_clear_direction,
        "limitations": "Synthetic application offered load; not IEEE saturated DCF, not SimPy event parity",
        "device": args.device,
        "scope": "Direct Warp GPU finite offered-load comparison (NOT saturated DCF) comparing cw_inclusive_upper=1 ([0,CW]) vs 0 ([0,CW-1]) against finite-retry Bianchi (total_attempts=5)",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if nonzero_traffic else 2


if __name__ == "__main__":
    raise SystemExit(main())
