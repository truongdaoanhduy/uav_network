#!/usr/bin/env python3
"""Controlled CPU or real-CUDA Warp PHY vs NumPy reference (not RL/packet parity)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]


def run_phy_gate(device: str = "cuda:0", *, q_cap: int = 64) -> dict:
    try:
        import warp as wp
    except ImportError as exc:
        return {"status": "BLOCKED_NO_WARP", "reason": str(exc), "gpu_executed": False}
    if device.startswith("cuda:"):
        try:
            import torch
            cuda_ok = torch.cuda.is_available() and len(wp.get_cuda_devices()) > int(device.split(':')[1])
        except (ImportError, ValueError, RuntimeError):
            cuda_ok = False
        if not cuda_ok:
            return {"status": "BLOCKED_CUDA_REQUIRED", "device": device, "gpu_executed": False}
    elif device != "cpu":
        raise ValueError("device must be cpu or cuda:N")

    from zhang_xue.communication.common import CsmaProtocolConfig
    from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67

    positions = np.zeros((2, 3, 3), dtype=np.float32)
    positions[:, 0, :] = (100., 100., 50.)
    positions[:, 1, :] = (200., 100., 50.)
    positions[:, 2, :] = (100., 200., 50.)
    proto = CsmaProtocolConfig(difs_us=70, ack_timeout_us=200)
    radio = WarpCompressedRadioNetworkV67(
        2, num_uavs=2, grid_w=20, q_cap=q_cap, device=device,
        protocol=proto, seed=46,
    )
    cases = (
        ("los", (), (), "binary_nlos"),
        ("building_nlos", ((150., 100., 15., 100.),), (), "binary_nlos"),
        ("blackout", ((150., 100., 15., 100.),), ((2, 2),), "binary_nlos"),
        ("multi_obstacle", ((125., 100., 10., 100.),(150., 100., 10., 100.),
                             (175., 100., 10., 100.)), (), "multi_obstacle"),
        ("rf_opaque", ((150., 100., 15., 100.),), (), "rf_opaque"),
    )
    results = {}
    for label, buildings, blackout, mode in cases:
        radio.update_phy_from_positions(positions, buildings, blackout, 0, 50.0,
                                        obstacle_model=mode, per_building_loss_db=12.)
        got = radio.data["gain"].numpy()  # Allowed in a one-off validation, NOT a hot loop
        expected = compute_channel_gain_v67_np(positions, buildings, blackout, 0, 20, 50.0,
                                                 obstacle_model=mode, per_building_loss_db=12.)
        np.testing.assert_allclose(got, expected, atol=1e-15, rtol=2e-4)
        results[label] = {"max_abs_diff": float(np.max(np.abs(got - expected))),
                          "gain_u0_u1": float(got[0, 0, 1]),
                          "gain_u0_u2": float(got[0, 0, 2])}
    assert 0 < results["building_nlos"]["gain_u0_u1"] < results["los"]["gain_u0_u1"]
    assert results["blackout"]["gain_u0_u1"] == 0.0
    assert results["blackout"]["gain_u0_u2"] == 0.0
    assert results["rf_opaque"]["gain_u0_u1"] == 0.0
    assert results["multi_obstacle"]["gain_u0_u1"] > 0.0
    return {"status": "PHY_REFERENCE_PASS_NOT_FULL_RL", "device": device,
            "gpu_executed": device.startswith("cuda:"), "q_cap": q_cap,
            "cases": results, "full_gpu_rl_training": False,
            "limitations": ["Controlled geometry only", "No packet MAC event parity",
                            "No fused flight/sensing/reward GPU rollout"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warp-q-cap", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_phy_gate(args.device, q_cap=args.warp_q_cap)
    except Exception:
        result = {"status": "FAIL_RUNTIME", "traceback": traceback.format_exc()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PHY_REFERENCE_PASS_NOT_FULL_RL" else (
        3 if result["status"].startswith("BLOCKED") else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
