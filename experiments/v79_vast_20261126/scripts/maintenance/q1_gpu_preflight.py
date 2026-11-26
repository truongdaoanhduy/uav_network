#!/usr/bin/env python3
"""Q1-Clean 2D + compact-3 Warp readiness gate (v6.9).

Fail closed: reports both legacy v6.1/v6.5 fields (for backward-compatible tests)
and the executable v6.7/v6.9 compressed Data-Mule Warp runtime (`ack_origin_gen` int16),
while remaining honestly BLOCKED_NO_CUDA_HOST until verified on a real CUDA Tesla T4.
"""
from __future__ import annotations

import json


def q1_gpu_readiness() -> dict:
    from zhang_xue.gpu.layout_v67 import backing_bytes_per_environment_v67
    from zhang_xue.gpu.parity_v67_ref import NumPySoARefV67
    from zhang_xue.gpu.runtime import WarpSevenRadioNetwork
    from zhang_xue.gpu.runtime_v65 import WarpCompressedRadioNetwork
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    from zhang_xue.gpu.state import HAS_WARP

    layout = WarpSevenRadioNetwork.active_layout_id()
    compressed_layout = WarpCompressedRadioNetwork.active_layout_id()
    v65_packet_step_implemented = callable(getattr(WarpCompressedRadioNetwork, "step", None))
    v67_packet_step_implemented = callable(getattr(WarpCompressedRadioNetworkV67, "step", None))

    # Run lightweight CPU NumPy SoA 2-contact sanity check for v6.7 schema
    ref = NumPySoARefV67(num_envs=1, num_uavs=2, grid_w=20, enable_data_mule=True)
    import numpy as np
    ref.sense_batch(np.array([[10, -1]]), np.array([[1, 0]]), t_s=1)
    pos = np.array([[[0.0, 0.0], [10.0, 0.0]]], dtype=np.float32)
    ref.inject_and_deliver_ideal(ref.resolve_compact3(np.array([[2, 0]]), pos))
    ref.inject_and_deliver_ideal(ref.resolve_compact3(np.array([[0, 1]]), pos))
    v67_cpu_parity = "IDEAL_HOP_SANITY_PASS" if int(ref.data["ledger_gen"][0, 2, 0, 10]) == 1 else "IDEAL_HOP_SANITY_FAIL"

    blockers = []
    if not HAS_WARP:
        blockers.append("warp_lang_unavailable")
        blockers.append("no_cuda_host")
    if not v65_packet_step_implemented:
        blockers.append("compressed_runtime_has_no_packet_step")
    blockers.extend((
        "cuda_t4_kernel_execution_and_benchmark_unverified",
        "gpu_cpu_packet_event_parity_unverified",
        "full_rl_warp_rollout_update_eval_not_integrated",
        "warp_queue_32_vs_cpu_configurable_1_to_200_vs_uavnetsim_200",
        "warp_single_tx_power_vs_cpu_per_uav_learned_tx",
        "warp_static_phy_single_update_vs_cpu_network_substeps",
        "int32_microsecond_timestamp_overflow_after_2147_seconds",
    ))
    if layout == "v61_uncompressed":
        blockers.extend((
            "compressed_layout_inactive",
            "hardcoded_6_uavs",
            "hardcoded_7_radio_actions",
            "cpu_ewma_2d_compact3_vs_gpu_kernel_parity_unverified",
        ))
    return {
        "profile": "v69-unified-v67-warp-exec-fixed-tx-2d-compact-3",
        "status": "BLOCKED_NO_WARP" if not HAS_WARP else "BLOCKED_CUDA_PARITY_UNVERIFIED",
        "ready": not blockers,
        "cuda_t4_benchmark_verified": False,
        "active_layout": layout,
        "v65_layout": compressed_layout,
        "v65_packet_step_implemented": v65_packet_step_implemented,
        "v67_layout": WarpCompressedRadioNetworkV67.active_layout_id(),
        "v67_packet_step_implemented": v67_packet_step_implemented,
        "v67_ack_origin_gen_dtype": "int16",
        "v67_ack_origin_gen_bytes_per_env": 6 * 7 * 6 * 400 * 2,
        "v67_soa_bytes_per_env": backing_bytes_per_environment_v67(6, 20),
        "v67_numpy_soa_ref_cpu_parity": v67_cpu_parity,
        "v67_ref_scope": "ideal-hop Data-Mule value transfer only; NOT packet-event/CCA/ACK parity",
        "full_rl_warp_training_integrated": False,
        "total_memory_is_estimate_not_vram_measurement": True,
        "active_soa_bytes_per_env": WarpSevenRadioNetwork.active_soa_bytes_per_environment(),
        "blockers": blockers,
    }


def main():
    result = q1_gpu_readiness()
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        print("BLOCKED: CUDA T4 kernel execution/benchmark is NOT verified on this host.")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
