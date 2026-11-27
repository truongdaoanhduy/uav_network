#!/usr/bin/env python3
"""Independent real-CUDA Warp packet-network smoke and *device-wide* VRAM check.

Not a full RL environment, not MATD3/MASAC training, not upstream parity.
Exit 3 when GPU prerequisites are missing, exit 1 on execution assertion/error.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research/zhang_xue/src')]

from scripts.gpu.packet_workload import run_packet_workload, summarize_snapshot


def _check_cuda():
    try:
        import torch
        import warp as wp
        if not torch.cuda.is_available() or not wp.get_cuda_devices():
            return None, None, 'CUDA not available to PyTorch and Warp'
        return torch, wp, None
    except (ImportError, RuntimeError) as exc:
        return None, None, str(exc)


def exercise(num_envs: int, device: str, num_uavs: int = 6, grid_w: int = 20):
    torch, wp, error = _check_cuda()
    if error:
        return {'status': 'BLOCKED_NO_CUDA_HOST', 'reason': error}, 3
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    from zhang_xue.gpu.layout_v67 import backing_bytes_per_environment_v67, assert_network_allocation_fits
    idx = int(device.rsplit(':', 1)[-1])
    torch.cuda.set_device(idx)
    wp.init()
    free0, total0 = torch.cuda.mem_get_info(idx)
    memory_gate = assert_network_allocation_fits(num_envs, num_uavs, grid_w, free_bytes=int(free0))
    net = WarpCompressedRadioNetworkV67(num_envs=num_envs, num_uavs=num_uavs, grid_w=grid_w,
              device=device, rx_lock_on_cca=False, seed=44)
    started = time.perf_counter()
    snapshot = run_packet_workload(net)
    wp.synchronize_device(device)
    torch.cuda.synchronize(idx)
    free1, total1 = torch.cuda.mem_get_info(idx)
    # Device-wide free-memory delta covers Warp allocations; torch's own peak does not.
    return {
        'status':'CUDA_NETWORK_EXECUTED_NOT_RL_TRAINING',
        'cuda_executed':True, 'full_rl_training':False,
        'workload_wall_seconds_including_jit':time.perf_counter()-started,
        'device':device, 'num_envs':num_envs, 'num_uavs':num_uavs, 'grid_w':grid_w,
        'gpu_name':torch.cuda.get_device_name(idx),
        'gpu_mem_free_before_bytes':free0,
        'gpu_mem_free_after_bytes':free1,
        'device_wide_delta_bytes':free0-free1,
        'total_vram_bytes':total0,
        'estimated_network_soa_bytes':backing_bytes_per_environment_v67(num_uavs,grid_w)*num_envs,
        'preallocation_memory_gate':memory_gate,
        'torch_only_peak_bytes_not_total_vram':torch.cuda.max_memory_allocated(idx),
        'stats': summarize_snapshot(snapshot),
        'warnings':['No CPU packet-event parity tested',
                    'No actor/replay/update/eval path on Warp',
                    'High env count is only a packet radio memory gate'],
    }, 0


def exercise_cpu(num_envs: int, num_uavs: int = 6, grid_w: int = 20):
    """Execute shared Warp kernels on CPU; never label this CUDA or RL."""
    try:
        from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
        from zhang_xue.gpu.layout_v67 import backing_bytes_per_environment_v67
        import warp as wp
    except ImportError as exc:
        return {'status':'BLOCKED_NO_WARP', 'reason':str(exc)}, 3
    started = time.perf_counter()
    net = WarpCompressedRadioNetworkV67(num_envs, num_uavs=num_uavs,
                                      grid_w=grid_w, device='cpu', seed=44)
    snapshot = run_packet_workload(net)
    wp.synchronize_device('cpu')
    return {'status':'WARP_CPU_NETWORK_EXECUTED_NOT_CUDA_OR_RL',
            'cuda_executed':False, 'full_rl_training':False, 'device':'cpu',
            'num_envs':num_envs, 'num_uavs':num_uavs, 'grid_w':grid_w,
            'workload_wall_seconds_including_jit':time.perf_counter()-started,
            'estimated_network_soa_bytes':backing_bytes_per_environment_v67(num_uavs,grid_w)*num_envs,
            'stats':summarize_snapshot(snapshot),
            'warnings':['Not CPU-vs-Warp packet-event parity',
                        'Not upstream UavNetSim validation', 'Not GPU throughput']}, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--uavs', type=int, default=6)
    parser.add_argument('--grid', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.envs < 1 or args.envs > 4096:
        parser.error('envs must be within [1,4096]')
    if not 1 <= args.uavs <= 30:
        parser.error('uavs must be within [1,30]')
    if args.grid < 1 or args.grid**2 > 4096:
        parser.error('grid cells must fit 12-bit packet IDs')
    if args.device != 'cpu' and (not args.device.startswith('cuda:') or not args.device.rsplit(':',1)[-1].isdigit()):
        parser.error('device must be cpu or cuda:N')
    try:
        if args.device == 'cpu':
            result, exit_code = exercise_cpu(args.envs, num_uavs=args.uavs, grid_w=args.grid)
        else:
            result, exit_code = exercise(args.envs, args.device, num_uavs=args.uavs, grid_w=args.grid)
    except Exception:
        result, exit_code = {'status':'FAIL_RUNTIME','traceback':traceback.format_exc()}, 1
    print(json.dumps(result,indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
