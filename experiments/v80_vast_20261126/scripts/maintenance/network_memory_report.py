#!/usr/bin/env python3
"""Reproducible network SoA byte-only estimates, never CUDA VRAM measurements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/zhang_xue/src'))
from zhang_xue.gpu.layout_v67 import backing_bytes_per_environment_v67

GIB = 1024**3
USABLE_T4_BYTES = int(15.0 * GIB)
SAFE_T4_NET_BUDGET_BYTES = int(USABLE_T4_BYTES * 0.85) - 3 * GIB
DEFAULT_CASES = ((6, 20), (6, 40), (6, 60), (12, 40), (30, 40))


def memory_inventory(env_counts=(2048, 4096), cases=DEFAULT_CASES) -> dict:
    profiles = []
    for uavs, grid in cases:
        bytes_per_env = backing_bytes_per_environment_v67(uavs, grid)
        for envs in env_counts:
            if not isinstance(envs, int) or envs < 1:
                raise ValueError('env_counts must be positive ints')
            total = bytes_per_env * envs
            profiles.append({
                'num_uavs': uavs, 'grid_w': grid, 'cells': grid**2,
                'num_envs': envs, 'network_soa_bytes_per_env': bytes_per_env,
                'network_soa_total_bytes': total,
                'network_soa_gib': round(total / GIB, 3),
                'single_t4_16gib_network_only_possible': total < USABLE_T4_BYTES,
                'fits_t4_with_3gib_reserve': total <= SAFE_T4_NET_BUDGET_BYTES,
            })
    return {
        'formula': 'sum(product(array.shape)*dtype.itemsize) from v67 SoA schema',
        'measured_cuda_peak': False,
        'limit_t4_vram_gib': 16,
        'usable_t4_vram_gib_est': 15.0,
        'safe_t4_network_budget_gib': round(SAFE_T4_NET_BUDGET_BYTES / GIB, 3),
        'profiles': profiles,
        'limitations': [
            'GPU measurement NOT performed: these are schema backing bytes only',
            'Includes queue payload and protocol SoA; excludes allocation overhead, input/destination buffers, kernel caches, PyTorch activations, optimizer, replay and full environment rollout',
            'Two Tesla T4 cards do not combine into one 32-GiB address space',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.dumps(memory_inventory(), indent=2, ensure_ascii=False) + '\n'
    if args.output:
        out = args.output if args.output.is_absolute() else ROOT / args.output
        if not out.resolve().is_relative_to((ROOT/'outputs').resolve()):
            parser.error('Generated memory report must stay under outputs/')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(data, encoding='utf-8')
        print(f'WROTE {out}: {len(memory_inventory()["profiles"])} estimated profiles (NOT CUDA measured)')
    else:
        print(data, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
