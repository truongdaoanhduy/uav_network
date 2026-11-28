#!/usr/bin/env python3
"""Paired CPU/Warp application delivery fixture, explicitly not event parity."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research/zhang_xue/src')]
from scripts.gpu.packet_workload import compare_contract_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uavs', type=int, choices=range(1, 31), default=6)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        import warp as wp
    except ImportError:
        print(json.dumps({'status':'BLOCKED_NO_WARP'}))
        return 3
    if args.device != 'cpu':
        if args.device not in [str(d) for d in wp.get_cuda_devices()]:
            print(json.dumps({'status':'BLOCKED_NO_CUDA_HOST'}))
            return 3
    print(json.dumps(compare_contract_fixture(args.uavs, args.device), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
