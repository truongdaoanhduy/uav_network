#!/usr/bin/env python
"""Compare MASAC critic TX-power ordering with matched simulator returns."""
import argparse
import json

from uav_network.experiments.power_calibration import probe_power_calibration, save_calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--scan-steps", type=int, default=500)
    args = parser.parse_args()
    if min(args.snapshots, args.horizon, args.scan_steps) < 1:
        raise ValueError("snapshots, horizon and scan-steps must be positive")
    result = probe_power_calibration(
        args.checkpoint, seed_start=args.seed_start, snapshots=args.snapshots,
        horizon=args.horizon, scan_steps=args.scan_steps)
    if args.output:
        save_calibration(result, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
