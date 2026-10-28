"""Compare an overflow snapshot in FP32/FP16 without launching or resuming training."""
import argparse
import json
from pathlib import Path
from uav_network.experiments.overflow_capture import replay_overflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = replay_overflow(args.snapshot, args.device)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
