#!/usr/bin/env python3
"""Prepare or verify the public Framefree CUDA exactness gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_network.experiments.framefree_cuda_gate import prepare_gate, verify_remote


def main(argv=None):
    parser = argparse.ArgumentParser(description="Public Framefree CUDA exactness gate")
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--kernel-ref", required=True)
    prepare.add_argument("--commit", required=True)
    prepare.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--kernel-ref", required=True)
    verify.add_argument("--commit", required=True)
    verify.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        path = prepare_gate(Path(args.output), args.kernel_ref, args.commit)
        result = {"status": "prepared", "output": str(path), "kernel_ref": args.kernel_ref,
                  "commit": args.commit, "is_private": False}
    else:
        payload = verify_remote(args.kernel_ref, args.commit, Path(args.output))
        result = {"status": "verified", "kernel_ref": args.kernel_ref,
                  "commit": payload["commit"], "gpu_names": payload.get("gpu_names", []),
                  "pytest_stdout": payload.get("pytest_stdout", "")}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
