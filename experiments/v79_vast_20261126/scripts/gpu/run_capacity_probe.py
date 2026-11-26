#!/usr/bin/env python3
"""Logged small functional or direct exact-size CUDA capacity check; not full RL."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

from uav_network.algorithms.configuration.loader import load_learner_config
from uav_network.experiments.capacity_probe import run_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/learners/e4096_b8192.yaml")
    parser.add_argument("--scenario", type=Path, default=ROOT / "research/zhang_xue/configs/scenario_v67_datamule_1km.yaml")
    parser.add_argument("--set", action="append", default=[],
                        help="Learner YAML override, e.g. runtime.num_envs=4096")
    parser.add_argument("--scenario-set", action="append", default=[],
                        help="Scenario override; unsupported Warp MAC timing fails closed")
    parser.add_argument("--warp-q-cap", type=int, default=32,
                        help="Warp queue capacity (1..127); use 64 for current CPU queue parity")
    parser.add_argument("--allow-dynamic-mac-timing", action="store_true",
                        help="Send validated scenario DIFS/SIFS/slot/CW/retries/ACK to Warp kernels")
    parser.add_argument("--require-queue-parity", action="store_true",
                        help="Fail closed when scenario CPU queue and Warp queue capacities differ")
    parser.add_argument("--mode", choices=["functional", "capacity"], default="capacity")
    parser.add_argument("--algorithm", choices=["matd3", "masac"], default="matd3")
    parser.add_argument("--network-device", default="cuda:0")
    parser.add_argument("--learner-device", default="cuda:1")
    parser.add_argument("--functional-envs", type=int, default=4)
    parser.add_argument("--functional-batch", type=int, default=16)
    parser.add_argument("--functional-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        parser.error("--output-dir must be under repository outputs/")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh --output-dir to preserve phase-log provenance")
    cfg = load_learner_config(args.config, overrides=args.set)
    result = run_probe(cfg, scenario_path=args.scenario, output_dir=output,
                       mode=args.mode, algorithm=args.algorithm, network_device=args.network_device,
                       learner_device=args.learner_device, seed=args.seed,
                       functional_envs=args.functional_envs, functional_batch=args.functional_batch,
                       functional_hidden=args.functional_hidden,
                       scenario_overrides=args.scenario_set,
                       warp_q_cap=args.warp_q_cap,
                       allow_dynamic_mac_timing=args.allow_dynamic_mac_timing,
                       require_queue_parity=args.require_queue_parity)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"].endswith("PASS_NOT_FULL_RL") else (
        3 if result["status"].startswith("BLOCKED") else 1)


if __name__ == "__main__":
    raise SystemExit(main())
