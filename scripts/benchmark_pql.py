"""Run explicit candidate batches; never silently replace an OOM configuration."""
import argparse
from copy import deepcopy
import json
import os
from uuid import uuid4
from pathlib import Path

import torch

from uav_network.configuration import load_config
from uav_network.cloud.identities import run_name
from uav_network.experiments.hardware import hardware_info
from uav_network.experiments.pql_benchmark import benchmark_pql


def main():
    os.environ["UAV_STAGE"] = "benchmark"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/motion-pql.yaml")
    parser.add_argument("--algorithms", nargs="+", choices=["masac", "matd3"], default=["masac", "matd3"])
    parser.add_argument("--batches", nargs="+", type=int, default=[8192])
    parser.add_argument("--envs", nargs="+", type=int, default=[4096])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--output", default="outputs/pql-benchmark/report.json")
    args = parser.parse_args()
    c = load_config(args.config, args.overrides)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if c.runtime.device == "cuda" and not torch.cuda.is_available():
        report = {"status": "cuda_unavailable", "cuda_pipeline_verified": False, "target_t4_scenario_verified": False,
                  "hardware": hardware_info(), "results": [],
                  "message": "No T4/GPU measurement performed; candidate configuration remains unverified"}
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return 2
    results = []
    for algorithm in args.algorithms:
        for count in args.envs:
            for batch in args.batches:
                case = deepcopy(c)
                case.runtime.algorithm, case.runtime.num_envs = algorithm, count
                case.algorithm_config.batch_size = batch
                case.runtime.output_dir = str(output.parent / f"{algorithm}-E{count}-B{batch}")
                if args.single_gpu:
                    case.pql.collector_device = case.pql.learner_device = case.pql.policy_device = "cuda:0"
                case.tracking.run_name = run_name(case, algorithm, "pql", "benchmark", uuid4().hex[:8])
                result = benchmark_pql(case, args.steps)
                results.append(result)
                output.write_text(json.dumps({"hardware": hardware_info(), "results": results}, indent=2))
                print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
