"""Small CLI; experiment parameters live in config rather than source edits."""
import argparse
import json
from pathlib import Path
from ..configuration import load_config


def main():
    parser = argparse.ArgumentParser(description="Battery-aware UAV swarm: UavNetSim, MASAC/MATD3 and process collection")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train MASAC or MATD3")
    training.add_argument("--config", help="YAML config path; unspecified fields use validated defaults")
    training.add_argument("--set", dest="overrides", action="append", default=[], metavar="GROUP.KEY=VALUE")
    training.add_argument("--algorithm", choices=["masac", "matd3"])
    training.add_argument("--num-envs", type=int)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"])
    inspection = commands.add_parser("config", help="Print resolved configuration without training")
    inspection.add_argument("--config")
    inspection.add_argument("--set", dest="overrides", action="append", default=[])
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved checkpoint on CPU")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--episodes", type=int)
    evaluation.add_argument("--seed", type=int)
    evaluation.add_argument("--output-dir")
    commands.add_parser("hardware", help="Show CPU affinity and CUDA devices usable by PyTorch")
    cloud = commands.add_parser("kaggle", help="Quota, private CUDA jobs and online W&B")
    cloud.add_argument("cloud_args", nargs=argparse.REMAINDER)
    dashboard = commands.add_parser("dashboard", help="Create or update a saved online W&B project view")
    dashboard.add_argument("--config")
    dashboard.add_argument("--set", dest="overrides", action="append", default=[])
    dashboard.add_argument("--run-ids", nargs="*", default=[])
    dashboard.add_argument("--name", default="UAV Network 2D - MASAC and MATD3")
    dashboard.add_argument("--workspace-url")
    speed = commands.add_parser("benchmark", help="Measure env and learner throughput separately")
    speed.add_argument("--config")
    speed.add_argument("--set", dest="overrides", action="append", default=[])
    speed.add_argument("--env-counts", nargs="+", type=int, default=[1, 2, 4])
    speed.add_argument("--steps", type=int, default=32)
    speed.add_argument("--learner-updates", type=int, default=10)
    speed.add_argument("--output", help="Optional JSON report path; must not exist")
    args = parser.parse_args()
    if args.command == "kaggle":
        from ..cloud.cli import main as cloud_main
        result = cloud_main(args.cloud_args)
    elif args.command == "hardware":
        from .hardware import hardware_info
        result = hardware_info()
    elif args.command == "dashboard":
        from .dashboard import create_dashboard
        result = create_dashboard(load_config(args.config, args.overrides), args.run_ids,
                                  args.name, args.workspace_url)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.checkpoint, args.episodes, args.output_dir, args.seed)
    else:
        overrides = list(args.overrides)
        if args.command == "train":
            for name in ("algorithm", "num_envs", "device"):
                value = getattr(args, name)
                if value is not None:
                    overrides.append(f"runtime.{name}={value}")
        config = load_config(args.config, overrides)
        if args.command == "config":
            result = config.to_dict()
        elif args.command == "benchmark":
            if args.output and Path(args.output).exists():
                raise FileExistsError(f"Benchmark output already exists: {args.output}")
            from .benchmark import benchmark
            result = benchmark(config, args.env_counts, args.steps, args.learner_updates)
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(result, indent=2, allow_nan=False))
        else:
            from .training import train
            result = train(config)
    print(json.dumps(result, indent=2, allow_nan=False))
