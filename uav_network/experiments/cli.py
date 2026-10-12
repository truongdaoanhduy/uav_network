"""Small CLI; experiment parameters live in config rather than source edits."""
import argparse
import json
from ..configuration import load_config


def main():
    parser = argparse.ArgumentParser(description="2D homogeneous UAV search with UavNetSim and Hybrid MASAC")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train independent MASAC actors")
    training.add_argument("--config", help="YAML config path; unspecified fields use validated defaults")
    training.add_argument("--set", dest="overrides", action="append", default=[], metavar="GROUP.KEY=VALUE")
    inspection = commands.add_parser("config", help="Print resolved configuration without training")
    inspection.add_argument("--config")
    inspection.add_argument("--set", dest="overrides", action="append", default=[])
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved checkpoint on CPU")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--episodes", type=int)
    evaluation.add_argument("--seed", type=int)
    evaluation.add_argument("--output-dir")
    args = parser.parse_args()
    if args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.checkpoint, args.episodes, args.output_dir, args.seed)
    else:
        config = load_config(args.config, args.overrides)
        if args.command == "config":
            result = config.to_dict()
        else:
            from .training import train
            result = train(config)
    print(json.dumps(result, indent=2, allow_nan=False))
