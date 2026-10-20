"""CPU paired communication audit of a checkpoint's exact saved scenario."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uav_network.experiments.relay_audit import MODES, run_relay_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit 1..30 unique seeds; default: 30 from saved eval_seed")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--fixed-power-w", type=float, help="Default: saved maximum radiated power")
    parser.add_argument("--trace", action="store_true", help="Save successful whole-report hops, not per-packet logs")
    args = parser.parse_args()
    def progress(value):
        if value["status"] != "running" or value["step"] % 50 == 0:
            print(json.dumps(value), flush=True)
    result = run_relay_audit(args.checkpoint, args.seeds, args.modes, args.fixed_power_w,
                             args.output_dir, args.trace, progress)
    print(json.dumps({"complete": result["complete"], "completed_cases": result["completed_cases"],
                      "expected_cases": result["expected_cases"],
                      "summary": str(args.output_dir / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
