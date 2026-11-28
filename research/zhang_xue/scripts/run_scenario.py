#!/usr/bin/env python3
"""Reproducible CPU-only seven-radio smoke; not a MATD3/MASAC trainer."""
from __future__ import annotations
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zhang_xue.configuration.loader import load_scenario, make_parser
from zhang_xue.environment.core import ZhangXue7NodePacketEnv


def run() -> int:
    parser = make_parser()
    parser.add_argument("--output", type=Path, default=None, help="Must be under outputs/")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    cfg = load_scenario(args.config, tuple(args.set), preset=args.preset)
    report = {
        "kind": "cpu-smoke-not-training",
        "seed": args.seed,
        "preset": args.preset,
        "config": asdict(cfg),
        "steps_requested": args.steps,
        "communication": args.communication,
    }
    if not args.dry_run:
        env = ZhangXue7NodePacketEnv(cfg, seed=args.seed)
        env.reset(seed=args.seed)
        returns = 0.0
        for _ in range(min(args.steps, cfg.safety_horizon_s)):
            velocities = np.zeros((env.actor_n, 2), dtype=np.float64)
            gcs_action = 1 if cfg.comm_action_mode == "compact_3" else env.actor_n
            action = gcs_action if args.communication == "gcs" else 0
            dest = np.full((env.actor_n,), action, dtype=np.int32)
            _, _, _, reward, terminated, truncated, info = env.step(velocities, dest)
            returns += float(reward)
            if terminated or truncated:
                break
        report["steps_executed"] = int(env.t)
        report["total_reward"] = returns
        report["terminated"] = bool(terminated)
        report["truncated"] = bool(truncated)
        report["statistics"] = info
    if args.output:
        out = args.output if args.output.is_absolute() else ROOT / args.output
        out = out.resolve()
        outputs = (ROOT / "outputs").resolve()
        if not out.is_relative_to(outputs):
            parser.error("--output must be under the gitignored outputs/ directory")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=_json_safe) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=_json_safe))
    return 0


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not serializable: {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(run())
