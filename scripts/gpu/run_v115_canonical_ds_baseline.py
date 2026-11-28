#!/usr/bin/env python3
"""V115 Fail-Closed Canonical DS-Only Baseline Launcher (fixes C42 runner default risk).

Hard-locks active DS-only scenario, canonical_continuous MATD3/MASAC mode, and
v85_canonical_matd3_masac_baseline.yaml before delegating to run_v81_real_rollout.py.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

DEFAULT_SCENARIO = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"
DEFAULT_LEARNER_CFG = ROOT / "configs/learners/v85_canonical_matd3_masac_baseline.yaml"
RUNNER = ROOT / "scripts/gpu/run_v81_real_rollout.py"

# Pinned to the two byte-identical V118 canonical YAMLs. Ablations belong in
# dedicated runners and must not be silently relabeled as the canonical baseline.
V118_CANONICAL_SCENARIO_SHA256 = "3fcfe434b35c29310835c2aa99f982a96c4af58375dc5e916e1e4d017a832fc5"
V118_CANONICAL_LEARNER_SHA256 = "f75ab58ad939ae829225ca65f623ed4d289d52bc89c7b9f03745d84482138e94"


def _validate_pinned(path: Path, expected: str, label: str) -> None:
    """Reject missing/edited/swapped YAML rather than training a mislabeled baseline."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Fail-closed canonical {label}: SHA256 mismatch for {path} ({digest})")


def build_command(args: argparse.Namespace) -> list[str]:
    from zhang_xue.configuration.loader import load_scenario

    # The delegated GPU runner explicitly blocks unacknowledged experimental
    # physics. The canonical wrapper must preserve that consent boundary.
    if not getattr(args, "allow_experimental_physics", False):
        raise ValueError("V119 C48: explicit --allow-experimental-physics required (V83 scope)")
    _validate_pinned(args.scenario, V118_CANONICAL_SCENARIO_SHA256, "scenario")
    _validate_pinned(args.learner_config, V118_CANONICAL_LEARNER_SHA256, "learner")
    cfg = load_scenario(args.scenario)
    if cfg.belief_inference_mode != "dempster_shafer":
        raise ValueError(f"Fail-closed C42: expected dempster_shafer, got {cfg.belief_inference_mode}")
    if cfg.enable_tx_power_control:
        raise ValueError("Fail-closed C42: canonical baseline requires fixed tx_power_w (enable_tx_power_control=False)")
    if cfg.enable_peer_dissemination or cfg.enable_data_mule_forwarding:
        raise ValueError("Fail-closed C42: pure-GCS baseline disables PEER and MULE")
    if (cfg.tx_power_w != 0.1 or cfg.comm_action_mode != "compact_3" or
            cfg.ds_fusion_mode != "sum" or cfg.ds_aoi_eta != 0.0 or
            cfg.static_evidence_mode != "raw_u16_count" or
            cfg.exploration_reward_mode != "first_visit"):
        raise ValueError("Fail-closed C42: unexpected active radio, DS, or reward contract")
    if (cfg.num_uavs != 6 or cfg.grid_w != 20 or cfg.num_targets != 15 or
            cfg.map_size_m != 1000.0 or cfg.uav_alt_m != 50.0 or
            cfg.safety_horizon_s != 900 or not cfg.safety_failure_terminate):
        raise ValueError("Fail-closed C42: unexpected mission scale or safety contract")

    return [
        sys.executable,
        str(RUNNER),
        "--algorithm", args.algorithm,
        "--scenario", str(args.scenario),
        "--learner-config", str(args.learner_config),
        "--baseline-mode", "canonical_continuous",
        "--replay-action-mode", "raw",
        "--allow-experimental-physics",
        "--envs", str(args.envs),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--replay-capacity", str(args.replay_capacity),
        "--world-device", args.world_device,
        "--learner-device", args.learner_device,
        "--seed", str(args.seed),
        "--eval-episodes", str(args.eval_episodes),
        "--output", str(args.output),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algorithm", choices=["matd3", "masac"], required=True)
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--learner-config", type=Path, default=DEFAULT_LEARNER_CFG)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--replay-capacity", type=int, default=65536)
    ap.add_argument("--world-device", default="cuda:0")
    ap.add_argument("--learner-device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--eval-episodes", type=int, default=0)
    ap.add_argument("--allow-experimental-physics", action="store_true", help="Explicitly acknowledge V83 physics limitations")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cmd = build_command(args)
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    return int(subprocess.run(cmd, cwd=ROOT).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
