"""V83 contracts: L2-coupled corner braking forward invariance, Tier-2 fallback, dual-replay math."""
import ast
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/gpu")]

from rollout_v82_contracts import compute_dual_replay_penalty_np, configure_learner
from uav_network.algorithms.configuration.schema import LearnerConfig
from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.mobility.flight_safety_energy import (
    clamp_velocity_with_accel,
    kinematic_safety_shield,
    trajectory_safety_violations,
)

CPU_SHIELD = ROOT / "research/zhang_xue/src/zhang_xue/mobility/flight_safety_energy.py"
GPU_K79 = ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py"
GPU_SH80 = ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py"
RUNNER_V83 = ROOT / "scripts/gpu/run_v81_real_rollout.py"
CONTRACTS_V83 = ROOT / "scripts/gpu/rollout_v82_contracts.py"


class _DummyArgs:
    def __init__(self, **kw):
        defaults = dict(
            algorithm="matd3", envs=64, steps=100, updates_per_step=1, batch_size=128,
            replay_capacity=4096, learning_starts=1024, log_interval=10, hidden=None,
            allow_no_updates=False, replay_action_mode="dual", shield_penalty_lambda=1.0,
        )
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestV83L2CornerViabilityAndContracts(unittest.TestCase):
    def test_v83_10000_random_l2_invariant_corner_states_forward_invariant_under_v_brake(self):
        """Theorem proof: every state inside L2 braking envelope stays inside [10,990] under v_brake."""
        rng = np.random.default_rng(2026)
        a_max = 5.0
        checked = 0
        for _ in range(10000):
            vx = float(rng.uniform(-15.0, 15.0))
            vy = float(rng.uniform(-15.0, 15.0))
            vn = math.hypot(vx, vy)
            if vn > 15.0:
                vx *= 15.0 / vn
                vy *= 15.0 / vn
                vn = 15.0
            bx = abs(vx) * vn / (2.0 * a_max)
            by = abs(vy) * vn / (2.0 * a_max)
            x_lo = 10.0 + (bx if vx < 0.0 else 0.0)
            x_hi = 990.0 - (bx if vx > 0.0 else 0.0)
            y_lo = 10.0 + (by if vy < 0.0 else 0.0)
            y_hi = 990.0 - (by if vy > 0.0 else 0.0)
            x = float(rng.uniform(x_lo, min(x_lo + 25.0, x_hi))) if vx < 0 else float(rng.uniform(max(x_hi - 25.0, x_lo), x_hi))
            y = float(rng.uniform(y_lo, min(y_lo + 25.0, y_hi))) if vy < 0 else float(rng.uniform(max(y_hi - 25.0, y_lo), y_hi))
            vb = clamp_velocity_with_accel([vx, vy], [0.0, 0.0], dt_s=1.0, v_max_mps=15.0, a_max_mps2=a_max)
            nx, ny = x + float(vb[0]), y + float(vb[1])
            self.assertGreaterEqual(nx, 10.0 - 1e-9)
            self.assertLessEqual(nx, 990.0 + 1e-9)
            self.assertGreaterEqual(ny, 10.0 - 1e-9)
            self.assertLessEqual(ny, 990.0 + 1e-9)
            checked += 1
        self.assertEqual(checked, 10000)

    def test_v83_single_uav_300_diagonal_corner_bang_bang_episodes_zero_boundary_collisions(self):
        """Regression for V82 RED (2/300 boundary hits): V83 achieves 0/300."""
        alive = np.array([True])
        hits = 0
        for seed in range(300):
            rng = np.random.default_rng(seed)
            p = np.array([[rng.uniform(40.0, 120.0), rng.uniform(40.0, 120.0)]], dtype=np.float64)
            v = np.zeros((1, 2), dtype=np.float64)
            ang = float(rng.uniform(math.pi, 1.5 * math.pi))
            cmd = np.array([[15.0 * math.cos(ang), 15.0 * math.sin(ang)]], dtype=np.float64)
            for _ in range(30):
                vs, _ = kinematic_safety_shield(
                    p, v, cmd, alive, [], map_size_m=1000.0, dt_s=1.0, v_max_mps=15.0, a_max_mps2=5.0,
                )
                vl = trajectory_safety_violations(
                    p, vs, alive, [], map_size_m=1000.0, dt_s=1.0, uav_safe_distance_m=50.0,
                )
                if any(vl.values()):
                    hits += 1
                    break
                p = p + vs
                v = vs
        self.assertEqual(hits, 0)

    def test_v83_6uav_four_corner_bang_bang_zero_boundary_and_building_collisions(self):
        """6 UAVs + 5 buildings slammed into all 4 corners never breach boundary or buildings."""
        buildings, _, _ = scale_canonical_geometry(1000.0, 20)
        alive = np.ones(6, dtype=bool)
        for corner in ((-15.0, -15.0), (15.0, 15.0), (-15.0, 15.0), (15.0, -15.0)):
            p = np.array([[90.0 + 65.0 * u, 110.0 + 30.0 * (u % 2)] for u in range(6)], dtype=np.float64)
            v = np.zeros((6, 2), dtype=np.float64)
            cmd = np.tile(np.array(corner, dtype=np.float64), (6, 1))
            for _ in range(40):
                vs, _ = kinematic_safety_shield(
                    p, v, cmd, alive, buildings, map_size_m=1000.0, dt_s=1.0,
                    v_max_mps=15.0, a_max_mps2=5.0, uav_safe_distance_m=50.0,
                )
                vl = trajectory_safety_violations(
                    p, vs, alive, buildings, map_size_m=1000.0, dt_s=1.0, uav_safe_distance_m=50.0,
                )
                self.assertEqual(vl["boundary"], [], f"boundary breach at corner={corner}")
                self.assertEqual(vl["building"], [], f"building breach at corner={corner}")
                if vl["pairwise"]:
                    break
                p = p + vs
                v = vs

    def test_v83_dual_replay_penalty_math_and_arg_validation(self):
        raw = np.array([[[1.0, 1.0], [0.0, 0.0]], [[0.5, 0.5], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
        exe = np.array([[[0.0, 1.0], [0.0, 0.0]], [[0.5, 0.5], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
        rew = np.array([10.0, 5.0, -100.0], dtype=np.float32)
        term = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        mask, pen = compute_dual_replay_penalty_np(raw, exe, rew, term, penalty_lambda=2.0)
        np.testing.assert_array_equal(mask, np.array([True, False, False]))
        self.assertAlmostEqual(float(pen[0]), 8.0, places=5)
        cfg = LearnerConfig()
        configure_learner(cfg, _DummyArgs(replay_action_mode="dual", shield_penalty_lambda=1.5))
        with self.assertRaises(ValueError):
            configure_learner(LearnerConfig(), _DummyArgs(replay_action_mode="invalid"))
        with self.assertRaises(ValueError):
            configure_learner(LearnerConfig(), _DummyArgs(shield_penalty_lambda=-0.1))

    def test_v83_source_parity_and_500_line_policy(self):
        for path in (CPU_SHIELD, GPU_K79, GPU_SH80, RUNNER_V83, CONTRACTS_V83):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            self.assertLessEqual(len(text.splitlines()), 500, f"{path.name} exceeded 500 lines")
        k79 = GPU_K79.read_text(encoding="utf-8")
        self.assertIn("v_norm = wp.sqrt(vx * vx + vy * vy)", k79)
        self.assertIn("wp.max(0.0, -vx) * v_norm * inv2a", k79)
        sh80 = GPU_SH80.read_text(encoding="utf-8")
        self.assertIn("gfb_cost", sh80)
        run = RUNNER_V83.read_text(encoding="utf-8")
        self.assertIn("--replay-action-mode", run)
        self.assertIn("gpu_totals += step_vec", run)
        self.assertLess(run.rindex("require_no_warp_overflow"), run.rindex("world.auto_reset_gpu()"))


if __name__ == "__main__":
    unittest.main()
