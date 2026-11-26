"""V80 feasible kinematic safety shield tests (rear-end trap, head-on evasion, contracts)."""
import ast
from pathlib import Path
import unittest

import numpy as np

from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params
from zhang_xue.mobility.flight_safety_energy import (
    kinematic_safety_shield,
    trajectory_safety_violations,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_V77 = ROOT / "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml"
SHIELD_V80 = ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py"
MISSION_V80 = ROOT / "research/zhang_xue/src/zhang_xue/gpu/mission_v80.py"
RUNNER_V80 = ROOT / "scripts/gpu/run_v80_real_rollout.py"


class TestV80FeasibleShieldAndContracts(unittest.TestCase):
    def test_v80_rear_end_braking_trap_resolved_by_66cand_projection(self):
        """Mechanism B: rear-end braking trap fails 1D braking but passes 66-cand projection."""
        pos = np.array([[200.0, 200.0], [255.0, 200.0]], dtype=np.float64)
        vel = np.array([[15.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        cmd = np.array([[15.0, 0.0], [5.0, 0.0]], dtype=np.float64)
        alive = np.array([True, True])
        v_safe, _ = kinematic_safety_shield(
            pos, vel, cmd, alive, [], map_size_m=1000.0, dt_s=1.0,
            v_max_mps=15.0, a_max_mps2=5.0, uav_safe_distance_m=50.0,
        )
        viol = trajectory_safety_violations(
            pos, v_safe, alive, [], map_size_m=1000.0, dt_s=1.0, uav_safe_distance_m=50.0,
        )
        self.assertFalse(any(viol.values()), f"66-cand shield should prevent collision: {viol}")

    def test_v80_random_policy_120_steps_on_tight_6uav_spawn_zero_collisions(self):
        """With 66-cand projection + 3-sweep CD, 6 UAVs from 71.59m spawn survive 120 random steps."""
        buildings, _, _ = scale_canonical_geometry(1000.0, 20)
        rng = np.random.default_rng(44)
        pos = np.array([[90.0 + 65.0 * u, 110.0 + 30.0 * (u % 2)] for u in range(6)], dtype=np.float64)
        vel = np.zeros((6, 2), dtype=np.float64)
        alive = np.ones(6, dtype=bool)
        collisions = 0
        for _ in range(120):
            cmd = rng.uniform(-15.0, 15.0, size=(6, 2))
            v_safe, _ = kinematic_safety_shield(
                pos, vel, cmd, alive, buildings, map_size_m=1000.0, dt_s=1.0,
                v_max_mps=15.0, a_max_mps2=5.0, uav_safe_distance_m=50.0,
            )
            viol = trajectory_safety_violations(
                pos, v_safe, alive, buildings, map_size_m=1000.0, dt_s=1.0, uav_safe_distance_m=50.0,
            )
            if any(viol.values()):
                collisions += 1
            pos = pos + v_safe
            vel = v_safe
        self.assertEqual(collisions, 0, "66-cand feasible shield must achieve 0 collisions over 120 random steps")

    def test_v80_params_from_yaml_and_files_under_500_lines(self):
        params = GPUWorldV80Params.from_yaml(SCENARIO_V77, envs=64, device="cuda:0")
        self.assertAlmostEqual(params.per_building_loss_db, 12.0)
        self.assertAlmostEqual(params.dt_s, 1.0)
        self.assertFalse(params.strict_braking_terminate)
        for path in (SHIELD_V80, MISSION_V80, RUNNER_V80):
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
            self.assertLessEqual(len(text.splitlines()), 500, f"{path.name} exceeded 500 lines")

    def test_v80_kernel_and_runner_contracts(self):
        s_src = SHIELD_V80.read_text(encoding="utf-8")
        self.assertIn("def _candidate_v80", s_src)
        self.assertIn("def _project_uav_v80", s_src)
        self.assertIn("for _ in range(3):", s_src)
        self.assertIn("_geom_physical_flags_v80", s_src)
        self.assertIn("_pair_physical_safe_v80", s_src)
        r_src = RUNNER_V80.read_text(encoding="utf-8")
        self.assertIn("WarpParallelMissionV80", r_src)
        self.assertIn("braking_margin_warning", r_src)
        self.assertLess(r_src.rindex("overflow"), r_src.rindex("world.auto_reset_gpu()"))


if __name__ == "__main__":
    unittest.main()
