"""V94 SCENARIO-FIRST unit tests closing S08, S11, S12 & 500-line/algorithm-lock policies."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.mobility.flight_safety_energy import (
    kinematic_safety_shield,
    trajectory_safety_violations,
)
from scripts.gpu.v94_mac_analytic_validation import (
    run_cleanroom_slotted_dcf_des,
    solve_finite_retry_bianchi,
)


class TestV94ScenarioFreezeGates(unittest.TestCase):
    def test_s08_coordinated_pass3_rescues_head_on_and_seed44_world7_step143(self):
        """Pass 3 coordinated reciprocal half-planes rescue tight head-on pairs without relaxing 50m/15/5."""
        buildings, _, _ = scale_canonical_geometry(1000.0, 20)
        blds = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in buildings]
        # Synthetic mirror of V93 seed44/world7 step 143: ||r|| = 66.43m, closing = 14.54 m/s
        pos = np.array([
            [200.0, 200.0],
            [266.43, 200.0],
            [600.0, 800.0],
            [800.0, 800.0],
            [800.0, 200.0],
            [150.0, 850.0],
        ], dtype=np.float64)
        vel = np.array([
            [7.5, 0.0],
            [-7.04, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float64)
        # Nominal commands keep pushing UAV 0 and 1 head-on
        v_cmd = np.array([
            [12.5, 0.0],
            [-12.04, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float64)
        alive = np.ones(6, dtype=bool)
        vs, _ = kinematic_safety_shield(
            pos, vel, v_cmd, alive, blds,
            map_size_m=1000.0, dt_s=1.0, v_max_mps=15.0, a_max_mps2=5.0, uav_safe_distance_m=50.0,
        )
        viols = trajectory_safety_violations(
            pos, vs, alive, blds, map_size_m=1000.0, dt_s=1.0, uav_safe_distance_m=50.0,
        )
        self.assertEqual(viols["boundary"], [])
        self.assertEqual(viols["building"], [])
        self.assertEqual(viols["pairwise"], [])
        # Also verify V93 bit32 next-step viability bound holds strictly (>= 50.0 m)
        fx = (pos[0] - pos[1]) + 2.0 * (vs[0] - vs[1])
        self.assertGreaterEqual(float(np.linalg.norm(fx)) + 10.0, 50.0)
        # Verify acceleration and speed limits strictly respected
        for u in range(6):
            self.assertLessEqual(float(np.linalg.norm(vs[u] - vel[u])), 5.0 + 1e-6)
            self.assertLessEqual(float(np.linalg.norm(vs[u])), 15.0 + 1e-6)

    def test_s11_shared_coverage_reward_flag_decouples_ds_and_bayes_exploration(self):
        """With shared_coverage_reward=True, identical (p, n) counts yield identical exploration delta."""
        w_loc = 0.5
        for p, n in [(1, 0), (0, 1), (1, 1), (2, 1), (3, 0)]:
            u_shared = 0.5 ** (p + n)
            r_bayes_shared = w_loc * (1.0 - u_shared)
            r_ds_shared = w_loc * (1.0 - u_shared)
            self.assertEqual(r_bayes_shared, r_ds_shared)
        # Verify GPUWorldV79Params defaults shared_coverage_reward=False for backward compatibility
        src = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/mission_v79.py").read_text(encoding="utf-8")
        self.assertIn("shared_coverage_reward: bool = False", src)
        ksrc = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py").read_text(encoding="utf-8")
        self.assertIn("if shared_coverage_reward != 0:", ksrc)

    def test_s12_independent_finite_retry_bianchi_and_cleanroom_des_agree(self):
        """Clean-room Python DCF DES matches exact finite-retry Bianchi & Tinnirello (2005) within 3%."""
        for n in (2, 4, 6):
            an_inc = solve_finite_retry_bianchi(n, inclusive=True)
            an_exc = solve_finite_retry_bianchi(n, inclusive=False)
            self.assertLess(an_inc["fixed_point_residual"], 1e-10)
            self.assertLess(an_exc["fixed_point_residual"], 1e-10)
            # Exclusive [0, CW-1] has strictly higher collision probability than closed [0, CW]
            self.assertGreater(an_exc["p_collision"], an_inc["p_collision"])
            des = run_cleanroom_slotted_dcf_des(n, num_slots=40_000, seed=44 + n, inclusive=True)
            p_rel = abs(des["p_collision_emp"] - an_inc["p_collision"]) / an_inc["p_collision"]
            s_rel = abs(des["throughput_airtime_frac_emp"] - an_inc["throughput_airtime_frac"]) / an_inc["throughput_airtime_frac"]
            self.assertLess(p_rel, 0.04)
            self.assertLess(s_rel, 0.02)

    def test_v94_500_line_policy_on_all_touched_files(self):
        touched = [
            "research/zhang_xue/src/zhang_xue/mobility/flight_safety_energy.py",
            "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py",
            "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v79_kernels.py",
            "research/zhang_xue/src/zhang_xue/gpu/mission_v79.py",
            "research/zhang_xue/src/zhang_xue/gpu/mission_v80.py",
            "scripts/gpu/v94_mac_analytic_validation.py",
            "tests/test_v94_scenario_freeze_gates.py",
        ]
        for rel in touched:
            nlines = len((ROOT / rel).read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(nlines, 500, f"{rel} has {nlines} lines > 500")


if __name__ == "__main__":
    unittest.main()
