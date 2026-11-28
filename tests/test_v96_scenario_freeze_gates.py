"""V96 SCENARIO-FIRST unit tests: S08 deficit-transfer Pass-3, S12 cw_inclusive_upper, policy & lock."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src"), str(ROOT / "scripts/gpu")]

from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params
from zhang_xue.mobility.flight_safety_energy import (
    kinematic_safety_shield,
    trajectory_safety_violations,
)
from v94_mac_analytic_validation import solve_finite_retry_bianchi


class TestV96ScenarioFreezeGates(unittest.TestCase):
    def test_s08_v96_proactive_deficit_transfer_rescues_asymmetric_clipped_pair(self):
        """When UAV i hits its 5 m/s accel cap, V96 Pass 3 transfers residual deficit to UAV j."""
        buildings, _, _ = scale_canonical_geometry(1000.0, 20)
        blds = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in buildings]
        # Mirror V94/V95 seed44/world7 step 144 & seed44/world9 tight asymmetric pair:
        # d0 = 51.50 m, closing = 3.63 m/s, UAV 0 at v_max=15.0 (limited accel slack)
        pos = np.array([
            [200.0, 200.0],
            [251.50, 200.0],
            [600.0, 800.0],
            [800.0, 800.0],
            [800.0, 200.0],
            [150.0, 850.0],
        ], dtype=np.float64)
        vel = np.array([
            [15.0, 0.0],
            [11.37, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float64)
        v_cmd = np.array([
            [15.0, 0.0],
            [8.0, 0.0],
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
        fx = (pos[0] - pos[1]) + 2.0 * (vs[0] - vs[1])
        self.assertGreaterEqual(float(np.linalg.norm(fx)) + 10.0, 50.0)
        for u in range(6):
            self.assertLessEqual(float(np.linalg.norm(vs[u] - vel[u])), 5.0 + 1e-6)
            self.assertLessEqual(float(np.linalg.norm(vs[u])), 15.0 + 1e-6)

    def test_s11_v95_shared_coverage_yamls_wire_cleanly_to_params(self):
        cfg_dir = ROOT / "research/zhang_xue/configs"
        sc_ds = load_scenario(cfg_dir / "scenario_v95_ds_shared_coverage_1km.yaml")
        self.assertTrue(sc_ds.shared_coverage_reward)
        p = GPUWorldV79Params.from_scenario(sc_ds, envs=2, device="cpu", seed=44)
        self.assertTrue(p.shared_coverage_reward)
        with self.assertRaisesRegex(ValueError, "belief_inference_mode"):
            load_scenario(cfg_dir / "scenario_v95_bayes_shared_coverage_1km.yaml")

    def test_s12_cw_inclusive_upper_contract_and_att5_bianchi(self):
        s67 = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/state_v67.py").read_text(encoding="utf-8")
        r67 = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/runtime_v67.py").read_text(encoding="utf-8")
        m67 = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mac_v67.py").read_text(encoding="utf-8")
        self.assertIn("cw_inclusive_upper: wp.int32", s67)
        self.assertIn("cw_inclusive_upper: int = 1", r67)
        self.assertIn("dims.cw_inclusive_upper", m67)
        # Verify total_attempts=5 matches events_v67.py max_retx=5 drop rule
        an_1 = solve_finite_retry_bianchi(6, inclusive=True, total_attempts=5)
        an_0 = solve_finite_retry_bianchi(6, inclusive=False, total_attempts=5)
        self.assertGreater(an_0["p_collision"], an_1["p_collision"])
        self.assertLess(an_1["fixed_point_residual"], 1e-10)

    def test_v96_500_line_policy_on_all_touched_files(self):
        touched = [
            "research/zhang_xue/src/zhang_xue/mobility/flight_safety_energy.py",
            "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py",
            "research/zhang_xue/src/zhang_xue/gpu/state_v67.py",
            "research/zhang_xue/src/zhang_xue/gpu/runtime_v67.py",
            "research/zhang_xue/src/zhang_xue/gpu/kernels/mac_v67.py",
            "scripts/gpu/v94_mac_analytic_validation.py",
            "scripts/gpu/v96_s13_rf_ood_sweep.py",
            "scripts/gpu/v96_warp_mac_validation.py",
            "tests/test_v96_s11_end_to_end_runtime.py",
            "tests/test_v96_scenario_freeze_gates.py",
        ]
        for rel in touched:
            nlines = len((ROOT / rel).read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(nlines, 500, f"{rel} has {nlines} lines > 500")


if __name__ == "__main__":
    unittest.main()
