"""V122 RED->GREEN regression gates for C83 (Zeng 2019 Appendix A), C84 (CPU/GPU
terminal safety reward parity), C04/C66 (DS temporal correlation mitigation), and
canonical Phase-1 invariants (30/30 algorithms + YAML SHA256 untouched).
"""
from __future__ import annotations

import hashlib
import math
import unittest
from pathlib import Path

import numpy as np
import yaml

from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.mobility.flight_safety_energy import (
    ZengBatteryConfig,
    zeng_propulsion_power_w,
    zeng_propulsion_power_with_accel_w,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_YAML = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"
RF_YAML = ROOT / "research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml"
EXPECTED_BASE_SHA = "3fcfe434b35c29310835c2aa99f982a96c4af58375dc5e916e1e4d017a832fc5"
EXPECTED_RF_SHA = "3ee084e793716baa9ea83d842af0b51345690fdb8156ab0e5c527739ad033b76"
EXPECTED_ALGO_FP = "96d57cd9c4b60679fe8f74eb0f8cb2d67ced12b9dbacec99858f8c9551ad1074"


class TestV122ScenarioFixesAndGates(unittest.TestCase):
    def test_c83_zeng_2019_appendix_a_exact_induced_power(self) -> None:
        cfg = ZengBatteryConfig()
        # Steady-level (a=0) must match zeng_propulsion_power_w exactly
        for v in (0.0, 5.0, 10.0, 15.0):
            p_steady = zeng_propulsion_power_w(v, cfg)
            p_acc0 = zeng_propulsion_power_with_accel_w([v, 0.0], [v, 0.0], 1.0, cfg)
            self.assertAlmostEqual(p_acc0, p_steady, places=9)

        # Horizontal acceleration (v_prev=[7.5,0], v_new=[12.5,0], a=5 m/s^2, v_avg=10 m/s)
        # Zeng 2019 Appendix A Eq. (69): P_ind = P_i * kappa * sqrt(sqrt(kappa^2 + x^2) - x)
        vp = np.array([7.5, 0.0])
        vn = np.array([12.5, 0.0])
        dt = 1.0
        v_avg = 10.0
        a_norm = 5.0
        kappa = math.hypot(1.0, a_norm / cfg.gravity_mps2)
        x = (v_avg * v_avg) / (2.0 * cfg.v0_mps * cfg.v0_mps)
        expected_blade = cfg.p0_w * (1.0 + 3.0 * (v_avg ** 2) / (cfg.u_tip_mps ** 2))
        expected_ind = cfg.pi_w * kappa * math.sqrt(math.hypot(kappa, x) - x)
        expected_par = 0.5 * cfg.d0 * cfg.rho * cfg.s * cfg.area_m2 * (v_avg ** 3)
        expected_ke = max(0.0, 0.5 * cfg.uav_mass_kg * (12.5 ** 2 - 7.5 ** 2)) / (cfg.prop_efficiency * dt)
        expected_total = expected_blade + expected_ind + expected_par + expected_ke
        actual_total = zeng_propulsion_power_with_accel_w(vp, vn, dt, cfg)
        self.assertAlmostEqual(actual_total, expected_total, places=7)

    def test_c84_terminal_safety_failure_zeros_blocked_motion_reward(self) -> None:
        with open(BASE_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = EnvConfigV54(**data)
        env = ZhangXue7NodePacketEnv(cfg)
        env.reset(seed=44)
        # Force an unavoidable pairwise collision on UAV 0 and UAV 1 so safety_violation=True
        # while the shield also intervenes (shield_intervention > 0).
        env.positions[0, :2] = np.array([200.0, 200.0])
        env.positions[1, :2] = np.array([210.0, 200.0])  # < 50 m safe_d
        env.velocities[0] = np.array([15.0, 0.0])
        env.velocities[1] = np.array([-15.0, 0.0])
        act_xy = np.zeros((cfg.num_uavs, 2), dtype=np.float64)
        act_xy[0] = np.array([15.0, 0.0])
        act_xy[1] = np.array([-15.0, 0.0])
        act_comm = np.ones((cfg.num_uavs,), dtype=np.int32)
        _, _, _, _, terminated, _, info = env.step(act_xy, act_comm)
        self.assertTrue(info["safety_violation"])
        self.assertTrue(terminated)
        # Parity with Warp _move_v80 (mission_v80_shield.py:446): intervention[e] = 0.0 on terminal fail
        self.assertEqual(float(info["r_blocked_motion"]), 0.0)

    def test_c04_c66_ds_correlation_mitigation_module(self) -> None:
        from zhang_xue.belief.ds_correlation_mitigation import (
            DSCorrelationMitigator,
            ar1_effective_sample_weight,
            revisit_cooldown_horizon_s,
        )

        # AR(1) effective independent sample weight properties
        self.assertAlmostEqual(ar1_effective_sample_weight(1, 0.0), 1.0, places=9)
        self.assertAlmostEqual(ar1_effective_sample_weight(None, 0.7), 1.0, places=9)
        w1 = ar1_effective_sample_weight(1, 0.7)
        w5 = ar1_effective_sample_weight(5, 0.7)
        w20 = ar1_effective_sample_weight(20, 0.7)
        self.assertTrue(0.0 < w1 < w5 < w20 < 1.0)
        self.assertAlmostEqual(w1, (1.0 - 0.7) / (1.0 + 0.7), places=9)

        # Revisit cooldown for rho=0.7, eps=0.05 -> ceil(ln(0.05)/ln(0.7)) = 9 s
        self.assertEqual(revisit_cooldown_horizon_s(0.7, 0.05), 9)
        self.assertEqual(revisit_cooldown_horizon_s(0.0, 0.05), 0)

        # Burst of 2 consecutive correlated FPs from same UAV must NOT confirm under cooldown=9
        mit = DSCorrelationMitigator(num_uavs=6, num_cells=400, mode="revisit_cooldown", cooldown_s=9)
        mit.record(uav_id=0, cell_id=10, det=True, step_s=1)
        mit.record(uav_id=0, cell_id=10, det=True, step_s=2)  # suppressed by cooldown
        m_t, _, m_u = mit.fused_cell(10, mf=0.80, me=0.80)
        self.assertLess(m_t, 0.90)
        self.assertAlmostEqual(m_u, 0.20, places=6)

        # A second distinct UAV observing the same cell IS independent and confirms
        mit.record(uav_id=1, cell_id=10, det=True, step_s=2)
        m_t2, _, m_u2 = mit.fused_cell(10, mf=0.80, me=0.80)
        self.assertGreaterEqual(m_t2, 0.90)
        self.assertLessEqual(m_u2, 0.20)

    def test_v122_canonical_invariants_preserved(self) -> None:
        self.assertEqual(hashlib.sha256(BASE_YAML.read_bytes()).hexdigest(), EXPECTED_BASE_SHA)
        self.assertEqual(hashlib.sha256(RF_YAML.read_bytes()).hexdigest(), EXPECTED_RF_SHA)
        algo_files = sorted((ROOT / "uav_network/algorithms").rglob("*.py"))
        self.assertEqual(len(algo_files), 30)
        h = hashlib.sha256()
        for f in algo_files:
            fh = hashlib.sha256(f.read_bytes()).hexdigest()
            h.update(f"{f.relative_to(ROOT).as_posix()}:{fh}\n".encode())
        self.assertEqual(h.hexdigest(), EXPECTED_ALGO_FP)


if __name__ == "__main__":
    unittest.main()
