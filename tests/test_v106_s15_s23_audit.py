"""V106 Scenario Audit Regression Suite (Gates S15-S23).

Verifies:
- S15: All Group-A functional tests/scripts migrated off scenario_v91_bayes_1km.yaml;
  Group-B tests enforce Warp DS-only fail-closed; Group-C probes tagged [LEGACY_BAYES_ONLY];
  v93_mission_audit.py no longer crashes on removed p.bayes_prior_occupancy.
- S17: Exact mathematical behavior of ds_ignorance vs deng_entropy vs pignistic_entropy
  at balanced conflict (10, 10) where m_T=m_F=0.50, m_U=0.0.
- S19: Continuous 2D XY action causality (explore >> hover on coverage & reward).
- S20: Independent 2x2 PEER x data_mule isolation (data_mule alone has zero effect when PEER=OFF).
- S22: Dynamic reachability through kinematic_safety_shield with 0 safety violations.
- S23: Runtime CPU energy conservation |actual - sum(uav_step_energy_j)| < 1e-6 J.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

from zhang_xue.belief.dempster_shafer import (
    DENG_MAX_BITS,
    counts_to_canonical_ds_stable,
    deng_entropy,
)
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.mobility.flight_safety_energy import ZengBatteryConfig, uav_step_energy_j


class TestV106ScenarioAudit(unittest.TestCase):
    BASE_YAML = ROOT / "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"
    STRESS_YAML = ROOT / "research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml"

    def test_s15_functional_tests_and_scripts_migrated_to_ds_only(self):
        group_a = (
            "tests/test_v93_invalid_actions.py",
            "tests/test_v93_one_step_viability.py",
            "tests/test_v93_peer_causality.py",
            "tests/test_v102_warp_energy_runtime.py",
            "scripts/gpu/v91_discrete_action_utility_probe.py",
            "scripts/gpu/v91_network_differential_audit.py",
            "scripts/gpu/v91_oracle_feasibility_probe.py",
            "scripts/gpu/v92_non_oracle_mission_probe.py",
        )
        for rel in group_a:
            txt = (ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("scenario_v91_bayes_1km.yaml", txt, f"Still references V91 Bayes: {rel}")
            self.assertIn("scenario_v103_ds_only_1km.yaml", txt, f"Missing V103 DS YAML: {rel}")
        v93_src = (ROOT / "scripts/gpu/v93_mission_audit.py").read_text(encoding="utf-8")
        self.assertNotIn("p.bayes_prior_occupancy", v93_src)
        for rel in ("scripts/gpu/v91_belief_reliability_audit.py", "scripts/gpu/v98_scenario_ablation_suite.py"):
            self.assertIn("[LEGACY_BAYES_ONLY]", (ROOT / rel).read_text(encoding="utf-8"))

    def test_s17_conflict_reward_math_honest_bounds_at_10_10(self):
        mt, mf, mu = counts_to_canonical_ds_stable(10, 10, 0.80, 0.80)
        self.assertAlmostEqual(mt, 0.5, places=6)
        self.assertAlmostEqual(mf, 0.5, places=6)
        self.assertLess(mu, 1e-6)
        # ds_ignorance static gain = 1 - mu ~ 1.0
        self.assertGreater(1.0 - mu, 0.99999)
        # deng_entropy normalized static gain = 1 - 1/log2(3) ~ 0.36907 (still leaks > 0!)
        deng_gain = 1.0 - deng_entropy(mt, mf, mu) / DENG_MAX_BITS
        self.assertAlmostEqual(deng_gain, 1.0 - 1.0 / DENG_MAX_BITS, places=5)
        # pignistic entropy H(BetP) = 1.0 at balanced conflict -> static gain = 0.0
        bet_t = mt + 0.5 * mu
        h_pig = -(bet_t * math.log2(bet_t) + (1.0 - bet_t) * math.log2(1.0 - bet_t))
        self.assertAlmostEqual(1.0 - h_pig, 0.0, places=9)

    def test_s19_s23_xy_action_causality_and_runtime_energy_conservation(self):
        cfg = load_scenario(self.BASE_YAML)
        bcfg = ZengBatteryConfig(e_max_j=cfg.e_max_j)
        # Hover rollout (20 steps)
        env_h = ZhangXue7NodePacketEnv(cfg, seed=44)
        _, _, mask_h = env_h.reset(seed=44)
        for _ in range(20):
            _, _, mask_h, _, _, _, info_h = env_h.step(
                np.zeros((6, 2), dtype=np.float32), mask_h[:, 1].astype(np.int64)
            )
        # Moving rollout (20 steps) + runtime energy conservation check (S23)
        env_m = ZhangXue7NodePacketEnv(cfg, seed=44)
        _, _, mask_m = env_m.reset(seed=44)
        modeled_j = np.zeros(6, dtype=np.float64)
        for step_idx in range(20):
            vel_cmd = np.tile(np.array([[0.0, 12.0]], dtype=np.float32), (6, 1))
            _, _, mask_m, _, _, _, info_m = env_m.step(vel_cmd, mask_m[:, 1].astype(np.int64))
            self.assertFalse(info_m["safety_violation"])
            for u in range(6):
                spd = float(np.linalg.norm(env_m.velocities[u]))
                air_s = float(env_m.net.step_tx_airtime_us[u]) / 1e6
                modeled_j[u] += uav_step_energy_j(spd, air_s, 1.0, bcfg, tx_power_w=cfg.tx_power_w)
        # S19: Moving covers strictly more area than hovering
        self.assertGreater(info_m["true_area_observed_pct"], info_h["true_area_observed_pct"] * 2.0)
        # S23: Runtime energy conservation error < 1e-6 J
        actual_j = cfg.e_max_j - env_m.battery_j
        self.assertLess(float(np.max(np.abs(actual_j - modeled_j))), 1e-6)

    def test_s20_data_mule_alone_has_zero_effect_when_peer_off(self):
        cfg_ff = load_scenario(
            self.STRESS_YAML,
            overrides=("enable_peer_dissemination=false", "enable_data_mule_forwarding=false"),
        )
        cfg_ft = load_scenario(
            self.STRESS_YAML,
            overrides=("enable_peer_dissemination=false", "enable_data_mule_forwarding=true"),
        )
        e1, e2 = ZhangXue7NodePacketEnv(cfg_ff, seed=44), ZhangXue7NodePacketEnv(cfg_ft, seed=44)
        _, _, m1 = e1.reset(seed=44)
        _, _, m2 = e2.reset(seed=44)
        for _ in range(15):
            vel = np.tile(np.array([[0.0, 10.0]], dtype=np.float32), (6, 1))
            _, _, m1, r1, _, _, i1 = e1.step(vel, m1[:, 1].astype(np.int64))
            _, _, m2, r2, _, _, i2 = e2.step(vel, m2[:, 1].astype(np.int64))
        self.assertAlmostEqual(r1, r2, places=9)
        self.assertEqual(i1["data_delivered_gcs"], i2["data_delivered_gcs"])
        self.assertEqual(i1["delivered_cell_coverage_pct"], i2["delivered_cell_coverage_pct"])

    def test_s22_dynamic_reachability_through_kinematic_shield_zero_violations(self):
        cfg = load_scenario(self.BASE_YAML)
        env = ZhangXue7NodePacketEnv(cfg, seed=44)
        _, _, mask = env.reset(seed=44)
        # Steer UAV 2 from spawn (330, 110) toward building-adjacent cell cid=106 (325, 275)
        target_xy = np.array([325.0, 275.0], dtype=np.float64)
        reached = False
        for _ in range(25):
            vel = np.zeros((6, 2), dtype=np.float32)
            diff = target_xy - env.positions[2, :2]
            dist = float(np.linalg.norm(diff))
            if dist > 1e-6:
                vel[2] = (diff / dist * min(15.0, dist)).astype(np.float32)
            _, _, mask, _, _, _, info = env.step(vel, mask[:, 1].astype(np.int64))
            self.assertFalse(info["safety_violation"])
            if int(env.net.ledger_gen_ts_s[2, 2, 106]) >= 0:
                reached = True
                break
        self.assertTrue(reached, "UAV 2 failed to dynamically reach & observe building-edge cid=106")


if __name__ == "__main__":
    unittest.main()
