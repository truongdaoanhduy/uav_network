"""Opt-in CPU six-radio battery pilot, with no Warp/GCS parity claim."""
import unittest
from pathlib import Path
from dataclasses import fields
import tempfile

import numpy as np

from uav_network.environment.zhang_xue_battery import (
    steady_level_propulsion_power_w, ZengPropulsionParameters,
)
from uav_network.environment.zhang_xue_coop import (
    ZhangXueConfig, ZhangXueCooperativeEnv,
)


class TestSteadyLevelPropulsion(unittest.TestCase):
    def test_matches_zeng_eq12_audit_table(self):
        # Independent rounded reference values supplied in the v5.2 audit.
        speeds = np.array([0.0, 10.21, 15.0, 18.30, 25.0])
        expected_w = np.array([168.49, 126.01, 138.55, 161.57, 248.96])
        np.testing.assert_allclose(
            steady_level_propulsion_power_w(speeds), expected_w,
            rtol=0.0, atol=0.005,
        )

    def test_rejects_negative_nonfinite_speed_and_invalid_parameters(self):
        for speed in (-1.0, np.nan, np.inf, [0.0, np.nan]):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                steady_level_propulsion_power_w(speed)
        for field in fields(ZengPropulsionParameters):
            with self.subTest(parameter=field.name), self.assertRaises(ValueError):
                ZengPropulsionParameters(**{field.name: np.nan})


class TestBatteryConfig(unittest.TestCase):
    def test_explicit_flat_yaml_selects_physical_battery(self):
        profile = (
            "termination_regime: physical_battery\n"
            "horizon_steps: 1500\n"
            "v_max_mps: 25.0\n"
            "usable_battery_e_max_j: 180000.0\n"
            "avionics_baseline_w: 15.0\n"
            "rf_tx_pa_baseband_w: 0.0\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cpu_battery.yaml"
            path.write_text(profile)
            config = ZhangXueConfig.from_yaml(path)
        self.assertEqual(config.termination_regime, "physical_battery")
        self.assertEqual(config.horizon_steps, 1500)
        self.assertEqual(config.usable_battery_e_max_j, 180000.0)
        self.assertEqual(config.p0_blade_hover_w, 79.86)
        self.assertEqual(ZhangXueConfig().termination_regime, "finite_horizon")
        self.assertEqual(ZhangXueConfig().v_max_mps, 50.0)

    def test_rejects_unknown_nested_duplicate_and_nonphysical_values(self):
        invalid_profiles = [
            "typo_energy_j: 180000\n",
            "environment:\n  termination_regime: regime_c_physical_battery_driven\n"
            "topology:\n  radio_n: 7\n",
            "horizon_steps: 1000\nhorizon_steps: 1500\n",
            "[]\n", "null\n", "dt_seconds: .nan\n", "v_max_mps: .inf\n",
            "horizon_steps: 1.5\n", "horizon_steps: true\n",
            "dt_seconds: 0\n", "v_max_mps: 0\n", "safe_distance_meter: -1\n",
            "no_fly_zone_count: -1\n", "tx_power_w: .nan\n",
            "termination_regime: physical_battery\nusable_battery_e_max_j: 0\n",
            "termination_regime: physical_battery\nrotor_solidity_s: 2\n",
            "termination_regime: physical_battery\nu_tip_mps: 0\n",
            "termination_regime: physical_battery\nrf_tx_pa_baseband_w: 3.0\n",
            "termination_regime: physical_battery\navionics_baseline_w: -.inf\n",
            "termination_regime: unknown\n",
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.yaml"
            for profile in invalid_profiles:
                with self.subTest(profile=profile):
                    path.write_text(profile)
                    with self.assertRaises(ValueError):
                        ZhangXueConfig.from_yaml(path)


class TestBatteryEnvironment(unittest.TestCase):
    def fresh(self, **overrides):
        options = dict(
            termination_regime="physical_battery", v_max_mps=25.0,
            horizon_steps=10, no_fly_zone_count=0, blackout_zone_count=0,
        )
        options.update(overrides)
        return ZhangXueCooperativeEnv(ZhangXueConfig(**options), seed=44)

    def test_soc_and_energy_follow_each_accepted_displacement(self):
        env = self.fresh(dt_seconds=2.0)
        obs, state, mask = env.reset(seed=44)
        np.testing.assert_array_equal(obs[:, 4], np.ones(6))
        self.assertEqual(obs.shape, (6, 121))
        self.assertEqual(state.shape, (6 * 121,))
        self.assertEqual(mask.shape, (6, 6))
        env.positions[0] = [995.0, 500.0]
        actions = np.zeros((6, 2), np.float32)
        actions[:2, 0] = 1.0  # UAV 0 hits boundary; UAV 1 accepts its 50 m move.
        obs, _, _, _, done, info = env.step(actions, np.zeros(6, np.int32))
        expected_j = np.array([366.98, 527.92, 366.98, 366.98, 366.98, 366.98])
        np.testing.assert_allclose(info["battery_energy_used_j"], expected_j,
                                   rtol=0.0, atol=0.01)
        np.testing.assert_allclose(obs[:, 4], 1.0 - expected_j / 180000.0,
                                   rtol=0.0, atol=1e-7)
        self.assertEqual(info["boundary_violations"], 1)
        self.assertEqual(env.velocity[0, 0], 0.0)
        self.assertEqual(env.velocity[1, 0], 25.0)
        self.assertFalse(done)
        self.assertEqual(info["battery_rf_energy_status"], "unavailable_no_per_node_tx_airtime")

    def test_true_battery_stop_and_safety_cap_have_different_bootstrap_flags(self):
        cases = [
            ("finite_horizon", 1, 180000.0, False, True, False),
            ("physical_battery", 1, 180000.0, False, False, True),
            ("physical_battery", 1, 183.49, True, True, False),
            ("physical_battery", 10, 100.0, True, True, False),
        ]
        for regime, horizon, energy_j, exhausted, terminated, truncated in cases:
            with self.subTest(regime=regime, energy_j=energy_j):
                env = self.fresh(termination_regime=regime, horizon_steps=horizon,
                                 usable_battery_e_max_j=energy_j)
                obs, _, _, _, done, info = env.step(
                    np.zeros((6, 2), np.float32), np.zeros(6, np.int32),
                )
                self.assertTrue(done)
                self.assertEqual(info["terminated"], terminated)
                self.assertEqual(info["truncated"], truncated)
                self.assertEqual(info["bootstrap_allowed"], not terminated)
                self.assertEqual(info["battery_exhausted"], exhausted)
                if regime == "physical_battery":
                    self.assertTrue(np.all(obs[:, 4] >= 0.0))
                with self.assertRaises(RuntimeError):
                    env.step(np.zeros((6, 2)), np.zeros(6, np.int32))

    def test_first_depleted_uav_ends_sortie_and_seeded_reset_restores_energy(self):
        env = self.fresh(usable_battery_e_max_j=200.0)
        outputs = []
        for _ in range(2):
            obs, _, _ = env.reset(seed=73)
            np.testing.assert_array_equal(obs[:, 4], np.ones(6))
            np.testing.assert_array_equal(env.battery_remaining_j, np.full(6, 200.0))
            actions = np.zeros((6, 2), np.float32)
            actions[0, 0] = 1.0
            result = env.step(actions, np.zeros(6, np.int32))
            next_obs, _, _, _, done, info = result
            self.assertTrue(done)
            self.assertEqual(info["termination_reason"], "battery_exhaustion")
            self.assertEqual(info["battery_energy_used_j"][0], 200.0)
            self.assertEqual(next_obs[0, 4], 0.0)
            self.assertAlmostEqual(info["battery_remaining_j"][1], 16.51, places=8)
            self.assertGreater(next_obs[1, 4], 0.0)
            outputs.append(result)
        for left, right in zip(outputs[0][:3], outputs[1][:3]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(outputs[0][3:], outputs[1][3:])

    def test_target_confirmation_is_a_true_stop_in_battery_mode(self):
        env = self.fresh(target_count=1, pd=1.0, pf=0.0, confirm_threshold=0.8)
        # Cell (12,10), centered at (625,525), is inside UAV 0's initial camera.
        env.target_mask[:] = False
        env.target_mask[212] = True
        _, _, _, _, done, info = env.step(
            np.zeros((6, 2), np.float32), np.zeros(6, np.int32),
        )
        self.assertTrue(done)
        self.assertTrue(info["mission_success"])
        self.assertTrue(info["terminated"])
        self.assertFalse(info["truncated"])
        self.assertFalse(info["bootstrap_allowed"])
        self.assertFalse(info["battery_exhausted"])
        self.assertEqual(info["termination_reason"], "team_target_confirmation")

    def test_legacy_clock_and_energy_are_unchanged_by_battery_capacity(self):
        env = self.fresh(termination_regime="finite_horizon", horizon_steps=2,
                         usable_battery_e_max_j=1.0)
        obs, _, _, _, done, info = env.step(
            np.zeros((6, 2), np.float32), np.zeros(6, np.int32),
        )
        np.testing.assert_array_equal(obs[:, 4], np.full(6, 0.5))
        np.testing.assert_array_equal(env.battery_remaining_j, np.ones(6))
        self.assertFalse(done)
        self.assertFalse(info["battery_exhausted"])
        self.assertNotIn("battery_energy_used_j", info)


if __name__ == "__main__":
    unittest.main()
