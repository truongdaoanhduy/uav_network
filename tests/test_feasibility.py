"""Privileged feasibility controller tests; this is not a learned-policy baseline."""
from pathlib import Path
import unittest

import numpy as np

from uav_network.configuration import load_config
from uav_network.environment.env import SwarmSearchEnv

ROOT = Path(__file__).resolve().parents[1]


class FeasibilityOracleTests(unittest.TestCase):
    def config(self, uavs=3, targets=2):
        c = load_config(ROOT / "configs/comparison-25k-scenario-v3.yaml")
        c.world.num_uavs = uavs
        c.world.num_targets = targets
        c.world.num_obstacles = 0
        c.world.max_steps = 120
        c.sensor.altitude_m = 20
        c.sensor.obstacle_occlusion = False
        c.runtime.num_envs = 1
        c.runtime.device = c.runtime.replay_device = "cpu"
        c.tracking.enabled = False
        return c.validate()

    def test_power_action_inverts_log_mapping_with_safety_margin(self):
        from uav_network.experiments.feasibility import power_action_for_required
        c = self.config().network
        required = 0.01
        action = power_action_for_required(required, c, margin_db=3.0)
        from uav_network.communication.link_budget import power_from_action
        selected = power_from_action(action, c)
        self.assertGreaterEqual(selected, required * 10 ** (3.0 / 10.0) - 1e-12)
        self.assertLessEqual(selected, c.tx_power_max_w)

    def test_target_assignments_are_unique_when_enough_targets_exist(self):
        from uav_network.experiments.feasibility import assign_search_targets
        positions = np.array([[0., 0.], [9., 0.], [20., 0.]])
        targets = np.array([[1., 0.], [10., 0.], [21., 0.]])
        assigned = assign_search_targets(positions, targets, np.array([False, False, False]),
                                         np.array([True, True, True]))
        self.assertEqual(assigned, [0, 1, 2])

    def test_oracle_prefers_direct_gcs_when_report_has_feasible_direct_link(self):
        from uav_network.experiments.feasibility import oracle_action
        c = self.config(uavs=2, targets=1)
        env = SwarmSearchEnv(c)
        self.addCleanup(env.close)
        env.reset(seed=44, options={"positions": [[500, 100], [850, 850]],
                                    "targets": [[500, 100]], "obstacles": []})
        action = oracle_action(env)
        self.assertEqual(int(action.destination[0]), c.world.num_uavs)
        from uav_network.communication.link_budget import nominal_required_power_w, power_from_action
        required = nominal_required_power_w(env.positions[0], env.gcs, env.obstacles, c.network,
                                           c.sensor.altitude_m, True)
        selected = power_from_action(action.continuous[0, 2], c.network)
        self.assertGreaterEqual(selected + 1e-12,
                                min(c.network.tx_power_max_w,
                                    max(c.network.tx_power_min_w, required * 10 ** (3.0 / 10.0))))

    def test_oracle_uses_progress_peer_when_gcs_is_out_of_range(self):
        from uav_network.experiments.feasibility import oracle_action
        c = self.config(uavs=3, targets=1)
        env = SwarmSearchEnv(c)
        self.addCleanup(env.close)
        # GCS=(500,0): source is >500 m away; peer 1 is closer to GCS and within peer range.
        env.reset(seed=44, options={"positions": [[500, 800], [500, 400], [900, 900]],
                                    "targets": [[500, 800]], "obstacles": []})
        action = oracle_action(env)
        # For source 0, ordered peers are [1,2], so peer node 1 is destination index 1.
        self.assertEqual(int(action.destination[0]), 1)

    def test_report_carrier_without_route_biases_motion_toward_gcs(self):
        from uav_network.experiments.feasibility import oracle_action
        c = self.config(uavs=2, targets=2)
        env = SwarmSearchEnv(c)
        self.addCleanup(env.close)
        # Source holds T0 at y=900; T1 is farther away from the GCS at y=950.
        # No peer/direct route exists, so communication urgency must win over T1 search.
        env.reset(seed=44, options={"positions": [[500, 900], [950, 950]],
                                    "targets": [[500, 900], [500, 950]], "obstacles": []})
        action = oracle_action(env)
        self.assertLess(action.continuous[0, 1], 0.0)

    def test_small_deterministic_mission_delivers_with_battery_remaining(self):
        from uav_network.experiments.feasibility import run_explicit_feasibility
        c = self.config(uavs=2, targets=2)
        c.world.max_steps = 100
        result = run_explicit_feasibility(c, seed=44,
            positions=np.array([[500., 100.], [560., 100.]]),
            targets=np.array([[500., 250.], [560., 260.]]), obstacles=np.empty((0, 3)))
        self.assertTrue(result["success"])
        self.assertEqual(result["search_rate_percent"], 100.0)
        self.assertEqual(result["delivery_rate_percent"], 100.0)
        self.assertGreater(min(result["battery_fraction"]), 0.0)


if __name__ == "__main__":
    unittest.main()
