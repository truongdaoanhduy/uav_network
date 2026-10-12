"""Six living UAVs can choose another relay after an obstructed peer fails."""
import unittest
import numpy as np

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv


class MissionTests(unittest.TestCase):
    def test_six_uav_obstacle_battery_and_policy_selected_three_hop_delivery(self):
        c = Config()
        c.world.num_targets = 1
        c.world.num_obstacles = 1
        c.sensor.altitude_m = 20
        c.network.peer_range_m = 210
        c.network.gcs_range_m = 210
        c.network.obstacle_extra_loss_db = 20
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options=dict(
                positions=[[500, 330], [500, 170], [600, 330],
                           [600, 170], [400, 330], [400, 170]],
                targets=[[500, 330]], obstacles=[[500, 250, 20]]))
            initial_charge = env.battery_j.copy()

            def send(source, destination):
                choices = np.zeros(6, dtype=np.int64)
                choices[source] = destination
                return env.step(Action(np.zeros((6, 3)), choices))

            # UAV0->UAV1 crosses the building. It may be attempted but fails SINR.
            obs, _, _, _, first = send(0, 1)
            self.assertGreater(first["network"]["phy_failures"], 0)
            self.assertIn(0, env.reports.buffers[0])
            self.assertEqual(obs[0, -6], 1)  # NLOS is not hidden by the eligibility mask.
            # The policy chooses UAV2, then UAV3, then GCS; no automatic rerouting.
            send(0, 2)
            self.assertIn(0, env.reports.buffers[2])
            send(2, 3)
            self.assertIn(0, env.reports.buffers[3])
            _, _, terminated, _, final = send(3, 6)
            self.assertTrue(terminated)
            self.assertTrue(final["success"])
            self.assertEqual(final["network"]["report_average_hops"], 3)
            self.assertTrue(np.all(env.battery_j < initial_charge))
            self.assertEqual(final["energy"]["active_uavs"], 6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
