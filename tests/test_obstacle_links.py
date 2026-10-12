"""Obstacles attenuate real UAV peer links, rather than just GCS links."""
import unittest

from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv


class ObstacleLinkTests(unittest.TestCase):
    def gain(self, blocked, extra_loss=0):
        c = Config()
        c.world.num_uavs = 2
        c.world.num_targets = 1
        c.world.num_obstacles = int(blocked)
        c.sensor.altitude_m = 20
        c.network.obstacle_extra_loss_db = extra_loss
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options=dict(positions=[[450, 200], [550, 200]],
                      targets=[[450, 200]], obstacles=[[500, 200, 10]] if blocked else []))
            simulator = env.network.simulator
            return simulator.channel.channel_model.estimates(0, simulator.drones, [0], [1])[(0, 1)]
        finally:
            env.close()

    def test_peer_obstacle_changes_los_to_lower_nlos_gain(self):
        clear, blocked = self.gain(False), self.gain(True)
        self.assertTrue(clear.line_of_sight)
        self.assertFalse(blocked.line_of_sight)
        self.assertLess(blocked.nominal, clear.nominal)

    def test_extra_loss_db_scales_only_obstructed_links(self):
        blocked, extra = self.gain(True), self.gain(True, 10)
        self.assertAlmostEqual(extra.nominal / blocked.nominal, .1, places=10)
        self.assertEqual(self.gain(False).nominal, self.gain(False, 10).nominal)


if __name__ == "__main__":
    unittest.main()
