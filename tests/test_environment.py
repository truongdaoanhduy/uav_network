"""Behavioral checks for the new world, without a network simulator."""
import unittest
from dataclasses import replace

import numpy as np

from uav_network.configuration import Config, load_config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.environment.geometry import segment_hits_circle


def search_config(**world):
    c = Config()
    return replace(c, world=replace(c.world, **world),
                   network=replace(c.network, enabled=False))


class EnvironmentTests(unittest.TestCase):
    def test_same_seed_replays_full_random_rollout(self):
        config = search_config()
        rng = np.random.default_rng(4)
        actions = [Action(rng.uniform(-1, 1, (6, 3)), rng.integers(0, 7, 6))
                   for _ in range(12)]
        env = SwarmSearchEnv(config)
        first = [env.reset(seed=44)[0]]
        first.extend(env.step(a)[0] for a in actions)
        second = [env.reset(seed=44)[0]]
        second.extend(env.step(a)[0] for a in actions)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(first[0].shape, (6, env.observation_dim))

    def test_fov_hides_target_outside_camera_disk(self):
        config = search_config(num_uavs=1, num_targets=1, num_obstacles=0)
        env = SwarmSearchEnv(config)
        layout = dict(positions=[[500, 500]], targets=[[800, 800]], obstacles=[])
        before = env.reset(seed=44, options=layout)[0]
        other = dict(layout, targets=[[850, 800]])
        np.testing.assert_array_equal(before, env.reset(seed=44, options=other)[0])
        after, _ = env.reset(seed=44, options=dict(layout, targets=[[520, 500]]))
        self.assertFalse(np.array_equal(before, after))
        self.assertTrue(env.reports.known[0, 0])
        self.assertNotIn("belief", env.info())

    def test_occlusion_blocks_in_range_target(self):
        c = search_config(num_uavs=1, num_targets=1, num_obstacles=1,
                          obstacle_clearance_m=0)
        env = SwarmSearchEnv(c)
        env.reset(options=dict(positions=[[300, 500]], targets=[[380, 500]],
                               obstacles=[[340, 500, 10]]))
        self.assertFalse(env.reports.known[0, 0])

    def test_acceleration_is_xy_and_speed_is_bounded(self):
        c = search_config(num_uavs=1, num_targets=1, num_obstacles=0)
        env = SwarmSearchEnv(c)
        env.reset(options=dict(positions=[[500, 500]], targets=[[950, 950]], obstacles=[]))
        env.step(Action(np.array([[1., 0., 0.]]), np.array([0])))
        np.testing.assert_allclose(env.positions, [[502, 500]])
        np.testing.assert_allclose(env.velocities, [[2, 0]])
        for _ in range(20):
            env.step(Action(np.array([[1., 1., 0.]]), np.array([0])))
        self.assertLessEqual(np.linalg.norm(env.velocities[0]), c.world.max_speed_mps + 1e-8)

    def test_segment_obstacle_check_prevents_tunneling(self):
        self.assertTrue(segment_hits_circle([0, 0], [100, 0], [50, 0], 2))
        c = search_config(num_uavs=1, num_targets=1, num_obstacles=1,
                          obstacle_clearance_m=0, max_speed_mps=200)
        env = SwarmSearchEnv(c)
        env.reset(options=dict(positions=[[100, 500]], targets=[[900, 900]],
                               obstacles=[[150, 500, 5]]))
        env.velocities[:] = [100, 0]
        _, _, _, _, info = env.step(Action(np.zeros((1, 3)), np.array([0])))
        np.testing.assert_allclose(env.positions, [[100, 500]])
        self.assertEqual(info["blocked_moves"], 1)

    def test_crossing_uav_paths_are_rejected(self):
        c = search_config(num_uavs=2, num_targets=1, num_obstacles=0,
                          safety_distance_m=10, max_speed_mps=200)
        env = SwarmSearchEnv(c)
        env.reset(options=dict(positions=[[100, 500], [200, 500]],
                               targets=[[900, 900]], obstacles=[]))
        env.velocities[:] = [[100, 0], [-100, 0]]
        env.step(Action(np.zeros((2, 3)), np.array([0, 0])))
        np.testing.assert_allclose(env.positions, [[100, 500], [200, 500]])

    def test_mission_deadline_is_terminal_not_external_truncation(self):
        c = search_config(num_uavs=1, num_targets=1, num_obstacles=0, max_steps=1)
        env = SwarmSearchEnv(c)
        env.reset(options=dict(positions=[[100, 100]], targets=[[900, 900]], obstacles=[]))
        _, _, terminated, truncated, _ = env.step(Action(np.zeros((1, 3)), np.array([0])))
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_invalid_actions_fail_without_changing_world(self):
        env = SwarmSearchEnv(search_config())
        env.reset(seed=44)
        original = env.positions.copy()
        with self.assertRaises(ValueError):
            env.step(Action(np.full((6, 3), np.nan), np.zeros(6, dtype=int)))
        with self.assertRaises(ValueError):
            env.step(Action(np.zeros((6, 3)), np.full(6, 0.5)))
        np.testing.assert_array_equal(env.positions, original)

    def test_config_overrides_are_validated(self):
        c = load_config(overrides=["world.num_targets=7", "network.enabled=false"])
        self.assertEqual(c.world.num_targets, 7)
        self.assertFalse(c.network.enabled)
        for item in ["world.typo=1", "world.num_uavs=0", "world.dt_s=.nan"]:
            with self.assertRaises(ValueError):
                load_config(overrides=[item])

    def test_impossible_world_reports_clear_error(self):
        c = search_config(map_size_m=100, num_obstacles=2,
                          obstacle_radius_min_m=80, obstacle_radius_max_m=90)
        with self.assertRaises(ValueError):
            SwarmSearchEnv(c).reset(seed=44)


if __name__ == "__main__":
    unittest.main()
