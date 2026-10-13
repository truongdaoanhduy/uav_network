"""Safety must preserve search access, dynamic limits and identical rollouts."""
import unittest
import numpy as np

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.environment.geometry import segment_distance
from uav_network.environment.motion import move_uavs


def scenario(positions, targets, obstacles, network=False):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = (
        len(positions), len(targets), len(obstacles))
    c.world.motion_safety = "dynamic_window"
    c.network.enabled = network
    env = SwarmSearchEnv(c)
    env.reset(seed=44, options=dict(positions=positions, targets=targets, obstacles=obstacles))
    return env


class AvoidanceTests(unittest.TestCase):
    def test_occluded_target_next_to_obstacle_is_found_without_hard_stops(self):
        env = scenario([[300, 500]], [[561, 500]], [[500, 500, 50]])
        self.addCleanup(env.close)
        self.assertFalse(env.reports.known[0, 0])
        # Scripted desired heading is a probe, not an omniscient MARL policy.
        for _ in range(160):
            old, previous = env.positions.copy(), env.velocities.copy()
            desired = env.targets[0] - env.positions[0]
            desired *= min(1, 12 / max(np.linalg.norm(desired), 1e-12))
            command = np.clip((desired - env.velocities[0]) / 2, -1, 1)
            env.step(Action(np.array([[*command, 0]]), np.array([0])))
            self.assertGreaterEqual(segment_distance(old[0], env.positions[0], [500, 500]), 60 - 1e-7)
            self.assertTrue(np.all(np.abs(env.velocities - previous) <= 2 + 1e-7))
            if env.reports.known[0, 0]:
                break
        self.assertTrue(env.reports.known[0, 0], "UAV is still stuck in front of the obstacle")
        self.assertEqual(env.blocked_moves, 0)

    def test_predictive_braking_prevents_rejection_from_an_initially_safe_state(self):
        env = scenario([[300, 500]], [[900, 900]], [[500, 500, 50]])
        self.addCleanup(env.close)
        for _ in range(90):
            old, previous = env.positions.copy(), env.velocities.copy()
            env.step(Action(np.array([[1., 0, 0]]), np.array([0])))
            self.assertGreaterEqual(segment_distance(old[0], env.positions[0], [500, 500]), 60 - 1e-7)
            self.assertTrue(np.all(np.abs(env.velocities - previous) <= 2 + 1e-7))
        self.assertGreater(env.positions[0, 0], 570)
        self.assertEqual(env.blocked_moves, 0)

    def test_safe_motion_in_narrow_corridor_is_not_repulsed(self):
        env = scenario([[400, 500]], [[900, 900]], [[500, 449, 40], [500, 551, 40]])
        self.addCleanup(env.close)
        env.velocities[:] = [10, 0]
        for _ in range(20):
            env.step(Action(np.zeros((1, 3)), np.array([0])))
        np.testing.assert_allclose(env.positions, [[600, 500]])
        np.testing.assert_allclose(env.velocities, [[10, 0]])

    def test_head_on_uavs_pass_with_continuous_separation(self):
        env = scenario([[300, 500], [700, 500]], [[900, 900]], [])
        self.addCleanup(env.close)
        for _ in range(80):
            old, previous = env.positions.copy(), env.velocities.copy()
            env.step(Action(np.array([[1., 0, 0], [-1., 0, 0]]), np.array([0, 0])))
            self.assertGreaterEqual(segment_distance(old[0] - old[1],
                env.positions[0] - env.positions[1], [0, 0]), 15 - 1e-7)
            self.assertTrue(np.all(np.abs(env.velocities - previous) <= 2 + 1e-7))
        self.assertGreater(env.positions[0, 0], env.positions[1, 0])
        self.assertEqual(env.blocked_moves, 0)

    def test_identical_commands_repeat_the_avoidance_trajectory_exactly(self):
        traces = []
        for _ in range(2):
            env = scenario([[300, 500]], [[900, 900]], [[500, 500, 50]])
            self.addCleanup(env.close)
            trace = []
            for _ in range(60):
                _, reward, _, _, info = env.step(Action(np.array([[1., 0, 0]]), np.array([0])))
                trace.append(np.r_[env.positions.ravel(), env.velocities.ravel(), reward,
                                  info["blocked_moves"]])
            traces.append(np.array(trace))
        np.testing.assert_array_equal(*traces)

    def test_legacy_rejection_can_be_selected_for_a_controlled_baseline(self):
        c = Config()
        c.world.motion_safety = "reject"
        p, v, blocked, _ = move_uavs(np.array([[439., 500]]), np.zeros((1, 2)),
            np.array([[1., 0]]), np.array([[500., 500, 50]]), c)
        np.testing.assert_array_equal(p, [[439., 500]])
        np.testing.assert_array_equal(v, [[0, 0]])
        self.assertEqual(blocked, 1)

    def test_speed_cap_preserves_per_axis_acceleration_at_an_oblique_heading(self):
        c = Config()
        previous = np.array([[np.sqrt(336), 8.]])
        _, actual, blocked, _ = move_uavs(np.array([[500., 500.]]), previous,
            np.array([[1., -1.]]), np.empty((0, 3)), c)
        self.assertTrue(np.all(np.abs(actual - previous) <= 2 + 1e-9))
        self.assertLessEqual(np.linalg.norm(actual), 20 + 1e-9)
        self.assertEqual(blocked, 0)

    def test_dead_peer_is_avoided_without_moving_it(self):
        env = scenario([[300, 500], [500, 500]], [[900, 900]], [])
        self.addCleanup(env.close)
        env.battery_j[1] = 0
        for _ in range(80):
            old = env.positions.copy()
            env.step(Action(np.array([[1., 0, 0], [1., 0, 0]]), np.array([0, 0])))
            self.assertGreaterEqual(segment_distance(old[0] - old[1],
                env.positions[0] - env.positions[1], [0, 0]), 15 - 1e-7)
        np.testing.assert_array_equal(env.positions[1], [500, 500])
        self.assertGreater(env.positions[0, 0], 550)
        self.assertEqual(env.blocked_moves, 0)

    def test_map_boundary_is_approached_with_bounded_braking(self):
        env = scenario([[850, 500]], [[100, 100]], [])
        self.addCleanup(env.close)
        for _ in range(40):
            previous = env.velocities.copy()
            env.step(Action(np.array([[1., 0, 0]]), np.array([0])))
            self.assertTrue(np.all(np.abs(env.velocities - previous) <= 2 + 1e-7))
            self.assertTrue(np.all((env.positions >= 0) & (env.positions <= 1000)))
        self.assertEqual(env.boundary_clips, 0)

    def test_tangential_route_uses_the_open_side_of_a_boundary_adjacent_obstacle(self):
        for center_y in (55., 60.):
            with self.subTest(center_y=center_y):
                env = scenario([[300, center_y]], [[900, 900]], [[500, center_y, 50]])
                self.addCleanup(env.close)
                for _ in range(90):
                    env.step(Action(np.array([[1., 0, 0]]), np.array([0])))
                self.assertGreater(env.positions[0, 0], 600, "UAV chose the sealed side and stalled")
                self.assertEqual(env.blocked_moves, 0)

    def test_target_adjacent_detour_can_complete_native_packet_delivery(self):
        env = scenario([[300, 500]], [[561, 500]], [[500, 500, 50]], network=True)
        self.addCleanup(env.close)
        for _ in range(160):
            desired = env.targets[0] - env.positions[0]
            desired *= min(1, 12 / max(np.linalg.norm(desired), 1e-12))
            command = np.clip((desired - env.velocities[0]) / 2, -1, 1)
            destination = int(env.reports.known[0, 0])
            _, _, terminated, _, info = env.step(Action(np.array([[*command, 0]]),
                                                        np.array([destination])))
            if terminated:
                break
        self.assertTrue(info["success"])
        self.assertEqual(info["targets_delivered"], 1)
        self.assertGreater(info["network"]["received_packets"], 0)
        self.assertEqual(info["blocked_moves"], 0)
        self.assertEqual(info["energy"]["active_uavs"], 1)

    def test_old_checkpoint_keeps_historical_motion_mode(self):
        import torch
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from uav_network.algorithms.checkpoints import make_learner, load_policy
        from uav_network.environment.observations import observation_dim, state_dim
        c = Config()
        c.matd3.hidden_sizes = [16, 16]
        c.runtime.algorithm = "matd3"
        learner = make_learner(c, observation_dim(c), state_dim(c))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            learner.save(path)
            saved = torch.load(path, weights_only=True)
            del saved["config"]["world"]["motion_safety"]
            torch.save(saved, path)
            self.assertEqual(load_policy(path).config.world.motion_safety, "reject")

    def test_unknown_motion_mode_is_rejected(self):
        c = Config()
        c.world.motion_safety = "unknown"
        with self.assertRaisesRegex(ValueError, "motion_safety"):
            c.validate()


if __name__ == "__main__":
    unittest.main()
