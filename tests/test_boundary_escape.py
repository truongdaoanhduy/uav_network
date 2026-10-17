"""Boundary safety must not turn an open edge into a permanent stop."""
import unittest
import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None

from uav_network.configuration import Config
from uav_network.environment.geometry import segment_distance
from uav_network.environment.motion import move_uavs


def config(agents=1, obstacles=0):
    value = Config()
    value.world.map_size_m = 100
    value.world.num_uavs = agents
    value.world.num_targets = 1
    value.world.num_obstacles = obstacles
    value.network.enabled = False
    value.energy.enabled = False
    return value


def boundaries():
    return [([0., 0.], [-1., -1.]), ([0., 100.], [-1., 1.]),
            ([100., 100.], [1., 1.]), ([100., 0.], [1., -1.]),
            ([0., 50.], [-1., 0.]), ([100., 50.], [1., 0.]),
            ([50., 0.], [0., -1.]), ([50., 100.], [0., 1.]),
            ([50., 50.], [-1., -1.]), ([0., 0.], [-.5, -1.])]


class BoundaryEscapeTests(unittest.TestCase):
    def test_outward_commands_keep_progress_through_repeated_boundary_turns(self):
        c = config()
        for start, command in boundaries():
            with self.subTest(start=start, command=command):
                position = np.array([start]); velocity = np.zeros((1, 2))
                trace = [position[0].copy()]
                longest_stall = stalled = 0
                for _ in range(250):
                    old_velocity = velocity.copy()
                    position, velocity, blocked, clipped = move_uavs(position, velocity,
                        np.array([command], np.float32), np.empty((0, 3)), c)
                    trace.append(position[0].copy())
                    self.assertTrue(np.all(np.abs(velocity - old_velocity) <= 2 + 1e-8))
                    self.assertTrue(np.all((position >= 0) & (position <= 100)))
                    self.assertEqual((blocked, clipped), (0, 0))
                    stalled = stalled + 1 if np.linalg.norm(velocity) < 1e-8 else 0
                    longest_stall = max(longest_stall, stalled)
                trace = np.asarray(trace)
                # Distance alone would accept a 2m inward/outward jitter loop.
                self.assertGreater(float(np.ptp(trace, axis=0).max()), 50)
                self.assertGreater(float(np.linalg.norm(np.diff(trace[-101:], axis=0), axis=1).sum()), 25)
                self.assertGreater(float(np.ptp(trace[-101:], axis=0).max()), 25,
                                   'Escape must not settle into a small corner jitter loop')
                self.assertLessEqual(longest_stall, 2)

    def test_boundary_escape_respects_obstacles_peers_and_a_closed_cage(self):
        scenes = [
            ([[0., 50.]], [[5., 35.85, 5.]], [False]),
            ([[0., 50.], [0., 35.]], [], [False, True]),
            ([[0., 50.], [0., 35.], [0., 65.], [15., 50.]], [], [False, True, True, True])]
        for start, obstacles, disabled in scenes:
            with self.subTest(start=start, obstacles=obstacles):
                c = config(len(start), len(obstacles))
                position = np.array(start); velocity = np.zeros_like(position)
                circles = np.asarray(obstacles).reshape(-1, 3)
                actions = np.zeros_like(position, dtype=np.float32); actions[0] = [-1., 0.]
                for step in range(20):
                    old = position.copy(); old_velocity = velocity.copy()
                    position, velocity, blocked, clipped = move_uavs(position, velocity,
                        actions, circles, c, disabled=np.array(disabled))
                    self.assertEqual((blocked, clipped), (0, 0))
                    self.assertTrue(np.all(np.abs(velocity - old_velocity) <= 2 + 1e-8))
                    for circle in circles:
                        self.assertGreaterEqual(segment_distance(old[0], position[0], circle[:2]),
                                                circle[2] + c.world.obstacle_clearance_m - 1e-8)
                    for j in range(1, len(start)):
                        self.assertGreaterEqual(segment_distance(old[0] - old[j],
                            position[0] - position[j], [0., 0.]), c.world.safety_distance_m - 1e-8)
                    np.testing.assert_array_equal(position[1:], np.asarray(start)[1:])
                    if len(start) < 4 and step == 0:
                        self.assertGreater(position[0, 1], start[0][1], 'Use the unblocked tangent')
                if len(start) == 4:
                    np.testing.assert_array_equal(position, start)

    def test_safe_interior_velocity_is_unchanged(self):
        diagnostics = {}
        position, velocity, blocked, clipped = move_uavs(np.array([[50., 50.]]),
            np.array([[1., 2.]]), np.zeros((1, 2)), np.empty((0, 3)), config(), diagnostics=diagnostics)
        np.testing.assert_array_equal(velocity, [[1., 2.]])
        np.testing.assert_array_equal(position, [[51., 52.]])
        self.assertEqual((blocked, clipped, diagnostics['shield_interventions']), (0, 0, 0))
        self.assertEqual(diagnostics['shield_correction'], 0.0)

    def test_shield_correction_is_positive_on_intervention_and_zero_with_no_active_uav(self):
        c = config()
        diagnostics = {}
        move_uavs(np.array([[0., 50.]]), np.zeros((1,2)), np.array([[-1.,0.]]),
                  np.empty((0,3)), c, diagnostics=diagnostics)
        self.assertGreater(diagnostics['shield_correction'], 0.0)
        self.assertTrue(np.isfinite(diagnostics['shield_correction']))
        disabled = {}
        move_uavs(np.array([[0., 50.]]), np.zeros((1,2)), np.array([[-1.,0.]]),
                  np.empty((0,3)), c, disabled=np.array([True]), diagnostics=disabled)
        self.assertEqual(disabled['shield_correction'], 0.0)

    @unittest.skipIf(wp is None, 'optional warp-lang dependency required')
    def test_warp_matches_native_escape_and_constrained_safety(self):
        from uav_network.environment.gpu.state import allocate_state
        from uav_network.environment.gpu.motion import motion_step
        wp.init()
        c = config()
        starts = np.asarray([[start] for start, _ in boundaries()])
        commands = np.zeros((len(starts), 1, 3), np.float32)
        commands[..., :2] = np.asarray([[command] for _, command in boundaries()])
        cases = [(c, starts, commands, np.empty((0, 3)), np.array([False]), 250)]
        cage = np.array([[[0., 50.], [0., 35.], [0., 65.], [15., 50.]]])
        cage_commands = np.zeros((1, 4, 3), np.float32); cage_commands[0, 0, :2] = [-1., 0.]
        cage_config = config(4); cage_config.energy.enabled = True
        cases.append((cage_config, cage, cage_commands, np.empty((0, 3)), np.array([False, True, True, True]), 20))
        obstacle_commands = np.array([[[-1., 0., 0.]]], np.float32)
        cases.append((config(1, 1), np.array([[[0., 50.]]]), obstacle_commands,
                      np.array([[5., 35.85, 5.]]), np.array([False]), 20))
        for c, position, commands, obstacles, disabled, steps in cases:
            with self.subTest(agents=c.world.num_uavs, obstacles=c.world.num_obstacles):
                position = position.copy(); velocity = np.zeros_like(position)
                state, params = allocate_state(c, num_envs=len(position), device='cpu')
                state.positions.assign(position)
                battery = state.battery.numpy(); battery[:, disabled] = 0; state.battery.assign(battery)
                if len(obstacles):
                    state.obstacles.assign(obstacles[None])
                actions = wp.array(commands, dtype=wp.float32, device='cpu')
                for _ in range(steps):
                    for e in range(len(position)):
                        position[e], velocity[e], _, _ = move_uavs(position[e], velocity[e],
                            commands[e, :, :2], obstacles, c, disabled=disabled)
                    wp.launch(motion_step, dim=len(position), inputs=[state, params, actions], device='cpu')
                    np.testing.assert_allclose(state.positions.numpy(), position, rtol=0, atol=2e-8)
                    np.testing.assert_allclose(state.velocities.numpy(), velocity, rtol=0, atol=2e-8)
                self.assertFalse(state.blocked.numpy().any())
                self.assertFalse(state.clipped.numpy().any())


if __name__ == '__main__':
    unittest.main()
