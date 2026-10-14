"""Native oracle checks for the device geometry, motion and actor interface."""
import importlib
import importlib.util
import unittest

import numpy as np
try:
    import warp as wp
except ImportError:
    wp = None

from uav_network.communication.reports import Report
from uav_network.configuration import Config
from uav_network.environment.energy import funded_motion
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.observations import local_observations, centralized_state
from uav_network.environment.world import generate_scenario

if wp is not None:
    wp.init()
    pcg_uniform = None


    @wp.kernel
    def _draw_uniforms(rows: wp.array2d(dtype=wp.uint64), out: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        for k in range(out.shape[1]):
            out[e, k] = pcg_uniform(rows, e)

def _pcg_row(seed):
    state = np.random.PCG64(seed).state["state"]
    return np.array([state["state"] & ((1 << 64) - 1), state["state"] >> 64,
                     state["inc"] & ((1 << 64) - 1), state["inc"] >> 64], dtype=np.uint64)


@unittest.skipIf(wp is None, "optional warp-lang dependency required")
class GPUPhysicsTests(unittest.TestCase):
    def api(self, name):
        package = "uav_network.environment.gpu"
        self.assertIsNotNone(importlib.util.find_spec(package),
                             "GPU physics backend has not been implemented")
        module = f"{package}.{name}"
        self.assertIsNotNone(importlib.util.find_spec(module),
                             f"GPU {name} kernels have not been implemented")
        return importlib.import_module(module)

    def allocate(self, c, count=1, seeds=None):
        self.api("world")
        return self.api("state").allocate_state(c, num_envs=count,
                                               seeds=seeds, device="cpu")

    @staticmethod
    def assign(s, name, value):
        target = getattr(s, name)
        target.assign(np.asarray(value, dtype=target.numpy().dtype))

    def layout(self, positions, targets, obstacles, mode="dynamic_window"):
        c = Config()
        c.world.num_uavs = len(positions)
        c.world.num_targets = len(targets)
        c.world.num_obstacles = len(obstacles)
        c.world.motion_safety = mode
        c.network.enabled = False
        env = SwarmSearchEnv(c)
        env.reset(seed=44, options=dict(positions=positions, targets=targets, obstacles=obstacles))
        self.addCleanup(env.close)
        s, p = self.allocate(c, seeds=[44])
        self.assign(s, "positions", env.positions[None])
        self.assign(s, "targets", env.targets[None])
        self.assign(s, "obstacles", env.obstacles[None])
        self.assign(s, "gcs", env.gcs[None])
        self.assign(s, "battery", env.battery_j[None])
        return c, env, s, p

    def step_motion(self, env, s, p, command):
        command = np.asarray(command, dtype=np.float32)
        details = {}
        pos, vel, blocked, clipped, cost = funded_motion(
            env.positions, env.velocities, command[:, :2], env.obstacles,
            env.battery_j, env.config, diagnostics=details)
        env.positions, env.velocities = pos, vel
        env.propulsion_energy_j += cost
        env.blocked_moves += blocked
        env.boundary_clips += clipped
        for key, value in details.items():
            env.motion_counts[key] += value
        actions = wp.array(command[None], dtype=wp.float32, device="cpu")
        wp.launch(self.api("motion").motion_step, dim=1, inputs=[s, p, actions], device="cpu")
        np.testing.assert_allclose(s.positions.numpy()[0], pos, rtol=0, atol=2e-8)
        np.testing.assert_allclose(s.velocities.numpy()[0], vel, rtol=0, atol=2e-8)
        np.testing.assert_allclose(s.battery.numpy()[0], env.battery_j, rtol=0, atol=2e-7)
        np.testing.assert_allclose(s.flight_energy.numpy()[0], env.propulsion_energy_j,
                                   rtol=0, atol=2e-7)
        counts = s.motion_counts.numpy()[0, :8]
        np.testing.assert_array_equal(counts, list(env.motion_counts.values()))
        self.assertEqual(int(s.blocked.numpy()[0]), env.blocked_moves)
        self.assertEqual(int(s.clipped.numpy()[0]), env.boundary_clips)

    def test_pcg64_uniform_stream_and_final_state_match_numpy(self):
        global pcg_uniform
        pcg_uniform = self.api("world").pcg_uniform
        seeds = [0, 1, 44, 9451, (1 << 63) - 1]
        rows = wp.array(np.stack([_pcg_row(seed) for seed in seeds]), dtype=wp.uint64, device="cpu")
        out = wp.zeros((len(seeds), 1000), dtype=wp.float64, device="cpu")
        wp.launch(_draw_uniforms, dim=len(seeds), inputs=[rows, out], device="cpu")
        expected, final = [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            expected.append(rng.random(1000))
            state = rng.bit_generator.state["state"]
            final.append([state["state"] & ((1 << 64) - 1), state["state"] >> 64,
                          state["inc"] & ((1 << 64) - 1), state["inc"] >> 64])
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(rows.numpy(), np.asarray(final, dtype=np.uint64))

    def test_seeded_reset_replays_native_rejections_and_preserves_unmasked_world(self):
        c = Config()
        seeds = [0, 44, 8459]
        s, p = self.allocate(c, 3, seeds)
        self.assign(s, "pcg_state", np.stack([_pcg_row(seed) for seed in seeds]))
        mask = wp.array(np.array([1, 1, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
        wp.launch(self.api("world").reset_geometry, dim=3, inputs=[s, p, mask], device="cpu")
        for e, seed in enumerate(seeds):
            oracle = generate_scenario(c, np.random.default_rng(seed))
            for name in ("positions", "targets", "obstacles"):
                np.testing.assert_array_equal(getattr(s, name).numpy()[e], getattr(oracle, name))
            np.testing.assert_array_equal(s.gcs.numpy()[e], oracle.gcs)
        before = s.positions.numpy().copy()
        pcg_before = s.pcg_state.numpy().copy()
        mask.assign(np.array([0, 1, 0], dtype=np.int32))
        wp.launch(self.api("world").reset_geometry, dim=3, inputs=[s, p, mask], device="cpu")
        np.testing.assert_array_equal(s.positions.numpy()[[0, 2]], before[[0, 2]])
        np.testing.assert_array_equal(s.pcg_state.numpy()[[0, 2]], pcg_before[[0, 2]])

    def test_motion_safe_oblique_speed_projection(self):
        c, env, s, p = self.layout([[500, 500]], [[950, 950]], [])
        env.velocities[:] = [np.sqrt(336), 8]
        self.assign(s, "velocities", env.velocities[None])
        self.step_motion(env, s, p, [[1, -1, 0]])
        for _ in range(12):
            self.step_motion(env, s, p, [[1, 1, 0]])


    def test_nonbinary_dt_and_acceleration_keep_native_float32_action_rounding(self):
        c, env, s, p = self.layout([[500, 500]], [[950, 950]], [])
        c.world.dt_s = .7
        c.world.max_accel_mps2 = 2.3
        p = self.api("state").params_from_config(c)
        for _ in range(12):
            self.step_motion(env, s, p, [[.23456, -.45678, 0]])

    def test_obstacle_braking_tangent_matches_native_90_step_trace(self):
        c, env, s, p = self.layout([[300, 500]], [[900, 900]], [[500, 500, 50]])
        for _ in range(90):
            self.step_motion(env, s, p, [[1, 0, 0]])

    def test_boundary_braking_and_open_obstacle_side_match_native(self):
        for positions, obstacles in [([[850, 500]], []),
                                     ([[300, 55]], [[500, 55, 50]])]:
            with self.subTest(positions=positions):
                c, env, s, p = self.layout(positions, [[900, 900]], obstacles)
                for _ in range(90):
                    self.step_motion(env, s, p, [[1, 0, 0]])

    def test_head_on_peer_reservations_match_native(self):
        c, env, s, p = self.layout([[300, 500], [700, 500]], [[900, 900]], [])
        for _ in range(80):
            self.step_motion(env, s, p, [[1, 0, 0], [-1, 0, 0]])

    def test_partial_battery_stop_rechecks_peer_and_debits_remaining_charge(self):
        c, env, s, p = self.layout([[100, 500], [130, 500]], [[900, 900]], [])
        env.velocities[:] = [[20, 0], [-20, 0]]
        env.battery_j[0] = 1
        self.assign(s, "velocities", env.velocities[None])
        self.assign(s, "battery", env.battery_j[None])
        self.step_motion(env, s, p, [[0, 0, 0], [0, 0, 0]])

    def test_reject_mode_continuous_guard_and_clipping_match_native(self):
        c, env, s, p = self.layout([[439, 500], [900, 900]], [[950, 950]],
                                  [[500, 500, 50]], mode="reject")
        self.step_motion(env, s, p, [[1, 0, 0], [1, 0, 0]])
        env.positions[1] = [999, 900]
        env.velocities[1] = [20, 0]
        self.assign(s, "positions", env.positions[None])
        self.assign(s, "velocities", env.velocities[None])
        self.step_motion(env, s, p, [[1, 0, 0], [1, 0, 0]])

    def test_visibility_occlusion_fov_and_exhausted_sensor(self):
        c, env, s, p = self.layout([[300, 500], [700, 500]],
                                  [[380, 500], [300, 600], [300, 600.1], [700, 500]],
                                  [[340, 500, 10]])
        env.battery_j[1] = 0
        self.assign(s, "battery", env.battery_j[None])
        wp.launch(self.api("geometry").sense, dim=(1, 2, 4), inputs=[s, p], device="cpu")
        np.testing.assert_array_equal(s.visible.numpy()[0], [[0, 1, 0, 0], [0, 0, 0, 0]])

    def test_default_observation_and_state_layout_match_native(self):
        c = Config()
        c.network.enabled = False
        env = SwarmSearchEnv(c)
        env.reset(seed=44)
        self.addCleanup(env.close)
        s, p = self.allocate(c, seeds=[44])
        for name, value in [("positions", env.positions), ("velocities", env.velocities),
                            ("targets", env.targets), ("obstacles", env.obstacles),
                            ("battery", env.battery_j), ("known", env.reports.known),
                            ("recovery", env.reports.recovery_pending)]:
            self.assign(s, name, value[None])
        self.assign(s, "gcs", env.gcs[None])
        present = np.zeros((1, c.world.num_uavs, c.world.num_targets), np.int32)
        created = np.zeros_like(present, dtype=np.float64)
        for i, reports in enumerate(env.reports.buffers):
            for t, report in reports.items():
                present[0, i, t], created[0, i, t] = 1, report.created_s
        self.assign(s, "report_present", present)
        self.assign(s, "report_created", created)
        obs_module = self.api("observations")
        wp.launch(self.api("geometry").sense, dim=(1, p.n, p.targets_count), inputs=[s, p], device="cpu")
        wp.launch(obs_module.observe, dim=(1, p.n), inputs=[s, p], device="cpu")
        wp.launch(obs_module.centralized, dim=1, inputs=[s, p], device="cpu")
        self.assertEqual(s.observation.shape[-1], 95)
        self.assertEqual(s.state.shape[-1], 217)
        np.testing.assert_array_equal(s.observation.numpy()[0], local_observations(env))
        np.testing.assert_array_equal(s.state.numpy()[0], centralized_state(env))


    def copy_reports(self, env, s):
        self.assign(s, "known", env.reports.known[None])
        self.assign(s, "recovery", env.reports.recovery_pending[None])
        self.assign(s, "delivered", env.reports.delivered[None])
        present = np.zeros((1, env.config.world.num_uavs, env.config.world.num_targets), np.int32)
        created = np.zeros_like(present, dtype=np.float64)
        for i, reports in enumerate(env.reports.buffers):
            for t, report in reports.items():
                present[0, i, t], created[0, i, t] = 1, report.created_s
        self.assign(s, "report_present", present)
        self.assign(s, "report_created", created)
        self.assign(s, "steps", [env.steps])
        self.assign(s, "battery", env.battery_j[None])

    def compare_observations(self, env, s, p):
        self.copy_reports(env, s)
        wp.launch(self.api("geometry").sense, dim=(1, p.n, p.targets_count),
                  inputs=[s, p], device="cpu")
        module = self.api("observations")
        wp.launch(module.observe, dim=(1, p.n), inputs=[s, p], device="cpu")
        wp.launch(module.centralized, dim=1, inputs=[s, p], device="cpu")
        np.testing.assert_array_equal(s.observation.numpy()[0], local_observations(env))
        np.testing.assert_array_equal(s.state.numpy()[0], centralized_state(env))

    def test_recovery_ttl_receiver_admission_and_sorted_slots_match_native(self):
        c, env, s, p = self.layout(
            [[300, 500], [350, 450], [900, 900]],
            [[310, 500], [300, 510], [280, 500], [800, 800], [950, 950]],
            [[300, 550, 5], [360, 500, 5]])
        c.network.enabled = True
        c.network.gcs_range_m = 600
        p = self.api("state").params_from_config(c)
        env.steps = 37
        env.reports.known[:] = [[1, 1, 1, 1, 0], [1, 1, 0, 1, 0], [0, 0, 0, 0, 1]]
        env.reports.recovery_pending[0, :2] = True
        env.reports.buffers = [
            {0: Report(0, 0, 10, c.network.report_bytes),
             2: Report(2, 0, 5, c.network.report_bytes)},
            {0: Report(0, 1, 3, c.network.report_bytes),
             1: Report(1, 1, 9, c.network.report_bytes),
             3: Report(3, 1, 36, c.network.report_bytes)},
            {4: Report(4, 2, 1, c.network.report_bytes)}]
        env.battery_j[2] = 0
        self.compare_observations(env, s, p)
        obs = s.observation.numpy()[0]
        self.assertEqual(obs[0, 10 + p.targets_count + 1], -1)
        self.assertGreater(obs[0, 10 + p.targets_count], 0)
        np.testing.assert_array_equal(obs[0, -4:], [1, 0, 0, 1])
        np.testing.assert_array_equal(obs[2, -4:], [1, 0, 0, 0])
        offset = 10 + 2 * p.targets_count
        np.testing.assert_array_equal(obs[0, offset:offset + 8],
                                      np.array([.01, 0, 1, 1, 0, .01, 1, 1], np.float32))
        del env.reports.buffers[1][3]
        self.compare_observations(env, s, p)
        self.assertEqual(s.observation.numpy()[0, 0, -3], 1)
        c.network.receiver_admission_mask = False
        p = self.api("state").params_from_config(c)
        env.reports.buffers[1][3] = Report(3, 1, 36, c.network.report_bytes)
        self.compare_observations(env, s, p)
        self.assertEqual(s.observation.numpy()[0, 0, -3], 1)

    def test_actor_does_not_leak_hidden_target_coordinates(self):
        c, env, s, p = self.layout([[100, 100]], [[800, 800]], [])
        self.compare_observations(env, s, p)
        before = s.observation.numpy().copy()
        state_before = s.state.numpy().copy()
        env.targets[0] = [850, 800]
        self.assign(s, "targets", env.targets[None])
        self.compare_observations(env, s, p)
        np.testing.assert_array_equal(s.observation.numpy(), before)
        self.assertFalse(np.array_equal(s.state.numpy(), state_before))

    def test_infeasible_injected_velocity_still_uses_continuous_peer_guard(self):
        c, env, s, p = self.layout([[100, 500], [200, 500]], [[900, 900]], [])
        c.world.max_speed_mps = 200
        c.world.safety_distance_m = 10
        p = self.api("state").params_from_config(c)
        env.velocities[:] = [[100, 0], [-100, 0]]
        self.assign(s, "velocities", env.velocities[None])
        self.step_motion(env, s, p, [[0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(s.positions.numpy()[0], [[100, 500], [200, 500]])

    def test_reset_impossible_geometry_sets_explicit_error(self):
        c = Config()
        c.world.map_size_m = 100
        c.world.obstacle_radius_min_m = 80
        c.world.obstacle_radius_max_m = 90
        s, p = self.allocate(c, seeds=[44])
        mask = wp.array(np.array([1], np.int32), dtype=wp.int32, device="cpu")
        wp.launch(self.api("world").reset_geometry, dim=1, inputs=[s, p, mask], device="cpu")
        self.assertEqual(int(s.errors.numpy()[0]), 101)


    @unittest.skipUnless(wp is not None and wp.is_cuda_available(), "CUDA device required for parity")
    def test_cuda_32_seed_reset_and_16_action_physics_parity_repeatability(self):
        c = Config()
        c.network.enabled = False
        seeds = np.arange(32, dtype=np.int64)
        allocator = self.api("state").allocate_state
        cpu, p = allocator(c, num_envs=32, seeds=seeds, device="cpu")
        cuda, _ = allocator(c, num_envs=32, seeds=seeds, device="cuda")
        repeated, _ = allocator(c, num_envs=32, seeds=seeds, device="cuda")
        world = self.api("world")
        geometry = self.api("geometry")
        motion = self.api("motion")
        observations = self.api("observations")
        masks = {device: wp.array(np.ones(32, np.int32), dtype=wp.int32, device=device)
                 for device in ("cpu", "cuda")}
        for state, device in [(cpu, "cpu"), (cuda, "cuda"), (repeated, "cuda")]:
            wp.launch(world.reset_geometry, dim=32, inputs=[state, p, masks[device]], device=device)
        for name in ("positions", "targets", "obstacles", "pcg_state"):
            np.testing.assert_array_equal(getattr(cuda, name).numpy(), getattr(cpu, name).numpy())
        rng = np.random.default_rng(445)
        for step in range(16):
            values = rng.uniform(-1, 1, (32, p.n, 3)).astype(np.float32)
            for state, device in [(cpu, "cpu"), (cuda, "cuda"), (repeated, "cuda")]:
                actions = wp.array(values, dtype=wp.float32, device=device)
                wp.launch(motion.motion_step, dim=32, inputs=[state, p, actions], device=device)
                wp.launch(geometry.sense, dim=(32, p.n, p.targets_count),
                          inputs=[state, p], device=device)
                wp.launch(observations.observe, dim=(32, p.n), inputs=[state, p], device=device)
                wp.launch(observations.centralized, dim=32, inputs=[state, p], device=device)
        for name in ("positions", "velocities", "battery", "flight_energy", "observation", "state"):
            np.testing.assert_allclose(getattr(cuda, name).numpy(), getattr(cpu, name).numpy(),
                                       rtol=0, atol=1e-8)
            np.testing.assert_array_equal(getattr(cuda, name).numpy(), getattr(repeated, name).numpy())
        for name in ("motion_counts", "blocked", "clipped", "visible", "errors"):
            np.testing.assert_array_equal(getattr(cuda, name).numpy(), getattr(cpu, name).numpy())
            np.testing.assert_array_equal(getattr(cuda, name).numpy(), getattr(repeated, name).numpy())


if __name__ == "__main__":
    unittest.main()
