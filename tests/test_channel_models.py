"""Analytical LOS/NLOS must affect actual packets and fail unsupported settings early."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

from uav_network.configuration import Config, load_config
from uav_network.environment.env import Action, SwarmSearchEnv


class ChannelModelTests(unittest.TestCase):
    def test_v1_linear_and_v2_log_power_mapping_and_inverse_fraction(self):
        from uav_network.communication.link_budget import power_from_action, power_fraction
        c = Config()
        self.assertAlmostEqual(power_from_action(-1.0, c.network), 0.1, places=12)
        self.assertAlmostEqual(power_from_action(0.0, c.network), 0.25, places=12)
        self.assertAlmostEqual(power_from_action(1.0, c.network), 0.4, places=12)
        c.network.tx_power_min_w = 0.001
        c.network.tx_power_max_w = 0.4
        c.network.tx_power_mapping = "log"
        self.assertAlmostEqual(power_from_action(-1.0, c.network), 0.001, places=12)
        self.assertAlmostEqual(power_from_action(0.0, c.network), np.sqrt(0.001 * 0.4), places=12)
        self.assertAlmostEqual(power_from_action(1.0, c.network), 0.4, places=12)
        for action in (-1.0, -0.5, 0.0, 0.5, 1.0):
            self.assertAlmostEqual(power_fraction(power_from_action(action, c.network), c.network),
                                   (action + 1.0) / 2.0, places=12)

    def test_required_los_power_at_500m_matches_analytic_link_budget(self):
        from uav_network.communication.link_budget import required_power_w
        required = required_power_w(500.0, 2_400_000_000.0, 22_000_000.0,
                                    -174.0, 7.0, 6.0, "urban", True)
        self.assertAlmostEqual(required, 0.00442144, places=8)

    def test_scenario_v2_configuration_validation_rejects_bad_mapping_power_and_reward_weights(self):
        c = Config()
        c.network.tx_power_mapping = "quadratic"
        with self.assertRaisesRegex(ValueError, "tx_power_mapping"):
            c.validate()
        c = Config(); c.network.tx_power_min_w = 0.0
        with self.assertRaises(ValueError):
            c.validate()
        c = Config(); c.reward.communication_efficiency = -1.0
        with self.assertRaisesRegex(ValueError, "reward"):
            c.validate()
        c = Config(); c.reward.shield_correction = float("nan")
        with self.assertRaisesRegex(ValueError, "reward"):
            c.validate()

    def test_supported_upstream_nlos_models_are_accepted(self):
        for backend in ("native", "warp_packet"):
            for model in ("urban", "suburban"):
                with self.subTest(backend=backend, model=model):
                    config = Config()
                    config.runtime.environment_backend = backend
                    config.network.nlos_model = model
                    config.validate()
                    if backend == "warp_packet" and importlib.util.find_spec("warp"):
                        from uav_network.environment.gpu.state import params_from_config
                        params_from_config(config)

    def test_current_profiles_share_enlarged_scenario_and_packet_backend(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("comparison-20k.yaml", "motion-gpu.yaml", "motion-pql.yaml"):
            with self.subTest(profile=name):
                config = load_config(root / "configs" / name)
                self.assertEqual(config.world.num_uavs, 6)
                self.assertEqual(config.world.num_targets, 35)
                self.assertEqual(config.world.num_obstacles, 25)
                self.assertEqual(config.runtime.environment_backend, "warp_packet")
                self.assertTrue(config.network.propagation_delay_enabled)
                for algorithm in ("masac", "matd3"):
                    config.runtime.algorithm = algorithm
                    config.validate()

    def test_unsupported_nlos_model_is_rejected_before_backend_initialization(self):
        for backend in ("native", "warp_packet"):
            for model in ("free_space", "unknown"):
                with self.subTest(backend=backend, model=model):
                    config = Config()
                    config.runtime.environment_backend = backend
                    config.network.nlos_model = model
                    with self.assertRaisesRegex(ValueError, "nlos_model"):
                        config.validate()

    def backends(self):
        yield "native"
        if importlib.util.find_spec("warp"):
            yield "warp_packet"

    def prepared(self, backend, blocked, extra_loss):
        c = Config()
        c.world.num_uavs, c.world.num_targets = 2, 2
        c.world.num_obstacles = int(blocked)
        c.sensor.altitude_m = 20
        c.network.obstacle_extra_loss_db = extra_loss
        c.runtime.device = "cpu"
        positions = np.array([[450.,500.],[550.,500.]])
        targets = np.array([[450.,500.],[900.,900.]])
        obstacles = np.array([[500.,500.,20.]]) if blocked else np.empty((0,3))
        if backend == "native":
            env = SwarmSearchEnv(c)
            env.enable_communication_trace()
            env.reset(seed=44, options=dict(positions=positions, targets=targets, obstacles=obstacles))
        else:
            import warp as wp
            from uav_network.environment.gpu.vector import WarpMissionEnv
            from uav_network.environment.gpu.geometry import sense
            from uav_network.communication.gpu.mac import reset_network, sense_reports
            c.runtime.environment_backend = backend
            env = WarpMissionEnv(c)
            env.enable_communication_trace()
            env.reset(seed=44)
            s,p = env.vector.s,env.vector.p
            s.positions.assign(positions[None])
            s.targets.assign(targets[None])
            if blocked:
                s.obstacles.assign(obstacles[None])
            wp.launch(reset_network,dim=1,inputs=[s,p,wp.array([1],dtype=wp.int32)],device="cpu")
            wp.launch(sense,dim=(1,2,2),inputs=[s,p],device="cpu")
            wp.launch(sense_reports,dim=1,inputs=[s,p],device="cpu")
            env._refresh()
        self.addCleanup(env.close)
        return env

    def test_power_and_obstruction_change_real_reception_with_native_warp_parity(self):
        outcomes = []
        # At100m the default urban NLOS link needs high power; additional6dB defeats it.
        for blocked,power,extra,expected in ((False,-1,0,True),(True,-1,0,False),
                                              (True,1,0,True),(True,1,6,False)):
            results = []
            for backend in self.backends():
                with self.subTest(backend=backend,blocked=blocked,power=power,extra_loss=extra):
                    env = self.prepared(backend,blocked,extra)
                    controls = np.zeros((2,3),dtype=np.float32)
                    controls[:,2] = power
                    _,_,_,_,info = env.step(Action(controls,np.array([1,0],dtype=np.int64)))
                    events = env.drain_communication_events()
                    received = any(e["source"]==0 and e["destination"]==1 for e in events)
                    self.assertEqual(received,expected)
                    network = info["network"]
                    self.assertGreater(network["uav_communication_energy_j"],0)
                    if not expected:
                        self.assertGreater(network["phy_failures"],0)
                    results.append((received,network["received_packets"],network["phy_failures"],
                                    network["uav_communication_energy_j"]))
                    env.close()
            if len(results)==2:
                self.assertEqual(results[0][:3],results[1][:3])
                self.assertAlmostEqual(results[0][3],results[1][3],places=12)
            outcomes.append(results[0][0])
        self.assertEqual(outcomes,[True,False,True,False])

    @unittest.skipUnless(importlib.util.find_spec("warp"),"optional Warp unavailable")
    def test_scaled_production_scene_resets_and_steps_like_native(self):
        from uav_network.environment.gpu.vector import WarpMissionEnv
        c = load_config(Path(__file__).resolve().parents[1]/"configs/comparison-20k.yaml")
        c.runtime.device = c.runtime.replay_device = "cpu"
        c.runtime.num_envs = 1
        # This exercises enlarged world/state/report arrays rather than only YAML values.
        c.world.num_targets,c.world.num_obstacles = 35,25
        for seed in (0,1,44,99,2000000044,2000000073):
            native = SwarmSearchEnv(c)
            warp = WarpMissionEnv(c)
            self.addCleanup(native.close)
            self.addCleanup(warp.close)
            left,_ = native.reset(seed=seed)
            right,_ = warp.reset(seed=seed)
            self.assertEqual(native.targets.shape,(35,2))
            self.assertEqual(warp.obstacles.shape,(25,3))
            np.testing.assert_allclose(warp.positions,native.positions,rtol=0,atol=2e-8)
            np.testing.assert_allclose(warp.targets,native.targets,rtol=0,atol=2e-8)
            np.testing.assert_allclose(warp.obstacles,native.obstacles,rtol=0,atol=2e-8)
            np.testing.assert_allclose(left,right,rtol=0,atol=2e-7)
            command = Action(np.zeros((6,3),dtype=np.float32),np.zeros(6,dtype=np.int64))
            l = native.step(command)
            r = warp.step(command)
            np.testing.assert_allclose(l[0],r[0],rtol=0,atol=2e-7)
            self.assertAlmostEqual(l[1],r[1],places=7)
            self.assertEqual(l[2:4],r[2:4])
            self.assertGreater(warp.vector.allocated_bytes,0)
            native.close()
            warp.close()


if __name__ == "__main__":
    unittest.main()
