"""Device-vector storage, native-compatible policy outputs and real learner updates."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.algorithms.checkpoints import make_learner


def small_config(algorithm="masac"):
    c = Config()
    c.runtime.algorithm = algorithm
    c.runtime.num_envs = 2
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.environment_backend = "warp_packet"
    c.runtime.vectorize_critics = True
    c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 4
    a = c.algorithm_config
    a.hidden_sizes, a.batch_size = [32,32], 4
    a.replay_capacity, a.learning_starts = 32, 4
    c.runtime.episodes = 2
    return c


@unittest.skipUnless(importlib.util.find_spec("warp"), "optional Warp dependency not installed")
class DeviceVectorTests(unittest.TestCase):
    def test_native_default_and_backend_validation(self):
        c = Config()
        self.assertEqual(c.runtime.environment_backend, "native")
        c.runtime.environment_backend = "unknown"
        with self.assertRaises(ValueError):
            c.validate()

    def test_warp_packet_scheduler_defaults_to_gen1_and_rejects_unknown_mode(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        c = small_config()
        vector = WarpVectorEnv(c, device="cpu")
        try:
            self.assertEqual(vector.packet_scheduler, "gen1")
        finally:
            vector.close()
        rxindex = WarpVectorEnv(c, device="cpu", packet_scheduler="rxindex")
        try:
            self.assertEqual(rxindex.packet_scheduler, "rxindex")
        finally:
            rxindex.close()
        with self.assertRaisesRegex(ValueError, "packet scheduler"):
            WarpVectorEnv(c, device="cpu", packet_scheduler="unknown")

    def test_tensor_actions_equal_legacy_deterministic_actions_for_both_learners(self):
        torch.set_num_threads(1)
        for algorithm in ("masac","matd3"):
            c = small_config(algorithm)
            c.runtime.environment_backend = "native"
            env = SwarmSearchEnv(c)
            try:
                observations, _ = env.reset(seed=44)
                obs = np.stack([observations,observations])
                learner = make_learner(c,env.observation_dim,env.state_dim,"cpu")
                native = learner.act_batch(obs,deterministic=True)
                tensor = learner.act_batch(torch.as_tensor(obs),deterministic=True,return_tensors=True)
                self.assertIsInstance(tensor.continuous,torch.Tensor)
                self.assertIsInstance(tensor.destination,torch.Tensor)
                self.assertEqual(tensor.destination.dtype,torch.int64)
                np.testing.assert_array_equal(tensor.continuous.numpy(),native.continuous)
                np.testing.assert_array_equal(tensor.destination.numpy(),native.destination)
            finally:
                env.close()

    def test_vector_has_no_cpu_workers_and_does_not_overwrite_previous_observations(self):
        from uav_network.environment.vector import make_vector
        vector = make_vector(small_config())
        try:
            obs, _ = vector.reset(seed=44)
            self.assertEqual(vector.num_cpu_workers,0)
            self.assertIsInstance(obs["observation"],torch.Tensor)
            snapshot = {k:v.clone() for k,v in obs.items()}
            actions = {"continuous":torch.full((2,6,3),.2),"destination":torch.zeros((2,6),dtype=torch.int64)}
            following, reward, terminated, _, _ = vector.step(actions)
            for key in obs:
                torch.testing.assert_close(obs[key],snapshot[key],rtol=0,atol=0)
                self.assertNotEqual(obs[key].data_ptr(),following[key].data_ptr())
            self.assertIsInstance(reward,torch.Tensor)
            self.assertEqual(terminated.dtype,torch.bool)
        finally:
            vector.close()

    def test_masked_reset_retains_unselected_world_and_shapes(self):
        from uav_network.environment.vector import make_vector
        vector = make_vector(small_config())
        try:
            before, _ = vector.reset(seed=[44,45])
            saved = {k:v[1].clone() for k,v in before.items()}
            after, _ = vector.reset(seed=[99,None],options={"reset_mask":np.array([True,False])})
            for key in saved:
                torch.testing.assert_close(after[key][1],saved[key],rtol=0,atol=0)
            self.assertEqual(after["observation"].shape[:2],(2,6))
        finally:
            vector.close()

    def test_masked_reset_does_not_regenerate_expired_reports_in_running_world(self):
        from uav_network.environment.vector import make_vector
        c = small_config()
        c.network.enabled = False
        c.network.report_ttl_s = .5
        c.sensor.altitude_m = 10000
        c.sensor.obstacle_occlusion = False
        vector = make_vector(c)
        try:
            vector.reset(seed=[44,45])
            obs, _, _, _, _ = vector.step({
                "continuous":torch.zeros((2,6,3)),
                "destination":torch.zeros((2,6),dtype=torch.int64)})
            arrays = ("report_present","report_counts","report_created","report_sequence")
            saved = {name:getattr(vector.s,name).numpy()[1].copy() for name in arrays}
            observation = {k:v[1].clone() for k,v in obs.items()}
            actual,_ = vector.reset(seed=[99,None],options={"reset_mask":np.array([True,False])})
            for name in arrays:
                np.testing.assert_array_equal(getattr(vector.s,name).numpy()[1],saved[name])
            for key in observation:
                torch.testing.assert_close(actual[key][1],observation[key],rtol=0,atol=0)
        finally:
            vector.close()

    def test_vector_reset_propagates_impossible_geometry_placement(self):
        from uav_network.environment.vector import make_vector
        c = small_config()
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2,1,0
        c.world.launch_min_spacing_m = 1000
        c.world.placement_attempts = 10
        vector = make_vector(c)
        try:
            with self.assertRaisesRegex(RuntimeError,"103"):
                vector.reset(seed=44)
        finally:
            vector.close()

    def test_fractional_destination_is_rejected_without_advancing_world(self):
        from uav_network.environment.vector import make_vector
        vector = make_vector(small_config())
        try:
            vector.reset(seed=44)
            before = vector.s.positions.numpy().copy()
            with self.assertRaises(ValueError):
                vector.step({"continuous":torch.zeros((2,6,3)),
                             "destination":torch.full((2,6),1.5)})
            np.testing.assert_array_equal(before,vector.s.positions.numpy())
        finally:
            vector.close()

    def test_invalid_device_action_fails_before_motion_and_can_be_reset(self):
        from uav_network.environment.vector import make_vector
        vector = make_vector(small_config())
        try:
            vector.reset(seed=44)
            before = vector.s.positions.numpy().copy()
            actions = {"continuous":torch.zeros((2,6,3)),"destination":torch.zeros((2,6),dtype=torch.int64)}
            actions["continuous"][0,0,0] = float("nan")
            with self.assertRaises(RuntimeError):
                vector.step(actions)
            np.testing.assert_array_equal(before,vector.s.positions.numpy())
            vector.reset(seed=44)
            actions["continuous"].zero_()
            vector.step(actions)
        finally:
            vector.close()

    def test_full_environment_script_matches_native_motion_rewards_and_packets(self):
        from uav_network.environment.vector import make_vector
        for seed in (44,99):
            c = Config()
            c.world.max_steps = 60
            c.world.target_exclusion_radius_m = 0
            c.runtime.device = "cpu"
            c.runtime.num_envs = 1
            native = SwarmSearchEnv(c)
            c.runtime.environment_backend = "warp_packet"
            device = make_vector(c)
            try:
                expected,_ = native.reset(seed=seed)
                actual,_ = device.reset(seed=seed)
                np.testing.assert_array_equal(actual["observation"][0].numpy(),expected)
                for step in range(24):
                    continuous = np.tile(np.array([.25,.125,.5],dtype=np.float32),(6,1))
                    destination = np.full(6,6,dtype=np.int64)
                    from uav_network.environment.env import Action
                    expected,reward,ended,_,info = native.step(Action(continuous,destination))
                    actual,r,done,_,_ = device.step({"continuous":continuous[None],"destination":destination[None]})
                    np.testing.assert_allclose(actual["observation"][0].numpy(),expected,rtol=0,atol=2e-6)
                    np.testing.assert_allclose(actual["state"][0].numpy(),native.state(),rtol=0,atol=2e-6)
                    self.assertAlmostEqual(float(r[0]),reward,places=5)
                    self.assertEqual(bool(done[0]),ended)
                    from uav_network.communication.gpu.metrics import world_info
                    measured = world_info(device.s,device.p,0)
                    for key in ("targets_found","targets_delivered","expired_reports","dropped_reports"):
                        self.assertEqual(measured[key],info[key])
                    for key in ("generated_packets","received_packets","phy_successes","phy_failures","collisions"):
                        self.assertEqual(measured["network"][key],info["network"][key])
            finally:
                native.close()
                device.close()

    def test_progress_metrics_tensor_reductions_match_numpy_metrics(self):
        from uav_network.experiments.tracking import progress_metrics
        from uav_network.environment.vector import make_vector
        c = small_config()
        vector = make_vector(c)
        try:
            observations,_ = vector.reset(seed=44)
            numpy_values = {k:v.numpy() for k,v in observations.items()}
            expected = progress_metrics(numpy_values,c)
            actual = progress_metrics(observations,c)
            for key in expected:
                self.assertAlmostEqual(actual[key],expected[key],places=5)
        finally:
            vector.close()

    def test_both_learners_train_real_warp_worlds_and_evaluate_checkpoint_on_cpu(self):
        from uav_network.experiments.training import train
        from uav_network.experiments.evaluation import evaluate
        with TemporaryDirectory() as directory:
            for algorithm in ("masac","matd3"):
                c = small_config(algorithm)
                c.runtime.output_dir = str(Path(directory)/algorithm)
                result = train(c)
                self.assertGreater(result["updates"],0)
                self.assertEqual(result["environment_backend"],"warp_packet")
                self.assertEqual(result["num_cpu_env_workers"],0)
                checkpoint = torch.load(result["checkpoint"],map_location="cpu",weights_only=True)
                self.assertEqual(checkpoint["config"]["runtime"]["environment_backend"],"warp_packet")
                evaluation = evaluate(result["checkpoint"],episodes=1,seed=10044,
                                      output_dir=Path(directory)/(algorithm+"-eval"))
                self.assertEqual(evaluation["device"],"cpu")
                self.assertEqual(evaluation["environment_backend"],"warp_packet")


if __name__ == "__main__":
    unittest.main()
