"""Real CUDA packet oracles, device storage, repeatable training and CPU handoff."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

import test_gpu_packets as packets
from test_gpu_vector import small_config

CUDA_WARP = torch.cuda.is_available() and importlib.util.find_spec("warp") is not None


@unittest.skipUnless(CUDA_WARP, "requires real CUDA and optional Warp dependency")
class CudaPacketTests(packets.GPUPacketTests):
    device = "cuda"


@unittest.skipUnless(CUDA_WARP, "requires real CUDA and optional Warp dependency")
class CudaDeviceIntegrationTests(unittest.TestCase):
    def test_composed_cpu_cuda_worlds_match_identical_actions_and_reset_seeds(self):
        from uav_network.communication.gpu.metrics import world_info
        from uav_network.configuration import Config
        from uav_network.environment.gpu.vector import WarpVectorEnv
        c = Config()
        c.runtime.environment_backend = "warp_packet"
        c.runtime.num_envs = 4
        c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 3, 2, 60
        cpu, cuda = WarpVectorEnv(c, device="cpu"), WarpVectorEnv(c, device="cuda")
        try:
            expected, _ = cpu.reset(seed=44)
            actual, _ = cuda.reset(seed=44)
            for key in actual:
                self.assertTrue(actual[key].is_cuda)
                torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=0, atol=2e-6)
            self.assertEqual(cuda.num_cpu_workers, 0)
            rng = np.random.default_rng(991)
            for step in range(24):
                continuous = rng.uniform(-1, 1, (4, 6, 3)).astype(np.float32)
                destination = rng.integers(0, 7, (4, 6), dtype=np.int64)
                expected, reward, done, _, _ = cpu.step(dict(continuous=continuous, destination=destination))
                actual, r, ended, _, _ = cuda.step(dict(
                    continuous=torch.as_tensor(continuous, device="cuda"),
                    destination=torch.as_tensor(destination, device="cuda")))
                for key in actual:
                    torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=0, atol=2e-6)
                torch.testing.assert_close(r.cpu(), reward, rtol=0, atol=2e-6)
                torch.testing.assert_close(ended.cpu(), done, rtol=0, atol=0)
                for i in range(4):
                    left, right = world_info(cpu.s, cpu.p, i), world_info(cuda.s, cuda.p, i)
                    for key in ("targets_found", "targets_delivered", "dropped_reports",
                                "expired_reports", "blocked_moves", "boundary_clips"):
                        self.assertEqual(left[key], right[key])
                    for key in ("generated_packets", "received_packets", "phy_successes",
                                "phy_failures", "collisions"):
                        self.assertEqual(left["network"][key], right["network"][key])
            mask = np.array([True, False, True, False])
            seeds = [99, None, 101, None]
            a, _ = cpu.reset(seed=seeds, options={"reset_mask": mask})
            b, _ = cuda.reset(seed=seeds, options={"reset_mask": mask})
            for key in a:
                torch.testing.assert_close(b[key].cpu(), a[key], rtol=0, atol=2e-6)
        finally:
            cpu.close()
            cuda.close()

    def test_gen1_gen2_active_one_megabyte_session_is_bit_exact_on_cuda(self):
        import warp as wp
        from uav_network.configuration import load_config
        from uav_network.environment.gpu.vector import WarpVectorEnv
        from uav_network.environment.gpu.geometry import sense
        from uav_network.communication.gpu.mac import sense_reports

        root = Path(__file__).resolve().parents[1]
        c = load_config(root / "configs/comparison-25k-scenario-v3.yaml")
        c.runtime.device = c.runtime.replay_device = "cuda"
        c.runtime.num_envs = 2
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 0
        c.world.max_steps = 20
        c.tracking.enabled = False
        gen1 = WarpVectorEnv(c, device="cuda", packet_scheduler="gen1")
        gen2 = WarpVectorEnv(c, device="cuda", packet_scheduler="gen2")
        try:
            for vector in (gen1, gen2):
                vector.reset(seed=[44,45])
                positions = vector.s.positions.numpy()
                targets = vector.s.targets.numpy()
                targets[:,0,:] = positions[:,0,:]
                vector.s.targets.assign(targets)
                vector._launch(sense,(2,2,2),vector.s,vector.p)
                vector._launch(sense_reports,2,vector.s,vector.p)
                vector._publish(vector._buffers[vector._buffer])
                wp.synchronize_device(vector.wp_device)
            continuous = torch.zeros((2,2,3),device="cuda",dtype=torch.float32)
            continuous[:,0,2] = 1.0
            destination = torch.zeros((2,2),device="cuda",dtype=torch.int64)
            destination[:,0] = 2
            actions = {"continuous":continuous,"destination":destination}
            for _ in range(2):
                left = gen1.step(actions)
                right = gen2.step(actions)
                torch.testing.assert_close(left[1],right[1],rtol=0,atol=0,equal_nan=True)
                torch.testing.assert_close(left[2],right[2],rtol=0,atol=0)
                snap1,snap2 = gen1.snapshot_state(),gen2.snapshot_state()
                self.assertEqual(snap1["arrays"].keys(),snap2["arrays"].keys())
                for name,value in snap1["arrays"].items():
                    torch.testing.assert_close(value,snap2["arrays"][name],rtol=0,atol=0,
                                               equal_nan=True,msg=name)
            self.assertTrue(bool((gen2.snapshot_state()["arrays"]["session_active"] != 0).any()),
                            "CUDA exactness corpus must retain an active 1 MB session")
            self.assertEqual(gen2.scheduler_metrics(), {
                "attempted_chains": 0, "completed_chains": 0, "fused_events": 0,
                "competitor_fallbacks": 0, "boundary_fallbacks": 0, "max_chain_length": 0,
            })
        finally:
            gen1.close(); gen2.close()


    def test_gen1_rxindex_active_one_megabyte_session_is_bit_exact_on_cuda(self):
        import warp as wp
        from uav_network.configuration import load_config
        from uav_network.environment.gpu.vector import WarpVectorEnv
        from uav_network.environment.gpu.geometry import sense
        from uav_network.communication.gpu.mac import sense_reports

        root = Path(__file__).resolve().parents[1]
        c = load_config(root / "configs/comparison-25k-scenario-v3.yaml")
        c.runtime.device = c.runtime.replay_device = "cuda"
        c.runtime.num_envs = 2
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 0
        c.world.max_steps = 20
        c.tracking.enabled = False
        gen1 = WarpVectorEnv(c, device="cuda", packet_scheduler="gen1")
        rxindex = WarpVectorEnv(c, device="cuda", packet_scheduler="rxindex")
        try:
            for vector in (gen1, rxindex):
                vector.reset(seed=[44,45])
                positions = vector.s.positions.numpy()
                targets = vector.s.targets.numpy()
                targets[:,0,:] = positions[:,0,:]
                vector.s.targets.assign(targets)
                vector._launch(sense,(2,2,2),vector.s,vector.p)
                vector._launch(sense_reports,2,vector.s,vector.p)
                vector._publish(vector._buffers[vector._buffer])
                wp.synchronize_device(vector.wp_device)
            continuous = torch.zeros((2,2,3),device="cuda",dtype=torch.float32)
            continuous[:,0,2] = 1.0
            destination = torch.zeros((2,2),device="cuda",dtype=torch.int64)
            destination[:,0] = 2
            actions = {"continuous":continuous,"destination":destination}
            for _ in range(2):
                left = gen1.step(actions)
                right = rxindex.step(actions)
                torch.testing.assert_close(left[1],right[1],rtol=0,atol=0,equal_nan=True)
                torch.testing.assert_close(left[2],right[2],rtol=0,atol=0)
                snap1,snap2 = gen1.snapshot_state(),rxindex.snapshot_state()
                self.assertEqual(snap1["arrays"].keys(),snap2["arrays"].keys())
                for name,value in snap1["arrays"].items():
                    torch.testing.assert_close(value,snap2["arrays"][name],rtol=0,atol=0,
                                               equal_nan=True,msg=name)
            self.assertTrue(bool((rxindex.snapshot_state()["arrays"]["session_active"] != 0).any()),
                            "CUDA exactness corpus must retain an active 1 MB session")
            self.assertEqual(rxindex.scheduler_metrics(), {
                "attempted_chains": 0, "completed_chains": 0, "fused_events": 0,
                "competitor_fallbacks": 0, "boundary_fallbacks": 0, "max_chain_length": 0,
            })
            self.assertGreater(rxindex.allocated_bytes,gen1.allocated_bytes)
        finally:
            gen1.close(); rxindex.close()

    def test_vector_follows_nondefault_torch_stream_and_waits_for_async_step(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        c = small_config()
        c.runtime.device = "cuda"
        cuda = WarpVectorEnv(c)
        cpu = WarpVectorEnv(c,device="cpu")
        try:
            cpu.reset(seed=44)
            cuda.reset(seed=44)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                cuda.reset(seed=99)
                action = {"continuous":torch.full((2,6,3),.125,device="cuda"),
                          "destination":torch.full((2,6),6,device="cuda",dtype=torch.int64)}
                cuda.step_async(action)
            actual, reward, done, _, _ = cuda.step_wait()
            cpu.reset(seed=99)
            expected, r, ended, _, _ = cpu.step({
                "continuous":np.full((2,6,3),.125,dtype=np.float32),
                "destination":np.full((2,6),6,dtype=np.int64)})
            for key in actual:
                torch.testing.assert_close(actual[key].cpu(),expected[key],rtol=0,atol=2e-6)
            torch.testing.assert_close(reward.cpu(),r,rtol=0,atol=2e-6)
            torch.testing.assert_close(done.cpu(),ended,rtol=0,atol=0)
        finally:
            cuda.close()
            cpu.close()

    def test_gpu_packet_worlds_survive_1000_vector_steps_with_checked_capacity(self):
        from uav_network.configuration import Config
        from uav_network.environment.gpu.vector import WarpVectorEnv
        c = Config()
        c.runtime.environment_backend = "warp_packet"
        c.runtime.device = c.runtime.replay_device = "cuda"
        c.runtime.num_envs = 8
        c.runtime.coalesce_idle_polls = True
        c.world.max_steps = 1000
        vector = WarpVectorEnv(c)
        try:
            obs,_ = vector.reset(seed=44)
            next_seed = 52
            for step in range(1000):
                action = vector.random_actions(obs["observation"])
                obs, reward, done, _, infos = vector.step({
                    "continuous":action.continuous,"destination":action.destination})
                self.assertTrue(bool(torch.isfinite(reward).all()))
                if infos["device_episode_infos"]:
                    seeds = [None]*8
                    for index in infos["device_episode_infos"]:
                        seeds[index],next_seed = next_seed,next_seed+1
                    obs,_ = vector.reset(seed=seeds,options={"reset_mask":done.cpu().numpy()})
            self.assertTrue(obs["observation"].is_cuda)
            self.assertTrue(bool(torch.isfinite(obs["observation"]).all()))
            self.assertEqual(vector.num_cpu_workers,0)
        finally:
            vector.close()

    def test_both_cuda_learners_repeat_and_visualize_in_a_separate_cpu_process(self):
        from uav_network.experiments.training import train
        torch.set_num_threads(1)
        with TemporaryDirectory() as folder:
            for algorithm in ("masac", "matd3"):
                c = small_config(algorithm)
                c.runtime.device = c.runtime.replay_device = "cuda"
                histories = []
                for label in ("first", "repeat"):
                    c.runtime.output_dir = str(Path(folder) / algorithm / label)
                    result = train(c)
                    self.assertEqual(result["device"], "cuda")
                    self.assertEqual(result["replay_device"], "cuda")
                    self.assertEqual(result["environment_backend"], "warp_packet")
                    self.assertEqual(result["num_cpu_env_workers"], 0)
                    self.assertGreater(result["updates"], 0)
                    histories.append(result["reproducibility"])
                    saved = torch.load(result["checkpoint"], map_location="cuda", weights_only=True)
                    self.assertTrue(all(t.is_cuda and torch.isfinite(t).all()
                                        for t in saved["model"].values()))
                self.assertEqual(histories[0]["episode_hash"], histories[1]["episode_hash"])
                self.assertEqual(histories[0]["model_hash"], histories[1]["model_hash"])
                evaluation = Path(folder) / algorithm / "cpu-evaluation"
                code = """import json, sys, torch
from pathlib import Path
from uav_network.experiments.evaluation import evaluate
assert not torch.cuda.is_available()
result = evaluate(sys.argv[1], episodes=1, seed=10044, output_dir=sys.argv[2])
Path(sys.argv[2], "subprocess-result.json").write_text(json.dumps(result))
"""
                subprocess.run([sys.executable, "-c", code, result["checkpoint"], str(evaluation)],
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg"),
                               check=True, timeout=180)
                measured = json.loads((evaluation / "subprocess-result.json").read_text())
                self.assertEqual(measured["device"], "cpu")
                self.assertEqual(measured["environment_backend"], "warp_packet")
                self.assertTrue(Path(measured["render_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
