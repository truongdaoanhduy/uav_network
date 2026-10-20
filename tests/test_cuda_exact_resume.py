"""Real CUDA exact world/resume tests; full4096 cases require explicit opt-in.

Run only this file on two visible GPUs with UAV_CUDA_FULL4096=1 for the
production-profile checks. CPU discovery skips before allocating any world.
"""
import gc
import importlib.util
import json
import os
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
import unittest

import torch

from exact_resume_helpers import exact_world_config, assert_current_world
from uav_network.configuration import load_config
from uav_network.experiments.training import train

CUDA_WARP = torch.cuda.is_available() and importlib.util.find_spec("warp") is not None
FULL4096 = CUDA_WARP and torch.cuda.device_count() >= 2 and os.environ.get("UAV_CUDA_FULL4096") == "1"
PROFILE = "configs/comparison-20k-steps1800-scenario-v3.yaml"


def release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        for index in range(min(2, torch.cuda.device_count())):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()


def context(label, **values):
    """Emit bounded, credential-free evidence in the unittest/kernel log."""
    memory = {}
    if Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                memory[key] = value.strip()
    devices = []
    for index in range(min(2, torch.cuda.device_count())):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "free_bytes": free, "total_bytes": total})
    print(json.dumps({"cuda_exact_resume": label, "host_memory": memory,
                      "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "devices": devices, **values}), flush=True)


def world_bytes(snapshot):
    return sum(value.numel() * value.element_size() for value in snapshot["arrays"].values())


def progress_only(path):
    """Retain only small collector metadata, releasing the large CPU snapshot."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    environment = saved["training"]["environment"]
    result = {name: environment[name].clone() for name in ("lengths", "seeds")}
    result.update(next_seed=environment["next_seed"], progress=saved["progress"],
                  updates=saved["updates"], actor_updates=saved["actor_updates"],
                  world_bytes=world_bytes(environment["world"]))
    torch.testing.assert_close(environment["world"]["arrays"]["steps"].to(torch.int64),
                               result["lengths"], rtol=0, atol=0)
    return result


class ResumeChecks:
    def check_resume(self, config, *, active):
        config.tracking.enabled = False
        config.runtime.max_train_seconds = 1e-9
        first = train(config)
        self.assertEqual(first["episodes"], 0)
        self.assertEqual(first["initial_global_step"], 0)
        self.assertTrue(first["environment_state_saved"])
        old = progress_only(first["resume_checkpoint"])
        self.assertEqual(old["progress"]["records"], [])
        self.assertGreater(first["global_step"], 0)
        self.assertEqual(old["updates"], first["updates"])
        self.assertEqual(first["global_step"], config.runtime.num_envs)
        torch.testing.assert_close(old["lengths"], torch.ones_like(old["lengths"]), rtol=0, atol=0)
        expected_seeds = torch.arange(config.runtime.seed, config.runtime.seed + config.runtime.num_envs)
        torch.testing.assert_close(old["seeds"], expected_seeds, rtol=0, atol=0)
        if active:
            self.assertGreater(first["new_updates"], 0)
            self.assertGreater(old["actor_updates"], 0)
        else:
            self.assertEqual(first["updates"], 0)
        release_cuda()
        config.runtime.output_dir = str(Path(config.runtime.output_dir).parent / "second")
        config.runtime.warm_start_checkpoint = first["resume_checkpoint"]
        second = train(config)  # Fresh learner, replay, vector and checkpoint load.
        self.assertEqual(second["state_restore"], "training_state_exact_environment")
        self.assertTrue(second["optimizer_replay_restored"])
        self.assertTrue(second["environment_state_saved"])
        self.assertEqual(second["initial_global_step"], first["global_step"])
        self.assertEqual(second["initial_episodes"], 0)
        self.assertEqual(second["episodes"], 0)
        self.assertEqual(second["segment_episodes"], 0)
        self.assertEqual(second["stop_reason"], "time_budget")
        self.assertEqual(second["global_step"] - first["global_step"], config.runtime.num_envs)
        self.assertEqual(second["collection_steps"], first["collection_steps"] + 1)
        current = progress_only(second["resume_checkpoint"])
        self.assertEqual(current["updates"], second["updates"])
        self.assertEqual(second["updates"] - second["new_updates"], first["updates"])
        self.assertEqual(current["progress"]["records"], old["progress"]["records"])
        self.assertEqual(current["next_seed"], old["next_seed"])
        torch.testing.assert_close(current["seeds"], old["seeds"], rtol=0, atol=0)
        torch.testing.assert_close(current["lengths"], old["lengths"] + 1, rtol=0, atol=0)
        learner_role = "critic" if config.pql.enabled else "learner"
        learning_index = 1 if torch.cuda.device_count() >= 2 else 0
        for result in (first, second):
            self.assertEqual(result["devices"]["collector"], "cuda:0")
            self.assertEqual(result["devices"][learner_role], f"cuda:{learning_index}")
            self.assertEqual(result["replay_device"], f"cuda:{learning_index}" if config.pql.enabled else "cuda")
            if not config.pql.enabled:
                self.assertEqual(result["devices"]["replay"], f"cuda:{learning_index}")
            self.assertEqual(result["num_cpu_env_workers"], 0)
        if active:
            self.assertGreater(second["new_updates"], 0)
            self.assertGreater(current["updates"], old["updates"])
            self.assertGreater(current["actor_updates"], old["actor_updates"])
        else:
            self.assertEqual(second["updates"], 0)
        context("active-learning" if active else "full4096-warmup",
                algorithm=config.runtime.algorithm, pql=config.pql.enabled,
                world_array_bytes=old["world_bytes"], initial_global_step=second["initial_global_step"],
                global_step=second["global_step"], updates_before=old["updates"],
                updates_after=current["updates"], first_elapsed_s=first["elapsed_s"],
                second_elapsed_s=second["elapsed_s"],
                checkpoint_bytes=Path(second["resume_checkpoint"]).stat().st_size)

    def active_case(self, algorithm, pql):
        with TemporaryDirectory(prefix="uav-cuda-active-") as folder:
            try:
                config = exact_world_config(Path(folder) / "first", algorithm)
                config.runtime.algorithm = algorithm
                config.runtime.device = config.runtime.replay_device = "cuda"
                config.algorithm_config.batch_size = config.algorithm_config.learning_starts = 2
                config.algorithm_config.updates_per_step = 2
                config.pql.enabled, config.pql.mode = pql, "async"
                config.pql.warmup_collection_steps = 0
                config.pql.critic_sample_ratio = 2
                config.pql.collector_device = config.pql.learner_device = config.pql.policy_device = "auto"
                self.check_resume(config, active=True)
            finally:
                release_cuda()

    def full_case(self, algorithm, pql):
        with TemporaryDirectory(prefix="uav-cuda-full4096-") as folder:
            try:
                config = load_config(PROFILE)
                config.runtime.algorithm, config.pql.enabled = algorithm, pql
                config.runtime.output_dir = str(Path(folder) / "first")
                self.assertEqual((config.runtime.num_envs, config.runtime.episodes,
                                  config.world.max_steps, config.runtime.seed,
                                  config.world.num_targets, config.world.num_obstacles,
                                  config.network.report_bytes), (4096, 20000, 1800, 44, 35, 25, 1000000))
                # Only tracking, segment time and local output/resume paths change.
                # World, learner hyperparameters and production episode budget stay pinned.
                context("full4096-warmup-start", algorithm=algorithm, pql=pql)
                self.check_resume(config, active=False)
            finally:
                release_cuda()


    def roundtrip(self, device, *, reports=False, full=False):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        config = load_config(PROFILE) if full else exact_world_config()
        config.runtime.device = config.runtime.replay_device = "cuda"
        config.tracking.enabled = False
        if reports:
            config.network.bit_rate_bps = 100000
            config.network.report_bytes, config.network.buffer_bytes = 1024 * 1024, 3 * 1024 * 1024
            config.network.propagation_delay_enabled = True
        vector = None
        with TemporaryDirectory(prefix="uav-cuda-world-") as folder:
            try:
                if full:
                    context("full4096-world-start", device=device)
                vector = WarpVectorEnv(config, device=device)
                obs, _ = vector.reset(seed=config.runtime.seed)
                self.assertEqual(vector.device, torch.device(device))
                for value in obs.values():
                    self.assertEqual(value.device, torch.device(device))
                if reports:
                    from uav_network.environment.gpu.geometry import sense
                    from uav_network.communication.gpu.mac import sense_reports
                    targets = vector.s.targets.numpy().copy()
                    targets[:, 0] = vector.s.positions.numpy()[:, 0]
                    targets[:, 1] = [900, 900]
                    vector.s.targets.assign(targets)
                    vector._launch(sense, (2, 2, 2), vector.s, vector.p)
                    vector._launch(sense_reports, 2, vector.s, vector.p)
                    vector._publish(vector._buffers[vector._buffer])
                    actions = {"continuous": torch.zeros((2, 2, 3), device=device),
                               "destination": torch.full((2, 2), 2, dtype=torch.int64, device=device)}
                    obs, *_ = vector.step(actions)
                else:
                    for _ in range(3):
                        obs, *_ = vector.step(vars(vector.random_actions(obs["observation"])))
                snapshot = vector.snapshot_state()
                self.assertEqual(snapshot["device_type"], "cuda")
                size = world_bytes(snapshot)
                if full:
                    context("full4096-world-snapshot", device=device, world_array_bytes=size)
                saved_buffer = vector._buffer
                if reports:
                    self.assertTrue(bool((snapshot["arrays"]["packet_sequence"] > 0).any()))
                    self.assertTrue(bool((snapshot["arrays"]["session_active"] != 0).any()))
                start = Path(folder) / "start.pt"
                torch.save(snapshot, start)
                del snapshot
                # Save the next-step oracle to disk, never keep two full CPU snapshots.
                action = vector.random_actions(obs["observation"])
                expected_actions = (action.continuous.cpu().clone(), action.destination.cpu().clone())
                _, reward, done, truncated, _ = vector.step(vars(action))
                expected_result = (reward.cpu().clone(), done.cpu().clone(), truncated.cpu().clone())
                expected = Path(folder) / "next.pt"
                torch.save(vector.snapshot_state(), expected)
                del action, obs
                vector.close()
                vector = None
                release_cuda()
                vector = WarpVectorEnv(config, device=device)
                # Restore into the opposite double-buffer parity after unrelated state.
                other, _ = vector.reset(seed=config.runtime.seed + 100)
                for _ in range(2):
                    other, *_ = vector.step(vars(vector.random_actions(other["observation"])))
                self.assertNotEqual(vector._buffer, saved_buffer)
                snapshot = torch.load(start, map_location="cpu", weights_only=True)
                obs = vector.restore_state(snapshot)
                assert_current_world(self, vector, snapshot)
                del snapshot, other
                gc.collect()
                action = vector.random_actions(obs["observation"])
                for actual, reference in zip((action.continuous.cpu(), action.destination.cpu()), expected_actions):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                _, reward, done, truncated, _ = vector.step(vars(action))
                for actual, reference in zip((reward.cpu(), done.cpu(), truncated.cpu()), expected_result):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                snapshot = torch.load(expected, map_location="cpu", weights_only=True)
                # Compare one current array at a time instead of snapshotting a second world.
                assert_current_world(self, vector, snapshot)
                torch.testing.assert_close(vector.action_generator.get_state().cpu(),
                                           snapshot["action_generator"], rtol=0, atol=0)
                torch.testing.assert_close(torch.as_tensor(vector._seeds), snapshot["seeds"], rtol=0, atol=0)
                context("full4096-world" if full else "small-world", device=device,
                        live_reports=reports, world_array_bytes=size, serialized_bytes=start.stat().st_size)
                del snapshot
            finally:
                if vector is not None:
                    vector.close()
                release_cuda()



@unittest.skipUnless(CUDA_WARP, "requires real CUDA and optional Warp; no CPU world allocated")
class CudaExactResumeTests(ResumeChecks, unittest.TestCase):
    def test_serialized_world_and_future_rng_on_both_visible_devices(self):
        for index in range(min(2, torch.cuda.device_count())):
            with self.subTest(device=index):
                self.roundtrip(f"cuda:{index}")

    def test_live_packet_sessions_and_mac_rng_on_both_visible_devices(self):
        for index in range(min(2, torch.cuda.device_count())):
            with self.subTest(device=index):
                self.roundtrip(f"cuda:{index}", reports=True)

    def test_active_masac_base(self):
        self.active_case("masac", False)

    def test_active_matd3_base(self):
        self.active_case("matd3", False)

    def test_active_masac_pql(self):
        self.active_case("masac", True)

    def test_active_matd3_pql(self):
        self.active_case("matd3", True)


@unittest.skipUnless(FULL4096, "requires UAV_CUDA_FULL4096=1, real Warp CUDA and two visible GPUs")
class CudaFull4096ResumeTests(ResumeChecks, unittest.TestCase):
    def test_serialized_production_world_and_next_step(self):
        self.roundtrip("cuda:0", full=True)

    def test_masac_base(self):
        self.full_case("masac", False)

    def test_matd3_base(self):
        self.full_case("matd3", False)

    def test_masac_pql(self):
        self.full_case("masac", True)

    def test_matd3_pql(self):
        self.full_case("matd3", True)


if __name__ == "__main__":
    unittest.main()
