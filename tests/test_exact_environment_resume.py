"""Exact CPU Warp snapshots and unfinished BASE/PQL mission continuation."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import warp as wp

from test_continuation import small_config
from test_cloud_continuation import parent_spec, learner_result, write_resume
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.experiments.training import train
from uav_network.cloud.continuation import next_stage, build_continuation_spec
from uav_network.cloud import remote


class ExactWorldTests(unittest.TestCase):
    def config(self, folder="unused"):
        c = small_config(folder)
        c.runtime.environment_backend = "warp_packet"
        c.network.enabled = True
        c.world.max_steps = 12
        return c

    def test_split_world_preserves_every_array_and_future_random_actions(self):
        c = self.config()
        left, right = WarpVectorEnv(c), WarpVectorEnv(c)
        try:
            obs, _ = left.reset(seed=[44, 45])
            for _ in range(3):
                a = left.random_actions(obs["observation"])
                obs, *_ = left.step(vars(a))
            self.assertTrue(hasattr(left, "snapshot_state"), "Missing exact Warp snapshot API")
            snapshot = left.snapshot_state()
            # Exercise restoring after both double-buffer parities, not only a fresh vector.
            right.reset(seed=[144, 145])
            right.step(vars(right.random_actions(right._buffers[right._buffer]["observation"])))
            self.assertEqual(set(snapshot["arrays"]), {k for k, v in vars(left.s).items() if isinstance(v, wp.array)})
            restored = right.restore_state(snapshot)
            torch.testing.assert_close(restored["observation"], obs["observation"], rtol=0, atol=0)
            for _ in range(4):
                a, b = [v.random_actions(o["observation"]) for v, o in ((left, obs), (right, restored))]
                torch.testing.assert_close(a.continuous, b.continuous, rtol=0, atol=0)
                torch.testing.assert_close(a.destination, b.destination, rtol=0, atol=0)
                obs, reward_a, done_a, _, _ = left.step(vars(a))
                restored, reward_b, done_b, _, _ = right.step(vars(b))
                torch.testing.assert_close(reward_a, reward_b, rtol=0, atol=0)
                torch.testing.assert_close(done_a, done_b)
                for name, value in left.snapshot_state()["arrays"].items():
                    torch.testing.assert_close(value, right.snapshot_state()["arrays"][name], rtol=0, atol=0, equal_nan=True)
        finally:
            left.close()
            right.close()

    def test_split_world_preserves_live_packet_sessions_and_mac_rng(self):
        from uav_network.environment.gpu.geometry import sense
        from uav_network.communication.gpu.mac import sense_reports
        c = self.config()
        c.network.bit_rate_bps = 100000
        c.network.report_bytes, c.network.buffer_bytes = 1024 * 1024, 3 * 1024 * 1024
        c.network.propagation_delay_enabled = True
        left, right = WarpVectorEnv(c), WarpVectorEnv(c)
        try:
            left.reset(seed=44)
            targets = left.s.targets.numpy().copy()
            targets[:, 0] = left.s.positions.numpy()[:, 0]
            targets[:, 1] = [900, 900]
            left.s.targets.assign(targets)
            left._launch(sense, (2, 2, 2), left.s, left.p)
            left._launch(sense_reports, 2, left.s, left.p)
            left._publish(left._buffers[left._buffer])
            actions = {"continuous": torch.zeros((2, 2, 3)),
                       "destination": torch.full((2, 2), 2, dtype=torch.int64)}
            left.step(actions)
            snapshot = left.snapshot_state()
            self.assertTrue(bool((snapshot["arrays"]["packet_sequence"] > 0).any()))
            self.assertTrue(bool((snapshot["arrays"]["session_active"] != 0).any()))
            right.restore_state(snapshot)
            for _ in range(3):
                left.step(actions)
                right.step(actions)
            actual, expected = right.snapshot_state(), left.snapshot_state()
            for name, value in expected["arrays"].items():
                torch.testing.assert_close(actual["arrays"][name], value, rtol=0, atol=0, equal_nan=True)
        finally:
            left.close()
            right.close()

    def test_invalid_snapshot_rejected_before_any_world_mutation_and_waiting_rejected(self):
        v = WarpVectorEnv(self.config())
        try:
            obs, _ = v.reset(seed=44)
            self.assertTrue(hasattr(v, "snapshot_state"), "Missing exact Warp snapshot API")
            before = v.snapshot_state()
            for mutation in ("shape", "dtype", "layout", "missing", "extra", "config", "rng"):
                bad = deepcopy(before)
                if mutation == "shape":
                    bad["arrays"]["battery"] = bad["arrays"]["battery"][:1]
                elif mutation == "dtype":
                    bad["arrays"]["battery"] = bad["arrays"]["battery"].float()
                elif mutation == "layout":
                    bad["arrays"]["battery"] = bad["arrays"]["battery"].to_sparse()
                elif mutation == "missing":
                    del bad["arrays"]["pcg_state"]
                elif mutation == "extra":
                    bad["arrays"]["unknown"] = torch.zeros(2)
                elif mutation == "config":
                    bad["config"]["world"]["max_steps"] += 1
                else:
                    bad["action_generator"] = torch.zeros(3, dtype=torch.uint8)
                with self.assertRaises(ValueError):
                    v.restore_state(bad)
                for name, value in before["arrays"].items():
                    torch.testing.assert_close(value, v.snapshot_state()["arrays"][name], rtol=0, atol=0, equal_nan=True)
            v.step_async(vars(v.random_actions(obs["observation"])))
            with self.assertRaises(RuntimeError):
                v.snapshot_state()
            with self.assertRaises(RuntimeError):
                v.restore_state(before)
            v.step_wait()
        finally:
            v.close()

    def test_zero_episode_time_limited_base_and_pql_keep_active_worlds(self):
        for pql, algorithm, mode in ((False, "masac", "synchronous"), (False, "matd3", "synchronous"),
                                     (True, "masac", "synchronous"), (True, "matd3", "synchronous"),
                                     (True, "matd3", "async")):
            with self.subTest(pql=pql, algorithm=algorithm, mode=mode), TemporaryDirectory() as folder:
                c = self.config(Path(folder) / "first")
                c.runtime.algorithm = algorithm
                if pql:
                    c.pql.enabled, c.pql.mode = True, mode
                    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
                    c.pql.warmup_collection_steps = 100
                c.algorithm_config.learning_starts = 100
                c.runtime.max_train_seconds = 1e-9
                first = train(c)
                self.assertEqual(first["episodes"], 0)
                self.assertTrue(first.get("environment_state_saved"), "Full checkpoint lost in-flight mission state")
                saved = torch.load(first["resume_checkpoint"], weights_only=True)
                self.assertIn("environment", saved["training"])
                self.assertNotIn("training", torch.load(first["checkpoint"], weights_only=True))
                c.runtime.output_dir = str(Path(folder) / "second")
                c.runtime.warm_start_checkpoint = first["resume_checkpoint"]
                second = train(c)
                self.assertEqual(second["initial_global_step"], first["global_step"])
                self.assertEqual(second["state_restore"], "training_state_exact_environment")
                saved2 = torch.load(second["resume_checkpoint"], weights_only=True)
                state = saved2["training"]["environment"]
                self.assertEqual(state["next_seed"], saved["training"]["environment"]["next_seed"])
                torch.testing.assert_close(state["lengths"], torch.full((2,), 2, dtype=torch.int64))
                torch.testing.assert_close(state["seeds"], saved["training"]["environment"]["seeds"])
                self.assertEqual(state["world"]["arrays"]["steps"].tolist(), [2, 2])
                c.runtime.max_train_seconds = 0
                c.runtime.output_dir = str(Path(folder) / "third")
                c.runtime.warm_start_checkpoint = second["resume_checkpoint"]
                third = train(c)
                self.assertEqual(third["global_step"], 24)
                self.assertEqual(third["final_episode"]["episode_length"], 12)
                self.assertEqual(third["final_episode"]["seed"], 45)
                c.runtime.warm_start_checkpoint = "none"
                c.runtime.output_dir = str(Path(folder) / "uninterrupted")
                whole = train(c)
                self.assertEqual(third["reproducibility"]["episode_hash"], whole["reproducibility"]["episode_hash"])
                self.assertEqual(third["reproducibility"]["model_hash"], whole["reproducibility"]["model_hash"])
                # Resumed cumulative returns and packet/mission state equal an uninterrupted run.
                final_world = torch.load(third["resume_checkpoint"], weights_only=True)["training"]["environment"]
                whole_world = torch.load(whole["resume_checkpoint"], weights_only=True)["training"]["environment"]
                for name in ("returns", "lengths", "seeds"):
                    torch.testing.assert_close(final_world[name], whole_world[name], rtol=0, atol=0)

    def test_legacy_v3_remains_an_explicit_environment_reset(self):
        with TemporaryDirectory() as folder:
            c = self.config(Path(folder) / "first")
            c.runtime.max_train_seconds = 1e-9
            first = train(c)
            saved = torch.load(first["resume_checkpoint"], weights_only=True)
            saved["training"].pop("environment", None)
            path = Path(folder) / "legacy.pt"
            torch.save(saved, path)
            c.runtime.warm_start_checkpoint = str(path)
            c.runtime.output_dir = str(Path(folder) / "second")
            second = train(c)
            self.assertEqual(second["state_restore"], "training_state_environment_reset")


class TransitionGateTests(unittest.TestCase):
    def result(self):
        r = learner_result()
        r["masac"].update(episodes=0, initial_episodes=0, segment_episodes=0,
                          initial_global_step=0, global_step=8, environment_state_saved=True)
        return r

    def test_zero_episodes_need_exact_state_positive_steps_and_verified_parent_prefix(self):
        p, r = parent_spec(), self.result()
        self.assertEqual(next_stage(p, r), "gpu")
        child = build_continuation_spec(p, r)
        self.assertEqual(child["continuation_expected_global_step"], {"masac": 8})
        for changes in ({"global_step": 0}, {"environment_state_saved": False}, {"initial_global_step": 9}):
            bad = deepcopy(r)
            bad["masac"].update(changes)
            with self.assertRaises(ValueError):
                next_stage(p, bad)
        alias_parent = deepcopy(p)
        alias_parent["warm_start_artifacts"] = {"masac": "owner/project/masac-base-original:latest"}
        alias_parent["resolved_warm_start_artifacts"] = {"masac": r["masac"]["checkpoint_artifact"]}
        with self.assertRaises(ValueError):
            next_stage(alias_parent, r)
        child_result = deepcopy(r)
        child_result["masac"].update(initial_global_step=7, global_step=16,
                                     checkpoint_artifact="owner/project/masac-base-child:v0")
        with self.assertRaises(ValueError):
            next_stage(child, child_result)

    def test_segments_can_exceed_episode_target_with_positive_step_progress(self):
        p, r = parent_spec(), self.result()
        p["overrides"] = ["runtime.episodes=1"]
        p.update(requested_episodes=1, continuation_index=3,
                 continuation_expected_episodes={"masac": 0},
                 continuation_expected_global_step={"masac": 8})
        r["masac"].update(requested_episodes=1, initial_global_step=8, global_step=16)
        self.assertEqual(next_stage(p, r), "gpu")
        self.assertEqual(build_continuation_spec(p, r)["continuation_index"], 4)
        excessive = deepcopy(p)
        excessive["continuation_index"] = 100
        with self.assertRaises(ValueError):
            next_stage(excessive, r)
        del p["continuation_expected_global_step"]
        with self.assertRaises(ValueError):
            next_stage(p, r)

    def test_remote_input_rejects_wrong_step_prefix_and_missing_required_world(self):
        with TemporaryDirectory() as directory:
            c = small_config(Path(directory))
            c.runtime.algorithm, c.runtime.num_envs, c.runtime.episodes = "masac", 2, 10000
            source = Path(directory) / "source.pt"
            write_resume(c, source)
            def download(root):
                path = Path(root) / "resume.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.read_bytes())
                return str(path)
            artifact = SimpleNamespace(get_entry=lambda name: SimpleNamespace(download=download))
            api = SimpleNamespace(artifact=lambda *args, **kwargs: artifact)
            for expected, required, error in ((3, False, "global-step"), (4, True, "environment")):
                p = parent_spec()
                p.update(warm_start_artifacts={"masac": "owner/project/source:v0"},
                         continuation_expected_episodes={"masac": 2},
                         continuation_expected_global_step={"masac": expected},
                         continuation_requires_environment_state=required)
                with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=lambda: api)}):
                    with self.assertRaisesRegex(ValueError, error):
                        remote.prepare_checkpoints(p, Path(directory) / str(expected), c)
