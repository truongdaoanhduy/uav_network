"""Matched episode budgets and autonomous launch metadata."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from uav_network.configuration import Config
from uav_network.experiments.training import train


class Matched10kTests(unittest.TestCase):
    def test_common_profile_has_identical_parameters_and_held_out_cpu_bank(self):
        from uav_network.configuration import load_config
        base = load_config("configs/comparison-10k.yaml", ["pql.enabled=false"])
        pql = load_config("configs/comparison-10k.yaml", ["pql.enabled=true"])
        left, right = base.to_dict(), pql.to_dict()
        left["pql"]["enabled"] = right["pql"]["enabled"]
        self.assertEqual(left, right)
        self.assertEqual(base.runtime.episodes, 10000)
        self.assertEqual(base.runtime.num_envs, 4096)
        self.assertEqual(base.runtime.eval_episodes, 30)
        self.assertEqual(base.runtime.eval_seed, 2000000044)
        for learner in (base.masac, base.matd3):
            self.assertEqual(learner.batch_size, 8192)
            self.assertEqual(learner.replay_capacity, 500000)
            self.assertEqual(learner.learning_starts, base.pql.warmup_collection_steps * base.runtime.num_envs)
            self.assertEqual(learner.updates_per_step, base.pql.critic_sample_ratio)

    def test_pql_resume_continues_warp_warmup_action_rng(self):
        from uav_network.experiments import pql_training
        original = pql_training.make_vector
        before, after = [], []
        def traced_vector(*args, **kwargs):
            vector = original(*args, **kwargs)
            sample = vector.random_actions
            def traced(observation):
                before.append(vector.action_generator.get_state().clone())
                action = sample(observation)
                after.append(vector.action_generator.get_state().clone())
                return action
            vector.random_actions = traced
            return vector
        def profile(destination, target):
            c = Config()
            c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 3
            c.runtime.environment_backend = "warp_packet"
            c.runtime.device = c.runtime.replay_device = "cpu"
            c.runtime.num_envs, c.runtime.episodes = 4, target
            c.runtime.output_dir = str(destination)
            c.runtime.checkpoint_interval, c.runtime.log_interval_steps = 1000, 20
            c.masac.hidden_sizes = [16, 16]
            c.masac.batch_size, c.masac.replay_capacity = 4, 32
            c.pql.enabled, c.pql.mode = True, "synchronous"
            c.pql.critic_sample_ratio, c.pql.warmup_collection_steps = 2, 20
            c.tracking.enabled = False
            return c
        with TemporaryDirectory() as folder, patch.object(pql_training, "make_vector", side_effect=traced_vector):
            first = train(profile(Path(folder) / "first", 4))
            saved = torch.load(first["resume_checkpoint"], weights_only=True)
            self.assertIn("collector", saved["training"]["rng"])
            expected = saved["training"]["rng"]["collector"]["torch"]
            self.assertTrue(torch.equal(expected, after[-1]))
            previous_draws = len(before)
            c = profile(Path(folder) / "second", 8)
            c.runtime.warm_start_checkpoint = first["resume_checkpoint"]
            resumed = train(c)
            self.assertTrue(torch.equal(before[previous_draws], expected))
            self.assertEqual(resumed["initial_episodes"], 4)
            self.assertEqual(resumed["episodes"], 8)

    def test_policy_only_pql_resume_is_rejected_before_reading_training_state(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "policy.pt"
            torch.save({"version": 2}, checkpoint)
            c = Config()
            c.runtime.device = c.runtime.replay_device = "cpu"
            c.runtime.output_dir = str(Path(folder) / "invalid")
            c.runtime.warm_start_checkpoint = str(checkpoint)
            c.pql.enabled = True
            c.tracking.enabled = False
            try:
                train(c)
            except Exception as error:
                self.assertIsInstance(error, ValueError)
                self.assertIn("version-3", str(error))
            else:
                self.fail("PQL full resume must reject a policy-only checkpoint")

    def test_pql_stops_at_exact_nonmultiple_episode_budget(self):
        with TemporaryDirectory() as folder:
            c = Config()
            c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 3
            c.runtime.environment_backend = "warp_packet"
            c.runtime.device = c.runtime.replay_device = "cpu"
            c.runtime.num_envs, c.runtime.episodes = 4, 6
            c.runtime.output_dir = str(Path(folder) / "run")
            c.runtime.checkpoint_interval = 1000
            c.runtime.log_interval_steps = 20
            c.masac.hidden_sizes = [16, 16]
            c.masac.batch_size, c.masac.replay_capacity = 4, 32
            c.pql.enabled, c.pql.mode = True, "synchronous"
            c.pql.critic_sample_ratio, c.pql.warmup_collection_steps = 2, 1
            c.tracking.enabled = False
            result = train(c)
            self.assertEqual(result["episodes"], 6)
            self.assertTrue(result["episode_budget_complete"])
            state = torch.load(result["resume_checkpoint"], weights_only=True)
            self.assertEqual(len(state["progress"]["records"]), 6)
            self.assertEqual(state["progress"]["episodes"], 6)

    def test_fresh_account_owner_comes_from_authenticated_identity(self):
        from uav_network.cloud import client
        from uav_network.cloud.credentials import Account
        account = Account("fresh", "selected-token")
        def authenticate():
            self.assertEqual(os.environ["KAGGLE_API_TOKEN"], account.token)
            self.assertNotIn("KAGGLE_USERNAME", os.environ)
            self.assertNotIn("KAGGLE_KEY", os.environ)
        api = SimpleNamespace(CONFIG_NAME_USER="username", authenticate=authenticate,
                              get_config_value=lambda key: "fresh-owner")
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "old-token", "KAGGLE_USERNAME": "old-owner",
                                     "KAGGLE_KEY": "old-key"}), \
             patch.dict("sys.modules", {"kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=lambda: api)}), \
             patch.object(client, "run_kaggle", return_value="No kernels found"):
            try:
                owner = client.resolve_username(account)
            except Exception as error:
                self.fail("A fresh account has no resources but its authenticated owner is available: "
                          + type(error).__name__)
            self.assertEqual(owner, "fresh-owner")
            self.assertEqual(os.environ["KAGGLE_API_TOKEN"], "old-token")
            self.assertEqual(os.environ["KAGGLE_USERNAME"], "old-owner")
            self.assertEqual(os.environ["KAGGLE_KEY"], "old-key")

    def _launch_auto_continue(self, hours=30):
        from uav_network.cloud import cli
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/profile.yaml").write_text("runtime:\n  episodes: 10000\n")
            key = root / "key.txt"
            key.write_text(json.dumps({"kaggle": {"accounts": [{"name": "alias", "token": "secret-k"}]},
                                      "wandb": {"api_key": "secret-w", "entity": "owner", "project": "project"}}))
            with patch.object(cli, "ROOT", root), \
                 patch.object(cli, "account_quotas", return_value=[{"name": "alias", "ok": True, "remaining_hours": hours}]), \
                 patch.object(cli, "resolve_username", return_value="owner"), \
                 patch.object(cli, "verify_credential_dataset"), \
                 patch.object(cli, "ensure_pipeline_dataset", return_value="owner/private-launch"), \
                 patch.object(cli, "run_kaggle", return_value="Kernel version 1 successfully pushed. https://www.kaggle.com/code/owner/root-run") as push, \
                 patch.object(cli.subprocess, "check_output", side_effect=["", "a" * 40]):
                try:
                    result = cli.main(["--key-file", str(key), "launch", "--stage", "validate",
                                       "--algorithm", "masac", "--auto-continue",
                                       "--config", "configs/profile.yaml", "--credential-dataset", "owner/wandb-private"])
                except SystemExit:
                    self.fail("Missing --auto-continue CLI")
        return result, push.call_args[0][0]

    def test_auto_continue_cli_caps_initial_session_to_verified_quota(self):
        result, args = self._launch_auto_continue(hours=1)
        self.assertEqual(result["timeout_seconds"], 2700)
        self.assertEqual(args[args.index("--timeout") + 1], "2700")

    def test_auto_continue_cli_rejects_quota_without_safe_upload_window(self):
        with self.assertRaisesRegex(RuntimeError, "quota|safe"):
            self._launch_auto_continue(hours=.4)

    def test_auto_continue_cli_arms_handoff_and_total_budget(self):
        result, _ = self._launch_auto_continue()
        self.assertTrue(result["auto_continue"])
        self.assertTrue(result["cpu_visualize"])
        self.assertEqual(result["requested_episodes"], 10000)
        self.assertEqual(result["continuation_index"], 0)
        self.assertEqual(result["root_kernel_ref"], result["kernel_ref"])
        self.assertEqual(result["accelerator"], "NvidiaTeslaT4")
        self.assertEqual(result["training_mode"], "base")
        self.assertIn("MASAC-BASE", result["runs"]["masac"]["name"])

    def test_cpu_url_stays_at_root_after_gpu_continuations(self):
        from uav_network.cloud.visualize import build_visualization_spec
        parent = {"kernel_ref": "owner/root-c01", "root_kernel_ref": "owner/root",
                  "continuation_index": 1, "commit": "a" * 40, "account": "alias",
                  "entity": "owner", "project": "project", "credential_dataset": "owner/private",
                  "runs": {"masac": {"id": "child", "name": "MASAC-BASE"}},
                  "cpu_eval": {"episodes": 2, "seed": 2000000044, "frame_stride": 5, "fps": 10}}
        result = {"masac": {"tracking_status": "finished", "checkpoint_artifact": "owner/project/model:v1"}}
        child = build_visualization_spec(parent, result)
        self.assertEqual(child["kernel_ref"], "owner/root-cpu")


if __name__ == "__main__":
    unittest.main()
