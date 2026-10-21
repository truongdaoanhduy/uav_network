"""Server-side continuation gates preserve verified progress and final CPU handoff."""
import importlib
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from uav_network.algorithms.checkpoints import make_learner
from uav_network.algorithms.replay import TensorReplayBuffer
from uav_network.environment.env import Action
from uav_network.environment.observations import observation_dim, state_dim

from uav_network.cloud import remote, visualize
from uav_network.cloud.credentials import Account, Credentials
from uav_network.configuration import Config


def parent_spec():
    return {"stage": "train", "config": "configs/kaggle-smoke.yaml",
            "overrides": ["runtime.episodes=10000", "runtime.eval_seed=10044"],
            "commit": "a" * 40, "kernel_ref": "owner/masac-base-training-root",
            "root_kernel_ref": "owner/masac-base-training-root", "account": "selected",
            "entity": "owner", "project": "project", "requested_episodes": 10000,
            "credential_dataset": "owner/private-wandb",
            "cpu_credential_dataset": "owner/private-cpu-wandb",
            "pipeline_credential_dataset": "owner/private-launch",
            "storage_credential_dataset": "owner/private-hf", "checkpoint_repo": "owner/private-checkpoints",
            "auto_continue": True, "cpu_visualize": True, "continuation_index": 0,
            "timeout_seconds": 39600, "accelerator": "NvidiaTeslaT4",
            "cpu_eval": {"episodes": 2, "seed": 10044, "frame_stride": 5, "fps": 10},
            "runs": {"masac": {"id": "original", "name": "MASAC-base-seed44-root"}}}


def learner_result(complete=False):
    episodes = 10000 if complete else 1500
    return {"masac": {"algorithm": "masac", "requested_episodes": 10000,
            "episodes": episodes, "initial_episodes": 0, "segment_episodes": episodes,
            "global_step": 9000, "updates": 100, "tracking_status": "finished",
            "episode_budget_complete": complete,
            "stop_reason": "episode_budget" if complete else "time_budget",
            "checkpoint_artifact": "owner/project/masac-base-original:v4"}}


def write_summary(output, result):
    folder = output / "train" / "masac"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps(result["masac"]))


def write_resume(config, path):
    progress = {"episodes": 2, "global_step": 4, "collection_steps": 2, "next_seed": 3,
                "records": [{"episode": 1, "global_step": 2, "seed": 1},
                            {"episode": 2, "global_step": 4, "seed": 2}]}
    dims = (config.runtime.num_envs, config.world.num_uavs, observation_dim(config))
    obs = torch.zeros(dims)
    state = torch.zeros((config.runtime.num_envs, state_dim(config)))
    action = Action(torch.zeros((config.runtime.num_envs, config.world.num_uavs, 3)),
                    torch.zeros((config.runtime.num_envs, config.world.num_uavs), dtype=torch.int64))
    transition = (obs, state, action, torch.zeros(config.runtime.num_envs),
                  obs, state, torch.zeros(config.runtime.num_envs))
    if config.pql.enabled:
        from uav_network.experiments.pql_system import PQLSystem
        engine = PQLSystem(config)
        try:
            for _ in range(2):
                engine.submit(transition)
            engine.save(path, progress)
        finally:
            engine.close()
    else:
        from uav_network.experiments.fixed_checkpoints import save_fixed_checkpoint
        learner = make_learner(config, observation_dim(config), state_dim(config), "cpu")
        replay = TensorReplayBuffer(config.algorithm_config.replay_capacity,
                                    dims[1:], state_dim(config), seed=1)
        for _ in range(2):
            replay.add_batch(*transition)
        save_fixed_checkpoint(learner, path, progress, replay, np.random.default_rng(1),
                              warmup_origin=0, random_warmup=True, next_metrics_at=10, latest={},
                              pending_learning=False)
    return progress


class CloudContinuationTests(unittest.TestCase):
    def test_continuation_preserves_requested_dual_gpu_allocation(self):
        parent = parent_spec()
        parent["gpus_per_run"] = 2
        child = self.module().build_continuation_spec(parent, learner_result())
        self.assertEqual(child.get("gpus_per_run"), 2)
        self.assertEqual(remote.gpu_allocation(Config(), 2, 0,
                                             child["gpus_per_run"]), (1, "0,1"))

    def test_baseline_auto_continuation_reserves_time_for_full_replay_upload(self):
        config = Config()
        config.pql.enabled = False
        config.tracking.artifact_timeout_seconds = 900
        spec = parent_spec()
        spec["timeout_seconds"] = 2000
        self.assertEqual(remote.learner_time_limit(spec, config, 1, 0, 0), 980)
        spec["auto_continue"] = False
        self.assertEqual(remote.learner_time_limit(spec, config, 1, 0, 0), 1400)

    def test_submission_deadline_failure_retains_plan_without_push(self):
        module = self.module()
        with TemporaryDirectory() as directory, \
             patch.dict("os.environ", {"KAGGLE_API_TOKEN": "synthetic-kaggle-token",
                                      "WANDB_API_KEY": "synthetic-wandb-key",
                                      "UAV_JOB_STARTED_AT": "1"}), \
             patch.object(module.time, "monotonic", return_value=39600), \
             patch.object(module, "run_kaggle", side_effect=AssertionError("No time left to submit")):
            with self.assertRaises(TimeoutError):
                module.submit_continuation(parent_spec(), learner_result(), Path(directory))
            self.assertEqual(json.loads((Path(directory) / "continuation.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((Path(directory) / "continuation-failure.json").read_text()),
                             {"error_type": "TimeoutError"})

    def module(self):
        try:
            module = importlib.import_module("uav_network.cloud.continuation")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "Server-side GPU continuation is missing")
        return module

    def test_incomplete_segment_builds_fresh_gpu_run_with_same_total_and_pinned_inputs(self):
        module = self.module()
        parent, results = parent_spec(), learner_result()
        child = module.build_continuation_spec(parent, results)
        self.assertEqual(child["stage"], "train")
        self.assertEqual(child["warm_start_artifacts"], {"masac": results["masac"]["checkpoint_artifact"]})
        self.assertEqual(child["continuation_expected_episodes"], {"masac": 1500})
        self.assertEqual(child["continuation_index"], 1)
        self.assertEqual(child["root_kernel_ref"], parent["root_kernel_ref"])
        self.assertTrue(child["auto_continue"])
        self.assertTrue(child["cpu_visualize"])
        self.assertEqual(child["requested_episodes"], 10000)
        for key in ("config", "overrides", "commit", "account", "entity", "project",
                    "credential_dataset", "pipeline_credential_dataset", "cpu_credential_dataset",
                    "storage_credential_dataset", "checkpoint_repo", "cpu_eval", "accelerator"):
            self.assertEqual(child[key], parent[key], key)
        self.assertNotEqual(child["runs"]["masac"]["id"], parent["runs"]["masac"]["id"])
        self.assertIn("MASAC-base", child["runs"]["masac"]["name"])
        self.assertNotEqual(child["runs"]["masac"]["name"], parent["runs"]["masac"]["name"])
        from uav_network.cloud.kernel import kernel_metadata, kernel_script
        metadata, script = kernel_metadata(child), kernel_script(child)
        self.assertTrue(metadata["enable_gpu"])
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["dataset_sources"], ["owner/private-wandb", "owner/private-hf", "owner/private-launch"])
        self.assertIn(parent["commit"], script)
        self.assertIn('SPEC["commit"]', script)

    def test_refs_and_titles_fit_after_many_continuations_and_cpu_suffix(self):
        module = self.module()
        root = "longusername/" + "a-very-long-original-experiment-title-" * 3
        for index in (0, 1, 99, 100, 10000000000):
            with self.subTest(index=index):
                gpu = module.kernel_ref_for_segment(root, index)
                cpu = module.kernel_ref_for_segment(root, index, cpu=True)
                self.assertLessEqual(len(gpu), 50)
                self.assertLessEqual(len(cpu), 50)
                self.assertTrue(cpu.endswith("-cpu"))
                self.assertEqual(gpu.split("/")[0], "longusername")

    def test_complete_budget_submits_cpu_only_and_retains_exact_result(self):
        self.module()
        with TemporaryDirectory() as directory:
            output, result = Path(directory), learner_result(complete=True)
            write_summary(output, result)
            emitted = []
            def cpu(parent, completed, destination):
                emitted.append(("cpu", completed))
                return {"kernel_ref": "owner/final-cpu"}
            with patch.object(visualize, "submit_visualization", side_effect=cpu), \
                 patch("uav_network.cloud.continuation.submit_continuation",
                       side_effect=AssertionError("Complete budget must not launch a GPU")), \
                 patch.object(remote.subprocess, "run", side_effect=AssertionError("GPU must never render")):
                self.assertEqual(remote.finalize_runs(parent_spec(), [SimpleNamespace(returncode=0)], output), result)
            self.assertEqual(emitted, [("cpu", result)])
            self.assertEqual(json.loads((output / "job-result.json").read_text()), result)

    def test_existing_scheduling_failure_blocks_any_auto_handoff(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            output, result = Path(directory), learner_result()
            write_summary(output, result)
            (output / "job-scheduling-failure.json").write_text('{"error_type": "TimeoutError"}')
            with patch.object(module, "submit_continuation",
                              side_effect=AssertionError("Scheduling failure must not emit a child")), \
                 patch.object(visualize, "submit_visualization",
                              side_effect=AssertionError("Scheduling failure must not emit CPU")):
                with self.assertRaisesRegex(RuntimeError, "continuation"):
                    remote.finalize_runs(parent_spec(), [SimpleNamespace(returncode=0)], output)
            self.assertEqual(json.loads((output / "job-result.json").read_text()), result)
            self.assertEqual(json.loads((output / "continuation-failure.json").read_text()),
                             {"error_type": "ValueError"})

    def test_incomplete_budget_submits_gpu_only_after_results_are_saved(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            output, result = Path(directory), learner_result()
            write_summary(output, result)
            emitted = []
            def gpu(parent, completed, destination):
                self.assertEqual(json.loads((output / "job-result.json").read_text()), result)
                emitted.append(("gpu", completed))
                return {"kernel_ref": "owner/next-c01"}
            with patch.object(module, "submit_continuation", side_effect=gpu), \
                 patch.object(visualize, "submit_visualization",
                              side_effect=AssertionError("Incomplete budget must not launch CPU")), \
                 patch.object(remote.subprocess, "run", side_effect=AssertionError("GPU must never render")):
                self.assertEqual(remote.finalize_runs(parent_spec(), [SimpleNamespace(returncode=0)], output), result)
            self.assertEqual(emitted, [("gpu", result)])

    def test_bad_progress_or_failed_upload_cannot_launch_any_child(self):
        self.module()
        invalid = [
            {"algorithm": "matd3"}, {"requested_episodes": 9999},
            {"episodes": 0, "segment_episodes": 0},
            {"initial_episodes": 1500, "segment_episodes": 0},
            {"tracking_status": "failed"}, {"checkpoint_artifact": None},
            {"checkpoint_artifact": "owner/project/model:latest"},
            {"checkpoint_artifact": "other/project/model:v1"},
            {"stop_reason": "deadline_error"}, {"episode_budget_complete": True},
            {"episodes": 10001, "segment_episodes": 10001},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), TemporaryDirectory() as directory:
                output, result = Path(directory), learner_result()
                result["masac"].update(changes)
                write_summary(output, result)
                with patch.object(visualize, "submit_visualization",
                                  side_effect=AssertionError("Bad segment must not launch CPU")), \
                     patch("uav_network.cloud.continuation.submit_continuation",
                           side_effect=AssertionError("Bad segment must not launch GPU")):
                    with self.assertRaisesRegex(RuntimeError, "continuation"):
                        remote.finalize_runs(parent_spec(), [SimpleNamespace(returncode=0)], output)
                self.assertEqual(json.loads((output / "job-result.json").read_text()), result)
                failure = json.loads((output / "continuation-failure.json").read_text())
                self.assertEqual(failure["error_type"], "ValueError")
                manifest = json.loads((output / "continuation.json").read_text())
                self.assertEqual(manifest["status"], "failed")

    def test_missing_summary_and_nonzero_learner_exit_do_not_submit(self):
        self.module()
        for code in (0, 1):
            with self.subTest(code=code), TemporaryDirectory() as directory:
                output = Path(directory)
                with patch.object(visualize, "submit_visualization",
                                  side_effect=AssertionError("Failed learner must not launch CPU")), \
                     patch("uav_network.cloud.continuation.submit_continuation",
                           side_effect=AssertionError("Failed learner must not launch GPU")):
                    with self.assertRaisesRegex(RuntimeError, "continuation"):
                        remote.finalize_runs(parent_spec(), [SimpleNamespace(returncode=code)], output)
                self.assertEqual(json.loads((output / "job-result.json").read_text()), {})
                self.assertTrue((output / "continuation-failure.json").is_file())

    def test_continuation_rejects_wrong_initial_prefix_or_same_uploaded_artifact(self):
        module = self.module()
        parent, result = parent_spec(), learner_result()
        parent.update(continuation_index=1, continuation_expected_episodes={"masac": 1000},
                      warm_start_artifacts={"masac": "owner/project/old:v3"})
        with self.assertRaisesRegex(ValueError, "initial"):
            module.build_continuation_spec(parent, result)
        result["masac"].update(initial_episodes=1000, segment_episodes=500,
                              checkpoint_artifact="owner/project/old:v3")
        with self.assertRaisesRegex(ValueError, "artifact"):
            module.build_continuation_spec(parent, result)

    def test_gpu_push_verifies_private_inputs_and_never_embeds_credentials(self):
        module = self.module()
        account = Account("selected", "synthetic-kaggle-token")
        credentials = Credentials([account], "synthetic-wandb-key", "owner", "project")
        verifications, pushes = [], []
        def verify(selected, creds, ref, username, **kwargs):
            self.assertIs(selected, account)
            self.assertEqual(username, "owner")
            verifications.append((ref, kwargs.get("only_selected", False)))
        def launch_file(selected, ref, username, filenames, **kwargs):
            self.assertEqual(ref, "owner/private-launch")
            self.assertEqual(filenames, {"uav_cpu_launch_token.txt"})
            self.assertTrue(kwargs["only_selected"])
            return "uav_cpu_launch_token.txt", account.token
        with TemporaryDirectory() as directory:
            output = Path(directory)
            def push(args, selected, **kwargs):
                self.assertIs(selected, account)
                self.assertEqual(args[:2], ["kernels", "push"])
                self.assertEqual(args[args.index("--accelerator") + 1], "NvidiaTeslaT4")
                self.assertEqual(args[args.index("--timeout") + 1], "2700")
                folder = Path(args[args.index("-p") + 1])
                metadata = json.loads((folder / "kernel-metadata.json").read_text())
                pushes.append(metadata)
                self.assertTrue(metadata["enable_gpu"])
                for file in folder.glob("*"):
                    self.assertNotIn(account.token, file.read_text())
                    self.assertNotIn(credentials.wandb_key, file.read_text())
                return "Kernel version 1 successfully pushed. https://www.kaggle.com/code/owner/next-c01"
            with patch.object(module, "account_quotas", return_value=[
                    {"name": "selected", "ok": True, "remaining_hours": 1}]), \
                 patch.object(module, "verify_credential_dataset", side_effect=verify), \
                 patch.object(module, "private_dataset_file", side_effect=launch_file), \
                 patch.object(module, "run_kaggle", side_effect=push):
                manifest = module.submit_continuation(parent_spec(), learner_result(), output,
                                                     account=account, credentials=credentials)
            self.assertEqual(verifications, [("owner/private-wandb", False),
                                             ("owner/private-cpu-wandb", True)])
            self.assertEqual(len(pushes), 1)
            self.assertEqual(manifest["status"], "submitted")
            self.assertEqual(manifest["timeout_seconds"], 2700)
            serialized = (output / "continuation.json").read_text()
            self.assertNotIn(account.token, serialized)
            self.assertNotIn(credentials.wandb_key, serialized)

    def test_quota_failure_retains_manifest_and_explicit_failure_without_push(self):
        module = self.module()
        account = Account("selected", "synthetic-kaggle-token")
        credentials = Credentials([account], "synthetic-wandb-key", "owner", "project")
        for quota in ([], [{"name": "selected", "ok": False}],
                      [{"name": "selected", "ok": True, "remaining_hours": .24}]):
            with self.subTest(quota=quota), TemporaryDirectory() as directory:
                output = Path(directory)
                with patch.object(module, "account_quotas", return_value=quota), \
                     patch.object(module, "run_kaggle", side_effect=AssertionError("No GPU quota")):
                    with self.assertRaisesRegex(RuntimeError, "quota"):
                        module.submit_continuation(parent_spec(), learner_result(), output,
                                                   account=account, credentials=credentials)
                manifest = json.loads((output / "continuation.json").read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["warm_start_artifacts"],
                                 {"masac": "owner/project/masac-base-original:v4"})
                self.assertEqual(json.loads((output / "continuation-failure.json").read_text()),
                                 {"error_type": "RuntimeError"})
                self.assertNotIn(account.token, json.dumps(manifest))

    def test_baseline_and_pql_auto_resume_download_v3_while_cpu_uses_light_policy(self):
        self.module()
        entries = []
        with TemporaryDirectory() as directory:
            output = Path(directory)
            config = Config()
            config.runtime.algorithm = "masac"
            config.runtime.device = config.runtime.replay_device = "cpu"
            config.runtime.num_envs = 2
            config.runtime.episodes = 10000
            config.world.num_uavs = 2
            config.algorithm_config.hidden_sizes = [8, 8]
            config.algorithm_config.batch_size = config.algorithm_config.learning_starts = 4
            config.algorithm_config.replay_capacity = 16
            config.pql.mode = "synchronous"
            parent = parent_spec()
            parent.update(warm_start_artifacts={"masac": "owner/project/model:v4"},
                          continuation_index=1, continuation_expected_episodes={"masac": 2})
            for pql in (False, True):
                config.pql.enabled = pql
                source = output / f"source-{pql}.pt"
                write_resume(config, source)
                def entry(name):
                    entries.append(name)
                    def download(root):
                        path = Path(root) / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        payload = torch.load(source, weights_only=True)
                        if name == "checkpoint.pt":
                            payload.pop("training")
                            payload["version"] = 2
                        torch.save(payload, path)
                        return str(path)
                    return SimpleNamespace(download=download)
                api = SimpleNamespace(artifact=lambda *args, **kwargs: SimpleNamespace(get_entry=entry))
                with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=lambda: api)}):
                    paths = remote.prepare_checkpoints(deepcopy(parent), output / str(pql), config)
                    self.assertEqual(Path(paths["masac"]).name, "resume.pt")
                    if pql:
                        cpu = remote.prepare_checkpoints({"warm_start_artifacts": parent["warm_start_artifacts"]},
                                                         output / "cpu")
                        self.assertEqual(Path(cpu["masac"]).name, "checkpoint.pt")
            self.assertEqual(entries, ["resume.pt", "resume.pt", "checkpoint.pt"])

    def test_auto_resume_rejects_v2_or_a_stale_prefix(self):
        self.module()
        for payload in ({"version": 2}, {"version": 3, "training": {}}):
            with self.subTest(payload=payload), TemporaryDirectory() as directory:
                folder = Path(directory)
                payload = {**payload, "algorithm": "masac",
                           "progress": {"episodes": 1, "global_step": 2, "collection_steps": 1,
                                        "next_seed": 2, "records": [
                                            {"episode": 1, "global_step": 2, "seed": 1}]}}
                def download(root):
                    path = Path(root) / "resume.pt"
                    path.parent.mkdir(parents=True)
                    torch.save(payload, path)
                    return str(path)
                artifact = SimpleNamespace(get_entry=lambda name: SimpleNamespace(download=download))
                api = SimpleNamespace(artifact=lambda *args, **kwargs: artifact)
                parent = parent_spec()
                parent.update(warm_start_artifacts={"masac": "owner/project/model:v4"},
                              continuation_index=1, continuation_expected_episodes={"masac": 2})
                with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=lambda: api)}):
                    with self.assertRaises(ValueError):
                        remote.prepare_checkpoints(parent, folder, Config())


if __name__ == "__main__":
    unittest.main()
