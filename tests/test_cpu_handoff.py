"""Outbound CPU job allocation and preservation of uploaded GPU results."""
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.cloud.kernel import kernel_metadata
from uav_network.cloud.remote import finalize_runs


def parent_spec():
    return {"commit": "a" * 40, "kernel_ref": "owner/train-example", "account": "alias",
            "credential_dataset": "owner/wandb-private", "entity": "owner", "project": "project",
            "stage": "train", "config": "configs/motion-long.yaml", "overrides": [],
            "cpu_visualize": True, "runs": {"masac": {"id": "m001", "name": "MASAC"}},
            "cpu_eval": {"episodes": 1, "seed": 10044, "frame_stride": 5, "fps": 10}}


def gpu_result():
    return {"masac": {"algorithm": "masac", "episodes": 18, "global_step": 18000,
                      "updates": 7000, "episode_budget_complete": False, "stop_reason": "time_budget",
                      "tracking_status": "finished", "checkpoint_artifact": "owner/project/model:v8"}}


class CPUHandoffTests(unittest.TestCase):
    def test_cpu_child_uses_its_clean_wandb_dataset(self):
        parent = parent_spec()
        parent["cpu_credential_dataset"] = "owner/wandb-cpu-only"
        child = self.module().build_visualization_spec(parent, gpu_result())
        self.assertEqual(kernel_metadata(child)["dataset_sources"], ["owner/wandb-cpu-only"])

    def test_cpu_submission_rejects_wandb_dataset_with_unrelated_secrets(self):
        module = self.module()
        dataset = SimpleNamespace(ref="owner/wandb-private", is_private=True)
        api = SimpleNamespace(authenticate=lambda: None, dataset_list=lambda **kwargs: [dataset])
        with TemporaryDirectory() as directory:
            def transport(args, account, **kwargs):
                if args[:2] == ["datasets", "files"]:
                    return json.dumps([{"name": "wandb_api_key.txt"}, {"name": "hf_token.txt"}])
                if args[:2] == ["datasets", "download"]:
                    Path(args[args.index("-p") + 1], "wandb_api_key.txt").write_text("secret-w")
                    return ""
                raise AssertionError("CPU request must not mount unrelated secrets")
            with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "secret-k", "WANDB_API_KEY": "secret-w"}), \
                 patch.dict("sys.modules", {"kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=lambda: api)}), \
                 patch("uav_network.cloud.client.run_kaggle", side_effect=transport), \
                 patch.object(module, "run_kaggle", side_effect=transport):
                with self.assertRaises(RuntimeError):
                    module.submit_visualization(parent_spec(), gpu_result(), Path(directory))
            self.assertFalse((Path(directory) / "cpu-visualization.json").exists())

    def test_cpu_runner_preserves_render_when_another_uploaded_artifact_is_unavailable(self):
        import shutil
        import torch
        from uav_network.configuration import Config
        from uav_network.experiments.training import train
        module = self.module()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            c = Config()
            c.world.num_targets, c.world.num_obstacles, c.world.max_steps = 2, 1, 3
            c.runtime.device = c.runtime.replay_device = "cpu"
            c.runtime.episodes = 1
            c.runtime.output_dir = str(output / "policy")
            c.masac.hidden_sizes = [32, 32]
            trained = train(c)
            def artifact(ref, **kwargs):
                if "missing" in ref:
                    raise RuntimeError("Artifact unavailable")
                def download(root):
                    path = Path(root) / "checkpoint.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(trained["checkpoint"], path)
                    return str(path)
                return SimpleNamespace(get_entry=lambda name: SimpleNamespace(download=download))
            api = SimpleNamespace(artifact=artifact)
            spec = {"artifacts": {"matd3": "owner/project/missing:v1", "masac": "owner/project/model:v1"},
                    "cpu_eval": {"episodes": 1, "seed": 10044, "frame_stride": 5, "fps": 10},
                    "training_status": {"matd3": {}, "masac": {}}}
            with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=lambda: api)}), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "matd3"):
                    module.run_visualization(spec, output / "cpu")
            result = json.loads((output / "cpu/visualization-result.json").read_text())
            self.assertEqual(result["masac"]["device"], "cpu")
            self.assertTrue(Path(result["masac"]["animation_path"]).is_file())

    def test_cpu_runner_rejects_visible_cuda_before_artifact_download(self):
        import torch
        module = self.module()
        with patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "CPU-only"):
                module.run_visualization({}, "unused-output")

    def test_launch_cli_emits_private_cpu_handoff_and_pins_published_source(self):
        from uav_network.cloud import cli
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "configs").mkdir()
            (output / "configs/profile.yaml").write_text("runtime:\n  episodes: 500\nworld:\n  max_steps: 1000\n")
            key = output / "key.txt"
            key.write_text(json.dumps({"kaggle": {"accounts": [{"name": "alias", "token": "secret-k"}]},
                                       "wandb": {"api_key": "secret-w", "entity": "owner", "project": "project"}}))
            def transport(args, account, **kwargs):
                folder = Path(args[args.index("-p") + 1])
                metadata = json.loads((folder / "kernel-metadata.json").read_text())
                self.assertEqual(metadata["dataset_sources"], ["owner/wandb-private", "owner/private-launch"])
                self.assertTrue(metadata["enable_gpu"])
                return "Kernel version 1 successfully pushed. https://www.kaggle.com/code/owner/published-gpu-run"
            with patch.object(cli, "ROOT", output), \
                 patch.object(cli, "account_quotas", return_value=[{"name": "alias", "ok": True, "remaining_hours": 30}]), \
                 patch.object(cli, "resolve_username", return_value="owner"), \
                 patch.object(cli, "verify_credential_dataset"), \
                 patch.object(cli, "ensure_pipeline_dataset", create=True, return_value="owner/private-launch"), \
                 patch.object(cli, "ensure_cpu_wandb_dataset", create=True, return_value="owner/wandb-cpu-only"), \
                 patch.object(cli, "run_kaggle", side_effect=transport), \
                 patch.object(cli.subprocess, "check_output", side_effect=["", "a" * 40]):
                try:
                    result = cli.main(["--key-file", str(key), "launch", "--cpu-visualize",
                                       "--config", "configs/profile.yaml", "--credential-dataset", "owner/wandb-private"])
                except SystemExit as error:
                    self.fail(f"Autonomous CPU visualization CLI is missing: {error}")
            self.assertTrue(result["cpu_visualize"])
            self.assertEqual(result.get("cpu_credential_dataset"), "owner/wandb-cpu-only")
            self.assertEqual(result["commit"], "a" * 40)
            self.assertEqual(result["cpu_eval"], {"episodes": 1, "seed": 10044, "frame_stride": 5, "fps": 10})

    def module(self):
        try:
            module = importlib.import_module("uav_network.cloud.visualize")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "Separate CPU cloud visualization is missing")
        return module

    def test_outbound_cpu_job_pins_uploaded_version_without_gpu_or_launch_token(self):
        module = self.module()
        child = module.build_visualization_spec(parent_spec(), gpu_result())
        self.assertEqual(child["artifacts"], {"masac": "owner/project/model:v8"})
        self.assertEqual(child["stage"], "visualize")
        self.assertEqual(child["training_status"]["masac"]["stop_reason"], "time_budget")
        self.assertFalse(child["training_status"]["masac"]["episode_budget_complete"])
        metadata = kernel_metadata(child)
        self.assertFalse(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_tpu"])
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["dataset_sources"], ["owner/wandb-private"])

    def test_missing_or_mutable_checkpoint_is_rejected_before_cpu_job(self):
        module = self.module()
        for artifact in (None, "owner/project/model:latest", "owner/project/model:final"):
            result = gpu_result()
            result["masac"]["checkpoint_artifact"] = artifact
            with self.subTest(artifact=artifact), self.assertRaises(ValueError):
                module.build_visualization_spec(parent_spec(), result)

    def test_gpu_finalizer_saves_results_then_submits_separate_cpu_request(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            summary = output / "train/masac/summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps(gpu_result()["masac"]))
            dataset = SimpleNamespace(ref="owner/wandb-private", is_private=True)
            api = SimpleNamespace(authenticate=lambda: None, dataset_list=lambda **kwargs: [dataset])
            payloads = []

            def transport(args, account, **kwargs):
                if args[:2] == ["datasets", "files"]:
                    return json.dumps([{"name": "wandb_api_key.txt"}])
                if args[:2] == ["datasets", "download"]:
                    Path(args[args.index("-p") + 1], "wandb_api_key.txt").write_text("secret-w")
                    return ""
                self.assertEqual(args[:2], ["kernels", "push"])
                self.assertNotIn("--accelerator", args)
                self.assertEqual(json.loads((output / "job-result.json").read_text()), gpu_result())
                folder = Path(args[args.index("-p") + 1])
                payloads.append(json.loads((folder / "kernel-metadata.json").read_text()))
                return "Kernel version 1 successfully pushed. https://www.kaggle.com/code/owner/train-example-cpu-visualize"

            with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "secret-k", "WANDB_API_KEY": "secret-w"}):
                with patch.dict("sys.modules", {"kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=lambda: api)}):
                    with patch("uav_network.cloud.client.run_kaggle", side_effect=transport), \
                         patch.object(module, "run_kaggle", side_effect=transport), \
                         patch("uav_network.cloud.remote.subprocess.run", side_effect=AssertionError("GPU session must not render")):
                        result = finalize_runs(parent_spec(), [SimpleNamespace(returncode=0)], output)
            self.assertEqual(result, gpu_result())
            self.assertEqual(len(payloads), 1)
            self.assertFalse(payloads[0]["enable_gpu"])
            manifest = json.loads((output / "cpu-visualization.json").read_text())
            self.assertEqual(manifest["kernel_url"], "https://www.kaggle.com/code/owner/train-example-cpu-visualize")
            for file in (output / "cpu-kernel").glob("*"):
                self.assertNotIn("secret-k", file.read_text())
                self.assertNotIn("secret-w", file.read_text())

    def test_failed_handoff_retains_completed_training_and_explicit_failure(self):
        self.module()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            summary = output / "train/masac/summary.json"
            summary.parent.mkdir(parents=True)
            result = gpu_result()
            result["masac"]["checkpoint_artifact"] = None
            summary.write_text(json.dumps(result["masac"]))
            with self.assertRaisesRegex(RuntimeError, "visualization"):
                finalize_runs(parent_spec(), [SimpleNamespace(returncode=0)], output)
            self.assertEqual(json.loads((output / "job-result.json").read_text()), result)
            failure = json.loads((output / "cpu-visualization-failure.json").read_text())
            self.assertEqual(failure["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
