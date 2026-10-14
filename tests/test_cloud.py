"""Quota, pinned private kernel generation and non-secret job manifests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from uav_network.cloud.credentials import load_credentials, parse_quota, select_account
from uav_network.cloud.kernel import kernel_metadata, kernel_script
from uav_network.cloud.remote import learner_command, wait_all


class CloudTests(unittest.TestCase):
    def test_cli_uses_the_active_python_environment_without_a_global_executable(self):
        from uav_network.cloud.client import run_kaggle
        import os
        with TemporaryDirectory() as directory:
            package = Path(directory) / "kaggle"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "__main__.py").write_text(
                "import json,os,sys\n"
                "print(json.dumps({'args':sys.argv[1:],'selected':os.environ.get('KAGGLE_API_TOKEN')=='secret-a'}))\n")
            with patch.dict(os.environ, {"PATH": directory, "PYTHONPATH": directory}):
                response = json.loads(run_kaggle(["kernels", "status", "owner/child"], self.credentials().accounts[0]))
            self.assertEqual(response["args"], ["kernels", "status", "owner/child"])
            self.assertTrue(response["selected"])

    def test_artifact_mapping_rejects_duplicate_or_unselected_algorithms(self):
        from uav_network.cloud.cli import parse_warm_start_artifacts
        self.assertEqual(parse_warm_start_artifacts(["masac=owner/project/model:v0"], ["masac"]),
                         {"masac": "owner/project/model:v0"})
        for values in (["matd3=owner/project/model:v0"], ["masac="],
                       ["masac=first:v0", "masac=second:v0"]):
            with self.assertRaises(ValueError):
                parse_warm_start_artifacts(values, ["masac"])

    def test_soft_deadline_shares_a_single_gpu_budget_and_caps_two_gpu_learners(self):
        from uav_network.cloud.remote import learner_time_limit
        from uav_network.configuration import Config
        c = Config()
        c.runtime.max_train_seconds = 36000
        spec = {"timeout_seconds": 39600, "runs": {"masac": {}, "matd3": {}}}
        self.assertEqual(learner_time_limit(spec, c, 2, 0, 0), 36000)
        self.assertEqual(learner_time_limit(spec, c, 1, 0, 0), 19500)
        self.assertEqual(learner_time_limit(spec, c, 1, 1, 19500), 19500)
        with self.assertRaisesRegex(RuntimeError, "time"):
            learner_time_limit(spec, c, 2, 0, 40000)

    def test_missing_timeout_uses_the_safe_eleven_hour_ceiling(self):
        from uav_network.cloud.remote import learner_time_limit
        from uav_network.configuration import Config
        self.assertEqual(learner_time_limit({"runs": {"masac": {}, "matd3": {}}}, Config(), 1, 0, 0), 19500)

    def test_failed_upload_keeps_the_saved_summary_and_explicit_failure(self):
        from uav_network.cloud.remote import finalize_runs
        with TemporaryDirectory() as directory:
            output = Path(directory)
            folder = output / "train" / "masac"
            folder.mkdir(parents=True)
            summary = {"episodes": 2, "global_step": 10, "updates": 3, "wandb_url": "test",
                       "tracking_status": "failed"}
            (folder / "summary.json").write_text(json.dumps(summary))
            with patch("uav_network.cloud.remote.subprocess.run"):
                with self.assertRaisesRegex(RuntimeError, "Learner failures"):
                    finalize_runs({"stage": "train", "runs": {"masac": {}}},
                                  [SimpleNamespace(returncode=1)], output)
            self.assertEqual(json.loads((output / "job-result.json").read_text())["masac"], summary)
            self.assertEqual(json.loads((output / "job-failures.json").read_text())["masac"]["exit_code"], 1)

    def test_requested_quota_does_not_query_unrelated_accounts(self):
        from uav_network.cloud.client import account_quotas
        c = self.credentials()
        def quota(args, account, **kwargs):
            self.assertEqual(account.name, "a")
            return json.dumps({"resources": [{"resource": "GPU", "used": "1h", "remaining": "29h", "total": "30h"}]})
        with patch("uav_network.cloud.client.run_kaggle", side_effect=quota):
            result = account_quotas(c, requested="a")
        self.assertEqual([row["name"] for row in result], ["a"])

    def test_artifact_inputs_are_downloaded_and_passed_to_the_corresponding_learner(self):
        from uav_network.cloud.remote import prepare_checkpoints
        with TemporaryDirectory() as directory:
            output = Path(directory)
            def download(root):
                import torch
                path = Path(root) / "checkpoint.pt"
                path.parent.mkdir(parents=True)
                torch.save({"algorithm": "matd3", "progress": {"episodes": 940}}, path)
                return str(path)
            api = SimpleNamespace(artifact=lambda *args, **kwargs: SimpleNamespace(
                get_entry=lambda name: SimpleNamespace(download=download)))
            with patch.dict(sys.modules, {"wandb": SimpleNamespace(Api=lambda: api)}):
                paths = prepare_checkpoints({"warm_start_artifacts": {"matd3": "owner/project/model:v0"}}, output)
            spec = {"config": "configs/kaggle.yaml", "overrides": [], "warm_start_paths": paths}
            args, _ = learner_command(spec, "matd3", {"id": "test123", "name": "MATD3"}, 1, output / "train")
            self.assertIn(f"runtime.warm_start_checkpoint={paths['matd3']}", args)

    def test_credential_check_downloads_only_the_selected_wandb_file(self):
        from uav_network.cloud.client import verify_credential_dataset
        credentials = self.credentials()
        account = credentials.accounts[0]
        dataset = SimpleNamespace(ref="owner/private-wandb", is_private=True)
        api = SimpleNamespace(authenticate=lambda: None, dataset_list=lambda **kwargs: [dataset])
        calls = []

        def kaggle(args, selected, **kwargs):
            calls.append(args)
            if args[:2] == ["datasets", "files"]:
                if "--page-token" in args:
                    self.assertEqual(args[args.index("--page-token") + 1], "next-page")
                    return json.dumps([{"name": "hf_repo_id.txt"}])
                return "Next Page Token = next-page\n" + json.dumps([
                    {"name": "hf_token.txt"}, {"name": "wandb_api_key.txt"}])
            self.assertIn("-f", args)
            self.assertEqual(args[args.index("-f") + 1], "wandb_api_key.txt")
            folder = Path(args[args.index("-p") + 1])
            (folder / "wandb_api_key.txt").write_text(credentials.wandb_key)
            return ""

        with patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=lambda: api)}):
            with patch("uav_network.cloud.client.run_kaggle", side_effect=kaggle):
                verify_credential_dataset(account, credentials, dataset.ref, "owner")
        self.assertEqual(len(calls), 3)

    def test_duplicate_wandb_files_on_separate_pages_are_rejected_before_download(self):
        from uav_network.cloud.client import verify_credential_dataset
        credentials = self.credentials()
        dataset = SimpleNamespace(ref="owner/private-wandb", is_private=True)
        api = SimpleNamespace(authenticate=lambda: None, dataset_list=lambda **kwargs: [dataset])
        pages = ["Next Page Token = next-page\n" + json.dumps([{"name": "wandb_api_key.txt"}]),
                 json.dumps([{"name": "nested/wandb_secret.json"}])]
        with patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=lambda: api)}):
            with patch("uav_network.cloud.client.run_kaggle", side_effect=pages) as call:
                with self.assertRaisesRegex(RuntimeError, "exactly one"):
                    verify_credential_dataset(credentials.accounts[0], credentials, dataset.ref, "owner")
        self.assertEqual(call.call_count, 2)

    def test_evaluation_failure_preserves_training_results_and_later_evaluations(self):
        from uav_network.cloud.remote import finalize_runs
        with TemporaryDirectory() as directory:
            output = Path(directory)
            spec = {"stage": "train", "runs": {"masac": {}, "matd3": {}}}
            summaries = {a: {"episodes": 2, "global_step": 10, "updates": 3,
                              "wandb_url": f"https://wandb.ai/test/{a}"} for a in spec["runs"]}
            for algorithm, summary in summaries.items():
                folder = output / "train" / algorithm
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(json.dumps(summary))
            calls = []

            def evaluate(args, **kwargs):
                self.assertEqual(json.loads((output / "job-result.json").read_text()), summaries)
                calls.append(args)
                if len(calls) == 1:
                    raise subprocess.CalledProcessError(2, args)

            with patch("uav_network.cloud.remote.subprocess.run", side_effect=evaluate):
                with self.assertRaisesRegex(RuntimeError, "evaluation"):
                    finalize_runs(spec, [SimpleNamespace(returncode=0)] * 2, output)
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads((output / "evaluation-failures.json").read_text())["masac"]["exit_code"], 2)

    def test_failed_learner_does_not_interrupt_an_independent_sibling(self):
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "completed"
            failed = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
            failed.wait()
            sibling = subprocess.Popen([sys.executable, "-c",
                "import time,sys; from pathlib import Path; time.sleep(.2); Path(sys.argv[1]).touch()", str(marker)])
            try:
                codes = wait_all([failed, sibling])
                self.assertEqual(codes, [3, 0])
                self.assertTrue(marker.is_file())
            finally:
                if sibling.poll() is None:
                    sibling.terminate()
                sibling.wait()

    def credentials(self):
        data = {"kaggle": {"accounts": [{"name": "a", "token": "secret-a", "enabled": True},
                                          {"name": "b", "token": "secret-b", "enabled": True}]},
                "wandb": {"api_key": "secret-w", "entity": "owner", "project": "project"}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "key.txt"
            path.write_text(json.dumps(data))
            return load_credentials(path)

    def test_max_quota_selection_and_zero_quota_rejection(self):
        quota = parse_quota({"resources": [{"resource": "GPU", "used": "5h", "remaining": "25h", "total": "30h"}]})
        self.assertEqual(quota["remaining_hours"], 25)
        credentials = self.credentials()
        selected = select_account(credentials, [{"name": "a", "remaining_hours": 1, "ok": True},
                                                {"name": "b", "remaining_hours": 25, "ok": True}])
        self.assertEqual(selected.name, "b")
        with self.assertRaisesRegex(RuntimeError, "quota"):
            select_account(credentials, [{"name": "a", "remaining_hours": 0, "ok": True}])

    def test_generated_kernel_is_private_pinned_and_places_independent_learners(self):
        spec = {"commit": "a" * 40, "kernel_ref": "owner/uav-network-test",
                "credential_dataset": "owner/private-wandb", "entity": "owner", "project": "project",
                "stage": "train", "config": "configs/kaggle.yaml", "overrides": [],
                "runs": {"masac": {"id": "masac123", "name": "MASAC"},
                         "matd3": {"id": "matd3123", "name": "MATD3"}}}
        source = kernel_script(spec)
        compile(source, "kernel.py", "exec")
        self.assertIn(spec["commit"], source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("secret-a", source)
        self.assertNotIn("secret-b", source)
        self.assertNotIn("secret-w", source)
        metadata = kernel_metadata(spec)
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["dataset_sources"], ["owner/private-wandb"])
        self.assertEqual(metadata["title"].lower().replace(" ", "-"), "uav-network-test")

    def test_push_uses_actual_api_url_and_rejects_cli_success_with_server_error(self):
        from uav_network.cloud.client import pushed_kernel_ref
        response = ("Kernel version 1 successfully pushed.  Please check progress at "
                    "https://www.kaggle.com/code/owner/actual-title-slug")
        self.assertEqual(pushed_kernel_ref(response), "owner/actual-title-slug")
        for failure in ("Kernel push error: quota exhausted",
                        "The following are not valid dataset sources and could not be added to the kernel: owner/key\n" + response):
            with self.assertRaisesRegex(RuntimeError, "push"):
                pushed_kernel_ref(failure)

    def test_independent_learner_uses_requested_gpu_and_public_run_id(self):
        spec = {"config": "configs/kaggle.yaml", "overrides": ["runtime.episodes=10"]}
        args, env = learner_command(spec, "matd3", {"id": "run123", "name": "MATD3"},
                                    gpu_index=1, destination=Path("/kaggle/working/train/matd3"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["WANDB_RUN_ID"], "run123")
        self.assertIn("runtime.episodes=10", args)
        self.assertIn("runtime.output_dir=/kaggle/working/train/matd3", args)


if __name__ == "__main__":
    unittest.main()
