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
