"""Auto-continue launch is ready only after a cloud supervisor was submitted."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.cloud import cli, controller
from uav_network.cloud.credentials import Account, Credentials
from uav_network.configuration import Config


class ControllerLaunchTests(unittest.TestCase):
    def launch(self, folder, fail=False):
        account = Account("selected", "test-launch-token")
        credentials = Credentials([account], "test-wandb-secret", "owner", "main")
        config = Config()
        config.tracking.entity, config.tracking.project = "owner", "main"
        config.tracking.checkpoint_store = "wandb"
        args = SimpleNamespace(account="selected", required_gpu_hours=1,
            credential_dataset="owner/private", credential_timeout_seconds=120,
            config="configs/kaggle-smoke.yaml", stage="train", algorithm="masac",
            warm_start_artifact=[], overrides=[], auto_continue=True, cpu_visualize=True,
            compare_legacy=False, gpus_per_run=1, accelerator="NvidiaTeslaT4",
            timeout_seconds=39600, key_file="unused", diagnostic=None)
        def submit(spec, output, selected, creds):
            self.assertEqual(spec["state"]["current_gpu_ref"], "owner/actual-gpu")
            self.assertTrue(list(Path(folder).rglob("launch.json")), "GPU receipt must survive controller failure")
            if fail:
                raise RuntimeError("service unavailable")
            return {**spec, "status": "submitted"}
        with patch.object(cli, "ROOT", Path(folder)), \
             patch.object(cli.subprocess, "check_output", side_effect=["", "a"*40]), \
             patch.object(cli, "account_quotas", return_value=[{"name": "selected", "ok": True, "remaining_hours": 30}]), \
             patch.object(cli, "select_account", return_value=account), \
             patch.object(cli, "resolve_username", return_value="owner"), \
             patch.object(cli, "verify_credential_dataset"), \
             patch.object(cli, "ensure_pipeline_dataset", return_value="owner/pipeline"), \
             patch.object(cli, "load_config", return_value=config), \
             patch.object(cli, "run_kaggle", return_value="https://www.kaggle.com/code/owner/actual-gpu"), \
             patch.object(controller, "submit_controller", side_effect=submit):
            return cli.launch(args, credentials)

    def test_auto_continue_seeds_cloud_controller_before_reporting_readiness(self):
        with TemporaryDirectory() as directory:
            manifest = self.launch(directory)
        self.assertTrue(manifest.get("unattended_ready"), "Running GPU alone cannot survive a quota wait offline")
        self.assertTrue(manifest["controller_ref"].startswith("owner/uav-flow-"))

    def test_controller_launch_failure_keeps_gpu_receipt_and_reports_not_ready(self):
        with TemporaryDirectory() as directory:
            manifest = self.launch(directory, fail=True)
            saved = json.loads(Path(manifest["manifest"]).read_text())
        self.assertFalse(manifest.get("unattended_ready", True))
        self.assertEqual(saved["kernel_ref"], "owner/actual-gpu")
        self.assertEqual(saved["controller_error_type"], "RuntimeError")
