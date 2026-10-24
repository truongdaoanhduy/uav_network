"""Only actual cloud heartbeats establish the offline-ready signal."""
from datetime import datetime, timezone, timedelta
import importlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_cloud_continuation import parent_spec
from uav_network.cloud.credentials import Account, Credentials


class ControllerHealthTests(unittest.TestCase):
    def test_receipt_without_heartbeat_is_not_ready_and_adoption_uses_actual_run_url(self):
        module = importlib.import_module("uav_network.cloud.controller")
        self.assertTrue(callable(getattr(module, "wait_controller_ready", None)))
        account = Account("selected", "launch-test-token")
        credentials = Credentials([account], "wandb-test-token", "owner", "project")
        spec = module.build_controller_spec(parent_spec(), "a"*40)
        summary = {"phase": "training", "current_gpu_ref": parent_spec()["kernel_ref"]}
        run = SimpleNamespace(id="actual-existing-run", url="https://wandb.ai/owner/project/runs/actual-existing-run",
                              state="running", config={"root_kernel_ref": spec["root_kernel_ref"], "commit": spec["commit"]},
                              summary=summary)
        api = SimpleNamespace(runs=lambda *a, **kw: [run])
        sdk = SimpleNamespace(Api=lambda **kw: api)
        with patch.dict(sys.modules, {"wandb": sdk}), patch.object(module, "kernel_status", return_value="running"):
            absent = module.wait_controller_ready(spec, account, credentials, timeout=0)
            self.assertFalse(absent["ready"])
            summary["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
            healthy = module.wait_controller_ready(spec, account, credentials, timeout=0)
            self.assertTrue(healthy["ready"])
            self.assertEqual(healthy["wandb_url"], run.url)
            self.assertEqual(healthy["run_id"], "actual-existing-run")
            summary["heartbeat_utc"] = (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
            self.assertFalse(module.wait_controller_ready(spec, account, credentials, timeout=0)["ready"])
