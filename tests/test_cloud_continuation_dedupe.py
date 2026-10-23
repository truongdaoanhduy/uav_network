"""Kaggle missing-private-child dedupe regression tests."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from test_cloud_continuation import learner_result, parent_spec
from uav_network.cloud import continuation
from uav_network.cloud.credentials import Account, Credentials


class ContinuationDedupeTests(unittest.TestCase):
    def setUp(self):
        self.account = Account("selected", "synthetic-kaggle-token")
        self.credentials = Credentials([self.account], "synthetic-wandb-key", "owner", "project")
        self.parent, self.results = parent_spec(), learner_result()
        self.parent["kernel_ref"] = "owner/completed-root"
        self.parent["root_kernel_ref"] = self.parent["kernel_ref"]
        self.child = continuation.kernel_ref_for_segment(self.parent["root_kernel_ref"], 1)

    def _kaggle(self, list_result):
        def run(args, selected, **kwargs):
            self.assertIs(selected, self.account)
            if args[:2] == ["kernels", "output"]:
                destination = Path(args[args.index("-p") + 1])
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "job.json").write_text(json.dumps(self.parent))
                (destination / "job-result.json").write_text(json.dumps(self.results))
                (destination / "continuation.json").write_text(json.dumps(
                    {"status": "waiting_for_quota", "wait_reason": "insufficient_quota"}))
                return ""
            if args[:2] == ["kernels", "list"]:
                self.assertIn("--mine", args)
                self.assertEqual(args[args.index("--search") + 1], self.child.split("/", 1)[1])
                return list_result
            raise AssertionError(args)
        return run

    def test_permission_denied_missing_child_falls_back_to_owned_list(self):
        with TemporaryDirectory() as directory, \
             patch.object(continuation, "kernel_status", side_effect=[
                 "complete", RuntimeError("Permission 'kernels.get' was denied")]), \
             patch.object(continuation, "run_kaggle", side_effect=self._kaggle("Not found\n")), \
             patch.object(continuation, "submit_continuation", return_value={
                 "status": "submitted", "kernel_ref": self.child}) as submit:
            result = continuation.resume_waiting_continuation(
                self.parent["kernel_ref"], self.account, self.credentials, Path(directory) / "retry")
        self.assertEqual(result["kernel_ref"], self.child)
        submit.assert_called_once()

    def test_permission_denied_existing_child_is_still_deduplicated(self):
        with TemporaryDirectory() as directory, \
             patch.object(continuation, "kernel_status", side_effect=[
                 "complete", RuntimeError("Permission 'kernels.get' was denied")]), \
             patch.object(continuation, "run_kaggle", side_effect=self._kaggle(
                 json.dumps([{"ref": self.child}]))), \
             patch.object(continuation, "submit_continuation", side_effect=AssertionError("duplicate child")):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                continuation.resume_waiting_continuation(
                    self.parent["kernel_ref"], self.account, self.credentials, Path(directory) / "retry")


if __name__ == "__main__":
    unittest.main()
