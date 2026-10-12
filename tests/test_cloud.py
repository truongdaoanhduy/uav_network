"""Quota, pinned private kernel generation and non-secret job manifests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from uav_network.cloud.credentials import load_credentials, parse_quota, select_account
from uav_network.cloud.kernel import kernel_metadata, kernel_script
from uav_network.cloud.remote import learner_command


class CloudTests(unittest.TestCase):
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
