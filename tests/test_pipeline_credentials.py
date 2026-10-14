"""Autonomous CPU launch credentials stay in an owned private dataset."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.cloud import client
from uav_network.cloud.credentials import Account


class PipelineCredentialTests(unittest.TestCase):
    def test_existing_launch_dataset_with_extra_credentials_is_rejected(self):
        account = Account("alias", "selected-token")
        dataset = SimpleNamespace(ref="owner/uav-cpu-launch-credential", is_private=True)
        api = SimpleNamespace(authenticate=lambda: None, dataset_list=lambda **kwargs: [dataset])
        def transport(args, selected, **kwargs):
            if args[:2] == ["datasets", "files"]:
                return json.dumps([{"name": "uav_cpu_launch_token.txt"}, {"name": "other_account_token.txt"}])
            Path(args[args.index("-p") + 1], "uav_cpu_launch_token.txt").write_text("selected-token")
            return ""
        with patch.dict("sys.modules", {"kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=lambda: api)}):
            with patch.object(client, "run_kaggle", side_effect=transport):
                with self.assertRaises(RuntimeError):
                    client.ensure_pipeline_dataset(account, "owner")

    def test_selected_token_only_is_uploaded_privately_then_verified(self):
        ensure = getattr(client, "ensure_pipeline_dataset", None)
        self.assertIsNotNone(ensure, "Private credential preparation for CPU job submission is missing")
        account = Account("alias", "selected-token")
        created = []
        dataset = SimpleNamespace(ref="owner/uav-cpu-launch-credential", is_private=True)
        api = SimpleNamespace(authenticate=lambda: None,
                              dataset_list=lambda **kwargs: [dataset] if created else [])

        def transport(args, selected, **kwargs):
            if args[:2] == ["datasets", "create"]:
                self.assertNotIn("--public", args)
                folder = Path(args[args.index("-p") + 1])
                metadata = json.loads((folder / "dataset-metadata.json").read_text())
                self.assertEqual(metadata["id"], "owner/uav-cpu-launch-credential")
                self.assertEqual(sorted(p.name for p in folder.iterdir()),
                                 ["dataset-metadata.json", "uav_cpu_launch_token.txt"])
                self.assertEqual((folder / "uav_cpu_launch_token.txt").read_text(), "selected-token")
                created.append(metadata)
                return "Created privately"
            if args[:2] == ["datasets", "files"]:
                return json.dumps([{"name": "uav_cpu_launch_token.txt"}])
            Path(args[args.index("-p") + 1], "uav_cpu_launch_token.txt").write_text("selected-token")
            return ""

        with patch.dict("sys.modules", {"kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=lambda: api)}):
            with patch.object(client, "run_kaggle", side_effect=transport):
                self.assertEqual(ensure(account, "owner"), "owner/uav-cpu-launch-credential")
        self.assertEqual(len(created), 1)

    def test_public_existing_launch_credentials_are_rejected_without_upload(self):
        ensure = getattr(client, "ensure_pipeline_dataset", None)
        self.assertIsNotNone(ensure, "Private credential verification is missing")
        dataset = SimpleNamespace(ref="owner/uav-cpu-launch-credential", is_private=False)
        api = SimpleNamespace(authenticate=lambda: None, dataset_list=lambda **kwargs: [dataset])
        with patch.dict("sys.modules", {"kaggle.api.kaggle_api_extended": SimpleNamespace(KaggleApi=lambda: api)}):
            with patch.object(client, "run_kaggle", side_effect=AssertionError("Public credentials must not be used")):
                with self.assertRaises(RuntimeError):
                    ensure(Account("alias", "selected-token"), "owner")


if __name__ == "__main__":
    unittest.main()
