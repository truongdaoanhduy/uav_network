"""Autonomous CPU launch credentials stay in an owned private dataset."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.cloud import client
from uav_network.cloud.credentials import Account


class PipelineCredentialTests(unittest.TestCase):
    def test_new_private_dataset_waits_for_download_endpoint_readiness(self):
        account = Account('one', 'private-token')
        with patch.object(client, 'private_dataset_info', return_value={}), \
             patch.object(client, 'private_dataset_file', side_effect=[
                 RuntimeError('404 Client Error: Not Found for DownloadDataset'),
                 ('uav_cpu_launch_token.txt', account.token)]) as read, \
             patch.object(client.time, 'sleep') as pause:
            self.assertEqual(client.ensure_pipeline_dataset(account, 'owner'),
                             'owner/uav-cpu-launch-credential')
            self.assertEqual(read.call_count, 2)
            pause.assert_called_once()
            self.assertTrue(read.call_args.kwargs['only_selected'])


    def test_existing_private_credential_refreshes_when_value_is_stale(self):
        account = Account("one", "new-private-token")
        dataset = SimpleNamespace(ref="owner/uav-cpu-launch-credential", is_private=True)
        reads = [("uav_cpu_launch_token.txt", "old-private-token"),
                 ("uav_cpu_launch_token.txt", account.token)]
        versions = []

        def transport(args, selected, **kwargs):
            if args[:2] == ["datasets", "version"]:
                folder = Path(args[args.index("-p") + 1])
                self.assertEqual((folder / "uav_cpu_launch_token.txt").read_text(), account.token)
                versions.append(True)
                return "versioned"
            raise AssertionError(args)

        with patch.object(client, "private_dataset_info", return_value=dataset), \
             patch.object(client, "private_dataset_file", side_effect=reads) as read, \
             patch.object(client, "run_kaggle", side_effect=transport):
            self.assertEqual(client.ensure_pipeline_dataset(account, "owner", timeout=60), dataset.ref)
        self.assertEqual(len(versions), 1)
        self.assertEqual(read.call_count, 2)


    def test_existing_private_credential_refreshes_after_download_timeout(self):
        account = Account("one", "private-token")
        dataset = SimpleNamespace(ref="owner/uav-cpu-launch-credential", is_private=True)
        calls = []

        def transport(args, selected, **kwargs):
            calls.append(args[:2])
            if args[:2] == ["datasets", "version"]:
                folder = Path(args[args.index("-p") + 1])
                self.assertEqual((folder / "uav_cpu_launch_token.txt").read_text(), account.token)
                metadata = json.loads((folder / "dataset-metadata.json").read_text())
                self.assertEqual(metadata["id"], dataset.ref)
                return "versioned"
            if args[:2] == ["datasets", "files"]:
                return json.dumps([{"name": "uav_cpu_launch_token.txt"}])
            raise AssertionError(args)

        with patch.object(client, "private_dataset_info", return_value=dataset), \
             patch.object(client, "private_dataset_file",
                          side_effect=subprocess.TimeoutExpired(["kaggle", "datasets", "download"], 30)), \
             patch.object(client, "run_kaggle", side_effect=transport):
            self.assertEqual(client.ensure_pipeline_dataset(account, "owner", timeout=60), dataset.ref)
        self.assertIn(["datasets", "version"], calls)
        self.assertIn(["datasets", "files"], calls)



    def test_existing_private_credential_accepts_ambiguous_version_timeout_with_strict_shape(self):
        account = Account("one", "private-token")
        dataset = SimpleNamespace(ref="owner/uav-cpu-launch-credential", is_private=True)

        def transport(args, selected, **kwargs):
            if args[:2] == ["datasets", "version"]:
                raise subprocess.TimeoutExpired(["kaggle", "datasets", "version"], kwargs.get("timeout", 15))
            if args[:2] == ["datasets", "files"]:
                return json.dumps([{"name": "uav_cpu_launch_token.txt"}])
            raise AssertionError(args)

        with patch.object(client, "private_dataset_info", return_value=dataset), \
             patch.object(client, "private_dataset_file",
                          side_effect=subprocess.TimeoutExpired(["kaggle", "datasets", "download"], 15)), \
             patch.object(client, "run_kaggle", side_effect=transport):
            self.assertEqual(client.ensure_pipeline_dataset(account, "owner", timeout=60), dataset.ref)

    def test_download_probe_leaves_time_for_timeout_recovery(self):
        account = Account("one", "private-token")
        dataset = SimpleNamespace(ref="owner/uav-cpu-launch-credential", is_private=True)
        observed = []

        def read(*args, **kwargs):
            observed.append(args[4])
            raise subprocess.TimeoutExpired(["kaggle", "datasets", "download"], args[4])

        def transport(args, selected, **kwargs):
            if args[:2] == ["datasets", "version"]:
                return "versioned"
            if args[:2] == ["datasets", "files"]:
                return json.dumps([{"name": "uav_cpu_launch_token.txt"}])
            raise AssertionError(args)

        with patch.object(client, "private_dataset_info", return_value=dataset), \
             patch.object(client, "private_dataset_file", side_effect=read), \
             patch.object(client, "run_kaggle", side_effect=transport):
            self.assertEqual(client.ensure_pipeline_dataset(account, "owner", timeout=60), dataset.ref)
        self.assertLessEqual(observed[0], 15)


    def test_pipeline_token_verification_uses_strict_file_shape_when_download_stalls(self):
        verify = getattr(client, "verify_pipeline_credential_dataset", None)
        self.assertIsNotNone(verify, "Pipeline credential timeout fallback is missing")
        account = Account("one", "private-token")
        timeout = subprocess.TimeoutExpired(["kaggle", "datasets", "download"], 15)
        with patch.object(client, "private_dataset_file", side_effect=timeout), \
             patch.object(client, "private_dataset_files",
                          return_value=[{"name": "uav_cpu_launch_token.txt"}]):
            self.assertEqual(verify(account, "owner/uav-cpu-launch-credential", "owner", timeout=60),
                             account.token)

    def test_download_readiness_retry_expires_and_never_retries_auth_failures(self):
        account = Account('one', 'private-token')
        with patch.object(client, 'private_dataset_info', return_value={}), \
             patch.object(client, 'private_dataset_file', side_effect=RuntimeError('404 Client Error')), \
             patch.object(client.time, 'monotonic', side_effect=[0, 0, 121]), \
             patch.object(client.time, 'sleep'):
            with self.assertRaises(TimeoutError):
                client.ensure_pipeline_dataset(account, 'owner')
        with patch.object(client, 'private_dataset_info', return_value={}), \
             patch.object(client, 'private_dataset_file', side_effect=RuntimeError('403 Client Error')) as read, \
             patch.object(client.time, 'sleep') as pause:
            with self.assertRaisesRegex(RuntimeError, '403'):
                client.ensure_pipeline_dataset(account, 'owner')
            self.assertEqual(read.call_count, 1)
            pause.assert_not_called()

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
