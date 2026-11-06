"""Offline validation of Kaggle's file-free direct save_kernel() integration."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from uav_network.cloud.client import submit_kernel
from uav_network.cloud.direct_submit import prepare_save_request, submit_direct


class FakeRequest:
    def __init__(self):
        self.machine_shape = None


def _job(tmp_path, *, stage="train", ref="owner/train-matd3-base-unique"):
    source = tmp_path / "kernel.py"
    source.write_text("print('start training')\n")
    return {"kernel_ref": ref, "code_file": str(source),
            "enable_gpu": stage != "visualize",
            "dataset_sources": ["owner/wandb-private", "owner/hf-private"],
            "accelerator": "NvidiaTeslaT4", "timeout_seconds": 39600}


def test_direct_request_has_gpu_t4x2_private_network_and_credentials_without_metadata(tmp_path):
    job = _job(tmp_path)
    request = prepare_save_request(job, request_type=FakeRequest)
    assert request.slug == job["kernel_ref"]
    assert request.new_title == "train matd3 base unique"
    assert request.text == "print('start training')\n"
    assert request.enable_gpu is True and request.enable_internet is True
    assert request.enable_tpu is False and request.is_private is True
    assert request.machine_shape == "NvidiaTeslaT4"
    assert request.dataset_data_sources == job["dataset_sources"]
    assert request.session_timeout_seconds == 39600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kernel.py"]


def test_cpu_submit_does_not_request_gpu_shape(tmp_path):
    job = _job(tmp_path, stage="visualize")
    request = prepare_save_request(job, request_type=FakeRequest)
    assert request.enable_gpu is False and request.machine_shape is None


def test_direct_save_kernel_produces_verified_receipt(tmp_path):
    job = _job(tmp_path)
    calls = []

    @contextmanager
    def client():
        yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(
            save_kernel=lambda request: calls.append(request) or SimpleNamespace(
                url=f'https://www.kaggle.com/code/{job["kernel_ref"]}',
                versionNumber=1, error=None, invalidDatasetSources=[]))))

    fake_api = SimpleNamespace(authenticate=lambda: None, build_kaggle_client=client)
    receipt = submit_direct(job, api_factory=lambda: fake_api, request_type=FakeRequest)
    assert receipt["ref"] == job["kernel_ref"]
    assert receipt["version"] == 1
    assert len(calls) == 1
    assert not (tmp_path / "kernel-metadata.json").exists()


def test_direct_api_refuses_invalid_sources(tmp_path):
    job = _job(tmp_path)
    for url, rejected in ((f'https://www.kaggle.com/code/{job["kernel_ref"]}', ["missing-dataset"]),
                           ("https://www.kaggle.com/code/owner/wrong-slug", [])):
        @contextmanager
        def client():
            yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(
                save_kernel=lambda request: SimpleNamespace(
                    url=url, error=None, invalidDatasetSources=rejected))))
        fake_api = SimpleNamespace(authenticate=lambda: None, build_kaggle_client=client)
        with pytest.raises(RuntimeError):
            submit_direct(job, api_factory=lambda: fake_api, request_type=FakeRequest)


def test_old_sdk_must_fail_before_gpu_submitted(tmp_path):
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    if not hasattr(ApiSaveKernelRequest(), "machine_shape"):
        with pytest.raises(RuntimeError, match="kaggle>=2.2"):
            prepare_save_request(_job(tmp_path))


def test_launcher_child_process_has_token_only_in_env_and_no_metadata(tmp_path):
    spec = {"stage": "train", "kernel_ref": "owner/train-masac-base-unique",
            "credential_dataset": "owner/wandb-private",
            "storage_credential_dataset": "owner/hf-private",
            "timeout_seconds": 39600, "accelerator": "NvidiaTeslaT4"}
    (tmp_path / "kernel.py").write_text("print('training')\n")
    account = SimpleNamespace(token="not-a-real-secret-token")
    observed = []

    def fake_run(cmd, **kwargs):
        observed.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "ref": spec["kernel_ref"], "url": f'https://www.kaggle.com/code/{spec["kernel_ref"]}'}), stderr="")

    with patch("uav_network.cloud.client.subprocess.run", side_effect=fake_run):
        url = submit_kernel(spec, tmp_path, account)
    assert url.endswith(spec["kernel_ref"])
    assert len(observed) == 1
    command, kwargs = observed[0]
    assert command[-1] == "uav_network.cloud.direct_submit"
    assert account.token not in " ".join(command)
    assert kwargs["env"]["KAGGLE_API_TOKEN"] == account.token
    assert account.token not in kwargs["input"]
    job = json.loads(kwargs["input"])
    assert job["dataset_sources"] == ["owner/wandb-private", "owner/hf-private"]
    assert job["enable_gpu"] is True
    assert not (tmp_path / "kernel-metadata.json").exists()


def test_cpu_controller_credential_source_is_explicit(tmp_path):
    spec = {"stage": "visualize", "kernel_ref": "owner/run-cpu",
            "credential_dataset": "owner/wandb-private",
            "pipeline_credential_dataset": "owner/launch-private", "cpu_visualize": True,
            "timeout_seconds": 3600}
    (tmp_path / "kernel.py").write_text("pass\n")
    account = SimpleNamespace(token="fake-token")
    data = {}

    def fake_run(cmd, **kwargs):
        data.update(json.loads(kwargs["input"]))
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "url": f'https://www.kaggle.com/code/{spec["kernel_ref"]}', "ref": spec["kernel_ref"]}))

    with patch("uav_network.cloud.client.subprocess.run", side_effect=fake_run):
        submit_kernel(spec, tmp_path, account)
    assert data["enable_gpu"] is False
    assert data["dataset_sources"] == ["owner/wandb-private", "owner/launch-private"]


def test_controller_submission_has_no_local_metadata_file(tmp_path):
    from uav_network.cloud import controller
    account = SimpleNamespace(name="selected", enabled=True, token="dummy-cred-not-live")
    creds = SimpleNamespace(accounts=[account], wandb_key="dummy-wandb",
                            entity="owner", project="main")
    spec = {"root_kernel_ref": "owner/train-gpu", "kernel_ref": "owner/controller-stage",
            "stage": "controller", "account": "selected", "entity": "owner",
            "project": "main", "credential_dataset": "owner/wandb-private",
            "pipeline_credential_dataset": "owner/launch-private",
            "requires_kaggle_token": True, "timeout_seconds": 3600, "generation": 0}
    captured = []

    def submit(specification, folder, selected, timeout=50):
        assert selected is account
        assert not (folder / "kernel-metadata.json").exists()
        assert (folder / "kernel.py").is_file()
        assert (folder / "job.json").is_file()
        captured.append(specification)
        return "https://www.kaggle.com/code/owner/controller-stage"

    with patch.object(controller, "_owned_kernel_exists", return_value=False), \
         patch.object(controller, "verify_credential_dataset"), \
         patch.object(controller, "verify_pipeline_credential_dataset"), \
         patch.object(controller, "kernel_script", return_value="print('controller')\n"), \
         patch.object(controller, "submit_kernel", side_effect=submit):
        receipt = controller.submit_controller(spec, tmp_path, account, creds)
    assert receipt["kernel_ref"] == spec["kernel_ref"]
    assert captured == [spec]
    assert not list(tmp_path.rglob("kernel-metadata.json"))
