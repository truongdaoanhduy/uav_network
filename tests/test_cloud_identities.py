"""Cloud consumers must identify algorithm, replay layout and generator at a glance."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from uav_network.configuration import load_config
from uav_network.cloud import cli


@pytest.mark.parametrize("algorithm,pql", [("masac", False), ("matd3", True)])
@pytest.mark.parametrize("full_gpu", [False, True])
def test_launch_names_and_tags_are_authoritative(algorithm, pql, full_gpu):
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    c.pql.enabled = pql
    c.tracking.checkpoint_store = "wandb"
    if full_gpu:
        c.runtime.replay_device = "cuda"
        c.runtime.replay_cpu_fraction = 0
    account = SimpleNamespace(name="account_27", enabled=True, token="test-token")
    credentials = SimpleNamespace(accounts=[account], wandb_key="test-wandb",
                                  entity="truongdaoanhduy-uit", project="uav_search_target")
    args = SimpleNamespace(account=account.name, required_gpu_hours=1,
        credential_dataset="owner/private", credential_timeout_seconds=30,
        config="configs/comparison-20k-hybrid75-1kb.yaml", stage="train",
        algorithm=algorithm, warm_start_artifact=[], overrides=[], auto_continue=False,
        cpu_visualize=False, compare_legacy=False, gpus_per_run=2,
        accelerator="NvidiaTeslaT4", timeout_seconds=39600, key_file="unused",
        diagnostic=None)
    captured = {}
    def script(spec):
        captured.update(spec)
        return "pass\n"
    def kaggle(arguments, selected, timeout=45):
        assert arguments[:2] == ["kernels", "push"]
        metadata = json.loads((Path(arguments[arguments.index("-p") + 1]) /
                               "kernel-metadata.json").read_text())
        assert len(metadata["id"]) <= 50
        assert metadata["title"] == metadata["id"].split("/")[1].replace("-", " ")
        return "https://www.kaggle.com/code/" + metadata["id"]
    with TemporaryDirectory() as folder, \
         patch.object(cli, "ROOT", Path(folder)), \
         patch.object(cli.subprocess, "check_output", side_effect=["", "a" * 40]), \
         patch.object(cli, "account_quotas", return_value=[{"name":account.name,"ok":True,"remaining_hours":25}]), \
         patch.object(cli, "select_account", return_value=account), \
         patch.object(cli, "resolve_username", return_value="owner"), \
         patch.object(cli, "verify_credential_dataset"), \
         patch.object(cli, "load_config", return_value=c), \
         patch.object(cli, "kernel_script", side_effect=script), \
         patch.object(cli, "run_kaggle", side_effect=kaggle):
        cli.launch(args, credentials)
    name = captured["runs"][algorithm]["name"]
    for required in ["TRAIN", algorithm.upper(), "PQL" if pql else "BASE",
                     "FULLGPU" if full_gpu else "HYBRID-C75-G25",
                     "E4096", "B8192", "R1M", "G3", "S44"]:
        assert required in name
    assert algorithm in captured["kernel_ref"]
    actual_tags = json.loads(next(x.split("=",1)[1] for x in captured["overrides"]
                                 if x.startswith("tracking.tags=")))
    if full_gpu:
        assert "cpu75-gpu25" not in actual_tags
        assert "replay-fullgpu" in actual_tags
    else:
        assert "replay-hybrid-c75-g25" in actual_tags
