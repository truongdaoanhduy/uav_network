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
    def submit(spec, folder, selected, timeout=50):
        assert (Path(folder) / "kernel.py").is_file()
        assert not (Path(folder) / "kernel-metadata.json").exists()
        assert len(spec["kernel_ref"]) <= 50
        return "https://www.kaggle.com/code/" + spec["kernel_ref"]
    with TemporaryDirectory() as folder, \
         patch.object(cli, "ROOT", Path(folder)), \
         patch.object(cli.subprocess, "check_output", side_effect=["", "a" * 40]), \
         patch.object(cli, "account_quotas", return_value=[{"name":account.name,"ok":True,"remaining_hours":25}]), \
         patch.object(cli, "select_account", return_value=account), \
         patch.object(cli, "resolve_username", return_value="owner"), \
         patch.object(cli, "verify_credential_dataset"), \
         patch.object(cli, "load_config", return_value=c), \
         patch.object(cli, "kernel_script", side_effect=script), \
         patch.object(cli, "submit_kernel", side_effect=submit):
        cli.launch(args, credentials)
    name = captured["runs"][algorithm]["name"]
    assert name.startswith(f"{algorithm.upper()}-{'PQL' if pql else 'BASE'}-")
    assert "-S44-" in name
    assert "-SH-" in name if getattr(c, algorithm).share_actor_parameters and getattr(c, algorithm).share_critic_parameters else "-IND-" in name
    assert len(name) <= 38
    assert "HYBRID" not in name and "E4096" not in name and "B8192" not in name
    assert algorithm in captured["kernel_ref"]
    actual_tags = json.loads(next(x.split("=",1)[1] for x in captured["overrides"]
                                 if x.startswith("tracking.tags=")))
    if full_gpu:
        assert "cpu75-gpu25" not in actual_tags
        assert "replay-fullgpu" in actual_tags
    else:
        assert "replay-hybrid-c75-g25" in actual_tags


def test_actual_benchmark_entrypoint_uses_complete_unique_identity_and_stage(tmp_path, monkeypatch):
    import os
    import runpy
    import sys
    entry = runpy.run_path("scripts/benchmark_pql.py")
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    c.runtime.device = "cpu"
    c.runtime.replay_device = "cuda"
    c.runtime.replay_cpu_fraction = 0
    seen = []
    def benchmark(config, steps):
        seen.append((config.tracking.run_name, os.environ.get("UAV_STAGE")))
        return {"status": "passed"}
    monkeypatch.setitem(entry["main"].__globals__, "load_config", lambda *args: c)
    monkeypatch.setitem(entry["main"].__globals__, "benchmark_pql", benchmark)
    monkeypatch.setenv("UAV_STAGE", "train")
    monkeypatch.setattr(sys, "argv", ["benchmark_pql.py", "--algorithms", "masac",
                                    "--output", str(tmp_path / "benchmark.json")])
    assert entry["main"]() == 0
    assert entry["main"]() == 0
    for name, stage in seen:
        assert stage == "benchmark"
        assert name.startswith("BENCH-MASAC-PQL-")
        assert "-S44-" in name
        assert len(name) <= 39
    assert seen[0][0] != seen[1][0]


def test_mandatory_preflight_keeps_reserved_training_id_and_marks_benchmark(tmp_path, monkeypatch):
    import os
    from uav_network.cloud import remote
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    spec = {"stage": "train", "config": "configs/comparison-20k-hybrid75-1kb.yaml",
            "overrides": [], "runs": {"masac": {"id": "reserved-training"}}}
    seen = []
    def child(arguments, **kwargs):
        seen.append(kwargs["env"])
        Path(arguments[arguments.index("--output") + 1]).write_text(json.dumps(
            {"results": [{"status": "passed", "cuda_pipeline_verified": True}]}))
    monkeypatch.setenv("UAV_STAGE", "train")
    monkeypatch.setenv("WANDB_RUN_ID", "reserved-training")
    monkeypatch.setattr(remote.subprocess, "run", child)
    remote.pql_benchmarks(spec, c, tmp_path)
    assert seen[0]["UAV_STAGE"] == "benchmark"
    assert "WANDB_RUN_ID" not in seen[0]
    assert os.environ["WANDB_RUN_ID"] == "reserved-training"


def test_authoritative_tags_replace_stale_acceleration_tags_with_resolved_runtime():
    from uav_network.cloud.identities import authoritative_tags
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    c.tracking.tags = ["uav-network", "amp-fp16", "torch-compile", "fused-adam"]
    c.runtime.amp_mode = "fp16"
    c.runtime.compile_models = False
    c.runtime.fused_optimizers = False
    tags = authoritative_tags(c)
    assert "amp-fp16" in tags
    assert "compile-off" in tags
    assert "fused-off" in tags
    assert "torch-compile" not in tags
    assert "fused-adam" not in tags


def test_github_supervised_kernel_slug_has_reserved_marker_and_fits_limit():
    from uav_network.cloud.identities import kernel_slug
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    owner = "verylongownername"
    slug = kernel_slug(owner, c, "masac", "base", "train",
                       "20261027-101530", "abcdef", external_supervisor=True)
    assert slug.startswith("gh-")
    assert len(owner + "/" + slug) <= 50
    legacy = kernel_slug(owner, c, "masac", "base", "train",
                         "20261027-101530", "abcdef")
    assert not legacy.startswith("gh-")

@pytest.mark.parametrize(
    "share_actor,share_critic,marker",
    [
        (False, False, None),
        (True, False, "SHARED-ACTOR"),
        (False, True, "SHARED-CRITIC"),
        (True, True, "SHARED-ACTOR-CRITIC"),
    ],
)
def test_parameter_sharing_is_visible_without_renaming_default_runs(share_actor, share_critic, marker):
    from uav_network.cloud.identities import authoritative_tags, run_name
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    c.runtime.algorithm = "masac"
    c.masac.share_actor_parameters = share_actor
    c.masac.share_critic_parameters = share_critic
    name = run_name(c, "masac", "PQL", "train", "suffix")
    short = {(False, False): "IND", (True, False): "AS", (False, True): "CS", (True, True): "SH"}
    assert f"-{short[(share_actor,share_critic)]}-" in name
    assert "SHARED" not in name
    tags = authoritative_tags(c, "masac")
    assert ("actor-shared" if share_actor else "actor-independent") in tags
    assert ("critic-shared" if share_critic else "critic-independent") in tags
