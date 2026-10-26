"""Measured production acceleration policy must be explicit and narrowly scoped."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from uav_network.configuration import load_config
from uav_network.cloud.acceleration import apply_measured_acceleration_policy
from uav_network.cloud.identities import authoritative_tags


CANONICAL = Path("configs/comparison-20k-hybrid75-1kb.yaml").resolve()


def resolved(overrides):
    return load_config(CANONICAL, overrides)


@pytest.mark.parametrize(
    "algorithm,pql,expected_compile",
    [
        ("masac", False, True),
        ("matd3", False, True),
        ("masac", True, True),
        ("matd3", True, False),
    ],
)
def test_measured_policy_applies_the_four_production_modes(algorithm, pql, expected_compile):
    initial = [f"pql.enabled={str(pql).lower()}"]
    config = resolved(initial)
    effective = apply_measured_acceleration_policy(
        CANONICAL, "train", algorithm, config, initial, CANONICAL)
    final = resolved(effective)
    assert final.runtime.amp_mode == "fp16"
    assert final.runtime.compile_models is expected_compile
    assert final.runtime.fused_optimizers is False


def test_measured_policy_is_noop_for_benchmark_and_noncanonical_profile(tmp_path):
    initial = ["pql.enabled=true", 'runtime.amp_mode="off"', "runtime.compile_models=false"]
    config = resolved(initial)
    assert apply_measured_acceleration_policy(
        CANONICAL, "benchmark", "matd3", config, initial, CANONICAL) == initial
    copied = tmp_path / CANONICAL.name
    copied.write_text(CANONICAL.read_text())
    assert apply_measured_acceleration_policy(
        copied, "train", "matd3", config, initial, CANONICAL) == initial


def test_explicit_acceleration_overrides_win_over_measured_policy():
    initial = [
        "pql.enabled=true",
        'runtime.amp_mode="off"',
        "runtime.compile_models=true",
        "runtime.fused_optimizers=true",
    ]
    config = resolved(initial)
    effective = apply_measured_acceleration_policy(
        CANONICAL, "train", "matd3", config, initial, CANONICAL)
    assert effective == initial


def test_canonical_pql_both_production_is_refused_but_base_both_is_allowed():
    pql = resolved(["pql.enabled=true"])
    with pytest.raises(ValueError, match="separate MASAC and MATD3"):
        apply_measured_acceleration_policy(
            CANONICAL, "train", "both", pql, ["pql.enabled=true"], CANONICAL)
    base = resolved(["pql.enabled=false"])
    effective = apply_measured_acceleration_policy(
        CANONICAL, "train", "both", base, ["pql.enabled=false"], CANONICAL)
    final = resolved(effective)
    assert final.runtime.amp_mode == "fp16"
    assert final.runtime.compile_models is True
    assert final.runtime.fused_optimizers is False


def test_authoritative_acceleration_tags_follow_resolved_runtime():
    config = resolved([
        "pql.enabled=true", "runtime.algorithm=matd3", "runtime.amp_mode=fp16",
        "runtime.compile_models=false", "runtime.fused_optimizers=false",
    ])
    tags = authoritative_tags(config)
    assert "amp-fp16" in tags
    assert "compile-off" in tags
    assert "fused-off" in tags
    assert "torch-compile" not in tags
    assert "fused-adam" not in tags


def test_canonical_tracking_group_is_acceleration_neutral():
    group = load_config(CANONICAL).tracking.group.lower()
    assert "ampfp16" not in group
    assert "compile" not in group
    assert "accel-policy" in group


def _launch_canonical(tmp_path, algorithm, pql, extra_overrides=None, stage="train"):
    from uav_network.cloud import cli

    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANONICAL, tmp_path / "configs" / CANONICAL.name)
    account = SimpleNamespace(name="account_test", enabled=True, token="test-token")
    credentials = SimpleNamespace(
        accounts=[account], wandb_key="test-wandb",
        entity="truongdaoanhduy-uit", project="uav_search_target")
    overrides = [
        f"pql.enabled={str(pql).lower()}",
        'tracking.checkpoint_store="wandb"',
        *(extra_overrides or []),
    ]
    args = SimpleNamespace(
        account=account.name, required_gpu_hours=1,
        credential_dataset="owner/private", credential_timeout_seconds=30,
        config=f"configs/{CANONICAL.name}", stage=stage, algorithm=algorithm,
        warm_start_artifact=[], overrides=overrides, auto_continue=False,
        cpu_visualize=False, compare_legacy=False, gpus_per_run=2,
        accelerator="NvidiaTeslaT4", timeout_seconds=3600, key_file="unused",
        diagnostic=None, benchmark_steps=2, benchmark_learner_updates=2)
    with patch.object(cli, "ROOT", tmp_path), \
         patch.object(cli.subprocess, "check_output", side_effect=["", "a" * 40]), \
         patch.object(cli, "account_quotas", return_value=[{
             "name": account.name, "ok": True, "remaining_hours": 25}]), \
         patch.object(cli, "select_account", return_value=account), \
         patch.object(cli, "resolve_username", return_value="owner"), \
         patch.object(cli, "verify_credential_dataset"), \
         patch.object(cli, "kernel_script", return_value="pass\n"), \
         patch.object(cli, "run_kaggle", return_value="https://www.kaggle.com/code/owner/test-kernel"):
        return cli.launch(args, credentials)


def _override_map(manifest):
    result = {}
    for item in manifest["overrides"]:
        key, value = item.split("=", 1)
        result[key] = yaml.safe_load(value)
    return result


def _tracking_tags(manifest):
    return json.loads(next(item.split("=", 1)[1] for item in manifest["overrides"]
                           if item.startswith("tracking.tags=")))


def test_launcher_materializes_measured_pql_policy_and_truthful_tags(tmp_path):
    matd3 = _launch_canonical(tmp_path / "matd3", "matd3", True)
    masac = _launch_canonical(tmp_path / "masac", "masac", True)
    matd3_overrides = _override_map(matd3)
    masac_overrides = _override_map(masac)
    assert matd3_overrides["runtime.amp_mode"] == "fp16"
    assert matd3_overrides["runtime.compile_models"] is False
    assert matd3_overrides["runtime.fused_optimizers"] is False
    assert masac_overrides["runtime.amp_mode"] == "fp16"
    assert masac_overrides["runtime.compile_models"] is True
    assert masac_overrides["runtime.fused_optimizers"] is False
    assert "compile-off" in _tracking_tags(matd3)
    assert "torch-compile" not in _tracking_tags(matd3)
    assert "torch-compile" in _tracking_tags(masac)
    assert "compile-off" not in _tracking_tags(masac)


def test_launcher_preserves_explicit_compile_override_for_matd3_pql(tmp_path):
    manifest = _launch_canonical(
        tmp_path, "matd3", True, ["runtime.compile_models=true"])
    assert _override_map(manifest)["runtime.compile_models"] is True


def test_launcher_keeps_benchmark_policy_free(tmp_path):
    manifest = _launch_canonical(
        tmp_path, "matd3", True,
        ['runtime.amp_mode="off"', "runtime.compile_models=false", "runtime.fused_optimizers=false"],
        stage="benchmark")
    effective = _override_map(manifest)
    assert effective["runtime.amp_mode"] == "off"
    assert effective["runtime.compile_models"] is False
    assert effective["runtime.fused_optimizers"] is False


def test_launcher_refuses_canonical_pql_both_before_submission(tmp_path):
    with pytest.raises(ValueError, match="separate MASAC and MATD3"):
        _launch_canonical(tmp_path, "both", True)
