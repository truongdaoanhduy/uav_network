"""Per-launch packaging differences must never modify training behavior or Kaggle semantics."""
import ast
import json
from unittest.mock import patch

from uav_network.cloud.kernel import (
    kernel_metadata,
    kernel_script,
    serialize_kernel_metadata,
)


def _spec(slug):
    return {
        "kernel_ref": f"owner/{slug}",
        "stage": "train",
        "entity": "owner",
        "project": "project",
        "commit": "c" * 40,
        "credential_dataset": "owner/wandb-private",
        "storage_credential_dataset": "owner/hf-private",
        "training_only": True,
    }


def test_metadata_serialization_preserves_valid_kaggle_fields():
    spec = _spec("train-matd3-base-unique-a")
    expected = kernel_metadata(spec)
    for _ in range(5):
        raw = serialize_kernel_metadata(spec)
        assert raw.endswith("\n")
        parsed = json.loads(raw)
        for key, value in expected.items():
            if key in ("enable_tpu", "competition_sources", "kernel_sources", "model_sources"):
                assert parsed.get(key, value) == value
            else:
                assert parsed[key] == value
        assert parsed["title"].lower().replace(" ", "-") == parsed["id"].split("/")[1]
        assert parsed["is_private"] and parsed["enable_gpu"] and parsed["enable_internet"]
        assert parsed["dataset_sources"] == [
            "owner/wandb-private",
            "owner/hf-private",
        ]
        assert not any(field.startswith("random_") for field in parsed)


def test_new_launch_changes_metadata_identity_without_changing_runtime_flags():
    a = json.loads(serialize_kernel_metadata(_spec("train-matd3-base-unique-a")))
    b = json.loads(serialize_kernel_metadata(_spec("train-matd3-base-unique-b")))
    assert a != b
    required = ("code_file", "language", "kernel_type", "is_private", "enable_gpu",
                "enable_internet", "dataset_sources")
    assert {key: a[key] for key in required} == {key: b[key] for key in required}


def test_each_bootstrap_has_unique_comments_without_changing_executed_ast():
    spec = _spec("train-matd3-base-unique-a")
    with patch("uav_network.cloud.kernel.build_source_archive", return_value=b"fixture"):
        a = kernel_script(spec)
        b = kernel_script(spec)
    assert a != b
    assert all(line.startswith("# package ") for line in a.splitlines()[:4])
    assert all(line.startswith("# package ") for line in b.splitlines()[:4])
    assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    assert "owner/wandb-private" in a
    assert "owner/hf-private" in a
