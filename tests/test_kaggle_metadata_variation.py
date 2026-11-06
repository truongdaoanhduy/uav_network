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
    operational = {key: value for key, value in expected.items() if not key.startswith("x_")}
    for _ in range(5):
        raw = serialize_kernel_metadata(spec)
        assert raw.endswith("\n")
        parsed = json.loads(raw)
        for key, value in operational.items():
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


def test_masac_and_matd3_have_different_inert_fields_beyond_id_and_title():
    masac = _spec("new-run-masac")
    matd3 = _spec("new-run-matd3")
    masac["runs"] = {"masac": {"id": "aaaa0000"}}
    matd3["runs"] = {"matd3": {"id": "bbbb0000"}}
    a = kernel_metadata(masac)
    b = kernel_metadata(matd3)
    assert a["x_uav_annotations"]["algorithm_profile"] == "masac"
    assert b["x_uav_annotations"]["algorithm_profile"] == "matd3"
    assert "policy_note_stochastic" in a["x_uav_annotations"]["profile_fields"]
    assert "policy_note_deterministic" in b["x_uav_annotations"]["profile_fields"]
    assert a["x_uav_annotations"]["profile_fields"] != b["x_uav_annotations"]["profile_fields"]
    for d in (a, b):
        assert 5 <= sum(key.startswith("x_package_") for key in d) <= 11
        assert d["is_private"] is True and d["enable_gpu"] is True


def test_same_algorithm_each_launch_has_different_nonce_and_random_keys():
    from uav_network.cloud.kernel import _packaging_annotations
    spec = _spec("run-masac-unique")
    spec["runs"] = {"masac": {"id": "abcd0001"}}
    a, b = kernel_metadata(spec), kernel_metadata(spec)
    assert a["x_uav_annotations"]["nonce"] != b["x_uav_annotations"]["nonce"]
    assert set(a) != set(b)
    assert _packaging_annotations(spec)["algorithm_profile"] == "masac"


def test_arbitrary_nested_key_value_map_can_be_supplied_without_touching_kaggle_controls():
    from uav_network.cloud.kernel import validate_metadata_extras
    extras = validate_metadata_extras({
        "common": {"custom_number": 43, "settings": {"alpha": False, "layers": [1, 2, 3]}},
        "masac": {"masac_custom": "different-a"},
        "matd3": {"matd3_custom": "different-b"},
    })
    masac = {**_spec("unique-masac"), "runs": {"masac": {}}, "metadata_extras": extras}
    matd3 = {**_spec("unique-matd3"), "runs": {"matd3": {}}, "metadata_extras": extras}
    m = kernel_metadata(masac)
    d = kernel_metadata(matd3)
    assert m["x_uav_annotations"]["custom"]["masac_custom"] == "different-a"
    assert "matd3_custom" not in m["x_uav_annotations"]["custom"]
    assert d["x_uav_annotations"]["custom"]["matd3_custom"] == "different-b"
    assert m["x_uav_annotations"]["custom"]["settings"]["layers"] == [1, 2, 3]
    assert d["x_uav_annotations"]["custom"]["settings"]["alpha"] is False
    with patch("uav_network.cloud.kernel.build_source_archive", return_value=b"fixture"):
        script_m = kernel_script(masac)
        script_d = kernel_script(matd3)
    assert "masac_custom" in script_m and "matd3_custom" not in script_m
    assert "matd3_custom" in script_d and "masac_custom" not in script_d
    assert ast.dump(ast.parse(script_m)) != ast.dump(ast.parse(script_d))  # the run specification differs
    assert m["code_file"] == d["code_file"] == "kernel.py"
    assert m["dataset_sources"] == d["dataset_sources"]


def test_user_metadata_rejects_nonmap_and_secretlike_fields():
    import pytest
    from uav_network.cloud.kernel import validate_metadata_extras
    for bad in ({"masac": []}, {"other": {"value": 1}},
                {"common": {"api_key": "do-not-store"}},
                {"common": {"many": "x" * 10_000}}):
        with pytest.raises((ValueError, TypeError)):
            validate_metadata_extras(bad)


def test_cli_accepts_custom_metadata_file_option_without_launching():
    from uav_network.cloud.cli import main
    with patch("uav_network.cloud.cli.load_credentials", return_value=object()), \
         patch("uav_network.cloud.cli.launch", side_effect=lambda args, credentials: args):
        args = main(["launch", "--credential-dataset", "owner/private",
                     "--algorithm", "matd3", "--metadata-extras-file", "/tmp/extra.json"])
        assert args.metadata_extras_file == "/tmp/extra.json"
        assert args.algorithm == "matd3"
