"""Offline tests for the standalone V9 baseline-batched Eval100/Test4000 run."""
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.evaltest_baseline_batch_from_hf import (
    build_layout_summary,
    extract_actor_snapshot_archive,
    snapshot_hub_path,
    unique_benchmark_run_name,
)


def test_v9_74_checkpoints_fit_one_gpu_batch_each():
    result = build_layout_summary(
        checkpoints=74, validation_episodes=100, policies_per_batch=81,
        max_eval_worlds_per_gpu=8192, test_episodes=4000,
        max_test_worlds_per_gpu=4096,
    )
    assert result["validation"]["gpu0"]["batches"] == [3700]
    assert result["validation"]["gpu1"]["batches"] == [3700]
    assert result["test"]["gpu0"]["batches"] == [2000]
    assert result["test"]["gpu1"]["batches"] == [2000]
    assert result["validation"]["total_episode_rollouts"] == 7400


def test_smaller_batch_is_comparable_and_nonoverlapping():
    result = build_layout_summary(74, 100, 2, 256, 4000, 256)
    assert result["validation"]["gpu0"]["batches"] == [200] * 18 + [100]
    assert result["test"]["gpu0"]["batches"] == [256] * 7 + [208]


def test_reject_allocation_above_capacity():
    with pytest.raises(ValueError, match="capacity"):
        build_layout_summary(74, 100, 82, 8192, 4000, 4096)


def test_source_repository_path_is_per_run():
    assert snapshot_hub_path("e079eace") == "deferred-gpu-eval/e079eace/deferred-gpu-eval.tar.gz"
    with pytest.raises(ValueError):
        snapshot_hub_path("e079eace/../../other")


def test_benchmark_does_not_use_original_run_id():
    name = unique_benchmark_run_name("e079eace", "12345678")
    assert "e079eace" in name and "12345678" in name and name != "e079eace"


def _make_tar(path, names):
    with tarfile.open(path, "w:gz") as out:
        for name in names:
            payload = b"{}"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            out.addfile(info, io.BytesIO(payload))


def test_refuses_tar_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    _make_tar(archive, ["deferred-gpu-eval/manifest.json", "../secret.txt"])
    with pytest.raises(ValueError, match="archive"):
        extract_actor_snapshot_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "secret.txt").exists()


def test_extract_manifest_and_snapshot_without_rewriting_source(tmp_path):
    import hashlib
    archive = tmp_path / "snap.tar.gz"
    actor = b"actor bytes"
    manifest = {
        "schema": 1,
        "kind": "deferred_gpu_eval_actor_snapshots",
        "count": 1, "collected": 1, "pruned": 0,
        "keep_all_distinct_versions": True,
        "validation_episodes": 100,
        "validation_seed_first": 1000000044,
        "snapshots": [{"file": "actor_0.pt", "sha256": hashlib.sha256(actor).hexdigest(),
                       "global_step": 0, "episodes": 0, "policy_version": 0, "reasons": ["periodic"]}],
    }
    with tarfile.open(archive, "w:gz") as out:
        for name, payload in [("deferred-gpu-eval/manifest.json", json.dumps(manifest).encode()),
                              ("deferred-gpu-eval/actor_0.pt", actor)]:
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            out.addfile(item, io.BytesIO(payload))
    folder = extract_actor_snapshot_archive(archive, tmp_path / "extract")
    assert (folder / "actor_0.pt").read_bytes() == actor
    assert json.loads((folder / "manifest.json").read_text())["count"] == 1


def test_rejects_corrupt_snapshot_checksum(tmp_path):
    import hashlib
    archive = tmp_path / "corrupt.tar.gz"
    actor = b"tampered checkpoint"
    manifest = {
        "schema": 1, "kind": "deferred_gpu_eval_actor_snapshots", "count": 1,
        "collected": 1, "pruned": 0, "keep_all_distinct_versions": True,
        "snapshots": [{"file": "actor_0.pt", "sha256": hashlib.sha256(b"other").hexdigest()}],
    }
    with tarfile.open(archive, "w:gz") as out:
        for name,payload in [("deferred-gpu-eval/manifest.json",json.dumps(manifest).encode()),
                              ("deferred-gpu-eval/actor_0.pt",actor)]:
            item=tarfile.TarInfo(name);item.size=len(payload)
            out.addfile(item,io.BytesIO(payload))
    with pytest.raises(ValueError,match="SHA256"):
        extract_actor_snapshot_archive(archive,tmp_path/"extract")


def test_raises_on_tar_symlink(tmp_path):
    archive=tmp_path/"link.tar.gz"
    with tarfile.open(archive,"w:gz") as out:
        link=tarfile.TarInfo("deferred-gpu-eval/model.pt")
        link.type=tarfile.SYMTYPE
        link.linkname="../../key.txt"
        out.addfile(link)
    with pytest.raises(ValueError,match="archive"):
        extract_actor_snapshot_archive(archive,tmp_path/"extract")


def test_independent_kaggle_notebook_is_3_cell_and_stays_under_1mb():
    import ast
    import base64
    import hashlib
    import nbformat
    import re
    from scripts.build_matd3_v9_evaltest_baseline_batch_kaggle_ui import build
    notebook=build()
    assert [c.cell_type for c in notebook.cells]==["markdown","code","code"]
    for cell in notebook.cells[1:]:
        ast.parse(cell.source)
    output=nbformat.writes(notebook).encode()
    assert len(output)<1_000_000
    c0,c1,c2=[c.source for c in notebook.cells]
    assert 'e079eace' in c1
    assert 'CHECKPOINTS_PER_GPU_BATCH = 81' in c1
    assert 'MAX_WARP_EVAL_WORLDS_PER_GPU = 8192' in c1
    assert 'MAX_WARP_TEST_WORLDS_PER_GPU = 4096' in c1
    assert 'evaltest_baseline_batch_from_hf.py' in c2
    assert 'uav_network.cloud.remote' not in c2
    assert 'uav_network.experiments.one_session\"' not in c2
    assert notebook.metadata['does_not_modify_source_wandb_run'] is True
    assert notebook.metadata['no_training'] is True
    match=re.search(r"payload = base64.b85decode\('([^']+)'\)",c2)
    assert match is not None
    assert hashlib.sha256(base64.b85decode(match.group(1))).hexdigest()==notebook.metadata['original_v9_embedded_source_sha256']
