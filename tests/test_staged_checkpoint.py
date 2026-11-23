"""A resumed Kaggle segment must free its downloaded full checkpoint before saving."""
import os
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest
import torch

from test_async_evaluation import settings
from uav_network.cloud.remote import learner_command
from uav_network.experiments.training import train


@pytest.mark.parametrize("pql", [False, True])
@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_staged_resume_is_unmapped_and_removed_before_next_atomic_save(tmp_path, pql, algorithm):
    c = settings()
    c.runtime.algorithm, c.pql.enabled = algorithm, pql
    c.pql.mode = "synchronous"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.pql.warmup_collection_steps, c.pql.critic_sample_ratio = 1, 1
    c.runtime.episodes = 2
    c.runtime.output_dir = str(tmp_path / "first")
    c.tracking.evaluation_interval_steps = 0
    for algorithm_config in (c.masac, c.matd3):
        algorithm_config.learning_starts = 0
    first = train(c)
    source = Path(first["resume_checkpoint"])
    original = source.read_bytes()
    staged = tmp_path / "cloud-input.pt"
    shutil.copyfile(source, staged)
    c.runtime.warm_start_checkpoint = str(staged)
    c.runtime.output_dir = str(tmp_path / "second")
    c.runtime.episodes = 4
    real_save = torch.save
    writes = []

    def check_disk_before_save(payload, path, *args, **kwargs):
        assert not staged.exists(), "Downloaded input still consumes disk during the next checkpoint"
        maps = Path("/proc/self/maps")
        if maps.exists():
            assert str(staged) not in maps.read_text(), "Deleted input remains pinned by mmap tensors"
        writes.append(path)
        return real_save(payload, path, *args, **kwargs)

    with patch.dict(os.environ, {"UAV_STAGED_RESUME_CHECKPOINT": str(staged)}), \
         patch("uav_network.algorithms.checkpoints.torch.save", side_effect=check_disk_before_save):
        second = train(c)
    assert writes
    assert second["initial_global_step"] == first["global_step"]
    assert second["initial_episodes"] == first["episodes"]
    assert second["optimizer_replay_restored"]
    assert Path(second["resume_checkpoint"]).is_file()
    assert source.read_bytes() == original
    c.runtime.warm_start_checkpoint = str(source)
    c.runtime.output_dir = str(tmp_path / "local-resume")
    local = train(c)
    for key in ("model_hash", "episode_hash", "settings_hash"):
        assert second["reproducibility"][key] == local["reproducibility"][key]


@pytest.mark.parametrize("auto_continue", [False, True])
def test_cloud_warm_start_always_marks_downloaded_input_for_release(tmp_path, auto_continue):
    staged = tmp_path / "inputs" / "masac" / "resume.pt"
    spec = {"config": "configs/kaggle.yaml", "overrides": [],
            "auto_continue": auto_continue, "warm_start_paths": {"masac": staged}}
    with patch.dict(os.environ, {"UAV_STAGED_RESUME_CHECKPOINT": "/unrelated/input.pt"}):
        _, env = learner_command(spec, "masac", {"id": "id", "name": "name"}, 0, tmp_path / "train")
    assert env.get("UAV_STAGED_RESUME_CHECKPOINT") == str(staged)


@pytest.mark.parametrize("pql", [False, True])
def test_failed_restore_preserves_staged_input(tmp_path, pql):
    staged = tmp_path / "broken.pt"
    staged.write_bytes(b"invalid checkpoint")
    c = settings()
    c.pql.enabled = pql
    c.runtime.output_dir = str(tmp_path / "run")
    c.runtime.warm_start_checkpoint = str(staged)
    with patch.dict(os.environ, {"UAV_STAGED_RESUME_CHECKPOINT": str(staged)}):
        with pytest.raises(Exception):
            train(c)
    assert staged.read_bytes() == b"invalid checkpoint"
