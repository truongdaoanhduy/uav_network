"""Local storage is bounded without sacrificing the last good atomic checkpoint."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from uav_network.algorithms.checkpoints import make_learner
from uav_network.environment.observations import observation_dim, state_dim
from test_async_evaluation import settings


def test_failed_atomic_write_preserves_previous_checkpoint_and_removes_partial(tmp_path):
    c = settings()
    learner = make_learner(c, observation_dim(c), state_dim(c))
    path = tmp_path / "resume_latest.pt"
    learner.save(path)
    before = path.read_bytes()

    def failing_save(payload, temporary):
        Path(temporary).write_bytes(b"partial new checkpoint")
        raise OSError("disk exhausted at original write")

    with patch("uav_network.algorithms.checkpoints.torch.save", side_effect=failing_save):
        with pytest.raises(OSError, match="disk exhausted at original write"):
            learner.save(path)
    assert path.read_bytes() == before
    assert not path.with_suffix(".pt.tmp").exists()


def test_retention_removes_only_old_generated_episode_checkpoints_and_measures_disk(tmp_path):
    from uav_network.experiments.local_checkpoints import prune_episode_checkpoints, disk_metrics
    for i in range(7):
        (tmp_path / f"episode_{i:06d}.pt").write_bytes(bytes(10))
        prune_episode_checkpoints(tmp_path, keep=2)
        assert len(list(tmp_path.glob("episode_*.pt"))) <= 2
    untouched = ["final.pt", "latest.pt", "resume_latest.pt", "resume_latest.pt.tmp",
                 "episode_notes.pt", "input-warmstart.pt"]
    for name in untouched:
        (tmp_path / name).write_bytes(bytes(3))
    (tmp_path / "evaluations").mkdir()
    (tmp_path / "evaluations" / "eval_step_1.pt").write_bytes(bytes(5))
    prune_episode_checkpoints(tmp_path, keep=2)
    assert sorted(p.name for p in tmp_path.glob("episode_0*.pt")) == ["episode_000005.pt", "episode_000006.pt"]
    assert all((tmp_path / name).is_file() for name in untouched)
    assert (tmp_path / "evaluations" / "eval_step_1.pt").is_file()
    metrics = disk_metrics(tmp_path)
    assert metrics["storage/checkpoint_bytes"] == 43
    assert metrics["storage/disk_free_bytes"] > 0
    assert metrics["storage/disk_used_bytes"] >= metrics["storage/checkpoint_bytes"]


@pytest.mark.parametrize("pql", [False, True])
def test_original_checkpoint_error_survives_secondary_disk_full_reporting(tmp_path, capsys, pql):
    from uav_network.experiments.training import train
    c = settings()
    c.pql.enabled, c.pql.mode = pql, "synchronous"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.runtime.episodes, c.runtime.checkpoint_interval = 1, 1
    c.runtime.output_dir = str(tmp_path / "full")
    c.tracking.evaluation_interval_steps = 0
    real_write = Path.write_text
    def write_text(path, *args, **kwargs):
        if path.name == "failure.json":
            raise OSError("secondary report disk full")
        return real_write(path, *args, **kwargs)
    with patch("uav_network.algorithms.checkpoints.torch.save",
               side_effect=OSError("original checkpoint disk full")), patch.object(Path, "write_text", write_text):
        with pytest.raises(OSError, match="original checkpoint disk full"):
            train(c)
    assert "original checkpoint disk full" in capsys.readouterr().out


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_checkpoint_retention_count_is_a_positive_integer(value):
    c = settings()
    c.tracking.local_checkpoints_to_keep = value
    with pytest.raises(ValueError, match="local_checkpoints_to_keep"):
        c.validate()
