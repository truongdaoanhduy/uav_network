"""Measurement must preserve sample RNG, resume metadata and unknown legacy rows."""
from copy import deepcopy

import torch

from uav_network.algorithms.replay import TensorReplayBuffer
from uav_network.experiments.pql_replay import MeasuredReplay
from uav_network.experiments.fixed_checkpoints import replay_state, restore_replay
from test_split_updates import small_learner, replay_batch
from test_pql_system import transition


def test_measured_replay_has_identical_sample_draws_and_base_resume_ages():
    learner = small_learner()
    batch = transition(replay_batch(learner))
    plain = TensorReplayBuffer(16, batch[0].shape[1:], learner.state_dim, seed=44)
    measured = MeasuredReplay(16, batch[0].shape[1:], learner.state_dim, seed=44)
    for _ in range(2):
        plain.add_batch(*batch)
        measured.add_batch(*batch)
    for _ in range(2):
        a, b = plain.sample(4), measured.sample(4)
        for key in a:
            assert torch.equal(a[key], b[key])
    saved = replay_state(measured)
    assert "measurement" in saved
    restored = MeasuredReplay(16, batch[0].shape[1:], learner.state_dim, seed=44)
    restore_replay(restored, saved)
    assert restored.stats() == measured.stats()
    for key, value in measured.sample(4).items():
        # Compare the same one subsequent draw from both.
        if key == "observation":
            following = restored.sample(4)
        assert torch.equal(value, following[key])
    assert restored.stats() == measured.stats()
    assert measured.stats()["replay/policy_age_available"] == 0


def test_legacy_resume_never_marks_old_replay_as_born_now_or_unsampled():
    learner = small_learner()
    batch = transition(replay_batch(learner))
    plain = TensorReplayBuffer(4, batch[0].shape[1:], learner.state_dim, seed=44)
    plain.add_batch(*batch)
    measured = MeasuredReplay(4, batch[0].shape[1:], learner.state_dim, seed=44)
    restore_replay(measured, replay_state(plain))
    measured.sample(4)
    stats = measured.stats()
    assert stats["replay/measurement_unknown_rows"] == 4
    assert stats["replay/sample_age_measured_uses"] == 0
    assert stats["replay/sample_age_mean_collections"] is None
    measured.add_batch(*batch)
    stats = measured.stats()
    assert stats["replay/measurement_unknown_rows"] == 0
    assert stats["replay/evicted_unsampled_rows"] == 0
    assert stats["replay/eviction_measured_rows"] == 0
    assert stats["replay/evicted_unsampled_fraction"] is None
