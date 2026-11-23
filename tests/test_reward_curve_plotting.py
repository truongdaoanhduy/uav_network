"""Regression tests for plotting functions: read-only aggregation, no fabricated bins."""
import numpy as np

from scripts.plot_training_reward_from_wandb import (
    fixed_episode_blocks, epymarl_rolling_mean, transition_bin_means,
)


def test_fixed_episode_groups_keep_complete_flags_and_partial_tail():
    episodes = np.arange(1, 11)
    blocks = fixed_episode_blocks(episodes, episodes.astype(float), episodes*10, width=4)
    assert len(blocks) == 3
    assert [b['n'] for b in blocks] == [4,4,2]
    assert [b['complete'] for b in blocks] == [True,True,False]
    assert [b['reward'] for b in blocks] == [2.5,6.5,9.5]
    assert [b['episode'] for b in blocks] == [2.5,6.5,9.5]


def test_epymarl_mean_uses_correct_window_x_alignment():
    steps = np.arange(1,6)
    rewards = np.array([1.,3.,7.,13.,21.])
    result = epymarl_rolling_mean(steps,rewards,width=3,stride=1)
    assert np.allclose([r['episode'] for r in result], [2,3,4])
    assert np.allclose([r['reward'] for r in result], [11/3,23/3,41/3])
    assert epymarl_rolling_mean(steps,rewards,width=6) == []


def test_transition_bins_preserve_missing_and_sparse_gaps():
    bins=transition_bin_means([1,2,5,5,5,21], [1,3,7,9,11,20], width=5, min_count=2)
    assert [r['n'] for r in bins] == [2,3,0,0,1]
    assert [r['reward'] for r in bins] == [2,9,None,None,None]


def test_mismatched_window_rejected():
    import pytest
    with pytest.raises(ValueError):
        fixed_episode_blocks([1],[1],[1],width=0)
