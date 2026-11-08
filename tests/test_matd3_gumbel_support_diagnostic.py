"""Offline audits for trained Actor-only MATD3 Gumbel/action-support diagnosis."""
import numpy as np
import torch

from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.algorithms.matd3 import MATD3
from scripts.diagnose_matd3_gumbel_support import one_sample, summarize_metrics


def test_only_silent_valid_gumbel_is_exactly_hard_one_hot():
    c = Config()
    c.world.num_uavs = 2
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.runtime.algorithm = "matd3"
    c.matd3.hidden_sizes = [16, 16]
    c.matd3.share_actor_parameters = True
    c.matd3.share_critic_parameters = True
    learner = MATD3(c, observation_dim(c), state_dim(c))
    obs = np.zeros((2, observation_dim(c)), dtype=np.float32)
    obs[:, 9] = 1.0
    obs[:, -(c.world.num_uavs + 1)] = 1.0
    measured = one_sample(learner, obs, 44)
    np.testing.assert_array_equal(measured["mask_count"], [1, 1])
    np.testing.assert_allclose(measured["soft_hard_l1"], 0.0)
    np.testing.assert_allclose(measured["soft_entropy"], 0.0, atol=1e-6)


def test_eligible_metrics_are_explicitly_conditional():
    sample = {
        "mask_count": np.array([1, 2, 2]),
        "policy_max_prob": np.array([1.0, .8, .7]),
        "soft_hard_l1": np.array([0.0, .2, .4]),
        "soft_hard_l2": np.array([0.0, .1, .2]),
        "soft_selected_mass": np.array([1, .9, .8]),
        "soft_entropy": np.array([0, .3, .5]),
    }
    stats = summarize_metrics([sample])
    assert stats["total_agent_states"] == 3
    assert stats["eligible_agent_states"] == 2
    assert stats["forced_silent_fraction"] == 1 / 3
    assert np.isclose(stats["soft_hard_l1"]["eligible_mean"], .3)
    assert np.isclose(stats["soft_hard_l1"]["all_mean"], .2)
    assert stats["unique_valid_destination_counts"] == {"1": 1, "2": 2}


def test_no_alternative_destination_is_not_misread_as_policy_accuracy():
    a = {"mask_count": np.array([1, 1]), "policy_max_prob": np.array([1., 1.]),
         "soft_hard_l1": np.array([0., 0.]), "soft_hard_l2": np.array([0., 0.]),
         "soft_selected_mass": np.array([1., 1.]), "soft_entropy": np.array([0., 0.])}
    stats = summarize_metrics([a])
    assert stats["eligible_agent_states"] == 0
    assert stats["soft_hard_l1"]["eligible_mean"] is None
