"""Reject accidental common-setting differences in a paired UAV baseline.

Algorithm objectives stay distinct: MATD3 delay/smoothing and the optional
author output penalty; MASAC stochastic entropy and temperature learning.
"""
from dataclasses import fields
from copy import deepcopy

from ..configuration import OffPolicyConfig


def assert_matched_baseline(config):
    common = {}
    for field in fields(OffPolicyConfig):
        left = getattr(config.masac, field.name)
        right = getattr(config.matd3, field.name)
        if left != right:
            raise ValueError(f"Matched baseline mismatch: {field.name}: MASAC={left!r}, MATD3={right!r}")
        common[field.name] = deepcopy(left)
    if config.matd3.initialization != "default":
        raise ValueError("Matched initialization requires PyTorch default in both algorithms")
    config.validate()
    return {
        "common": common,
        "architecture": {
            "actor_layer_norm": config.masac.actor_layer_norm,
            "critic_layer_norm": False,
            "initialization": "pytorch_default",
            "actor_observation": "decentralized_local_plus_agent_id_if_shared",
            "critic_input": "centralized_state_and_joint_action",
        },
        "algorithm_specific": {
            "matd3": {
                "policy_delay": config.matd3.policy_delay,
                "target_noise_std": config.matd3.target_noise_std,
                "exploration_noise_std": config.matd3.exploration_noise_std,
                "output_normalization": config.matd3.output_normalization,
                "actor_output_regularization": config.matd3.actor_output_regularization,
            },
            "masac": {
                "temperature_loss": config.masac.temperature_loss,
                "initial_alpha": config.masac.initial_alpha,
                "mask_aware_discrete_entropy_target": config.masac.mask_aware_discrete_entropy_target,
                "shared_critic_mode": config.masac.shared_critic_mode,
            },
        },
    }
