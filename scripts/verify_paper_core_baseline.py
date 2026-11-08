"""Executable source-equation and interface contract for 35T25O UAV MARL.

This 'paper-core' reference is not a claim of byte-for-byte reproduction:
the network/action/belief/CBF are documented UAV adapters; MATD3 author
implementation uses its own MPE actions and replay representation.

No training, W&B access, HF calls or checkpoint mutation. Exits nonzero on
any deviation from the deliberately selected reference contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from uav_network.configuration import load_config
from uav_network.environment.observations import observation_dim, state_dim, observation_version
from uav_network.algorithms.masac import MASAC, soft_bellman_target
from uav_network.algorithms.matd3 import MATD3, td3_target
from uav_network.algorithms.models import LOG_STD_MIN, LOG_STD_MAX


SOURCE = {
    "td3": "https://spinningup.openai.com/en/latest/algorithms/td3.html",
    "sac": "https://spinningup.openai.com/en/latest/algorithms/sac.html",
    "matd3_paper": "https://arxiv.org/abs/1910.01465",
    "matd3_author": "https://github.com/JohannesAck/MATD3implementation",
    "hybrid_sac": "https://arxiv.org/abs/1912.11077",
    "hybrid_masac": "https://proceedings.mlr.press/v270/fu25c.html",
}
SCIENTIFIC = ("world", "sensor", "network", "energy", "reward")
PROFILE = "configs/comparison-paper-core-35t25o-xyz-sensing.yaml"
SOURCE_PROFILE = "configs/comparison-independent-35t25o-h512-xyz-sensing.yaml"
ARCHIVED_TEN = "docs/methods/evidence/wandb_latest_ten_2026-11-08.json"


def require(condition, text):
    if not condition:
        raise ValueError("Paper-core baseline contract: " + text)


def verify_config(profile=PROFILE, comparison=SOURCE_PROFILE, archived=ARCHIVED_TEN):
    c = load_config(profile)
    historical = load_config(comparison)
    curr, old = c.to_dict(), historical.to_dict()
    for key in SCIENTIFIC:
        require(curr[key] == old[key], f"scientific scenario changed from original 35T25O profile: {key}")
    world = c.world
    require((world.num_uavs, world.num_targets, world.num_obstacles, world.max_steps)
            == (6, 35, 25, 1500), "6UAV/35T/25O/1500-step scenario")
    require(world.map_size_m == 1000 and world.motion_dimensions == 3, "3D 1km map")
    require(c.sensor.model == "bayesian" and observation_version(c) == 5,
            "Bayesian/observation v5 protocol")
    require(c.network.power_control == "learned" and c.world.motion_safety == "cbf",
            "hybrid RF-power and CBF environment semantics")
    require(c.runtime.environment_backend == "warp_packet" and not c.pql.enabled,
            "non-PQL original Warp backend")
    require(c.runtime.seed == 44 and c.runtime.episodes == 20000, "seed44/20k target")
    require(c.runtime.num_envs == 4096 and c.runtime.critic_chunk_size == 0,
            "4096-env and no critic chunks")
    require(c.runtime.amp_mode == "off" and not c.runtime.compile_models
            and not c.runtime.vectorize_critics and not c.runtime.fused_optimizers,
            "FP32/eager/no-vmap/no-fused")
    require(c.runtime.deterministic, "deterministic infrastructure execution")
    require(c.runtime.eval_seed == 2000000044 and c.runtime.eval_episodes == 30,
            "held-out final evaluation bank")
    require(c.tracking.periodic_evaluation_seed == 1000000044
            and c.tracking.periodic_evaluation_episodes == 5,
            "periodic validation bank")
    for name in ("masac", "matd3"):
        a = getattr(c, name)
        require(not a.share_actor_parameters and not a.share_critic_parameters,
                f"{name} independent agent actors/critics")
        require(not a.actor_layer_norm, f"{name} excludes experimental hidden LayerNorm")
        require(a.hidden_sizes == [512, 512], f"{name} controlled H512x2")
        require(a.batch_size == 8192 and a.replay_capacity == 1000000
                and a.learning_starts == 131072 and a.updates_per_step == 8,
                f"{name} matched replay/batch/U8")
        require(a.gamma == .99 and a.tau == .005 and a.learning_rate == .0003,
                f"{name} fixed algorithm hyperparameters")
    ac, ad = c.masac, c.matd3
    require(ac.temperature_loss == "alpha" and ac.initial_alpha == 1.0,
            "SAC direct-alpha dual with positive exp(log_alpha) parameterization")
    require(ac.grad_clip_mode == "none" and ac.target_continuous_entropy == -4.0,
            "Hybrid-SAC unclipped gradient and nominal -4 continuous target")
    require(ac.mask_aware_discrete_entropy_target
            and ac.target_discrete_entropy_ratio == .2
            and ac.target_discrete_entropy_final_ratio == .2
            and ac.target_discrete_entropy_anneal_updates == 0,
            "masked-UAV adapter with fixed 0.2 target ratio and no annealing")
    require(ad.initialization == "default" and ad.output_normalization is False,
            "standard Linear initialization, no experimental output renormalization")
    require(ad.grad_clip_mode == "per_parameter" and ad.grad_clip_norm == .5,
            "Ackermann MATD3 author's per-variable clipping")
    require(ad.policy_delay == 2 and ad.target_noise_std == .2
            and ad.target_noise_clip == .5 and ad.exploration_noise_std == .1,
            "TD3 delayed policy and target smoothing/exploration")
    require((LOG_STD_MIN, LOG_STD_MAX) == (-20, 2), "SAC tanh Gaussian log-std bounds")

    archive = json.loads(Path(archived).read_text())
    assert len(archive["runs"]) == 10
    strict_cases = [
        run["id"] for run in archive["runs"]
        if (not run["scientific_config"]["share_actor_parameters"]
            and not run["scientific_config"]["share_critic_parameters"]
            and not run["scientific_config"]["actor_layer_norm"]
            and run["scientific_config"]["updates_per_step"] == 8
            and run["scientific_config"]["hidden_sizes"] == [512, 512])
    ]
    require(not strict_cases, "unexpected identical strict-U8-NoLN run in historical 10-case registry")
    return c, {"source_urls": SOURCE,
               "scientific_scenario_identical_to_existing_35t25o_profile": True,
               "historical_exact_reference_runs": strict_cases,
               "historical_cohort_code_sha": archive["protocol"]["source_git_commit"]}


def verify_numerics():
    reward = torch.tensor([1.0, -2.0, 7.0])
    terminal = torch.tensor([0.0, 1.0, 0.0])
    q1 = torch.tensor([5.0, 2.0, -3.0])
    q2 = torch.tensor([4.0, 3.0, 10.0])
    gamma = .99
    expected_td3 = reward + gamma * (1 - terminal) * torch.minimum(q1, q2)
    torch.testing.assert_close(td3_target(reward, terminal, q1, q2, gamma), expected_td3,
                               atol=0, rtol=0)
    soft_v = torch.tensor([3.5, 1.8, -1.0])
    expected_sac = reward + gamma * (1 - terminal) * soft_v
    torch.testing.assert_close(soft_bellman_target(reward, terminal, soft_v, gamma),
                               expected_sac, atol=0, rtol=0)
    require(terminal[1] == 1, "terminal control")
    return {"td3_clipped_twin_q_and_terminal": True,
            "sac_soft_bellman_and_terminal": True}


def verify_module_structure(c):
    dims = {"observation": observation_dim(c), "centralized_state": state_dim(c),
            "num_uavs": c.world.num_uavs, "categorical_destinations": c.world.num_uavs+1,
            "continuous_controls": c.continuous_dim}
    structures = {}
    for algorithm, cls in (("matd3", MATD3), ("masac", MASAC)):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(44)
            learner = cls(c, dims["observation"], dims["centralized_state"], device="cpu")
            require(len(learner.actors) == c.world.num_uavs
                    and len(learner.critics) == c.world.num_uavs,
                    f"{algorithm} must retain one actor and twin-critic module per UAV")
            require(len(learner.actor_optimizers) == c.world.num_uavs,
                    f"{algorithm} independent actor optimizers")
            require(not any(isinstance(module, nn.LayerNorm)
                            for module in learner.modules()),
                    f"{algorithm} no LayerNorm anywhere")
            require(all(hasattr(critic, "q1") and hasattr(critic, "q2")
                        for critic in learner.critics), f"{algorithm} centralized twin Q")
            first = next(layer for layer in learner.actors[0].modules()
                         if isinstance(layer, nn.Linear))
            require(first.in_features == dims["observation"],
                    f"{algorithm} actor input local obs only; no sharing agent-ID input")
            obs = np.zeros((1, c.world.num_uavs, dims["observation"]), np.float32)
            obs[..., 9] = 1.0
            obs[..., -(c.world.num_uavs+1)] = 1.0  # silent-only legal mask
            action = learner.act_batch(obs, deterministic=True)
            require(action.continuous.shape == (1,c.world.num_uavs,c.continuous_dim),
                    f"{algorithm} continuous action dimensions")
            require(action.destination.shape == (1,c.world.num_uavs),
                    f"{algorithm} discrete action dimensions")
            require(np.all(action.destination == 0) and np.isfinite(action.continuous).all(),
                    f"{algorithm} no invalid destination under forced-silent mask")
            require(np.all(action.continuous[..., -1] == 0),
                    f"{algorithm} silent RF power must be exactly zero")
            structures[algorithm] = {
                "actor_count": len(learner.actors), "centralized_twin_critic_count": len(learner.critics),
                "actor_optimizer_count": len(learner.actor_optimizers),
                "actor_parameters_per_agent": sum(p.numel() for p in learner.actors[0].parameters()),
                "critic_parameters_per_agent": sum(p.numel() for p in learner.critics[0].parameters()),
                "input_scope": "local observation for actor, centralized state+joint actions for critic",
            }
            del learner
    return dims, structures


def report(profile=PROFILE):
    c, provenance = verify_config(profile=profile)
    numeric = verify_numerics()
    dims, structure = verify_module_structure(c)
    serialized = {
        "status": "paper-core baseline source-aligned and contract-verified; training NOT RUN",
        "profile": profile,
        "algorithmic_reference_is_not_exact_author_environment_or_network": True,
        "algorithmic_core": {
            "masac": "stochastic tanh Gaussian + categorical exact finite support expectation; independent actors/Q_i, twin min, separated continuous/discrete temperatures",
            "matd3": "deterministic continuous actor + soft Gumbel categorical adapter; independent actors/twin Q_i, clipped min, smoothing, delayed updates, replay peer actions",
        },
        "necessary_uav_adaptations": [
            "CTDE critic uses global 3D UAV/world state rather than the authors' exact MPE joint-observation features",
            "Actor uses local belief/sensing/mask; no privileged hidden targets",
            "Hybrid continuous XYZ/RF power + masked destination; replay has hard one-hot destinations",
            "RF power gated silent; 3D dynamics, CBF, network, sensor and rewards untouched",
            "MASAC mask-aware discrete entropy target adapts attainable entropy to valid destination support",
        ],
        "controlled_hyperparameters_not_claimed_original_paper_defaults": [
            "H512x512", "B8192", "U8 per 4096-vector collection",
            "replay1M", "20k target completed episodes", "LR3e-4", "tau0.005",
        ],
        "provenance": provenance,
        "numerical_oracles": numeric,
        "shapes": dims,
        "models": structure,
        "not_run": "The 2026-11-07 ten-run W&B cohort does not contain the exact Independent+NoLN+U8 H512 pair; no training is silently started.",
    }
    serialized["contract_sha256"] = hashlib.sha256(
        json.dumps(serialized, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return serialized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--out", default="docs/methods/evidence/paper_core_35t25o_contract_2026-11-08.json")
    args = parser.parse_args()
    data = report(args.profile)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print("PAPER_CORE_VALIDATED",data["models"],flush=True)
    print("CONTRACT",out,"SHA256",data["contract_sha256"],flush=True)


if __name__ == "__main__":
    main()
