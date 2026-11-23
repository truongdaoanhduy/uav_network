"""Executable 35T25O source-core contract and negative drift checks.

Never relabel Shared/LayerNorm W&B runs as original per-agent MATD3.
"""
from pathlib import Path

import pytest
import torch

from scripts.verify_paper_core_baseline import (
    PROFILE, SOURCE_PROFILE, ARCHIVED_TEN,
    verify_config, verify_numerics, verify_module_structure,
)
from uav_network.configuration import load_config


def test_paper_core_matches_existing_scientific_35t25o_setup():
    c, recorded = verify_config()
    assert recorded["scientific_scenario_identical_to_existing_35t25o_profile"]
    assert recorded["historical_exact_reference_runs"] == []
    original = load_config(SOURCE_PROFILE)
    for group in ("world", "sensor", "network", "energy", "reward"):
        assert c.to_dict()[group] == original.to_dict()[group]
    assert c.runtime.seed == 44 and c.world.num_targets == 35
    assert c.world.num_obstacles == 25
    assert not c.pql.enabled
    assert not c.matd3.share_actor_parameters and not c.matd3.share_critic_parameters
    assert not c.masac.share_actor_parameters and not c.masac.share_critic_parameters


def test_source_core_numerical_bellman_oracles():
    assert verify_numerics() == {
        "td3_clipped_twin_q_and_terminal": True,
        "sac_soft_bellman_and_terminal": True,
    }


def test_source_core_resolves_actual_six_agent_networks_and_mask_safety():
    c=load_config(PROFILE)
    shapes, models=verify_module_structure(c)
    assert shapes["observation"] == 228
    assert shapes["centralized_state"] == 830
    assert shapes["categorical_destinations"] == 7
    assert shapes["continuous_controls"] == 4
    for spec in models.values():
        assert spec["actor_count"] == 6
        assert spec["centralized_twin_critic_count"] == 6
        assert spec["actor_optimizer_count"] == 6
        assert spec["actor_parameters_per_agent"] > 0
        assert spec["critic_parameters_per_agent"] > 0


@pytest.mark.parametrize("group", ["matd3", "masac"])
def test_reject_silent_switch_to_shared_and_layernorm(tmp_path, group):
    import yaml
    base=yaml.safe_load(Path(PROFILE).read_text())
    copy=tmp_path / "changed.yaml"

    changed=yaml.safe_load(Path(PROFILE).read_text())
    changed[group]["share_actor_parameters"]=True
    changed[group]["share_critic_parameters"]=True
    copy.write_text(yaml.safe_dump(changed,sort_keys=False))
    with pytest.raises(ValueError,match="independent agent actors/critics"):
        verify_config(profile=str(copy))

    changed=yaml.safe_load(Path(PROFILE).read_text())
    changed[group]["actor_layer_norm"]=True
    copy.write_text(yaml.safe_dump(changed,sort_keys=False))
    with pytest.raises(ValueError,match="excludes experimental hidden LayerNorm"):
        verify_config(profile=str(copy))


def test_reject_matd3_epsilon_or_late_update_drift(tmp_path):
    original=Path(PROFILE).read_text()
    assert "  updates_per_step: 8" in original
    altered=original.replace("  updates_per_step: 8","  updates_per_step: 16")
    new=tmp_path / "changed.yaml"
    new.write_text(altered)
    with pytest.raises(ValueError,match="matched replay/batch/U8"):
        verify_config(profile=str(new))


def test_mask_aware_entropy_is_declared_uav_adapter():
    c=load_config(PROFILE)
    assert c.masac.mask_aware_discrete_entropy_target is True
    assert c.masac.temperature_loss == "alpha"
    assert c.masac.target_discrete_entropy_anneal_updates == 0
    assert c.matd3.initialization == "default"
    assert c.matd3.grad_clip_norm == .5
    assert c.matd3.grad_clip_mode == "per_parameter"
    assert c.runtime.critic_chunk_size == 0
    assert c.runtime.amp_mode == "off"


def test_original_10_run_evidence_is_immutable_and_not_renamed():
    import json
    evidence=json.loads(Path(ARCHIVED_TEN).read_text())
    assert len(evidence["runs"])==10
    for r in evidence["runs"]:
        assert not (
            r["scientific_config"]["share_actor_parameters"] is False
            and r["scientific_config"]["share_critic_parameters"] is False
            and r["scientific_config"]["actor_layer_norm"] is False
            and r["scientific_config"]["updates_per_step"]==8
            and r["scientific_config"]["hidden_sizes"]==[512,512])


@pytest.mark.parametrize("algorithm,actor_updates", [("masac", 2), ("matd3", 1)])
def test_real_paper_core_warp_replay_update_and_decentralized_actor(algorithm, actor_updates):
    """Complete tiny CPU data path with the *actual* 35T25O/512x512 profile."""
    import numpy as np
    from uav_network.algorithms.masac import MASAC
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.replay import TensorReplayBuffer
    from uav_network.environment.gpu.vector import WarpVectorEnv
    from uav_network.environment.observations import observation_dim, state_dim

    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        c = load_config(PROFILE, [f"runtime.algorithm={algorithm}",
                                  "runtime.device=cpu", "runtime.num_envs=2"])
        cls = MASAC if algorithm == "masac" else MATD3
        model = cls(c, observation_dim(c), state_dim(c), device="cpu")
        replay = TensorReplayBuffer(8, (6, 228), 830, continuous_dim=4)
        env = WarpVectorEnv(c, device="cpu")
        try:
            obs, _ = env.reset(seed=44)
            assert obs["observation"].shape == (2, 6, 228)
            assert obs["state"].shape == (2, 830)
            for _ in range(4):
                before = {k: v.detach().clone() for k, v in obs.items()}
                action = model.act_batch(before["observation"],return_tensors=True)
                obs, reward, done, _, _ = env.step({
                    "continuous": action.continuous,
                    "destination": action.destination,
                })
                replay.add_batch(before["observation"], before["state"],
                                 action,reward,obs["observation"],
                                 obs["state"],done.float())
            first_actor = [x.detach().clone() for x in model.actors[0].parameters()]
            first_critic = [x.detach().clone() for x in model.critics[0].parameters()]
            for _ in range(2):
                result = model.update(replay.sample(8))
                assert all(np.isfinite(value) for value in result.values())
            assert model.updates == 2 and model.actor_updates == actor_updates
            assert any(not torch.equal(x,y) for x,y in
                       zip(first_actor, model.actors[0].parameters()))
            assert any(not torch.equal(x,y) for x,y in
                       zip(first_critic, model.critics[0].parameters()))
            assert all(torch.isfinite(v).all() for v in model.parameters())
        finally:
            env.close()
