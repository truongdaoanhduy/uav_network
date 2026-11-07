"""Execute the five requested UI identities and verify the active models."""
from dataclasses import asdict

import pytest

from uav_network.configuration import load_config

PROFILE = "configs/comparison-independent-35t25o-h512-xyz-sensing.yaml"
STAGE_START = 'if SPEC["stage"] in {"visualize", "archive"}:'
CASES = [
    ("01", True, 512, 8, True),
    ("02", True, 512, 16, True),
    ("03", False, 512, 16, True),
    ("04", False, 512, 16, False),
    ("05", True, 256, 8, True),
]


@pytest.mark.parametrize("case_id,shared,width,updates,layernorm", CASES)
def test_ui_case_launches_requested_recipe_and_edited_seed(case_id, shared, width, updates, layernorm):
    # Catches a correct-looking filename whose actual SPEC selects another recipe.
    from uav_network.cloud.matd3_ui_notebooks import build_matd3_ablation_notebook

    notebook = build_matd3_ablation_notebook(case_id, b"frozen", "pinned-source")
    original = load_config(PROFILE)
    ids = []
    for seed in (44, 45):
        scope = {"NOTEBOOK_REF": "owner/matd3-ablation", "TRAIN_SEED": seed}
        code = notebook.cells[-1].source
        exec(code[:code.index(STAGE_START)], scope)
        spec = scope["SPEC"]
        c = load_config(spec["config"], spec["overrides"])
        assert spec["config"] == PROFILE and spec["commit"] == "pinned-source"
        assert set(spec["runs"]) == {"matd3"}
        assert spec["training_only"] and not spec["auto_continue"]
        assert not spec["warm_start_artifacts"] and not spec["cpu_visualize"]
        assert c.runtime.algorithm == "matd3" and not c.pql.enabled
        assert c.runtime.seed == seed
        assert c.matd3.share_actor_parameters is shared
        assert c.matd3.share_critic_parameters is shared
        assert c.matd3.actor_layer_norm is layernorm
        assert c.matd3.hidden_sizes == [width, width]
        assert c.matd3.updates_per_step == updates
        assert c.matd3.policy_delay == 2
        assert c.matd3.initialization == "default"
        assert not c.matd3.output_normalization
        assert c.runtime.critic_chunk_size == 0
        assert not c.runtime.compile_models and not c.runtime.vectorize_critics
        for name in ("world", "sensor", "network", "reward", "energy", "masac", "pql"):
            assert asdict(getattr(c, name)) == asdict(getattr(original, name))
        baseline, actual = asdict(original.matd3), asdict(c.matd3)
        for key in ("share_actor_parameters", "share_critic_parameters", "actor_layer_norm",
                    "hidden_sizes", "updates_per_step"):
            baseline.pop(key)
            actual.pop(key)
        assert baseline == actual
        mode = "SH" if shared else "IND"
        norm = "ALN" if layernorm else "NOLN"
        run = spec["runs"]["matd3"]
        assert f"-{mode}-" in run["name"] and f"-{norm}-" in run["name"]
        assert f"-H{width}-U{updates}-" in run["name"]
        assert f"-S{seed}-" in run["name"]
        assert run["id"] in run["name"] and c.runtime.output_dir.endswith(run["id"])
        assert f"seed{seed}" in c.tracking.tags
        assert f"hidden{width}" in c.tracking.tags
        assert f"updates{updates}" in c.tracking.tags
        assert ("shared-actors" if shared else "independent-actors") in c.tracking.tags
        assert ("actor-layernorm" if layernorm else "actor-no-layernorm") in c.tracking.tags
        assert "critic-no-layernorm" in c.tracking.tags
        assert f"_H{width}_U{updates}_{norm}_" in notebook.metadata.uav_ablation.filename
        ids.append(run["id"])
    assert ids[0] != ids[1]


@pytest.mark.parametrize("case_id,seed", [("06", 44), ("01", -1), ("01", True)])
def test_unsupported_case_or_seed_is_rejected(case_id, seed):
    from uav_network.cloud.matd3_ui_notebooks import build_matd3_ablation_notebook
    with pytest.raises(ValueError):
        build_matd3_ablation_notebook(case_id, b"frozen", "source", seed=seed)


@pytest.mark.parametrize("case_id,shared,width,updates,layernorm", CASES)
def test_ui_recipe_runs_real_model_and_delayed_updates(case_id, shared, width, updates, layernorm):
    # Catches incorrect sharing/normalization and numerical failures in the emitted recipe.
    import numpy as np
    import torch
    from torch import nn

    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.replay import TensorReplayBuffer
    from uav_network.cloud.matd3_ui_notebooks import build_matd3_ablation_notebook
    from uav_network.environment.gpu.vector import WarpVectorEnv

    torch.set_num_threads(1)
    torch.manual_seed(44)
    nb = build_matd3_ablation_notebook(case_id, b"frozen", "source")
    scope = {"NOTEBOOK_REF": "owner/matd3-cpu-check", "TRAIN_SEED": 44}
    code = nb.cells[-1].source
    exec(code[:code.index(STAGE_START)], scope)
    spec = scope["SPEC"]
    c = load_config(spec["config"], spec["overrides"] + ["runtime.device=cpu", "runtime.num_envs=2"])
    model = MATD3(c, 228, 830)
    assert len(model.actors) == len(model.critics) == (1 if shared else 6)
    for actor in model.actors:
        linear = [m for m in actor.encoder.modules() if isinstance(m, nn.Linear)]
        assert linear[0].in_features == (234 if shared else 228)
        assert [m.out_features for m in linear] == [width, width]
        assert sum(isinstance(m, nn.LayerNorm) for m in actor.modules()) == (2 if layernorm else 0)
    assert not any(isinstance(m, nn.LayerNorm) for m in model.critics.modules())
    assert model.critics[0].q1[0].in_features == 896
    replay = TensorReplayBuffer(8, (6, 228), 830, continuous_dim=4)
    vector = WarpVectorEnv(c, device="cpu")
    try:
        obs, _ = vector.reset(seed=44)
        for _ in range(4):
            before = {key: value.clone() for key, value in obs.items()}
            action = model.act_batch(before["observation"], return_tensors=True)
            obs, reward, done, _, _ = vector.step(
                {"continuous": action.continuous, "destination": action.destination})
            replay.add_batch(before["observation"], before["state"], action, reward,
                             obs["observation"], obs["state"], done.float())
        old_actor = [p.detach().clone() for p in model.actors[0].parameters()]
        old_critic = [p.detach().clone() for p in model.critics[0].parameters()]
        for _ in range(c.matd3.updates_per_step):
            metrics = model.update(replay.sample(8))
            assert np.isfinite(list(metrics.values())).all()
        assert model.updates == updates
        assert model.actor_updates == (4 if updates == 8 else 8)
        for old, module in ((old_actor, model.actors[0]), (old_critic, model.critics[0])):
            assert any(not torch.equal(a, b) for a, b in zip(old, module.parameters()))
        assert all(torch.isfinite(p).all() for p in model.parameters())
    finally:
        vector.close()
