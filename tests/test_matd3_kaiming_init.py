"""MATD3 He/Kaiming normal initialization-only ablation tests."""
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from uav_network.algorithms.matd3 import MATD3
from uav_network.configuration import load_config


def config(mode="kaiming"):
    c = load_config(
        "configs/comparison-reference-masac-matd3-xyz-sensing.yaml",
        ["runtime.algorithm=matd3", "runtime.critic_chunk_size=0",
         f"matd3.initialization={mode}"],
    )
    c.matd3.hidden_sizes = [16, 16]  # CPU-only test, not the production profile
    return c


def test_config_rejects_unknown_initializer():
    with pytest.raises(ValueError, match="matd3.initialization"):
        config("not-an-init")
    for name in ("default", "orthogonal", "kaiming"):
        assert config(name).matd3.initialization == name


def test_kaiming_calls_cover_all_hidden_and_output_linear_layers():
    c = config()
    original = nn.init.kaiming_normal_
    called = []

    def intercept(tensor, *args, **kwargs):
        called.append((tensor, kwargs.copy()))
        return original(tensor, *args, **kwargs)

    torch.manual_seed(44)
    with patch.object(nn.init, "kaiming_normal_", side_effect=intercept):
        learner = MATD3(c, 198, 565)
    assert len(learner.actors) == len(learner.critics) == 1

    actor = learner.actors[0]
    encoder = [x for x in actor.encoder.modules() if isinstance(x, nn.Linear)]
    heads = [actor.continuous, actor.destination]
    expected = [
        *((layer, "relu") for layer in encoder),
        *((layer, "linear") for layer in heads),
    ]
    for twin in (learner.critics[0].q1, learner.critics[0].q2):
        linear = [x for x in twin.modules() if isinstance(x, nn.Linear)]
        expected.extend((layer, "relu") for layer in linear[:-1])
        expected.append((linear[-1], "linear"))
    assert len(called) == len(expected)
    for (tensor, kwargs), (layer, nonlinearity) in zip(called, expected):
        assert tensor is layer.weight
        assert kwargs == {"mode": "fan_in", "nonlinearity": nonlinearity}
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))
        assert torch.isfinite(layer.weight).all()
        assert layer.weight.abs().sum() > 0
    for layer in actor.encoder.modules():
        if isinstance(layer, nn.LayerNorm):
            torch.testing.assert_close(layer.weight, torch.ones_like(layer.weight))
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))

    for target, source in ((learner.target_actors, learner.actors),
                           (learner.target_critics, learner.critics)):
        for p, q in zip(target.parameters(), source.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert not any(p.requires_grad for p in target.parameters())


def test_rng_and_unchanged_algorithm_config():
    torch.manual_seed(44)
    baseline = MATD3(config("default"), 198, 565)
    baseline_state = torch.get_rng_state().clone()
    torch.manual_seed(44)
    he = MATD3(config("kaiming"), 198, 565)
    assert torch.equal(baseline_state, torch.get_rng_state())
    assert not torch.equal(baseline.actors[0].continuous.weight,
                           he.actors[0].continuous.weight)
    for attr in ("batch_size", "gamma", "tau", "policy_delay", "hidden_sizes",
                 "actor_layer_norm", "share_actor_parameters", "share_critic_parameters",
                 "output_normalization", "learning_rate", "target_noise_std",
                 "target_noise_clip", "exploration_noise_std"):
        assert getattr(baseline.config.matd3, attr) == getattr(he.config.matd3, attr)


def test_kaiming_actor_forward_and_update_critic_are_finite():
    c = config()
    torch.manual_seed(44)
    learner = MATD3(c, 198, 565)
    batch_size, num_uavs, obs_dim, state_dim = 4, c.world.num_uavs, 198, 565
    rng = np.random.default_rng(44)
    observations = rng.normal(size=(batch_size, num_uavs, obs_dim)).astype("float32")
    observations[:, :, -7:] = 1
    joint = learner.act_batch(observations, deterministic=True)
    assert joint.continuous.shape == (batch_size, num_uavs, c.continuous_dim)
    assert np.isfinite(joint.continuous).all()
    assert np.abs(joint.continuous).max() <= 1
    batch = dict(
        observation=observations,
        next_observation=observations.copy(),
        state=rng.normal(size=(batch_size, state_dim)).astype("float32"),
        next_state=rng.normal(size=(batch_size, state_dim)).astype("float32"),
        continuous=rng.uniform(-1, 1, size=(batch_size, num_uavs, c.continuous_dim)).astype("float32"),
        destination=rng.integers(0, c.world.num_uavs + 1, (batch_size, num_uavs)),
        reward=rng.normal(size=batch_size).astype("float32"),
        terminated=np.zeros(batch_size, dtype="float32"),
    )
    # One critic update and one actor update, without changing the reference losses.
    q = learner.update_critic(batch)
    a = learner.update_actor(batch)
    assert all(np.isfinite(float(v)) for v in (*q.values(), *a.values()) if not isinstance(v, dict))


@pytest.mark.parametrize("mode", ["default", "orthogonal", "kaiming"])
def test_generated_notebook_run_name_matches_resolved_initialization(monkeypatch, mode):
    import nbformat
    from scripts.build_matd3_init_ui_notebooks import build, HEADER, START
    from uav_network.cloud import kernel

    monkeypatch.setattr(kernel, "build_source_archive", lambda root: b"previous archive")
    spec = {
        "stage": "train", "training_only": True, "requested_episodes": 20000,
        "config": "configs/comparison-reference-masac-matd3-xyz-sensing.yaml",
        "checkpoint_repo": "duy95/uav-search-checkpoints",
        "entity": "test-entity", "project": "test-project",
    }
    code = kernel.kernel_script(spec).replace(START, HEADER + "\n" + START, 1)
    template = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("Template"),
        nbformat.v4.new_code_cell('NOTEBOOK_REF = "owner/template"'),
        nbformat.v4.new_code_cell(code),
    ])
    generated = build(mode, b"new archive", "source-pin", template)
    launch = "".join(generated["cells"][2]["source"])
    scope = {"NOTEBOOK_REF": "owner/test"}
    exec(compile(launch[:launch.index(START)], "<notebook header>", "exec"), scope)
    resolved = scope["SPEC"]
    assert f"matd3.initialization={mode}" in resolved["overrides"]
    assert f"-{mode.upper()}-S44-" in resolved["runs"]["matd3"]["name"]
