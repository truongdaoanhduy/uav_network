"""Run exported scaled notebooks far enough to resolve their real recipe."""
from dataclasses import asdict

import pytest

from uav_network.cloud.ui_notebooks import build_independent_notebook
from uav_network.configuration import load_config
from uav_network.environment.observations import observation_dim, state_dim

STAGE_START = 'if SPEC["stage"] in {"visualize", "archive"}:'
SCALED = "configs/comparison-independent-35t25o-h512-xyz-sensing.yaml"


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_scaled_notebook_selects_new_scene_without_changing_other_science(algorithm):
    # Catches a scaled label whose actual launch still trains20t15o/256.
    nb = build_independent_notebook(algorithm, b"frozen", "source", scaled=True)
    for seed in (44, 45):
        scope = {"NOTEBOOK_REF": "owner/scaled-test", "TRAIN_SEED": seed}
        code = nb.cells[-1].source
        exec(code[:code.index(STAGE_START)], scope)
        spec = scope["SPEC"]
        c = load_config(spec["config"], spec["overrides"])
        old = load_config("configs/comparison-independent-baseline-xyz-sensing.yaml")
        assert spec["config"] == SCALED
        assert c.runtime.seed == seed
        assert (c.world.num_targets, c.world.num_obstacles) == (35, 25)
        assert (observation_dim(c), state_dim(c)) == (228, 830)
        for name in ("sensor", "network", "energy", "reward"):
            assert asdict(getattr(c, name)) == asdict(getattr(old, name))
        left, right = asdict(c.world), asdict(old.world)
        for key in ("num_targets", "num_obstacles"):
            left.pop(key)
            right.pop(key)
        assert left == right
        for name in ("masac", "matd3"):
            options = getattr(c, name)
            assert options.hidden_sizes == [512, 512]
            assert options.updates_per_step == 8
            before, after = asdict(getattr(old, name)), asdict(options)
            before.pop("hidden_sizes")
            after.pop("hidden_sizes")
            assert before == after
        assert c.runtime.critic_chunk_size == 0
        assert c.tracking.group == "uav-independent-35t25o-h512-u8-1500"
        assert "targets35" in c.tracking.tags
        assert "obstacles25" in c.tracking.tags
        assert "hidden512" in c.tracking.tags
        assert "updates8" in c.tracking.tags
        assert "-T35-O25-H512-U8-" in spec["runs"][algorithm]["name"]
        assert f"-S{seed}-" in spec["runs"][algorithm]["name"]
        assert "35t25o-h512-u8" in c.runtime.output_dir
        assert spec["training_only"] and not spec["auto_continue"]
        assert not spec["warm_start_artifacts"] and not c.pql.enabled


@pytest.mark.parametrize("algorithm,actor_calls", [("masac", 2), ("matd3", 1)])
def test_scaled_world_transitions_support_real_512_network_updates(algorithm, actor_calls):
    # Catches shape/mask mismatches between the dense scene, replay and learners.
    import numpy as np
    import torch

    from uav_network.algorithms.masac import MASAC
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.replay import TensorReplayBuffer
    from uav_network.environment.gpu.vector import WarpVectorEnv

    torch.set_num_threads(1)
    torch.manual_seed(44)
    c = load_config(SCALED, [f"runtime.algorithm={algorithm}",
                            "runtime.device=cpu", "runtime.num_envs=2"])
    cls = MASAC if algorithm == "masac" else MATD3
    model = cls(c, observation_dim(c), state_dim(c))
    replay = TensorReplayBuffer(8, (6, 228), 830, continuous_dim=4)
    vector = WarpVectorEnv(c, device="cpu")
    try:
        obs, _ = vector.reset(seed=44)
        assert obs["observation"].shape == (2, 6, 228)
        assert obs["state"].shape == (2, 830)
        for _ in range(4):
            before = {key: value.clone() for key, value in obs.items()}
            action = model.act_batch(before["observation"], return_tensors=True)
            obs, reward, done, _, _ = vector.step({"continuous": action.continuous,
                                                    "destination": action.destination})
            replay.add_batch(before["observation"], before["state"], action, reward,
                             obs["observation"], obs["state"], done.float())
        before_actor = [p.detach().clone() for p in model.actors[0].parameters()]
        before_critic = [p.detach().clone() for p in model.critics[0].parameters()]
        for _ in range(2):
            metrics = model.update(replay.sample(8))
            assert np.isfinite(list(metrics.values())).all()
        assert (model.updates, model.actor_updates) == (2, actor_calls)
        for old, new in ((before_actor, model.actors[0]), (before_critic, model.critics[0])):
            assert any(not torch.equal(a, b) for a, b in zip(old, new.parameters()))
        assert all(torch.isfinite(p).all() for p in model.parameters())
    finally:
        vector.close()
