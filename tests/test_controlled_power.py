"""Controller-owned RF power must not remain a hidden learned action."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from uav_network.configuration import Config, load_config
from uav_network.algorithms.checkpoints import make_learner
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.environment.observations import observation_dim, state_dim


def controlled_config(algorithm="masac", shared=False):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 0
    c.world.max_steps = 4
    c.network.power_control = "link_budget"
    c.network.link_budget_margin_db = 3.0
    c.network.tx_power_min_w = .001
    c.network.tx_power_mapping = "log"
    c.network.power_limited_admission = True
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.algorithm = algorithm
    c.runtime.torch_threads = 1
    a = getattr(c, algorithm)
    a.hidden_sizes, a.batch_size, a.replay_capacity = [8, 8], 4, 16
    a.share_actor_parameters = a.share_critic_parameters = shared
    return c


def batch(c):
    d, s, n = observation_dim(c), state_dim(c), c.world.num_uavs
    obs = torch.zeros((4, n, d))
    obs[..., 9] = 1
    obs[..., -(n+1):] = 1
    return {"observation": obs, "next_observation": obs.clone(),
            "state": torch.zeros((4, s)), "next_state": torch.zeros((4, s)),
            "continuous": torch.zeros((4, n, 3)),
            "destination": torch.full((4, n), n, dtype=torch.long),
            "reward": torch.ones(4), "terminated": torch.zeros(4)}


def test_margin_power_clamps_and_preserves_legacy_default():
    from uav_network.communication.link_budget import link_budget_power_w
    c = controlled_config()
    assert link_budget_power_w(.05, c.network) == pytest.approx(.09976311574844399)
    assert link_budget_power_w(1e-12, c.network) == .001
    assert link_budget_power_w(5, c.network) == .4
    assert Config().network.power_control == "learned"
    resolved = load_config(overrides=["network.power_control=link_budget",
                                      "network.link_budget_margin_db=3"])
    assert resolved.to_dict()["network"]["power_control"] == "link_budget"


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("shared", [False, True])
def test_policy_and_pql_inference_do_not_emit_learned_power(algorithm, shared):
    from uav_network.experiments.pql_workers import inference_copy
    c = controlled_config(algorithm, shared)
    learner = make_learner(c, observation_dim(c), state_dim(c))
    for actor in learner.actors:
        head = actor.mean if algorithm == "masac" else actor.continuous
        with torch.no_grad():
            head.bias[2] = 2.
    obs = batch(c)["observation"]
    for policy in [learner, inference_copy(learner, "cpu")]:
        action = policy.act_batch(obs, deterministic=True, return_tensors=True)
        assert torch.count_nonzero(action.continuous[..., 2]) == 0
        assert torch.isfinite(action.continuous).all()


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
@pytest.mark.parametrize("shared", [False, True])
def test_critic_ignores_unused_replay_power(algorithm, shared):
    torch.manual_seed(8)
    c = controlled_config(algorithm, shared)
    left = make_learner(c, observation_dim(c), state_dim(c))
    right = make_learner(c, observation_dim(c), state_dim(c))
    right.load_state_dict(left.state_dict())
    a, b = batch(c), batch(c)
    a["continuous"][..., 2] = -1
    b["continuous"][..., 2] = 1
    torch.manual_seed(13)
    lm = left.update_critic(a)
    torch.manual_seed(13)
    rm = right.update_critic(b)
    assert lm["critic_loss"] == pytest.approx(rm["critic_loss"], abs=1e-8)
    for l, r in zip(left.critics.parameters(), right.critics.parameters()):
        torch.testing.assert_close(l, r, rtol=0, atol=0)


@pytest.mark.parametrize("shared", [False, True])
def test_masac_entropy_has_no_power_gradient(shared):
    c = controlled_config(shared=shared)
    learner = make_learner(c, observation_dim(c), state_dim(c))
    obs = batch(c)["observation"]
    continuous, logp, _, _ = learner._sample_agents(obs)
    assert torch.count_nonzero(continuous[..., 2]) == 0
    logp.sum().backward()
    for actor in learner.actors:
        assert torch.count_nonzero(actor.mean.weight.grad[2]) == 0
        assert torch.count_nonzero(actor.log_std.weight.grad[2]) == 0
        assert actor.mean.bias.grad[2] == 0
        assert actor.log_std.bias.grad[2] == 0


@pytest.mark.parametrize("power_action", [-1., 1.])
def test_native_radio_uses_controller_power_for_selected_receiver(power_action):
    from uav_network.communication.link_budget import nominal_required_power_w
    c = controlled_config()
    env = SwarmSearchEnv(c)
    try:
        env.enable_communication_trace()
        env.reset(seed=44, options={"positions": [[450, 300], [550, 200]],
                                    "targets": [[450, 350], [900, 900]], "obstacles": []})
        action = Action(np.array([[0., 0., power_action], [0., 0., 0.]]),
                        np.array([2, 0]))
        env.step(action)
        events = env.drain_communication_events()
        assert events
        required = nominal_required_power_w(env.positions[0], env.gcs, env.obstacles,
                                             c.network, c.sensor.altitude_m, True)
        expected = np.clip(required * 10**.3, .001, .4)
        assert events[0]["power_w"] == pytest.approx(expected)
        assert events[0]["destination"] == 2
        assert env.info()["network"]["diagnostics"]["data_below_required_power_attempts"] == 0
    finally:
        env.close()


def test_warp_packets_use_controller_power():
    global selected_power
    wp = pytest.importorskip("warp")
    from uav_network.environment.gpu.state import allocate_state
    from uav_network.communication.gpu.power import selected_power
    c = controlled_config()
    s, p = allocate_state(c, num_envs=1, device="cpu")
    s.positions.assign(np.array([[[450., 300.], [550., 200.]]]))
    result = wp.zeros(2, dtype=wp.float64, device="cpu")
    wp.launch(power_probe, dim=2, inputs=[s, p, result], device="cpu")
    assert result.numpy()[0] > 0
    np.testing.assert_allclose(result.numpy(), result.numpy()[0], rtol=0, atol=0)


# Lazy import keeps the module importable when Warp is not installed.
try:
    import warp as wp
    from uav_network.environment.gpu.state import WorldParams, WorldState

    @wp.kernel
    def power_probe(s: WorldState, p: WorldParams, out: wp.array(dtype=wp.float64)):
        from_value = wp.float32(-1.)
        if wp.tid() == 1:
            from_value = wp.float32(1.)
        out[wp.tid()] = selected_power(s, p, 0, 0, p.n, from_value)
except ImportError:
    pass


@pytest.mark.parametrize("power_action", [-1., 1.])
def test_warp_real_packet_custody_uses_selected_link_budget(power_action):
    wp = pytest.importorskip("warp")
    from uav_network.environment.gpu.state import allocate_state
    from uav_network.environment.gpu.geometry import sense
    from uav_network.communication.gpu.mac import reset_network, packet_step
    from uav_network.communication.gpu.reports import sense_reports, begin_step
    from uav_network.communication.link_budget import nominal_required_power_w
    c = controlled_config()
    s, p = allocate_state(c, num_envs=1, device="cpu", trace=True)
    mask = wp.array([1], dtype=wp.int32, device="cpu")
    wp.launch(reset_network, dim=1, inputs=[s, p, mask], device="cpu")
    s.positions.assign(np.array([[[450.,300.],[550.,200.]]]))
    s.targets.assign(np.array([[[450.,350.],[900.,900.]]]))
    wp.launch(sense, dim=(1,2,2), inputs=[s,p], device="cpu")
    wp.launch(sense_reports, dim=1, inputs=[s,p], device="cpu")
    wp.launch(begin_step, dim=1, inputs=[s,p], device="cpu")
    continuous = wp.array(np.array([[[0.,0.,power_action],[0.,0.,0.]]], dtype=np.float32),
                          dtype=wp.float32, device="cpu")
    destinations = wp.array(np.array([[2,0]],dtype=np.int64), dtype=wp.int64, device="cpu")
    wp.launch(packet_step, dim=1, inputs=[s,p,continuous,destinations], device="cpu")
    assert s.delivered.numpy()[0,0] == 1
    count = int(s.trace_count.numpy()[0])
    complete = s.trace_kind.numpy()[0,:count] == 2
    assert complete.any()
    required = nominal_required_power_w(np.array([450.,300.]), np.array([500.,0.]),
                                        np.empty((0,3)), c.network, 100., True)
    powers = s.trace_report.numpy()[0,:count,4][complete]
    np.testing.assert_allclose(powers,np.clip(required*10**.3,.001,.4),rtol=1e-6)
