"""Numerical source checks for independent critic updates and clipping telemetry."""
import pytest
import torch
from torch import nn

from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.optimization import step_optimizer
from uav_network.configuration import Config
from uav_network.environment.observations import observation_dim, state_dim


def _setup(algorithm, agents=2):
    c = Config()
    c.world.num_uavs = agents
    c.world.num_targets = 2
    c.world.num_obstacles = 1
    c.runtime.vectorize_critics = False
    options = getattr(c, algorithm)
    options.hidden_sizes = [8, 8]
    learner = (MASAC if algorithm == "masac" else MATD3)(
        c, observation_dim(c), state_dim(c))
    obs = torch.zeros(4, agents, observation_dim(c))
    obs[:, :, 9] = 1
    obs[:, :, -(agents + 1):] = 1
    batch = {
        "observation": obs, "next_observation": obs.clone(),
        "state": torch.zeros(4, state_dim(c)),
        "next_state": torch.zeros(4, state_dim(c)),
        "continuous": torch.zeros(4, agents, c.continuous_dim),
        "destination": torch.zeros(4, agents, dtype=torch.long),
        "reward": torch.ones(4), "terminated": torch.ones(4),
    }
    return learner, batch


class ScalarTwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Parameter(torch.tensor(0.0))
        self.q2 = nn.Parameter(torch.tensor(0.0))

    def forward(self, state, joint):
        return self.q1.expand(len(state)), self.q2.expand(len(state))


@pytest.mark.parametrize("agents", [2, 6])
def test_matd3_independent_critics_match_author_batch_mean_and_clip(agents):
    # Averaging independent critic gradients across agents before clipping must
    # fail: the second update crosses the source clipping threshold differently.
    learner, batch = _setup("matd3", agents)
    learner.config.matd3.grad_clip_mode = "per_parameter"
    learner.config.matd3.grad_clip_norm = 0.5
    learner.critics = nn.ModuleList([ScalarTwinCritic() for _ in range(agents)])
    learner.critic_optimizer = torch.optim.Adam(learner.critics.parameters(), lr=3e-4)
    expected = [nn.Parameter(torch.tensor(0.0)) for _ in range(2 * agents)]
    optimizers = [torch.optim.Adam([p], lr=3e-4) for p in expected]
    for reward in (3.0, 0.3):
        batch["reward"].fill_(reward)
        learner.update_critic(batch, collect_metrics=False)
        # Independent author reference: each twin/agent uses a batch mean,
        # then clips this variable's gradient before its own Adam update.
        for parameter, optimizer in zip(expected, optimizers):
            optimizer.zero_grad()
            ((parameter - reward) ** 2).backward()
            torch.nn.utils.clip_grad_norm_([parameter], 0.5)
            optimizer.step()
    for actual, reference in zip(learner.critics.parameters(), expected):
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-10)


def test_per_parameter_norm_reports_the_norm_used_for_clipping():
    # A global norm of sqrt(.4**2+.4**2) must not falsely indicate that
    # either variable crossed the source's individual threshold of .5.
    p, q = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([p, q], lr=1.0)
    measured = step_optimizer(0.4 * p + 0.4 * q, optimizer, [p, q], 0.5,
                              grad_clip_mode="per_parameter")
    torch.testing.assert_close(measured, torch.tensor(0.4))
    torch.testing.assert_close(torch.stack([p, q]), torch.tensor([-0.4, -0.4]))


def test_masac_unclipped_critic_does_not_report_clipping():
    learner, batch = _setup("masac")
    learner.config.masac.grad_clip_mode = "none"
    learner.config.masac.grad_clip_norm = 1e-12
    metrics = learner.update_critic(batch)
    assert metrics["critic_grad_norm"] > learner.config.masac.grad_clip_norm
    assert metrics["critic_gradient_clipped"] == 0


@pytest.mark.parametrize("agents", [2, 6])
def test_masac_independent_critics_match_per_agent_batch_mean(agents):
    """Hybrid-MASAC Eq. 8 defines J_Q(theta_i) independently for each agent."""
    learner, batch = _setup("masac", agents)
    learner.config.masac.grad_clip_mode = "none"
    learner.critics = nn.ModuleList([ScalarTwinCritic() for _ in range(agents)])
    learner.critic_optimizer = torch.optim.SGD(learner.critics.parameters(), lr=0.1)
    expected = [nn.Parameter(torch.tensor(0.0)) for _ in range(2 * agents)]
    optimizers = [torch.optim.SGD([p], lr=0.1) for p in expected]

    batch["reward"].fill_(3.0)
    learner.update_critic(batch, collect_metrics=False)
    for parameter, optimizer in zip(expected, optimizers):
        optimizer.zero_grad()
        (0.5 * (parameter - 3.0) ** 2).backward()
        optimizer.step()
    for actual, reference in zip(learner.critics.parameters(), expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-7)


def test_masac_temperature_update_matches_independent_per_agent_objectives(monkeypatch):
    """Hybrid-MASAC Eq. 10/11 gives each alpha_i its own objective, not 1/N of it."""
    import math
    import types
    import uav_network.algorithms.masac as masac_module

    learner, batch = _setup("masac", agents=2)
    learner.config.masac.grad_clip_mode = "none"
    learner.config.masac.target_continuous_entropy = -4.0
    learner.config.masac.target_discrete_entropy_ratio = 0.2
    learner.config.masac.target_discrete_entropy_final_ratio = 0.2
    learner.config.masac.target_discrete_entropy_anneal_updates = 0
    learner.alpha_optimizer = torch.optim.SGD(
        [learner.log_alpha_continuous, learner.log_alpha_discrete], lr=0.1
    )
    for optimizer in learner.actor_optimizers:
        for group in optimizer.param_groups:
            group["lr"] = 0.0

    def fixed_sample(self, actor, observation, deterministic=False, agent_index=0):
        b = len(observation)
        k = self.destinations
        continuous = torch.zeros(b, self.continuous_dim, device=observation.device)
        logp = torch.full((b,), -2.0, device=observation.device)
        probs = torch.full((b, k), 1.0 / k, device=observation.device)
        log_probs = torch.full((b, k), -math.log(k), device=observation.device)
        return continuous, logp, probs, log_probs

    learner._sample = types.MethodType(fixed_sample, learner)
    monkeypatch.setattr(
        masac_module,
        "step_actor_optimizers",
        lambda losses, optimizers, groups, *args, **kwargs: [torch.tensor(0.0)] * len(losses),
    )

    before_c = learner.log_alpha_continuous.detach().clone()
    before_d = learner.log_alpha_discrete.detach().clone()
    learner.update_actor(batch, collect_metrics=False)

    k = learner.destinations
    target_c = learner.config.masac.target_continuous_entropy / learner.continuous_dim * (
        learner.continuous_dim - 1.0 / k
    )
    entropy_c = 2.0
    entropy_d = math.log(k)
    target_d = 0.2 * math.log(k)
    expected_c = before_c - 0.1 * (entropy_c - target_c)
    expected_d = before_d - 0.1 * (entropy_d - target_d)
    torch.testing.assert_close(learner.log_alpha_continuous, expected_c, rtol=0, atol=1e-6)
    torch.testing.assert_close(learner.log_alpha_discrete, expected_d, rtol=0, atol=1e-6)


def test_independent_masac_targets_use_each_agents_own_entropy(monkeypatch):
    import math

    learner, batch = _setup("masac", agents=2)
    learner.config.masac.grad_clip_mode = "none"
    learner.critics = nn.ModuleList([ScalarTwinCritic(), ScalarTwinCritic()])
    learner.target_critics = nn.ModuleList([ScalarTwinCritic(), ScalarTwinCritic()])
    learner.critic_optimizer = torch.optim.SGD(learner.critics.parameters(), lr=0.1)
    with torch.no_grad():
        learner.log_alpha_continuous.copy_(torch.tensor([2.0, 3.0]).log())
        learner.log_alpha_discrete.copy_(torch.tensor([5.0, 7.0]).log())
    batch["terminated"].zero_()
    count = len(batch["reward"])
    choices = learner.destinations

    def samples(actor, observations, deterministic=False, agent_index=0):
        index = 0 if actor is learner.actors[0] else 1
        return (
            torch.zeros(count, learner.continuous_dim),
            torch.full((count,), -float(index + 1)),
            torch.full((count, choices), 1.0 / choices),
            torch.full((count, choices), -math.log(choices)),
        )

    monkeypatch.setattr(learner, "_sample", samples)
    learner.update_critic(batch, collect_metrics=False)
    # Direct per-agent source target, with zero next Q and a batch mean per critic.
    soft_values = torch.tensor([2 + 5 * math.log(choices), 6 + 7 * math.log(choices)])
    targets = 1.0 + learner.config.masac.gamma * soft_values
    for index, critic in enumerate(learner.critics):
        torch.testing.assert_close(critic.q1, 0.1 * targets[index], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(critic.q2, 0.1 * targets[index], rtol=1e-6, atol=1e-7)
