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
