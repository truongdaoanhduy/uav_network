import numpy as np
import torch

from uav_network.environment.env import Action
from uav_network.experiments.power_calibration import calibration_summary, rollout_from_snapshot


class FakeVector:
    def __init__(self):
        self.restores = 0

    def restore_state(self, snapshot):
        self.restores += 1
        return {"observation": torch.as_tensor(snapshot["observation"])[None]}


class FakeEnv:
    def __init__(self):
        self.vector = FakeVector()
        self.steps = 0

    def _refresh(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1
        power = float(np.asarray(action.continuous)[0, -1])
        reward = 2.0 if power > 0 else -1.0
        done = self.steps >= 1
        info = {
            "delivery_rate_percent": 10.0 if power > 0 else 0.0,
            "network": {"diagnostics": {"data_phy_successes": 2 if power > 0 else 0}},
        }
        return np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), reward, done, False, info


class FakeLearner:
    config = type("Config", (), {"masac": type("MASAC", (), {"gamma": 0.99})()})()

    def act(self, observation, deterministic=True):
        return Action(np.zeros((1, 3), dtype=np.float32), np.zeros(1, dtype=np.int64))


def test_rollout_restores_exact_snapshot_for_each_counterfactual_branch():
    env = FakeEnv()
    learner = FakeLearner()
    snapshot = {"observation": np.zeros((1, 3), dtype=np.float32)}
    low = Action(np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32), np.asarray([1]))
    high = Action(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), np.asarray([1]))
    low_result = rollout_from_snapshot(env, learner, snapshot, low, horizon=4)
    high_result = rollout_from_snapshot(env, learner, snapshot, high, horizon=4)
    assert env.vector.restores == 2
    assert low_result["return"] == -1.0
    assert high_result["return"] == 2.0
    assert high_result["delivery_rate_percent"] - low_result["delivery_rate_percent"] == 10.0
    assert high_result["phy_successes"] - low_result["phy_successes"] == 2


def test_calibration_summary_reports_ordering_agreement_and_deltas():
    rows = [
        {"q_low": 2.0, "q_high": 5.0, "low": {"return": 1.0, "discounted_return": 1.0, "delivery_rate_percent": 0.0, "phy_successes": 0},
         "high": {"return": 4.0, "discounted_return": 3.5, "delivery_rate_percent": 10.0, "phy_successes": 2}},
        {"q_low": 8.0, "q_high": 3.0, "low": {"return": 2.0, "discounted_return": 2.0, "delivery_rate_percent": 5.0, "phy_successes": 1},
         "high": {"return": 6.0, "discounted_return": 5.0, "delivery_rate_percent": 15.0, "phy_successes": 3}},
    ]
    result = calibration_summary(rows)
    assert result["snapshots"] == 2
    assert result["critic_q_high_minus_low_mean"] == -1.0
    assert result["environment_return_high_minus_low_mean"] == 3.5
    assert result["environment_discounted_return_high_minus_low_mean"] == 2.75
    assert result["delivery_high_minus_low_mean"] == 10.0
    assert result["phy_success_high_minus_low_mean"] == 2.0
    assert result["ordering_agreement_fraction"] == 0.5
