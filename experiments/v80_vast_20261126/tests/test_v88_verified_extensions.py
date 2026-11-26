"""V88: numerical replay/IQM and complete parameter checkpoint contracts."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/gpu"))
from episode_metrics_v87 import compute_iqm
from uav_network.algorithms.baselines.canonical_continuous import CanonicalContinuousMATD3, CanonicalContinuousMASAC
from uav_network.algorithms.configuration.loader import load_learner_config

YAML = ROOT / "configs/learners/v85_canonical_matd3_masac_baseline.yaml"

def _learner(algorithm):
    cfg = load_learner_config(YAML)
    cfg.world.num_uavs = 2
    cfg.matd3.hidden_sizes = [16, 16]
    cfg.masac.hidden_sizes = [16, 16]
    return (CanonicalContinuousMATD3 if algorithm == "matd3" else CanonicalContinuousMASAC)(cfg, 518, 1036)

def _batch():
    obs = torch.zeros((4, 2, 518))
    obs[:, :, 9] = 1.
    obs[:, :, 10] = 1.
    return dict(observation=obs, next_observation=obs.clone(), state=obs.flatten(start_dim=1),
                next_state=obs.flatten(start_dim=1).clone(), continuous=torch.zeros(4, 2, 2),
                reward=torch.ones(4), terminated=torch.zeros(4))

def test_fractional_iqm_for_five_samples():
    assert compute_iqm([0., 0., 0., 10., 100.]) == pytest.approx(3.0)
    assert compute_iqm([0., 10., 20., 30., 40., 50., 60., 1000.]) == pytest.approx(35.)

@pytest.mark.parametrize("algorithm", ["matd3", "masac"])
def test_learner_checkpoint_save_load_restores_parameters_optimizers_and_updates(tmp_path, algorithm):
    torch.manual_seed(23)
    l = _learner(algorithm)
    sample = _batch()
    l.update(sample)
    path = tmp_path / f"{algorithm}.pt"
    l.save(path)
    restored = type(l).load(path, device="cpu")
    assert restored.updates == l.updates == 1
    for a, b in zip(l.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(l.actor_opts) == len(restored.actor_opts)
    assert len(l.critic_opts) == len(restored.critic_opts)
    for a, b in zip(l.critic_opts, restored.critic_opts):
        assert a.state_dict()["param_groups"] == b.state_dict()["param_groups"]
        assert len(a.state_dict()["state"]) == len(b.state_dict()["state"])
    saved_rng = torch.random.get_rng_state()
    original_metrics = l.update(sample)
    torch.random.set_rng_state(saved_rng)
    restored_metrics = restored.update(sample)
    assert original_metrics.keys() == restored_metrics.keys()
    for a, b in zip(l.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)

def test_replay_cost_of_peer_features_not_underestimated():
    capacity, num_uavs, added = 65536, 6, 35
    bytes_added = capacity * (4 * num_uavs * added) * 4
    assert bytes_added == 220_200_960
    assert bytes_added / 2**20 == 210.0
