"""Four-action-mode audit must faithfully call the existing MASAC Actor.

No training, network access or external checkpoint needed.
"""
import numpy as np
import pytest
import torch

from scripts.diagnose_masac_policy_modes import MODES, policy_action
from uav_network.configuration import Config
from uav_network.algorithms.masac import MASAC
from uav_network.environment.observations import observation_dim, state_dim


@pytest.mark.parametrize("shared", [True, False])
def test_mean_argmax_matches_real_deterministic_masac(shared):
    torch.manual_seed(44)
    c=Config()
    c.world.num_uavs=2
    c.masac.hidden_sizes=[16,16]
    c.masac.share_actor_parameters=shared
    c.masac.share_critic_parameters=shared
    model=MASAC(c,observation_dim(c),state_dim(c),device="cpu")
    obs=torch.randn(2,observation_dim(c)).numpy().astype("float32")
    obs[:,9]=1.0
    obs[:,-3:]=np.array([[1,1,0],[1,0,1]],dtype=np.float32)
    actual,probs=policy_action(model,obs,True,True)
    official=model.act(obs,deterministic=True)
    np.testing.assert_array_equal(actual.continuous,official.continuous)
    np.testing.assert_array_equal(actual.destination,official.destination)
    np.testing.assert_allclose(probs.sum(-1),1.0,rtol=0,atol=2e-7)
    assert np.isin(actual.destination,[0,1,2]).all()


@pytest.mark.parametrize("mode", MODES,ids=[m[0] for m in MODES])
def test_all_modes_obey_mask_and_silent_power(mode):
    torch.manual_seed(123)
    c=Config()
    c.world.num_uavs=2
    c.masac.hidden_sizes=[16,16]
    c.masac.share_actor_parameters=True
    c.masac.share_critic_parameters=True
    model=MASAC(c,observation_dim(c),state_dim(c),device="cpu")
    obs=np.zeros((2,observation_dim(c)),dtype=np.float32)
    obs[:,9]=1.0
    obs[:,-3:]=np.array([[1,0,0],[1,0,1]])
    for _ in range(5):
        action,_=policy_action(model,obs,mode[1],mode[2])
        assert action.destination[0] == 0
        assert action.destination[1] in (0,2)
        assert action.continuous[0,-1]==0
        assert np.isfinite(action.continuous).all()


def test_factorial_is_complete():
    assert len(MODES)==4
    assert len({(c,d) for _,c,d in MODES})==4
