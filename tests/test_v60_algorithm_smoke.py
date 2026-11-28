"""Standalone actor and update tests, not dependent on old UAV environment."""
from __future__ import annotations
from tempfile import TemporaryDirectory
from pathlib import Path
import math
import numpy as np
import pytest
import torch

from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.configuration import LearnerConfig


def _config(algorithm):
    cfg=LearnerConfig()
    cfg.runtime.algorithm=algorithm
    cfg.matd3.hidden_sizes=[32,32]
    cfg.masac.hidden_sizes=[32,32]
    cfg.matd3.batch_size=4
    cfg.masac.batch_size=4
    return cfg.validate()


def _batch():
    rng=np.random.default_rng(44)
    n,b,o,d=6,4,21,3
    obs=rng.uniform(-0.2,0.2,(b,n,o)).astype(np.float32)
    obs[:,:,9]=1.0
    obs[:,:,-7:]=1.0
    return {
       "observation":obs,"next_observation":obs.copy(),
       "state":rng.normal(size=(b,n*o)).astype(np.float32),
       "next_state":rng.normal(size=(b,n*o)).astype(np.float32),
       "continuous":rng.uniform(-1,1,(b,n,d)).astype(np.float32),
       "destination":rng.integers(0,n+1,(b,n)).astype(np.int64),
       "reward":rng.normal(size=(b,)).astype(np.float32),
       "terminated":np.array([0,1,0,0],np.float32)
    }


@pytest.mark.parametrize("cls,kind",[(MATD3,"matd3"),(MASAC,"masac")])
def test_standalone_algorithm_action_and_update(cls,kind):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    cfg=_config(kind)
    model=cls(cfg,21,126)
    batch=_batch()
    act=model.act_batch(batch["observation"],deterministic=True)
    assert act.continuous.shape==(4,6,3)
    assert act.destination.shape==(4,6)
    assert np.isfinite(act.continuous).all()
    report=model.update(batch)
    assert isinstance(report,dict)
    for key,v in report.items():
        if isinstance(v,(int,float,np.number)):
            assert math.isfinite(v), (key,v)


@pytest.mark.parametrize("cls,kind",[(MATD3,"matd3"),(MASAC,"masac")])
def test_checkpoint_is_new_scenario_only(cls,kind):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    cfg=_config(kind)
    model=cls(cfg,21,126)
    obs=_batch()["observation"]
    before=model.act_batch(obs,deterministic=True)
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/"learner.pt"
        model.save(path)
        restored=cls.load(path)
        after=restored.act_batch(obs,deterministic=True)
        np.testing.assert_array_equal(before.continuous,after.continuous)
        np.testing.assert_array_equal(before.destination,after.destination)
        from uav_network.algorithms.checkpoints import load_policy
        other="masac" if kind=="matd3" else "matd3"
        with pytest.raises(ValueError,match="algorithm"):
            load_policy(path,expected_algorithm=other)


def test_action_shape_preserves_fixed_power_head():
    config=_config("matd3")
    assert config.motion_dim==2 and config.continuous_dim==3
    assert config.network.power_control=="fixed"
    assert config.to_dict()["scenario_id"]=="zhang-xue-7radio-v60"
