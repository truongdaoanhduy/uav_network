"""Malformed XYZ geometry/sensor state must fail before restoring any array."""
from copy import deepcopy
import pytest
import torch
from test_xyz_warp import cfg
from uav_network.environment.gpu.vector import WarpVectorEnv


@pytest.mark.parametrize("field,value",[
    ("obstacle_heights",-1),("obstacle_heights",0),
    ("visited",-1),("visited",2),("positive",-1),("positive",2),
    ("sensor_tick",-1),("sensor_counts",-1),
])
def test_invalid_xyz_snapshot_domains_are_rejected_atomically(field,value):
    vector=WarpVectorEnv(cfg(),"cpu")
    try:
        vector.reset(seed=44)
        before=vector.snapshot_state()
        invalid=deepcopy(before)
        invalid["arrays"][field].view(-1)[0]=value
        with pytest.raises(ValueError,match="Invalid XYZ"):
            vector.restore_state(invalid)
        after=vector.snapshot_state()
        for name,array in before["arrays"].items():
            assert torch.equal(array,after["arrays"][name]),name
        assert torch.equal(before["seeds"],after["seeds"])
        assert torch.equal(before["action_generator"],after["action_generator"])
    finally:
        vector.close()


def test_zero_obstacle_xyz_snapshot_keeps_valid_padding_on_restore():
    config=cfg()
    config.world.num_obstacles=0
    vector=WarpVectorEnv(config,"cpu")
    try:
        vector.reset(seed=44)
        saved=vector.snapshot_state()
        # Warp keeps one zero padding slot when the scene has no obstacles.
        assert saved["arrays"]["obstacle_heights"].shape==(1,1)
        vector.restore_state(saved)
        actual=vector.snapshot_state()
        for name,array in saved["arrays"].items():
            assert torch.equal(array,actual["arrays"][name]),name
    finally:
        vector.close()
