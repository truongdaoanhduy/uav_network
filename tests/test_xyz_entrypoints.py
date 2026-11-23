"""Ancillary entrypoints must not reinterpret XYZ az as RF power."""
import numpy as np
import pytest
from test_xyz_sensing import xyz_config,world
from uav_network.environment.env import Action
from uav_network.environment.observations import observation_dim
from uav_network.experiments.relay_audit import communication_control
from uav_network.experiments.feasibility import run_feasibility,run_explicit_feasibility,oracle_action


@pytest.mark.parametrize("mode",["learned","direct_only","fixed_power"])
def test_xyz_communication_audit_preserves_all_motion_slots(mode):
    c=xyz_config()
    action=Action(np.array([[.1,.2,.3,-.4]],np.float32),np.array([1]))
    before=action.continuous.copy()
    obs=np.zeros((1,observation_dim(c)),np.float32);obs[:,-2:]=1
    after=communication_control(action,obs,c,mode)
    np.testing.assert_array_equal(after.continuous[:,:3],before[:,:3])
    if mode=="fixed_power":assert after.continuous[0,-1]==1
    np.testing.assert_array_equal(action.continuous,before)


@pytest.mark.parametrize("entry",["seeds","explicit","action"])
def test_old_xy_feasibility_oracle_rejects_xyz_explicitly(entry):
    c=xyz_config()
    with pytest.raises(NotImplementedError,match="XY-only"):
        if entry=="seeds":run_feasibility(c,[44])
        elif entry=="explicit":run_explicit_feasibility(c,44,[[500,500,100]],[[950,950]],[])
        else:oracle_action(world(c,target=(950,950)))
