"""Requested two-substep mobility-phy synchronization must actually run."""
import numpy as np
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.environment.configuration import make_v62_config


def test_midpoint_channel_update_uses_midpoint_coordinates():
    env=ZhangXue7NodePacketEnv(make_v62_config(),seed=88)
    env.reset(seed=88)
    starting=env.positions.copy()
    original=env.net.run_until_us
    seen=[]
    def collect(t_us):
        seen.append((t_us,env.positions.copy()))
        return original(t_us)
    env.net.run_until_us=collect
    cmd=np.zeros((6,2));cmd[0,0]=5.0
    env.step(cmd,np.zeros(6,dtype=np.int64),tx_power_w=np.full(6,0.5))
    assert len(seen)==2
    assert seen[0][0] < seen[1][0]
    np.testing.assert_allclose(seen[0][1], starting)
    expected_midpoint=starting[0,0]+0.5*(env.positions[0,0]-starting[0,0])
    assert seen[1][1][0,0]==expected_midpoint
