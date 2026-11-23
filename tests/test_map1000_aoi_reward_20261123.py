"""Reward is freshness weighted only after an actually delivered cell report."""
import numpy as np
import pytest
import torch
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.environment.gpu.state import WorldState, WorldParams
from uav_network.communication.gpu.reports import transfer, begin_step, end_step

CONFIG = "configs/matd3-nooracle-map1000-25d-h35-130-vz4-20261123.yaml"

@wp.kernel(enable_backward=False)
def deliver_true_report_after_60s(s: WorldState, p: WorldParams):
    e = wp.tid()
    # Synthetic receipt after 60 s, with true data on the GCS path.
    ok = transfer(s,p,e,0,p.n,0,0,wp.float64(0),0,wp.float64(60),wp.float64(0.2))
    if not ok:
        s.errors[e] = 98


def test_true_report_reward_uses_age_not_raw_phy_joules():
    c = load_config(CONFIG)
    c.runtime.num_envs=1
    c.runtime.device="cpu"
    c.world.num_targets=1
    c.world.num_uavs=2
    c.world.num_obstacles=0
    c.network.enabled=False
    c.reward.information_gain=0
    c.reward.investigation_shaping=0
    env=WarpVectorEnv(c,device="cpu")
    try:
        env.reset(seed=44)
        target=wp.to_torch(env.s.targets)[0,0].cpu().numpy()
        x,y=(np.floor(target / c.sensor.grid_cell_m)).astype(int)
        wp.to_torch(env.s.report_cell)[0,0]=int(y*env.p.grid_n+x)
        wp.to_torch(env.s.report_present)[0,0,0]=1
        wp.to_torch(env.s.report_origin)[0,0,0]=0
        wp.to_torch(env.s.report_created)[0,0,0]=0.
        wp.to_torch(env.s.report_hops)[0,0,0]=0
        wp.to_torch(env.s.known)[0,0,0]=1
        wp.launch(begin_step,dim=1,inputs=[env.s,env.p],device="cpu")
        wp.launch(deliver_true_report_after_60s,dim=1,inputs=[env.s,env.p],device="cpu")
        wp.launch(end_step,dim=1,inputs=[env.s,env.p],device="cpu")
        assert int(wp.to_torch(env.s.errors)[0])==0
        assert int(wp.to_torch(env.s.delivered)[0,0])==1
        assert float(wp.to_torch(env.s.true_delivery_freshness)[0])==pytest.approx(.8)
        # Weighted score: 50 * (0.5 + 0.5 * (1 - 60/300)) = 45
        reward_delivery=float(wp.to_torch(env.s.reward_components)[0,1])
        assert reward_delivery==pytest.approx(45.)
    finally:
        env.close()
