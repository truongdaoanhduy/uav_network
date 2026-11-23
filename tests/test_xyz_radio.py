"""XYZ range, finite-height RF LOS and real report custody."""
import numpy as np
import pytest
import torch
import warp as wp
from test_xyz_sensing import xyz_config
from uav_network.communication.link_budget import polygon_line_of_sight,nominal_required_power_w,required_power_w
from uav_network.environment.env import SwarmSearchEnv,Action
from uav_network.environment.observations import destination_mask


def test_rf_prism_los_uses_endpoint_heights():
    obs=np.array([[50.,0.,5.]])
    assert polygon_line_of_sight([0,0,60],[100,0,60],obs,16,heights=[40])
    assert not polygon_line_of_sight([0,0,20],[100,0,20],obs,16,heights=[40])
    assert not polygon_line_of_sight([50,0,60],[50,0,20],obs,16,heights=[40])


def test_nominal_required_power_uses_real_3d_endpoints():
    c=xyz_config();n=c.network
    d=np.sqrt(30**2+40**2+100**2)
    expected=required_power_w(d,n.carrier_frequency_hz,n.bandwidth_hz,n.noise_density_dbm_hz,
                             n.noise_figure_db,n.sinr_threshold_db,n.nlos_model,True)
    assert nominal_required_power_w([0,0,10],[30,40,110],[],n,heights=[])==pytest.approx(expected)


def test_native_3d_range_rejects_horizontal_only_contact():
    c=xyz_config();c.network.enabled=True;c.network.gcs_range_m=220
    env=SwarmSearchEnv(c)
    try:
        env.reset(options=dict(positions=[[500,200,150]],targets=[[950,950]],obstacles=[]))
        env.reports.discover(0,0,0)
        assert not destination_mask(env)[0,-1]
        assert env.network.simulator.drones[0].coords[2]==150
        env.step(Action(np.array([[0,0,0,1.]],np.float32),np.array([1])))
        assert env.network.diagnostic_counts.sum()>=0
        assert not env.reports.delivered.any()
    finally:env.close()


@pytest.mark.parametrize("backend",["native","warp"])
def test_xyz_real_peer_then_gcs_transfer(backend):
    c=xyz_config(n=2);c.network.enabled=True
    c.network.peer_range_m=c.network.gcs_range_m=250
    c.network.tx_power_min_w=.001;c.network.tx_power_mapping="log"
    if backend=="native":
        env=SwarmSearchEnv(c)
        env.reset(options=dict(positions=[[500,400,25],[500,200,25]],targets=[[950,950]],obstacles=[]))
        env.reports.discover(0,0,0)
        step=lambda a:env.step(a)
        held=lambda i:bool(env.reports.buffers[i])
        delivered=lambda:bool(env.reports.delivered[0])
    else:
        from uav_network.environment.gpu.vector import WarpVectorEnv
        c.runtime.num_envs=1;c.runtime.environment_backend="warp_packet";c.runtime.device="cpu"
        env=WarpVectorEnv(c,"cpu");env.reset(seed=44)
        wp.to_torch(env.s.positions)[0]=torch.tensor([[500.,400.],[500.,200.]])
        wp.to_torch(env.s.altitudes)[0]=25
        wp.to_torch(env.s.targets)[0]=torch.tensor([[950.,950.]])
        wp.to_torch(env.s.known)[0,0,0]=1
        wp.to_torch(env.s.report_present)[0,0,0]=1
        wp.to_torch(env.s.report_order)[0,0,0]=1
        step=lambda a:env.step(dict(continuous=a.continuous[None],destination=a.destination[None]))
        held=lambda i:bool(wp.to_torch(env.s.report_present)[0,i,0])
        delivered=lambda:bool(wp.to_torch(env.s.delivered)[0,0])
    try:
        a=np.zeros((2,4),np.float32);a[:,-1]=1
        step(Action(a,np.array([1,0],np.int64)))
        assert held(1) and not held(0)
        if backend=="native":info=env.info()
        else:
            from uav_network.communication.gpu.metrics import world_info
            info=world_info(env.s,env.p)
        actual=info["network"]["diagnostics"]["relay_gcs_distance_improvement_m"]["mean"]
        assert actual==pytest.approx(np.hypot(400,25)-np.hypot(200,25))
        assert not delivered()
        step(Action(a,np.array([0,2],np.int64)))
        assert delivered()
    finally:env.close()

from uav_network.environment.gpu.state import WorldState,WorldParams
from uav_network.communication.gpu.phy import nominal_required_power,link_los,link_distance


@wp.kernel(enable_backward=False)
def inspect_xyz_link(s:WorldState,p:WorldParams,out:wp.array(dtype=wp.float64)):
    out[0]=nominal_required_power(s,p,0,0,1)
    out[1]=wp.float64(int(link_los(s,p,0,0,1)))
    out[2]=link_distance(s,p,0,0,1)


@pytest.mark.parametrize("z0,z1",[(20,20),(60,60),(25,75)])
def test_warp_rf_prism_matches_native_budget(z0,z1):
    from uav_network.environment.gpu.state import allocate_state
    c=xyz_config(n=2,obstacles=1)
    s,p=allocate_state(c,1,device="cpu")
    s.positions.assign(np.array([[[0,500],[100,500]]],float))
    s.altitudes.assign(np.array([[z0,z1]],float))
    s.obstacles.assign(np.array([[[50,500,5]]],float))
    s.obstacle_heights.assign(np.array([[40.]],float))
    out=wp.zeros(3,dtype=wp.float64,device="cpu")
    wp.launch(inspect_xyz_link,dim=1,inputs=[s,p,out],device="cpu")
    a,b=[0,500,z0],[100,500,z1]
    expected=nominal_required_power_w(a,b,[[50,500,5]],c.network,heights=[40])
    assert out.numpy()[0]==pytest.approx(expected,rel=1e-10)
    assert bool(out.numpy()[1])==polygon_line_of_sight(a,b,[[50,500,5]],16,heights=[40])
    assert out.numpy()[2]==pytest.approx(np.linalg.norm(np.subtract(a,b)))
