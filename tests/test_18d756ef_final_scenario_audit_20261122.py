"""Adversarial no-oracle mission audits: hidden peer telemetry and slot independence."""
import torch
import warp as wp
from uav_network.configuration import Config
from uav_network.environment.gpu.vector import WarpVectorEnv


def c(agents=2, targets=2):
    cfg=Config()
    cfg.world.motion_dimensions=3
    cfg.world.motion_safety='reject'
    cfg.world.num_uavs=agents
    cfg.world.num_targets=targets
    cfg.world.num_obstacles=0
    cfg.sensor.model='bayesian'
    cfg.sensor.report_cell_hypotheses=True
    cfg.runtime.environment_backend='warp_packet'
    cfg.runtime.num_envs=1
    cfg.runtime.device='cpu'
    return cfg


def published(v):
    result=v._publish(v._buffers[v._buffer])
    return result['observation'].clone(),result['state'].clone()


def test_peer_altitude_and_vertical_velocity_not_free_observation():
    v=WarpVectorEnv(c(),device='cpu')
    try:
        v.reset(seed=44)
        wp.to_torch(v.s.positions)[0]=torch.tensor([[500.,500.],[510.,500.]])
        wp.to_torch(v.s.altitudes)[0]=torch.tensor([50.,50.])
        before,_=published(v)
        wp.to_torch(v.s.altitudes)[0,1]=100.
        wp.to_torch(v.s.vertical_velocity)[0,1]=7.
        after,_=published(v)
        torch.testing.assert_close(after[0,0],before[0,0])
    finally:v.close()


def test_critic_report_slot_permutation_invariance_and_full_slot_recall():
    v=WarpVectorEnv(c(),device='cpu')
    try:
        v.reset(seed=44)
        far_slot=v.p.report_slots-1
        cell=821
        wp.to_torch(v.s.report_cell)[0,far_slot]=cell
        wp.to_torch(v.s.known)[0,1,far_slot]=1
        wp.to_torch(v.s.delivered)[0,far_slot]=1
        _,far_state=published(v)

        wp.to_torch(v.s.report_cell)[0,far_slot]=-1
        wp.to_torch(v.s.known)[0,1,far_slot]=0
        wp.to_torch(v.s.delivered)[0,far_slot]=0
        wp.to_torch(v.s.report_cell)[0,0]=cell
        wp.to_torch(v.s.known)[0,1,0]=1
        wp.to_torch(v.s.delivered)[0,0]=1
        _,first_state=published(v)
        torch.testing.assert_close(far_state,first_state)
        wp.to_torch(v.s.known)[0,1,0]=0
        wp.to_torch(v.s.delivered)[0,0]=0
        _,empty_state=published(v)
        assert (far_state-empty_state).abs().max().item()>0
    finally:v.close()


def test_true_report_delay_not_conflated_with_false_gcs_delay():
    from uav_network.communication.gpu.metrics import network_info
    v=WarpVectorEnv(c(targets=1),device='cpu')
    try:
        v.reset(seed=44)
        true_pos=wp.to_torch(v.s.targets)[0,0].numpy()
        cell=int(true_pos[0]//v.p.cell_width)+v.p.grid_n*int(true_pos[1]//v.p.cell_width)
        wp.to_torch(v.s.report_cell)[0,0]=cell
        wp.to_torch(v.s.report_cell)[0,1]=(cell+1)%(v.p.grid_n**2)
        wp.to_torch(v.s.delivered)[0,:2]=1
        wp.to_torch(v.s.report_delivery_count)[0]=2
        wp.to_torch(v.s.report_delay_sum)[0]=13
        wp.to_torch(v.s.true_report_delay_sum)[0]=5
        info=network_info(v.s,v.p,0)
        assert info['report_e2e_delay_s']==6.5
        assert info['true_report_e2e_delay_s']==5.
        assert info['raw_reports_delivered']==2
        assert info['targets_delivered']==1
    finally:v.close()


def test_paper_metric_uses_true_only_report_delay_for_cell_based_task():
    import numpy as np
    from uav_network.experiments.comparison_metrics import MissionMetrics
    cfg=c(agents=2,targets=1)
    tracker=MissionMetrics(cfg,np.array([[0.,0.,50.],[30.,0.,50.]]),np.empty((0,3)),[False],[False],obstacle_heights=np.empty(0))
    tracker.observe(np.array([[0.,0.,50.],[30.,0.,50.]]),[True],[True],1)
    info={
        'success':True,'report_identity':'observed_cell_id','blocked_moves':0,
        'energy':{'propulsion_energy_j':100.},'motion_counts':{},
        'network':{'enabled':True,'uav_communication_energy_j':2.,
                   'report_e2e_delay_s':6.5,'true_report_e2e_delay_s':5.}
    }
    assert tracker.finish(info,0.)['report_delay_delivered_mean_s']==5.


def test_partial_fov_marginal_camera_probability_matches_bayesian_likelihood():
    """P_D for physically visible true target is correct; marginalize its unknown position.

    A camera sees a random target in a partially covered cell with probability
    q*Pd + (1-q)*P0. The kernel conditional branch using Pd for a visible
    target must not be 'fixed' to use this marginal probability a second time.
    """
    import numpy as np
    cfg=c(agents=1,targets=1)
    cfg.sensor.altitude_m=50.
    cfg.runtime.num_envs=256
    cfg.network.enabled=False
    v=WarpVectorEnv(cfg,device='cpu')
    try:
        v.reset(seed=list(range(1000,1256)))
        targets=np.random.default_rng(20261122).uniform([550.,500.],[575.,525.],size=(256,2))
        wp.to_torch(v.s.targets)[:,0]=torch.as_tensor(targets,dtype=torch.float64)
        wp.to_torch(v.s.positions)[:,0]=torch.as_tensor([[510.,510.]],dtype=torch.float64)
        wp.to_torch(v.s.altitudes)[:,0]=50.
        v.step({'continuous':np.zeros((256,1,4),np.float32),
                'destination':np.zeros((256,1),np.int64)})
        empirical=wp.to_torch(v.s.positive)[:,0,822].float().mean().item()
        gx=np.linspace(.5/200,1-.5/200,200)
        x,y=np.meshgrid(550+25*gx,500+25*gx)
        q=((x-510)**2+(y-510)**2<=50**2).mean()
        effective_pf=1-(1-.1)**q
        marginal_p1=q*.9+(1-q)*effective_pf
        assert 0 < q < 1
        assert abs(empirical-marginal_p1)<.09
    finally:v.close()


def test_out_of_range_peer_report_attempt_consumes_radio_airtime_without_free_beacon():
    """Only the radio PHY knows the peer is out of range, not the source policy/MAC.

    In the no-beacon mode, a peer route action must transmit and fail (or time
    out) over the channel; silently selecting zero packets is an oracle link
    precheck and incorrectly saves TX energy.
    """
    import numpy as np
    from uav_network.communication.gpu.metrics import network_info
    v=WarpVectorEnv(c(agents=2,targets=1),device='cpu')
    try:
        v.reset(seed=44)
        wp.to_torch(v.s.positions)[0]=torch.tensor([[500.,200.],[950.,950.]])
        wp.to_torch(v.s.report_cell)[0,0]=820
        wp.to_torch(v.s.known)[0,0,0]=1
        wp.to_torch(v.s.report_present)[0,0,0]=1
        wp.to_torch(v.s.report_origin)[0,0,0]=0
        wp.to_torch(v.s.report_created)[0,0,0]=0
        cont=np.zeros((1,2,4),np.float32)
        cont[0,0,3]=1
        v.step({'continuous':cont,'destination':np.array([[1,0]],np.int64)})
        net=network_info(v.s,v.p,0)
        assert net['generated_packets']>=1
        assert net['uav_tx_energy_j']>0
        assert net['targets_delivered']==0
    finally:v.close()


def test_semantic_and_network_changes_get_new_checkpoint_version():
    from uav_network.environment.observations import observation_version
    assert observation_version(c())==7
    v=WarpVectorEnv(c(),device='cpu')
    try:
        v.reset(seed=44)
        assert v.snapshot_state()['version']==4
    finally:v.close()
