"""True radio HELLO, receiver-only cache, staleness and battery (TDD)."""
import numpy as np
import torch
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.gpu.observations import observe
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.environment.observations import observation_version

BASE='configs/matd3-nooracle-map1000-25d-aoi-zeng2019-h35-130-vz4-20261123.yaml'

def config():
    c=load_config(BASE);c.runtime.device='cpu';c.runtime.num_envs=1
    c.world.num_uavs=2;c.world.num_obstacles=0;c.world.num_targets=1
    c.network.hello_enabled=True;c.network.hello_period_s=1.0
    c.network.hello_freshness_s=2.0;c.network.hello_bits=512
    c.sensor.visible_target_slots=1
    return c.validate()

def make_env():
    v=WarpVectorEnv(config(),device='cpu');v.reset(seed=[44])
    wp.to_torch(v.s.positions)[0]=torch.tensor([[400.,500.],[550.,500.]],dtype=torch.float64)
    wp.to_torch(v.s.altitudes)[0]=80.;wp.to_torch(v.s.battery)[0]=v.p.capacity
    # A locally held report is required before a peer route can be selected;
    # never manufacture a remote report or private peer state.
    wp.to_torch(v.s.report_cell)[0,0]=0
    wp.to_torch(v.s.report_present)[0,0,0]=1
    wp.to_torch(v.s.report_created)[0,0,0]=0.
    v._launch(observe,(1,2),v.s,v.p)
    return v

def actions():
    return {'continuous':np.zeros((1,2,4),np.float32),'destination':np.zeros((1,2),np.int64)}

def test_hello_initial_observation_no_oracle_and_new_version():
    v=make_env()
    try:
        assert observation_version(v.config)!=observation_version(load_config(BASE))
        obs=wp.to_torch(v.s.observation)[0,0]
        assert obs[-2].item()==0
        peer_start=10+2*v.p.targets_count+4*(v.p.target_slots+v.p.obstacle_slots)
        assert obs[peer_start:peer_start+6].abs().sum().item()==0
    finally:v.close()

def test_hello_rf_cache_and_expiry_not_live_truth():
    v=make_env()
    try:
        v.step(actions())
        rx=wp.to_torch(v.s.hello_rx_time)
        assert rx[0,0,1].item()>=0.0 and rx[0,1,0].item()>=0.0
        cached=wp.to_torch(v.s.observation)[0,0].clone()
        assert cached[-2].item()==1.0
        peer_start=10+2*v.p.targets_count+4*(v.p.target_slots+v.p.obstacle_slots)
        # Sixth peer feature is advertised-link freshness (1=brand-new, 0=stale)
        # rather than an omniscient live UAV availability flag.
        assert 0 < cached[peer_start+5].item() < 1
        wp.to_torch(v.s.positions)[0,1]=torch.tensor([950.,800.],dtype=torch.float64)
        v._launch(observe,(1,2),v.s,v.p)
        observed=wp.to_torch(v.s.observation)[0,0]
        assert observed[peer_start].item()==cached[peer_start].item()
        assert observed[-2].item()==1.0
        rx[0,0,1]=-100.0
        v._launch(observe,(1,2),v.s,v.p)
        expired=wp.to_torch(v.s.observation)[0,0]
        assert expired[peer_start:peer_start+6].abs().sum().item()==0
        assert expired[-2].item()==0
    finally:v.close()

def test_hello_tx_energy_and_reset_clears_cache():
    v=make_env()
    try:
        before=wp.to_torch(v.s.tx_energy)[0,:2].clone()
        v.step(actions())
        assert torch.all(wp.to_torch(v.s.tx_energy)[0,:2]>before)
        v.reset(seed=[45])
        assert torch.all(wp.to_torch(v.s.hello_rx_time)<0)
    finally:v.close()

def test_hello_no_out_of_range_cache_and_diagnostics():
    from uav_network.communication.gpu.metrics import network_info
    v=make_env()
    try:
        wp.to_torch(v.s.positions)[0]=torch.tensor([[100.,100.],[900.,900.]],dtype=torch.float64)
        v.step(actions())
        rx=wp.to_torch(v.s.hello_rx_time)[0]
        assert torch.all(rx<0)
        info=network_info(v.s,v.p)
        assert info['hello_tx_frames']==2
        assert info['hello_rx_frames']==0
        assert info['hello_decode_failures']>=2
        assert wp.to_torch(v.s.observation)[0,0,-2].item()==0
    finally:v.close()


def test_hello_obstructed_radio_never_reveals_hidden_position():
    c=config();c.world.num_obstacles=1
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        wp.to_torch(v.s.positions)[0]=torch.tensor([[400.,500.],[550.,500.]],dtype=torch.float64)
        wp.to_torch(v.s.altitudes)[0]=80.
        wp.to_torch(v.s.obstacles)[0,0]=torch.tensor([475.,500.,40.],dtype=torch.float64)
        wp.to_torch(v.s.obstacle_heights)[0,0]=120.
        v.step(actions())
        assert torch.all(wp.to_torch(v.s.hello_rx_time)[0]<0)
        assert wp.to_torch(v.s.hello_rx_fail_count)[0].item()>=2
    finally:v.close()


def test_hello_repeatable_seeded_mac_state():
    worlds=[make_env(),make_env()]
    try:
        for _ in range(3):
            for v in worlds:v.step(actions())
        for field in ('hello_rx_time','hello_position','hello_tx_count','hello_rx_count','hello_rx_fail_count'):
            assert torch.equal(wp.to_torch(getattr(worlds[0].s,field)),
                               wp.to_torch(getattr(worlds[1].s,field))),field
    finally:
        for v in worlds:v.close()

def test_hello_data_gcs_ack_success_does_not_confuse_frame_types():
    v=make_env()
    try:
        wp.to_torch(v.s.positions)[0]=torch.tensor([[400.,100.],[550.,100.]],dtype=torch.float64)
        # Send source 0's 1024-B local report to known fixed GCS.
        data={'continuous':np.zeros((1,2,4),np.float32),
              'destination':np.array([[2,0]],dtype=np.int64)}
        v.step(data)
        counts=wp.to_torch(v.s.packet_counts)[0]
        assert counts[0].item()>=1 and counts[1].item()>=1
        assert wp.to_torch(v.s.report_delivery_count)[0].item()>=1
        assert wp.to_torch(v.s.hello_tx_count)[0].item()==2
        assert wp.to_torch(v.s.errors)[0].item()==0
    finally:v.close()


def test_hello_stales_in_real_step_progress_after_link_disappears():
    v=make_env()
    try:
        v.step(actions())
        assert wp.to_torch(v.s.hello_rx_time)[0,0,1].item()>=0
        wp.to_torch(v.s.positions)[0,1]=torch.tensor([950.,800.],dtype=torch.float64)
        for _ in range(3):v.step(actions())
        assert wp.to_torch(v.s.hello_rx_time)[0,0,1].item()<0.5
        assert wp.to_torch(v.s.observation)[0,0,-2].item()==0
    finally:v.close()


def test_hello_rejects_wrong_backend_or_time_configuration():
    import pytest
    c=config();c.runtime.environment_backend='native'
    with pytest.raises(ValueError,match='warp_packet|Warp'):
        c.validate()
    c=config();c.network.hello_freshness_s=0.5
    with pytest.raises(ValueError,match='HELLO period'):
        c.validate()

def test_hello_ablation_config_preserves_science_other_than_rf_discovery():
    old=load_config(BASE)
    new=load_config('configs/matd3-nooracle-map1000-25d-aoi-zeng2019-hello1hz-fresh2s-20261123.yaml')
    assert not old.network.hello_enabled and new.network.hello_enabled
    assert new.network.hello_period_s==1 and new.network.hello_freshness_s==2
    for part in ('world','sensor','energy','reward','matd3','masac','pql'):
        assert getattr(old,part)==getattr(new,part),part
    assert old.runtime.episodes==new.runtime.episodes==20000
    assert old.runtime.num_envs==new.runtime.num_envs==4096
    assert old.matd3.batch_size==new.matd3.batch_size==8192
    assert new.runtime.warm_start_checkpoint=='none'
    assert old.tracking.run_name!=new.tracking.run_name
