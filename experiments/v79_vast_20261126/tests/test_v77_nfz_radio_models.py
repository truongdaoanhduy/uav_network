"""V77 NFZ RF assumptions: explicit opt-in, independent CPU/NumPy/Warp checks."""
from __future__ import annotations
import math
import numpy as np
import pytest
from zhang_xue.communication.common import link_path_gain
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np
from zhang_xue.gpu.runtime_v67 import HAS_WARP

P0=(100.,100.,50.); P1=(400.,100.,50.); P2=(250.,230.,50.)
BUILDINGS=((175.,100.,20.,85.),(250.,100.,20.,85.),(325.,100.,20.,85.))


def test_default_binary_nlos_keeps_legacy_0_one_many_behavior():
    los=link_path_gain(P0,P1,[],[])
    one=link_path_gain(P0,P1,BUILDINGS[:1],[])
    many=link_path_gain(P0,P1,BUILDINGS,[])
    assert 0 < one < los
    assert many == pytest.approx(one)
    assert one==pytest.approx(link_path_gain(P0,P1,BUILDINGS[:1],[],obstacle_model='binary_nlos'))


def test_multi_obstacle_fspl_counts_intersected_buildings_without_double_counting_nlos():
    los=link_path_gain(P0,P1,[],[])
    vals=[link_path_gain(P0,P1,BUILDINGS[:n],[],obstacle_model='multi_obstacle',per_building_loss_db=12.) for n in (0,1,2,3)]
    assert vals[0]==pytest.approx(los)
    for n,val in enumerate(vals):
        assert val/los==pytest.approx(10**(-1.2*n),rel=2e-5)
    assert vals[3]<vals[2]<vals[1]<los


def test_opaque_blocks_only_intersecting_3d_link_and_rooftop_clearance():
    assert link_path_gain(P0,P1,BUILDINGS,[],obstacle_model='rf_opaque')==0
    assert link_path_gain(P0,P2,BUILDINGS,[],obstacle_model='rf_opaque')>0
    a=(100.,100.,105.); b=(400.,100.,105.)
    assert link_path_gain(a,b,BUILDINGS,[],obstacle_model='rf_opaque')>0
    assert link_path_gain(P0,P1,BUILDINGS,[(2,2)],obstacle_model='multi_obstacle')==0


@pytest.mark.parametrize('mode', ['binary_nlos','multi_obstacle','rf_opaque'])
def test_numpy_reference_matches_cpu_three_modes(mode):
    p=np.array([[P0,P1,P2]],dtype=np.float32)
    gain=compute_channel_gain_v67_np(p,BUILDINGS,(),0,20,50.0,obstacle_model=mode,per_building_loss_db=12.)
    assert gain.shape==(1,3,3)
    assert np.all(gain>=0)
    for i in range(3):
        for j in range(i+1,3):
            expected=link_path_gain(p[0,i],p[0,j],BUILDINGS,(),obstacle_model=mode,per_building_loss_db=12.)
            assert gain[0,i,j]==pytest.approx(expected,rel=1e-4,abs=1e-12)
            assert gain[0,j,i]==gain[0,i,j]


def test_scenario_contract_cpu_network_and_opt_in_rejects_typos(tmp_path):
    with pytest.raises(ValueError,match='rf_obstacle_model'):
        EnvConfigV54(rf_obstacle_model='always_hard')
    with pytest.raises(ValueError,match='rf_per_building_loss_db'):
        EnvConfigV54(rf_per_building_loss_db=-1)
    path=tmp_path/'s.yaml'
    path.write_text('rf_obstacle_model: multi_obstacle\nrf_per_building_loss_db: 16.0\n')
    cfg=load_scenario(path)
    assert cfg.rf_obstacle_model=='multi_obstacle'
    from zhang_xue.environment.core import ZhangXue7NodePacketEnv
    env = ZhangXue7NodePacketEnv(cfg, seed=55)
    assert env.net.rf_obstacle_model == "multi_obstacle"
    assert env.net.rf_per_building_loss_db == pytest.approx(16.0)


def test_hidden_terminal_geometry_carrier_sense_and_gcs_links():
    """Two senders cannot hear one another while both hear GCS; an RF-only predicate."""
    p=np.array([P0,P1,P2],dtype=float)
    net=SevenRadioCsmaCaNetwork(num_uavs=2,buildings=BUILDINGS,blackout_centers=[],
                                 tx_power_w=0.1,rf_obstacle_model='multi_obstacle',
                                 rf_per_building_loss_db=20.0)
    net.update_positions(p)
    assert net.tx_power_w*net.chan_gain[0,1]<net.cca_th_w
    assert net.tx_power_w*net.chan_gain[0,2]>net.cca_th_w
    assert net.tx_power_w*net.chan_gain[1,2]>net.cca_th_w


@pytest.mark.skipif(not HAS_WARP,reason='warp-lang unavailable in this Python runtime')
@pytest.mark.parametrize('mode', ['binary_nlos','multi_obstacle','rf_opaque'])
def test_warp_cpu_matches_reference(mode):
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    p=np.array([[P0,P1,P2]],dtype=np.float32)
    radio=WarpCompressedRadioNetworkV67(1,num_uavs=2,grid_w=20,device='cpu',q_cap=64)
    radio.update_phy_from_positions(p,BUILDINGS,(),0,50.,obstacle_model=mode,per_building_loss_db=12.)
    ref=compute_channel_gain_v67_np(p,BUILDINGS,(),0,20,50.,obstacle_model=mode,per_building_loss_db=12.)
    np.testing.assert_allclose(radio.data['gain'].numpy(),ref,rtol=2e-4,atol=1e-15)


def test_hidden_terminal_causes_real_sinr_collision_on_overlapping_packets():
    from zhang_xue.communication.common import Packet, PKT_DATA, ST_TX_DATA, INF_TIME_US
    net=SevenRadioCsmaCaNetwork(num_uavs=2,buildings=BUILDINGS,blackout_centers=[],
                                tx_power_w=0.1,rf_obstacle_model='multi_obstacle',
                                rf_per_building_loss_db=20.,noise_power_w=1e-13)
    net.reset(np.asarray([P0,P1,P2]),episode_id=3)
    net.hello_next_us=[INF_TIME_US]*3
    packets=[]
    for src in (0,1):
        pkt=Packet(pkt_id=1+src,episode_id=3,pkt_type=PKT_DATA,app_src=src,
                   app_dst=net.gcs_id,app_seq=1+src,create_us=0,expire_us=1000000,
                   visited_mask=1<<src)
        net.radios[src].active_pkt=pkt
        net.radios[src].active_next_hop=net.gcs_id
        packets.append(pkt)
    net._start_tx(0,is_ack=False)
    assert net.radios[0].mac_state==ST_TX_DATA
    # The other UAV is hidden to CCA, while GCS is decoding transmitter zero.
    assert not net._is_channel_busy_at(1)
    assert net.radios[2].rx_locked_tx==0
    net._start_tx(1,is_ack=False)
    assert net.radios[2].rx_max_interf_w>0
    # Completing the first overlapping packet corrupts it at GCS via SINR.
    net._finish_tx(0)
    assert net.stats['sinr_drops']>=1
    assert net.stats['data_delivered_gcs']==0


def test_reproducible_cpu_gate_independent_reference_and_sinr_overlap():
    from scripts.maintenance.nfz_radio_gate import cpu_gate
    result=cpu_gate(20.)
    assert result['status'].startswith('CPU_PHY_')
    assert result['hidden_terminal_cca']
    assert result['sinr_drops_controlled_overlap'] >= 1
    assert result['cases']['rf_opaque']['gain_tx0_tx1'] == 0


@pytest.mark.parametrize('name,model,loss,highnoise', [
    ('scenario_v77_nfz_multi_1km.yaml','multi_obstacle',12.,False),
    ('scenario_v77_nfz_opaque_1km.yaml','rf_opaque',12.,False),
    ('scenario_v77_nfz_multi_noatten_1km.yaml','multi_obstacle',0.,False),
    ('scenario_v77_nfz_multi_highnoise_1km.yaml','multi_obstacle',12.,True),
    ('scenario_v77_nfz_multi_commreward_1km.yaml','multi_obstacle',12.,False),
])
def test_v77_scenario_ablation_profiles_parse(name,model,loss,highnoise):
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    file=root/'research/zhang_xue/configs'/name
    cfg=load_scenario(file)
    assert cfg.rf_obstacle_model==model
    assert cfg.rf_per_building_loss_db==pytest.approx(loss)
    assert cfg.mac_noise_power_w==pytest.approx(4.e-11 if highnoise else 4.38957709293155e-13,rel=1e-7)
    assert cfg.map_size_m==1000
    assert cfg.num_uavs==6
    assert cfg.uav_alt_m==50
