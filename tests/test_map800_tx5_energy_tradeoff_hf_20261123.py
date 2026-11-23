"""Learned TX5W ablation: physically coupled energy, fixed reward reference, deep forensics."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.communication.link_budget import power_from_action, required_power_w, report_reference_tx_energy_j
from uav_network.communication.diagnostics import network_diagnostics
from uav_network.environment.gpu.state import params_from_config,WorldState,WorldParams
from uav_network.communication.gpu.diagnostics import emission
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.experiments.forensic_transitions import ForensicTransitionRecorder
from uav_network.experiments.telemetry import episode_telemetry
from uav_network.experiments.storage_retention import ARCHIVE_SOURCES

BASE='configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-20261123.yaml'
TX5='configs/matd3-nooracle-lr4-map800-h256x256-orthogonal-compile-tx5-learned-hf-20261123.yaml'


def test_new_tx5_config_learns_power_and_destination_but_preserves_map_actor_and_hf():
    baseline=load_config(BASE)
    c=load_config(TX5)
    assert c.network.power_control=='learned'
    assert c.network.tx_power_min_w==.001 and c.network.tx_power_max_w==5.0
    assert c.network.tx_power_mapping=='log'
    assert c.network.pa_efficiency==.35 and c.network.ack_uses_data_power
    assert c.network.peer_range_m==c.network.gcs_range_m==500
    assert c.energy.battery_capacity_j==277200
    assert c.world.map_size_m==800 and c.sensor.grid_cell_m==20
    assert c.matd3.hidden_sizes==[256,256] and c.matd3.initialization=='orthogonal'
    assert c.runtime.compile_models and c.runtime.num_envs==4096
    assert c.tracking.enabled and c.tracking.checkpoint_store=='huggingface'
    assert c.tracking.checkpoint_repo=='duy95/uav-search-checkpoints'
    assert c.reward.communication_reference_tx_power_w==.4
    assert c.reward.communication_efficiency==baseline.reward.communication_efficiency==1
    assert c.tracking.run_name!=baseline.tracking.run_name


def test_tx_energy_reward_normalization_is_invariant_to_power_cap_and_yields_real_cost():
    old=load_config(BASE)
    new=load_config(TX5)
    ref=report_reference_tx_energy_j(old.network)
    assert report_reference_tx_energy_j(new.network,reference_power_w=.4)==pytest.approx(ref)
    assert report_reference_tx_energy_j(new.network)>ref*10
    assert params_from_config(new).report_reference_tx_energy==pytest.approx(ref)
    assert params_from_config(old).report_reference_tx_energy==pytest.approx(ref)
    assert new.network.tx_power_max_w==5
    # TX actually spends airtime*(circuit+P/efficiency), from the SAME battery.
    time_s=(1024*8+464)/new.network.bit_rate_bps
    low=time_s*(new.network.tx_circuit_power_w+.4/new.network.pa_efficiency)
    high=time_s*(new.network.tx_circuit_power_w+5/new.network.pa_efficiency)
    assert (high-low)>0.05
    assert (high-low)/ref>10


def test_power_choice_really_spans_low_power_and_nlos_250m_with_relay_remains_needed():
    c=load_config(TX5).network
    assert power_from_action(-1,c)==pytest.approx(.001)
    assert power_from_action(1,c)==pytest.approx(5.0)
    assert .05<power_from_action(0,c)<.1
    p250=required_power_w(250,c.carrier_frequency_hz,c.bandwidth_hz,c.noise_density_dbm_hz,
                          c.noise_figure_db,c.sinr_threshold_db,c.nlos_model,False,c.obstacle_extra_loss_db)
    p500=required_power_w(500,c.carrier_frequency_hz,c.bandwidth_hz,c.noise_density_dbm_hz,
                          c.noise_figure_db,c.sinr_threshold_db,c.nlos_model,False,c.obstacle_extra_loss_db)
    assert .4<p250<5<p500


def test_high_power_telemetry_has_noncollapsed_bins_with_labeled_watts():
    counts=[0]*46
    stats=[[0,0,0,0] for _ in range(11)]
    powers=[[1,2,3,4,5,6],[0,0,0,0,0,0]]
    raw=network_diagnostics(counts,stats,powers,[0]*8,tx_power_max_w=5)
    assert raw['data_tx_power_histogram']==[1,2,3,4,5,6]
    assert raw['tx_power_histogram_edges_w']==[.01,.1,.5,1.,3.]
    assert 'network/diagnostics/tx_power_histogram_edges_w/bin_4' in episode_telemetry(
        dict(energy={},network=dict(diagnostics=raw),report_diagnostics=None))


def test_forensic_capture_includes_exact_tx_rx_energy_and_queued_power(tmp_path):
    c=load_config(TX5);c.runtime.device='cpu';c.runtime.num_envs=1
    env=WarpVectorEnv(c,device='cpu')
    try:
        env.reset(seed=[44])
        row=ForensicTransitionRecorder._world(env,0,deep=True)
        for name in ('tx_energy','rx_energy','pkt_power','tx_power','communication_energy',
                     'diagnostic_counts','diagnostic_stats','power_histogram'):
            assert name in row and isinstance(row[name],list),name
        assert 'forensic-transitions.jsonl' in ARCHIVE_SOURCES
        assert 'forensic-minibatches.jsonl' in ARCHIVE_SOURCES
        assert 'training-step-optimizer.jsonl' in ARCHIVE_SOURCES
    finally:env.close()


@wp.kernel(enable_backward=False)
def _emit_power_probe(s:WorldState,p:WorldParams,power:wp.float64):
    emission(s,p,0,0,p.n,0,power)


def test_actual_warp_high_power_emissions_populate_each_power_band():
    from uav_network.communication.gpu.metrics import network_info
    c=load_config(TX5);c.runtime.device='cpu';c.runtime.num_envs=1
    v=WarpVectorEnv(c,device='cpu')
    try:
        v.reset(seed=[44])
        for power in (.005,.02,.2,.75,1.5,4.5):
            v._launch(_emit_power_probe,1,v.s,v.p,float(power))
        observed=network_info(v.s,v.p,0)['diagnostics']
        assert observed['data_attempts']==6
        assert observed['data_tx_power_histogram']==[1]*6
        assert observed['tx_power_histogram_edges_w']==[.01,.1,.5,1.,3.]
        assert observed['data_tx_power_w']['mean']==pytest.approx(sum((.005,.02,.2,.75,1.5,4.5))/6)
    finally:v.close()
