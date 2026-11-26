"""V78 test contracts; CUDA functional gate is separate, not faked by pytest."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
from scripts.maintenance.nfz_packet_workload_v78 import check_physics, cpu_experiment, MODES, NFZ, POSITIONS

@pytest.mark.parametrize('mode',MODES)
def test_real_nfz_two_hop_cpu_packet_and_provenance(mode):
    physical=check_physics(mode,20.)
    assert physical['direct_u0_gcs_rx_power_w']<physical['u0_peer_rx_power_w']
    actual=cpu_experiment(mode,20.,44)
    assert actual['gcs_evidence_count_uq88']==256
    assert actual['gcs_generation_s']==1
    assert actual['network_stats']['e2e_ack_rx']>=2

@pytest.mark.parametrize('mode',MODES)
def test_blockers_do_not_forge_positive_link(mode):
    from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np
    g=compute_channel_gain_v67_np(POSITIONS,NFZ,(),0,20,50.,obstacle_model=mode,per_building_loss_db=20.)
    assert np.all(np.isfinite(g))
    assert np.all(g>=0)
    assert np.allclose(g,np.swapaxes(g,1,2),rtol=2e-4,atol=1e-15)
    assert g[0,0,2]<g[0,0,1]


def test_mission_gate_is_honestly_blocked_without_opt_in(tmp_path):
    cmd=[sys.executable,str(ROOT/'scripts/gpu/run_v78_real_rollout.py'),
         '--algorithm','matd3','--output',str(tmp_path/'blocked.json')]
    p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==3
    status=json.loads((tmp_path/'blocked.json').read_text())
    assert status['status']=='BLOCKED_EXPERIMENTAL_PHYSICS_REQUIRES_ACK'
    assert status['verified_exact_cpu_model_parity'] is False
    assert status['gpu_training_converged'] is False


def test_randomized_hidden_terminal_produces_excess_collisions_and_mac_timeouts():
    from scripts.maintenance.v78_hidden_terminal_mac_stress import run
    result=run(12,offset=100)
    assert result['status']=='PASS'
    assert result['paired_mean_excess_sinr_drops']>0
    assert result['rf_obstacle_nfz']['gcs_packets']['mean']==2
