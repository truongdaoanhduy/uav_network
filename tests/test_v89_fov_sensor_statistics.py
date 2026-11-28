"""V89 sensing QA: FoV, 3D building occlusion, and measured Pd/Pf statistical calibration."""
import numpy as np
import pytest
import torch
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.gpu.kernels.mission_v79_kernels import _sense_visible_v79
from zhang_xue.gpu.state_v67 import wp

CONFIG="research/zhang_xue/configs/scenario_v89_static_1km.yaml"

def _sense(world):
    n,w,p=world.network,world.buffers,world.params
    wp.launch(_sense_visible_v79, dim=(p.envs,6),inputs=[
        n.state,w["positions"],w["truth"],w["episode_step"],w["episode_id"],
        w["battery"],w["unsafe_mask"],w["failed"],world.build_wp,
        len(world.buildings),int(p.seed),float(p.pd),float(p.pf),
        float(p.ds_aoi_eta),int(p.static_raw_count)],device=p.device)

@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_target_cell_requires_real_fov_visibility_and_unblocked_ray(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p=GPUWorldV80Params.from_yaml(CONFIG,envs=1,device=device,seed=44)
    world=WarpParallelMissionV80(p)
    cid=5*20+6  # target center (325,275): just outside building at (280,280), r45
    pos=wp.to_torch(world.buffers["positions"])
    ticks=wp.to_torch(world.buffers["episode_step"])
    ledger=world.network.data["ledger_gen"]
    pos[0,0,0]=300.
    pos[0,0,1]=235.
    ticks[0]=0
    _sense(world)
    assert int(ledger.numpy()[0,0,0,cid])==-1, "Occluded cell was sensed through a building"
    pos[0,0,0]=350.
    pos[0,0,1]=275.
    ticks[0]=1
    _sense(world)
    assert int(ledger.numpy()[0,0,0,cid])==2, "Unoccluded target cell inside FoV not measured"
    pos[0,0,0]=410.
    ticks[0]=2
    _sense(world)
    assert int(ledger.numpy()[0,0,0,cid])==2, "Out-of-FoV cell unexpectedly measured"

def test_empirical_pd_pf_on_visible_cell_across_gpu_worlds():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    envs=1024
    p=GPUWorldV80Params.from_yaml(CONFIG,envs=envs,device="cuda:0",seed=44)
    world=WarpParallelMissionV80(p)
    cid=2*20+1  # cell center 75,125, visible from spawn (90,110)
    truth=wp.to_torch(world.buffers["truth"])
    truth[:envs//2,cid]=1
    truth[envs//2:,cid]=0
    _sense(world)
    gen=world.network.data["ledger_gen"].numpy()[:,0,0,cid]
    detected=world.network.data["ledger_pos"].numpy()[:,0,0,cid]
    missed=world.network.data["ledger_neg"].numpy()[:,0,0,cid]
    assert np.all(gen==1)
    assert np.all(detected.astype(int)+missed.astype(int)==1)
    empirical_pd=float(np.mean(detected[:envs//2]))
    empirical_pf=float(np.mean(detected[envs//2:]))
    assert abs(empirical_pd-p.pd)<0.075, (empirical_pd,p.pd)
    assert abs(empirical_pf-p.pf)<0.045, (empirical_pf,p.pf)
