"""V89 static-target evidence physics: 270 samples without Q8.8 saturation or AoI-induced forgetting."""
import numpy as np
import pytest
import torch
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.gpu.kernels.mission_v79_kernels import _sense_visible_v79, _fuse_uq88_v79
from zhang_xue.gpu.state_v67 import HAS_WARP, NetworkBuffersV67, wp

if HAS_WARP:
    @wp.kernel
    def _probe_static_fusion_v89(
        net: NetworkBuffersV67, receiver: int, cell: int, time_s: int,
        logmf: float, logme: float, eta: float, static_mode: int,
        out: wp.array(dtype=wp.float32),
    ):
        ds = _fuse_uq88_v79(net, 0, receiver, cell, time_s, logmf, logme, eta, static_mode)
        out[0] = ds[0]
        out[1] = ds[1]
        out[2] = ds[2]

@pytest.mark.parametrize("device", ["cpu","cuda:0"])
def test_static_270_visible_samples_and_no_blackout_forgetting(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    params=GPUWorldV80Params.from_yaml(
        "research/zhang_xue/configs/scenario_v89_static_1km.yaml",
        envs=1,device=device,seed=44
    )
    world=WarpParallelMissionV80(params)
    n,w=world.network,world.buffers
    cid=2*20+1   # (75m,125m) seen from UAV0 spawn (90m,110m)
    s=wp.to_torch(w["episode_step"])
    for i in range(270):
        s[0]=i
        wp.launch(_sense_visible_v79,dim=(1,6),inputs=[
            n.state,w["positions"],w["truth"],w["episode_step"],w["episode_id"],
            w["battery"],w["unsafe_mask"],w["failed"],world.build_wp,len(world.buildings),
            int(params.seed),float(params.pd),float(params.pf),
            float(params.ds_aoi_eta),int(params.static_raw_count)],device=device)
    pos=int(n.data["ledger_pos"].numpy()[0,0,0,cid])
    neg=int(n.data["ledger_neg"].numpy()[0,0,0,cid])
    assert pos+neg == 270, (pos,neg)
    assert int(n.data["ledger_gen"].numpy()[0,0,0,cid]) == 270
    out=wp.zeros(3,dtype=wp.float32,device=device)
    def read_fusion(at_time):
        wp.launch(_probe_static_fusion_v89,dim=1,inputs=[
            n.state,0,cid,at_time,world.logmf,world.logme,
            float(params.ds_aoi_eta),int(params.static_raw_count),out],device=device)
        return out.numpy().copy()
    old=read_fusion(270)
    future=read_fusion(900)
    np.testing.assert_allclose(old,future,rtol=0,atol=1e-7)
    assert np.isfinite(old).all()
    assert float(np.sum(old)) == pytest.approx(1.0,abs=1e-6)

@pytest.mark.parametrize("device", ["cpu","cuda:0"])
def test_packeted_raw_counts_do_not_exceed_original_sensor_ledger(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    params=GPUWorldV80Params.from_yaml(
        "research/zhang_xue/configs/scenario_v89_static_1km.yaml",
        envs=1,device=device,seed=67
    )
    world=WarpParallelMissionV80(params)
    a=wp.zeros((1,6,2),dtype=wp.float32,device=device)
    dest=wp.full((1,6),1,dtype=wp.int32,device=device)
    for _ in range(25):
        world.step(a,dest)
    src_p=world.network.data["ledger_pos"].numpy()[0,0,0].astype(np.int64)
    src_n=world.network.data["ledger_neg"].numpy()[0,0,0].astype(np.int64)
    received_p=world.network.data["ledger_pos"].numpy()[0,6,0].astype(np.int64)
    received_n=world.network.data["ledger_neg"].numpy()[0,6,0].astype(np.int64)
    assert np.all(received_p<=src_p) and np.all(received_n<=src_n)
    assert int(np.sum(src_p+src_n))>0
    assert int(np.sum(received_p+received_n))>0, "No evidence arrived at GCS through actual packets"
    assert int(world.network.data["stats_gcs"].numpy()[0])>0
