"""V91 Bayes opt-in GPU/CPU posterior verification, keeping V90 DS ablation unmodified."""
import math
import numpy as np
import pytest
import torch

from zhang_xue.gpu.state_v67 import HAS_WARP, NetworkBuffersV67, wp
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.gpu.kernels.mission_v79_kernels import _fuse_belief_v91

ROOT="research/zhang_xue/configs/"
if HAS_WARP:
    @wp.kernel
    def _probe_belief_v91(
        n: NetworkBuffersV67,
        cell: int, bayesian: int,
        logmf: float, logme: float, log_prior: float, llrp: float, llrn: float,
        out: wp.array(dtype=wp.float32),
    ):
        b=_fuse_belief_v91(n,0,6,cell,5,logmf,logme,0.0,1,bayesian,log_prior,llrp,llrn)
        out[0]=b[0]
        out[1]=b[1]
        out[2]=b[2]

def posterior(p,n,pd=.85,pf=.03,prior=15/392):
    logit=math.log(prior/(1-prior))+p*math.log(pd/pf)+n*math.log((1-pd)/(1-pf))
    return 1/(1+math.exp(-logit))

@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_v91_bayesian_posterior_matches_analytical_counts(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    params=GPUWorldV80Params.from_yaml(ROOT+"scenario_v91_bayes_1km.yaml",
                                      envs=1,device=device,seed=44)
    w=WarpParallelMissionV80(params)
    cid=2*20+1
    count_p=wp.to_torch(w.network.data["ledger_pos"])
    count_n=wp.to_torch(w.network.data["ledger_neg"])
    gen=wp.to_torch(w.network.data["ledger_gen"])
    out=wp.zeros(3,dtype=wp.float32,device=device)
    for p,n in ((0,0),(1,0),(2,0),(1,1),(4,1),(0,7)):
        gen[0,6,0,cid]=5 if p+n else -1
        count_p[0,6,0,cid]=p
        count_n[0,6,0,cid]=n
        w._launch(_probe_belief_v91,1,[
            w.network.state,cid,1,w.logmf,w.logme,w.bayes_log_prior,
            w.bayes_llr_pos,w.bayes_llr_neg,out,
        ])
        vals=out.numpy()
        assert vals[0]==pytest.approx(posterior(p,n),abs=1e-5)
        assert vals[1]==pytest.approx(1-posterior(p,n),abs=1e-5)
        assert vals[2]==pytest.approx(.5**(p+n),abs=1e-5)
    # Different origins carry disjoint samples; repeated same-generation data
    # must overwrite an origin slot rather than increase a global counter.
    gen[0,6,0,cid]=5;gen[0,6,1,cid]=5
    count_p[0,6,0,cid]=1;count_n[0,6,0,cid]=0
    count_p[0,6,1,cid]=1;count_n[0,6,1,cid]=0
    w._launch(_probe_belief_v91,1,[
        w.network.state,cid,1,w.logmf,w.logme,w.bayes_log_prior,
        w.bayes_llr_pos,w.bayes_llr_neg,out,
    ])
    assert out.numpy()[0]==pytest.approx(posterior(2,0),abs=1e-5)

@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_v91_bayesian_reward_confirms_at_two_hits_despite_u_above_ds_ceiling(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    params=GPUWorldV80Params.from_yaml(ROOT+"scenario_v91_bayes_1km.yaml",
                                      envs=1,device=device,seed=44)
    w=WarpParallelMissionV80(params)
    cid=int(np.flatnonzero(w.buffers["truth"].numpy()[0])[0])
    gen=wp.to_torch(w.network.data["ledger_gen"])
    pos=wp.to_torch(w.network.data["ledger_pos"])
    neg=wp.to_torch(w.network.data["ledger_neg"])
    gen[0,6,0,cid]=1;pos[0,6,0,cid]=2;neg[0,6,0,cid]=0
    # Synthetic GCS evidence bypasses MAC deliberately: this test isolates
    # the Bayesian threshold contract. Actual packet path has other tests.
    cont=wp.zeros((1,6,2),dtype=wp.float32,device=device)
    disc=wp.zeros((1,6),dtype=wp.int32,device=device)
    w.step(cont,disc)
    assert int(w.buffers["gcs_tp"].numpy()[0,cid])==1

def test_v91_ds_ablation_keeps_original_params_and_guard():
    d=GPUWorldV80Params.from_yaml(ROOT+"scenario_v89_static_1km.yaml",envs=1,device="cpu")
    b=GPUWorldV80Params.from_yaml(ROOT+"scenario_v91_bayes_1km.yaml",envs=1,device="cpu")
    assert not d.bayesian_count_mode and b.bayesian_count_mode
    assert b.ds_aoi_eta==0.0 and b.static_raw_count
    assert b.pd==d.pd and b.pf==d.pf
    assert b.horizon==d.horizon==900
