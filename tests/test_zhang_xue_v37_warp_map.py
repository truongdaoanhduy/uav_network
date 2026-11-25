"""Map-only Warp batched sensor/DS validation, independent of networking/training."""
import math

import numpy as np
import pytest

wp=pytest.importorskip("warp")
from uav_network.environment.gpu.zhang_xue_map_warp import ZhangXueWarpMapBatch
from uav_network.communication.zhang_xue_ref.ds_belief import ds_log_space_safe


def _scene(e=2):
    pos=np.zeros((e,6,2),np.float32)
    xy=np.array([[525,525],[125,125],[875,125],[125,875],[875,875],[825,525]],np.float32)
    pos[:]=xy
    truth=np.zeros((e,400),np.int32)
    truth[:,210]=1
    return pos,truth


def test_warp_camera_footprint_nadir_fov90_counts_and_prior():
    pos,truth=_scene(2)
    batch=ZhangXueWarpMapBatch(2,device="cpu")
    batch.reset()
    initial=batch.read()
    np.testing.assert_array_equal(initial["pos"],0)
    np.testing.assert_array_equal(initial["neg"],0)
    np.testing.assert_allclose(initial["unknown"],1.)
    draws=np.ones((2,6,400),np.float32)*.9
    draws[:,0,210]=.1
    batch.step(pos,truth,uniform_draws=draws)
    state=batch.read()
    radius=100.
    for e in range(2):
        for u in range(6):
            for c in range(400):
                x,y=(c%20+.5)*50,(c//20+.5)*50
                expected=int((x-pos[e,u,0])**2+(y-pos[e,u,1])**2<=radius**2+1e-6)
                self_count=int(state["pos"][e,u,c]+state["neg"][e,u,c])
                assert self_count==expected
    assert int(state["pos"][0,0,210])==1
    assert float(state["mass_target"][0,0,210])==pytest.approx(.8,abs=1e-5)
    assert float(state["unknown"][0,0,210])==pytest.approx(.2,abs=1e-5)


def test_warp_ds_matches_numpy_own_and_team_over_repeated_steps():
    e=16
    rng=np.random.default_rng(20261125)
    pos=rng.uniform(0,999,(e,6,2)).astype(np.float32)
    truth=(rng.uniform(size=(e,400))<.03).astype(np.int32)
    batch=ZhangXueWarpMapBatch(e,device="cpu")
    batch.reset()
    ownp=np.zeros((e,6,400),np.int32)
    ownn=np.zeros_like(ownp)
    for step in range(4):
        draws=rng.random((e,6,400),dtype=np.float32)
        batch.step(pos,truth,uniform_draws=draws)
        for env in range(e):
            for agent in range(6):
                x,y=pos[env,agent]
                for cid in range(400):
                    dx=(cid%20+.5)*50-x
                    dy=(cid//20+.5)*50-y
                    if dx*dx+dy*dy>10000.+1e-5:continue
                    detected=draws[env,agent,cid] < (.85 if truth[env,cid] else .03)
                    if detected:ownp[env,agent,cid]+=1
                    else:ownn[env,agent,cid]+=1
        state=batch.read()
        np.testing.assert_array_equal(state["pos"],ownp)
        np.testing.assert_array_equal(state["neg"],ownn)
        f,e_mass,u=ds_log_space_safe(ownp,ownn,.8,.8)
        np.testing.assert_allclose(state["mass_target"],f,atol=2e-6,rtol=0)
        np.testing.assert_allclose(state["unknown"],u,atol=2e-6,rtol=0)
        ft,et,ut=ds_log_space_safe(ownp.sum(axis=1),ownn.sum(axis=1),.8,.8)
        np.testing.assert_allclose(state["team_mass_target"],ft,atol=2e-6,rtol=0)
        np.testing.assert_allclose(state["team_unknown"],ut,atol=2e-6,rtol=0)


def test_parallel_reset_no_state_leak_and_seeded_gpu_random():
    pos,truth=_scene(4)
    b=ZhangXueWarpMapBatch(4,device="cpu")
    b.reset()
    b.step(pos,truth,seed=123,step_index=1)
    first=b.read()["pos"].copy()
    b.reset()
    b.step(pos,truth,seed=123,step_index=1)
    np.testing.assert_array_equal(first,b.read()["pos"])
    b.reset()
    assert int(b.read()["pos"].sum()+b.read()["neg"].sum())==0


def test_validates_shapes_and_requires_actual_warp_array():
    b=ZhangXueWarpMapBatch(2,device="cpu")
    pos,truth=_scene(2)
    with pytest.raises(ValueError):
        b.step(pos[:1],truth,seed=1,step_index=0)
    with pytest.raises(ValueError):
        b.step(pos,truth,uniform_draws=np.zeros((2,400)))


def test_warp_sensor_pd_pf_empirical_1024_parallel_worlds():
    e=1024
    pos,_=_scene(e)
    b=ZhangXueWarpMapBatch(e,device="cpu")
    # All visible cells are ground targets (test Pd conditional probability).
    true_world=np.ones((e,400),np.int32)
    b.reset()
    b.step(pos,true_world,seed=119,step_index=0)
    state=b.read()
    observed=int((state["pos"]+state["neg"]).sum())
    pd=float(state["pos"].sum()/observed)
    assert abs(pd-.85)<.016,(observed,pd)
    # Same number of visits, no ground targets, estimate Pf.
    b.reset()
    b.step(pos,np.zeros_like(true_world),seed=521,step_index=0)
    state=b.read()
    observed=int((state["pos"]+state["neg"]).sum())
    pf=float(state["pos"].sum()/observed)
    assert abs(pf-.03)<.009,(observed,pf)
