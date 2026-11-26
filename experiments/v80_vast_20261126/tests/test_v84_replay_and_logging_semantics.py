"""Behavioral replay, warmup and logging regressions against the V83 implementation."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,sys
import numpy as np
import pytest
import torch
import warp as wp
from uav_network.algorithms.core.action_types import Action
from uav_network.algorithms.core.replay import TensorReplayBuffer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/gpu"))
def module():
    spec=importlib.util.spec_from_file_location("runner_v84_probe",ROOT/"scripts/gpu/run_v81_real_rollout.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def setup_store(nominal,safe,terminated=0):
    shape=(1,6,2)
    nom=np.full(shape,nominal*15,np.float32);exe=np.full(shape,safe*15,np.float32)
    intervention=float(np.linalg.norm(nom-exe,axis=-1).sum()/15)
    world=SimpleNamespace(buffers={
        "candidates":wp.array(nom,dtype=wp.float32),
        "v_safe":wp.array(exe,dtype=wp.float32),
        "velocities":wp.array(exe,dtype=wp.float32),
        "failed":wp.zeros(1,dtype=wp.uint8),
        "intervention":wp.array([intervention],dtype=wp.float32)})
    params=SimpleNamespace(v_max_mps=15.,w_blocked_motion=.2)
    replay=TensorReplayBuffer(64,(6,21),126,continuous_dim=2)
    action=Action(torch.ones(shape),torch.zeros((1,6),dtype=torch.long))
    obs=torch.zeros(1,6,21);obs[...,9]=1.
    args=(replay,obs,torch.zeros(1,126),action,
          torch.tensor([10.]),torch.zeros(1,6,21),torch.zeros(1,126),
          torch.tensor([float(terminated)]),world,params,"cpu")
    return replay,args
def test_dual_does_not_classify_plain_acceleration_limit_as_safety_intervention():
    r,args=setup_store(1/3,1/3)
    added=module()._store_transitions(*args,"dual",1.)
    assert added==0,"ordinary actuator acceleration clipping must not add unsafe-action tuples"
    assert len(r)==1
def test_executed_label_uses_committed_motion_on_successful_terminal_step():
    r,args=setup_store(1/3,1/3,terminated=1)
    module()._store_transitions(*args,"executed",1.)
    torch.testing.assert_close(r.data["continuous"][0],torch.full((6,2),1/3))
def test_dual_penalty_is_geometric_correction_and_safe_reward_removes_old_intervention_cost():
    r,args=setup_store(1/3,0)
    added=module()._store_transitions(*args,"dual",1.)
    assert added==1 and len(r)==2
    expected_cost=6*2*(1/3)**2
    torch.testing.assert_close(r.data["reward"][1],torch.tensor(10.-expected_cost))
    old_intervention=6*np.sqrt(2)/3
    torch.testing.assert_close(r.data["reward"][0],torch.tensor(10.+.2*old_intervention,dtype=torch.float32))
def argv(tmp_path,**kw):
    opts=dict(algorithm="matd3",envs=2,steps=4,horizon=2,batch_size=4,hidden=8,
              learning_starts=4,replay_capacity=128,log_interval=3,
              world_device="cpu",learner_device="cpu")
    opts.update(kw);out=tmp_path/"result.json"
    args=["run_v81_real_rollout.py","--allow-experimental-physics","--output",str(out)]
    for k,v in opts.items():args+=["--"+k.replace("_","-"),str(v)]
    return args,out
def test_sparse_progress_preserves_all_episode_counts(tmp_path,monkeypatch):
    m=module();args,out=argv(tmp_path);monkeypatch.setattr(sys,"argv",args)
    assert m.main()==0
    d=json.loads(out.read_text())
    rows=[json.loads(s) for s in out.with_suffix(".progress.jsonl").read_text().splitlines()]
    assert sum(r["termination_reason_counts"]["horizon_truncated"] for r in rows)==d["termination_reason_counts"]["horizon_truncated"]==4
def test_exception_retains_completed_episode_counts(tmp_path,monkeypatch):
    m=module();args,out=argv(tmp_path)
    check=m.require_no_warp_overflow
    def injected(w,step):
        check(w,step)
        if step==3:raise RuntimeError("injected overflow boundary for diagnostics")
    monkeypatch.setattr(m,"require_no_warp_overflow",injected)
    monkeypatch.setattr(sys,"argv",args)
    assert m.main()==1
    d=json.loads(out.read_text())
    assert d["termination_reason_counts"]["horizon_truncated"]==2
    assert d["world_steps"]==2
def test_warmup_counts_real_world_transitions_even_with_extra_replay_rows(tmp_path,monkeypatch):
    m=module();args,out=argv(tmp_path,steps=3,horizon=900,log_interval=1)
    orig=m._store_transitions
    def duplicate(*a,**k):
        orig(*a,**k);return orig(*a,**k)
    monkeypatch.setattr(m,"_store_transitions",duplicate)
    monkeypatch.setattr(sys,"argv",args)
    assert m.main()==0
    d=json.loads(out.read_text())
    assert d["updates"]==2,"synthetic rows must not advance the real-experience warmup clock"
