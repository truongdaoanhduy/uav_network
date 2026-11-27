"""Behavioral V82 regressions: reject ambiguous configuration and impossible warmup."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import json
import os
import numpy as np
import pytest
import torch
from uav_network.algorithms.configuration.schema import LearnerConfig
from uav_network.algorithms.configuration.loader import load_learner_config
from uav_network.algorithms.core.models import DeterministicActor, TwinCritic
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("algorithm", ["matd3", "masac"])
def test_rejects_conflicting_critic_normalizers(algorithm):
    cfg = LearnerConfig()
    hp = getattr(cfg, algorithm)
    hp.critic_layer_norm = hp.critic_batch_norm = True
    with pytest.raises(ValueError, match="normal"):
        cfg.validate()

@pytest.mark.parametrize("algorithm", ["matd3", "masac"])
@pytest.mark.parametrize("key", ["critic_layer_norm", "sync_critic_loss_scale"])
def test_checkpoint_bool_fields_are_typed(algorithm, key):
    data = LearnerConfig().to_dict()
    data[algorithm][key] = "false"
    with pytest.raises(ValueError, match="boolean"):
        LearnerConfig.from_dict(data)

def runner_module():
    spec = importlib.util.spec_from_file_location("v81_runner", ROOT/"scripts/gpu/run_v81_real_rollout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_runner_rejects_unreachable_learning_threshold(tmp_path):
    path = tmp_path/"bad.json"
    cmd = [sys.executable, str(ROOT/"scripts/gpu/run_v81_real_rollout.py"),
           "--algorithm", "matd3", "--envs", "1", "--steps", "1",
           "--learning-starts", "10000", "--replay-capacity", "512",
           "--world-device", "cpu", "--learner-device", "cpu",
           "--allow-experimental-physics", "--output", str(path)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0, p.stdout
    assert "learning_starts" in p.stdout+p.stderr or "learning-starts" in p.stdout+p.stderr

def test_pretanh_penalty_excludes_routing_and_reaches_saturated_motion():
    actor = DeterministicActor(7, 3, [8, 8], continuous_dim=2, activation="silu")
    with torch.no_grad():
        actor.continuous.weight.zero_()
        actor.continuous.bias.fill_(10)
    penalty = actor.pre_tanh_penalty(torch.zeros(4, 7)).mean()
    penalty.backward()
    assert actor.destination.weight.grad is None
    torch.testing.assert_close(actor.continuous.bias.grad, torch.full((2,), 10.))
    assert penalty.item() == 100.

@pytest.mark.parametrize("activation", ["relu", "silu"])
def test_layernorm_critic_broadcast_equals_dense(activation):
    torch.manual_seed(8)
    c = TwinCritic(9, 2, 3, [12, 10], continuous_dim=2,
                   layer_norm=True, activation=activation)
    s, j = torch.randn(5, 9), torch.randn(3, 5, 2, 5)
    optimized = c.forward_shared_state(s, j)
    dense = [torch.stack([c(s,j[k])[head] for k in range(3)]) for head in range(2)]
    for x,y in zip(optimized,dense):
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        assert torch.isfinite(x).all()

@pytest.mark.parametrize("dead_first", [True, False])
def test_live_dead_pair_collision_fails_closed(dead_first):
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
    device = os.getenv("V82_TEST_DEVICE", "cpu")
    p=GPUWorldV80Params.from_yaml(ROOT/"research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml",
                                envs=1,device=device)
    w=WarpParallelMissionV80(p)
    pos=w.buffers["positions"].numpy().copy()
    pos[0,:6,:2]=np.array([[100,100],[125,100],[400,100],[600,100],[800,100],[900,100]])
    vel=w.buffers["velocities"].numpy().copy()
    vel[:]=0
    bat=np.zeros((1,6),np.float32)
    bat[0,1 if dead_first else 0]=100000.
    wp.copy(w.buffers["positions"],wp.array(pos,dtype=wp.float32,device=device))
    wp.copy(w.buffers["velocities"],wp.array(vel,dtype=wp.float32,device=device))
    wp.copy(w.buffers["battery"],wp.array(bat,dtype=wp.float32,device=device))
    w.step(wp.zeros((1,6,2),dtype=wp.float32,device=device),wp.zeros((1,6),dtype=wp.int32,device=device))
    assert int(w.buffers["failed"].numpy()[0]) == 1
    assert int(w.buffers["safety_reason"].numpy()[0]) & 4
    np.testing.assert_array_equal(w.buffers["positions"].numpy(),pos)
    np.testing.assert_array_equal(w.buffers["velocities"].numpy(),vel)

@pytest.mark.parametrize("algorithm", ["matd3", "masac"])
@pytest.mark.parametrize("shared", [False, True])
def test_stability_controls_execute_real_updates_and_checkpoint_roundtrip(algorithm, shared, tmp_path):
    from uav_network.algorithms.matd3 import MATD3
    from uav_network.algorithms.masac import MASAC
    torch.set_num_threads(1)
    torch.manual_seed(44)
    cfg=load_learner_config(ROOT/"configs/learners/v81_matd3_masac_stable.yaml")
    cfg.runtime.algorithm=algorithm
    hp=getattr(cfg,algorithm)
    hp.hidden_sizes=[16,16]
    hp.batch_size=4
    hp.share_actor_parameters=hp.share_critic_parameters=shared
    cls=MATD3 if algorithm=="matd3" else MASAC
    learner=cls(cfg,21,126)
    obs=torch.randn(4,6,21)*.1
    obs[:,:,9]=1;obs[:,:,-3:]=1
    batch=dict(observation=obs,next_observation=obs.clone(),
               state=torch.randn(4,126),next_state=torch.randn(4,126),
               continuous=torch.randn(4,6,2).tanh(),destination=torch.zeros(4,6,dtype=torch.long),
               reward=torch.tensor([-100.,0.,25.,-100.]),terminated=torch.tensor([1.,0.,0.,1.]))
    before=[p.detach().clone() for p in learner.actors.parameters()]
    for _ in range(2):
        metrics=learner.update(batch)
        assert all(np.isfinite(float(v)) for v in metrics.values())
    assert any(not torch.equal(a,b) for a,b in zip(before,learner.actors.parameters()))
    path=tmp_path/"model.pt";learner.save(path)
    restored=cls.load(path)
    for a,b in zip(learner.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)

def test_runner_log_interval_preserves_updates_and_final_record(tmp_path):
    common=[sys.executable,str(ROOT/"scripts/gpu/run_v81_real_rollout.py"),
            "--algorithm","matd3","--envs","2","--steps","3","--horizon","900",
            "--batch-size","4","--hidden","16","--learning-starts","4","--replay-capacity","128",
            "--world-device","cpu","--learner-device","cpu","--allow-experimental-physics"]
    results=[]
    for interval in (1,2):
        path=tmp_path/f"interval_{interval}.json"
        p=subprocess.run(common+["--log-interval",str(interval),"--output",str(path)],
                         cwd=ROOT,capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
        data=json.loads(path.read_text())
        rows=[json.loads(x) for x in path.with_suffix(".progress.jsonl").read_text().splitlines()]
        assert [r["vector_step"] for r in rows]==([1,2,3] if interval==1 else [2,3])
        assert data["overflow_checked_steps"]==3
        results.append(data)
    assert results[0]["updates"]==results[1]["updates"]==2
    assert results[0]["final_record"]==results[1]["final_record"]
