"""Small real XYZ BASE/PQL training and CPU fixed-policy evaluation."""
from pathlib import Path
import numpy as np
import pytest
import torch
from test_xyz_sensing import xyz_config
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate


@pytest.mark.parametrize("algorithm",["masac","matd3"])
@pytest.mark.parametrize("mode",["base","synchronous","async"])
@pytest.mark.parametrize("shared",[False,True])
def test_xyz_train_eval_checkpoint_roundtrip(tmp_path,algorithm,mode,shared):
    torch.set_num_threads(1)
    c=xyz_config(n=2,targets=2);c.runtime.algorithm=algorithm
    c.runtime.device=c.runtime.replay_device="cpu";c.runtime.environment_backend="warp_packet"
    c.runtime.num_envs=2;c.runtime.episodes=2;c.world.max_steps=4
    c.runtime.output_dir=str(tmp_path/"train");c.runtime.checkpoint_interval=2
    c.runtime.eval_episodes=1;c.runtime.eval_seed=2000000044
    c.tracking.enabled=False;c.tracking.evaluation_interval_steps=0
    a=c.algorithm_config;a.hidden_sizes=[8,8];a.batch_size=2;a.replay_capacity=16
    a.learning_starts=1;a.updates_per_step=1
    a.share_actor_parameters=a.share_critic_parameters=shared
    c.masac.target_continuous_entropy=-4
    c.pql.enabled=mode!="base";c.pql.mode="async" if mode=="async" else "synchronous"
    c.pql.collector_device=c.pql.learner_device=c.pql.policy_device="cpu"
    c.pql.critic_sample_ratio=1;c.pql.warmup_collection_steps=1;c.pql.worker_timeout_s=30
    r=train(c)
    assert r["episodes"]==2 and r["updates"]>0
    assert Path(r["resume_checkpoint"]).is_file()
    saved=torch.load(r["resume_checkpoint"],weights_only=True)
    assert saved["observation_version"]==4
    ev=evaluate(r["checkpoint"],episodes=1,seed=2000000044,output_dir=tmp_path/"eval")
    assert ev["episodes"]==1
    assert np.isfinite(ev["mean_return"])
    assert ev["protocol"]["scenario"]["observation_version"]==4


def test_xyz_best_selector_accepts_schema4_snapshot(tmp_path):
    from test_best_checkpoints import config,candidate
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();c.world.motion_dimensions=3;c.world.motion_safety="cbf";c.sensor.model="bayesian"
    result,source=candidate(c,tmp_path/"candidate")
    selector=BestCheckpoint(c,tmp_path/"selection")
    assert selector.consider(result,source)
    assert selector.result(tmp_path/"final.pt")["checkpoint_selection"]["kind"]=="validation_best"

@pytest.mark.parametrize("algorithm",["masac","matd3"])
@pytest.mark.parametrize("pql",[False,True])
def test_xyz_full_resume_matches_uninterrupted_training(tmp_path,algorithm,pql):
    from copy import deepcopy
    torch.set_num_threads(1)
    c=xyz_config(n=2,targets=2)
    c.runtime.algorithm=algorithm;c.runtime.device=c.runtime.replay_device="cpu"
    c.runtime.environment_backend="warp_packet";c.runtime.num_envs=2
    c.runtime.episodes=4;c.world.max_steps=3;c.tracking.enabled=False
    c.runtime.collector_overlap=False  # Serial BASE and synchronous PQL have a fixed interleaving.
    c.tracking.evaluation_interval_steps=0;c.runtime.compile_models=False
    a=c.algorithm_config;a.hidden_sizes=[8,8];a.batch_size=2;a.replay_capacity=16
    a.learning_starts=1;a.updates_per_step=1
    a.share_actor_parameters=a.share_critic_parameters=True
    c.masac.target_continuous_entropy=-4
    c.pql.enabled=pql;c.pql.mode="synchronous";c.pql.critic_sample_ratio=1
    c.pql.warmup_collection_steps=1;c.pql.collector_device=c.pql.learner_device=c.pql.policy_device="cpu"
    c.runtime.output_dir=str(tmp_path/"whole")
    whole=train(c)
    first=deepcopy(c);first.runtime.episodes=2;first.runtime.output_dir=str(tmp_path/"first")
    before=train(first)
    again=deepcopy(c);again.runtime.output_dir=str(tmp_path/"again")
    again.runtime.warm_start_checkpoint=before["resume_checkpoint"]
    resumed=train(again)
    assert resumed["episodes"]==4
    assert resumed["reproducibility"]["model_hash"]==whole["reproducibility"]["model_hash"]
    left=torch.load(whole["resume_checkpoint"],weights_only=True)
    right=torch.load(resumed["resume_checkpoint"],weights_only=True)
    for name,value in left["training"]["environment"]["world"]["arrays"].items():
        assert torch.equal(value,right["training"]["environment"]["world"]["arrays"][name]),name
    # Final-segment metrics are sampled freshly; compare mission outcomes independently.
    clean=lambda rows:[{k:v for k,v in r.items() if k not in {"learner","worker_pid"}} for r in rows]
    assert clean(left["progress"]["records"])==clean(right["progress"]["records"])

@pytest.mark.parametrize("algorithm",["masac","matd3"])
def test_native_xyz_pql_warmup_uses_four_slots(tmp_path,algorithm):
    torch.set_num_threads(1)
    c=xyz_config(n=1,targets=1);c.runtime.algorithm=algorithm
    c.runtime.device=c.runtime.replay_device="cpu";c.runtime.environment_backend="native"
    c.runtime.num_envs=c.runtime.episodes=1;c.world.max_steps=3
    c.runtime.output_dir=str(tmp_path/"native");c.tracking.enabled=False
    c.tracking.evaluation_interval_steps=0
    a=c.algorithm_config;a.hidden_sizes=[8,8];a.batch_size=2;a.replay_capacity=8;a.learning_starts=1
    c.pql.enabled=True;c.pql.mode="synchronous";c.pql.critic_sample_ratio=1
    c.pql.warmup_collection_steps=1
    c.pql.collector_device=c.pql.learner_device=c.pql.policy_device="cpu"
    assert train(c)["updates"]>0
