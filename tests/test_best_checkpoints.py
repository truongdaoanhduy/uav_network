"""A later policy or continuation must not overwrite an evaluated better actor."""
from copy import deepcopy
from pathlib import Path
import json

import pytest
import torch

from uav_network.configuration import Config, config_fingerprint
from uav_network.algorithms.checkpoints import make_learner
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.experiments.checkpoint_store import file_sha256
from uav_network.experiments.comparison_metrics import evaluation_protocol
from uav_network.experiments.evaluation_snapshots import save_evaluation_snapshot


def config():
    c=Config()
    c.world.num_uavs,c.world.num_targets,c.world.num_obstacles=2,2,0
    c.world.max_steps=4
    c.network.enabled=False
    c.runtime.device=c.runtime.replay_device="cpu"
    c.runtime.num_envs=1
    c.runtime.episodes=1
    c.runtime.compile_models=False
    c.runtime.max_train_seconds=0
    c.runtime.eval_seed=2000000044
    c.runtime.eval_episodes=2
    c.tracking.enabled=False
    c.tracking.evaluation_interval_steps=2
    c.tracking.periodic_evaluation_seed=1000000044
    c.tracking.periodic_evaluation_episodes=2
    for a in (c.masac,c.matd3):
        a.hidden_sizes,a.batch_size,a.replay_capacity,a.learning_starts=[8,8],2,16,1
    return c


def candidate(c,folder,step=2,score=(0.,80.,90.,100.)):
    model=make_learner(c,observation_dim(c),state_dim(c))
    progress=dict(episodes=0,global_step=step,collection_steps=step,next_seed=45)
    checkpoint=save_evaluation_snapshot(c,folder,model.actors.state_dict(),step,progress)
    seeds=list(range(c.tracking.periodic_evaluation_seed,
                     c.tracking.periodic_evaluation_seed+c.tracking.periodic_evaluation_episodes))
    result=dict(zip(("success_rate","mean_delivery_rate_percent","mean_search_rate_percent","mean_return"),score))
    result.update(algorithm=c.runtime.algorithm,episodes=len(seeds),
                  protocol=evaluation_protocol(c,seeds,progress),
                  records=[{"seed":s} for s in seeds],
                  policy_provenance={"checkpoint_sha256":file_sha256(checkpoint),
                                    "config_sha256":config_fingerprint(c),"algorithm":c.runtime.algorithm,
                                    "training_mode":"pql" if c.pql.enabled else "base",
                                    "amp_mode":c.runtime.amp_mode})
    source=dict(kind="periodic_validation",checkpoint=str(checkpoint),
                checkpoint_sha256=file_sha256(checkpoint),global_step=step,episodes=0,policy_version=step,
                evaluation_episodes=len(seeds),seed_first=seeds[0],seed_last=seeds[-1])
    return result,source


def test_best_is_the_evaluated_earlier_actor_and_equal_score_keeps_it(tmp_path):
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();selection=BestCheckpoint(c,tmp_path)
    good,source=candidate(c,tmp_path/"snapshots",2,(0.,80.,90.,100.))
    assert selection.consider(good,source)
    original=Path(source["checkpoint"]).read_bytes()
    for step,score in [(3,(0.,70.,100.,10000.)),(4,(0.,80.,90.,100.))]:
        result,later=candidate(c,tmp_path/"snapshots",step,score)
        assert not selection.consider(result,later)
    assert (tmp_path/"best.pt").read_bytes()==original
    selected=selection.result(tmp_path/"final.pt")
    assert selected["checkpoint"]==str(tmp_path/"best.pt")
    assert selected["checkpoint_selection"]["global_step"]==2
    assert selected["checkpoint_selection"]["kind"]=="validation_best"


def test_completion_precedes_delivery_and_delivery_precedes_search_and_return(tmp_path):
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();selection=BestCheckpoint(c,tmp_path)
    for step,score in [(2,(0.,90.,99.,9000.)),(3,(.5,80.,80.,1.)),(4,(.5,81.,81.,0.))]:
        result,source=candidate(c,tmp_path/"snapshots",step,score)
        assert selection.consider(result,source)
    assert selection.result(tmp_path/"final.pt")["checkpoint_selection"]["global_step"]==4


@pytest.mark.parametrize("corruption",["policy_sha","config_sha","step","seeds","overlap","nan"])
def test_wrong_policy_or_validation_protocol_cannot_win(tmp_path,corruption):
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();selection=BestCheckpoint(c,tmp_path)
    result,source=candidate(c,tmp_path/"snapshots")
    if corruption=="policy_sha":result["policy_provenance"]["checkpoint_sha256"]="0"*64
    if corruption=="config_sha":result["policy_provenance"]["config_sha256"]="0"*64
    if corruption=="step":result["protocol"]["training_global_step"]=999
    if corruption=="seeds":result["records"][0]["seed"]=c.runtime.eval_seed
    if corruption=="overlap":result["protocol"]["training_seed_overlap"]=True
    if corruption=="nan":result["mean_return"]=float("nan")
    with pytest.raises(ValueError):selection.consider(result,source)
    assert not (tmp_path/"best.pt").exists()


def test_legacy_test_bank_is_never_used_for_selection_and_fallback_is_honest(tmp_path):
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();selection=BestCheckpoint(c,tmp_path)
    result,source=candidate(c,tmp_path/"snapshots")
    source["kind"]="legacy_validation"
    assert not selection.consider(result,source)
    assert selection.state_dict() is None
    fallback=selection.result(tmp_path/"final.pt")
    assert fallback["checkpoint_selection"]["kind"]=="unvalidated_final_fallback"
    assert fallback["checkpoint"]==str(tmp_path/"final.pt")


def test_resume_in_a_new_directory_preserves_best_bytes_and_parent_rng(tmp_path):
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();selection=BestCheckpoint(c,tmp_path/"first")
    result,source=candidate(c,tmp_path/"snapshots")
    selection.consider(result,source)
    state=selection.state_dict()
    resumed=deepcopy(c);resumed.runtime.output_dir=str(tmp_path/"second")
    resumed.runtime.warm_start_checkpoint=str(tmp_path/"resume_latest.pt")
    before=torch.get_rng_state().clone()
    new=BestCheckpoint(resumed,tmp_path/"second",state=state)
    assert torch.equal(before,torch.get_rng_state())
    assert (tmp_path/"second"/"best.pt").read_bytes()==Path(source["checkpoint"]).read_bytes()
    assert new.result("last.pt")["checkpoint_selection"]["global_step"]==2
    bad=deepcopy(state);bad["checkpoint_bytes"][0]^=1
    with pytest.raises(ValueError):BestCheckpoint(resumed,tmp_path/"corrupt",state=bad)
    changed=deepcopy(resumed);changed.network.power_control="link_budget"
    with pytest.raises(ValueError):BestCheckpoint(changed,tmp_path/"changed",state=state)


@pytest.mark.parametrize("algorithm,pql",[("masac",False),("matd3",False),("masac",True),("matd3",True)])
def test_real_training_selects_validated_snapshot_and_carries_it_in_full_resume(tmp_path,algorithm,pql):
    from uav_network.experiments.training import train
    from uav_network.experiments.evaluation import load_evaluation_policy,evaluate
    c=config();c.runtime.algorithm=algorithm;c.pql.enabled=pql
    c.runtime.output_dir=str(tmp_path/"train")
    c.pql.mode="synchronous";c.pql.critic_sample_ratio=1;c.pql.warmup_collection_steps=0
    c.pql.collector_device=c.pql.learner_device=c.pql.policy_device="cpu"
    result=train(c)
    assert Path(result["checkpoint"]).name=="best.pt",result
    selection=result["checkpoint_selection"]
    assert selection["kind"]=="validation_best"
    assert selection["global_step"]<=result["global_step"]
    model,progress,_=load_evaluation_policy(result["checkpoint"])
    assert progress["global_step"]==selection["global_step"]
    full=torch.load(result["resume_checkpoint"],map_location="cpu",weights_only=True)
    assert full["training"]["best_checkpoint"] is not None
    restored=tmp_path/"restored"
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    manager=BestCheckpoint(c,restored,state=full["training"]["best_checkpoint"])
    assert file_sha256(restored/"best.pt")==file_sha256(result["checkpoint"])
    final=evaluate(result["checkpoint"],episodes=2,output_dir=tmp_path/"eval",seed=c.runtime.eval_seed)
    assert final["protocol"]["training_global_step"]==selection["global_step"]
    c.runtime.episodes=2
    c.runtime.output_dir=str(tmp_path/"continued")
    c.runtime.warm_start_checkpoint=result["resume_checkpoint"]
    c.tracking.evaluation_interval_steps=0
    continued=train(c)
    assert continued["optimizer_replay_restored"] is True
    assert continued["initial_episodes"]==1
    assert continued["checkpoint_selection"]["global_step"]==selection["global_step"]
    assert file_sha256(continued["checkpoint"])==file_sha256(result["checkpoint"])


def test_final_test_bank_cannot_select_best_even_if_labeled_periodic(tmp_path):
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();c.runtime.eval_seed=c.tracking.periodic_evaluation_seed
    selection=BestCheckpoint(c,tmp_path)
    result,source=candidate(c,tmp_path/"snapshots")
    with pytest.raises(ValueError):selection.consider(result,source)


@pytest.mark.parametrize("damage",["overlap","unbounded_score","missing_metrics","inconsistent_metrics"])
def test_restore_rejects_invalid_winner_protocol_or_ranking(tmp_path,damage):
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();manager=BestCheckpoint(c,tmp_path/"first")
    result,source=candidate(c,tmp_path/"snapshots")
    manager.consider(result,source);state=manager.state_dict()
    if damage=="overlap":c.runtime.eval_seed=c.tracking.periodic_evaluation_seed
    if damage=="unbounded_score":state["metadata"]["score"][0]=999.
    if damage=="missing_metrics":del state["metadata"]["metrics"]
    if damage=="inconsistent_metrics":state["metadata"]["metrics"]["mean_return"]=999.
    with pytest.raises(ValueError):BestCheckpoint(c,tmp_path/"restored",state=state)


def test_result_repairs_missing_best_metadata_without_changing_winner(tmp_path):
    from uav_network.experiments.best_checkpoints import BestCheckpoint
    c=config();manager=BestCheckpoint(c,tmp_path)
    result,source=candidate(c,tmp_path/"snapshots")
    manager.consider(result,source);original=(tmp_path/"best.pt").read_bytes()
    (tmp_path/"best.json").unlink()
    selected=manager.result("last.pt")
    assert json.loads((tmp_path/"best.json").read_text())==selected["checkpoint_selection"]
    assert (tmp_path/"best.pt").read_bytes()==original


@pytest.mark.parametrize("algorithm",["masac","matd3"])
def test_manual_base_warm_start_from_best_actors_resets_training_and_critics(tmp_path,algorithm):
    from uav_network.experiments.continuation import restore_learner
    c=config();c.runtime.algorithm=algorithm
    result,source=candidate(c,tmp_path/"snapshots")
    saved=torch.load(source["checkpoint"],weights_only=True)
    c.runtime.warm_start_checkpoint=source["checkpoint"]
    torch.manual_seed(777)
    fresh=make_learner(c,observation_dim(c),state_dim(c))
    torch.manual_seed(777)
    learner,progress=restore_learner(c,"cpu")
    assert progress["state_restore"]=="actor_only_warm_start_fresh_training"
    assert progress["episodes"]==progress["global_step"]==progress["collection_steps"]==0
    assert progress["records"]==[] and "_training" not in progress
    assert learner.updates==learner.actor_updates==0
    for name,value in learner.actors.state_dict().items():
        torch.testing.assert_close(value,saved["actors"][name],rtol=0,atol=0)
    for name,value in fresh.state_dict().items():
        if not name.startswith(("actors.","target_actors.")):
            torch.testing.assert_close(learner.state_dict()[name],value,rtol=0,atol=0)
    if hasattr(learner,"target_actors"):
        for name,value in learner.target_actors.state_dict().items():
            torch.testing.assert_close(value,saved["actors"][name],rtol=0,atol=0)
