"""A study is a controlled matrix with honest, seed-level final summaries."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from uav_network.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/comparison-20k-hybrid75-1kb.yaml"


def flatten(data, prefix=""):
    out={}
    for key,value in data.items():
        name=prefix+key
        if isinstance(value,dict):out.update(flatten(value,name+"."))
        else:out[name]=value
    return out


def test_canonical_easier_profile_and_historical_profile_are_distinct():
    c = load_config(PROFILE)
    assert (c.world.num_uavs,c.world.map_size_m,c.world.num_targets,c.world.num_obstacles)==(6,1000,20,15)
    assert c.world.max_steps==1200
    assert "steps1200" in c.tracking.tags and "steps1800" not in c.tracking.tags
    assert c.network.power_control=="link_budget"
    assert c.network.link_budget_margin_db==3.
    assert c.network.buffer_bytes==3072 and c.network.report_ttl_s==300
    assert c.energy.enabled and c.network.enabled
    assert c.tracking.evaluation_interval_steps==3686400
    assert c.tracking.periodic_evaluation_seed==1000000044
    assert c.runtime.eval_seed==2000000044 and c.runtime.eval_episodes==30
    assert c.runtime.amp_mode=="off"
    old=load_config(ROOT/"configs/comparison-20k-hybrid75-1kb-35t25o.yaml")
    assert (old.world.num_targets,old.world.num_obstacles)==(35,25)
    assert old.world.max_steps==1800
    assert old.network.power_control=="learned"
    assert old.tracking.evaluation_interval_steps==0


def test_default_matrix_is_84_cases_with_independent_seed_banks(tmp_path):
    from uav_network.experiments.study import plan_study
    manifest=plan_study(PROFILE,tmp_path/"study")
    assert len(manifest["cases"])==84
    assert len({c["id"] for c in manifest["cases"]})==84
    assert manifest["training_seeds"]==[44,100044,200044]
    assert manifest["submitted_cloud_jobs"]==0
    for case in manifest["cases"]:
        c=load_config(tmp_path/"study"/case["config_path"])
        assert c.runtime.seed==case["seed"]
        assert c.runtime.algorithm==case["algorithm"]
        assert c.pql.enabled==(case["mode"]=="pql")
        assert c.runtime.amp_mode=="off"
        assert "20t15o" in c.tracking.run_name
        assert str(c.runtime.seed) in c.tracking.run_name
        assert c.tracking.evaluation_interval_steps>0


def test_each_variant_changes_only_one_scientific_factor(tmp_path):
    from uav_network.experiments.study import plan_study
    m=plan_study(PROFILE,tmp_path/"study",seeds=[44],algorithms=["masac"],modes=["base"])
    expected={"learned_power":"network.power_control","buffer16":"network.buffer_bytes",
              "buffer32":"network.buffer_bytes","ttl600":"network.report_ttl_s",
              "coverage8":"sensor.coverage_grid_cells","battery2x":"energy.battery_capacity_j"}
    science={}
    for case in m["cases"]:
        c=load_config(tmp_path/"study"/case["config_path"]).to_dict()
        science[case["variant"]]=flatten({k:c[k] for k in ["world","sensor","network","energy","reward"]})
    base=science["baseline"]
    for name,key in expected.items():
        changed={k for k,v in science[name].items() if base[k]!=v}
        assert changed=={key}
    assert science["buffer16"]["network.buffer_bytes"]==16384
    assert science["buffer32"]["network.buffer_bytes"]==32768
    assert science["ttl600"]["network.report_ttl_s"]==600
    assert science["coverage8"]["sensor.coverage_grid_cells"]==8
    assert science["battery2x"]["energy.battery_capacity_j"]==554400


@pytest.mark.parametrize("seeds",[[44,44],[44,45],[2000000044],[1000000044]])
def test_study_rejects_duplicate_or_overlapping_seed_banks(tmp_path,seeds):
    from uav_network.experiments.study import plan_study
    with pytest.raises(ValueError,match="seed"):
        plan_study(PROFILE,tmp_path/"study",seeds=seeds)


def test_plan_rejects_fp16_and_preserves_existing_output(tmp_path):
    from uav_network.experiments.study import plan_study
    with pytest.raises(ValueError,match="FP32"):
        plan_study(PROFILE,tmp_path/"bad",overrides=["runtime.amp_mode=fp16"])
    dest=tmp_path/"existing";dest.mkdir();(dest/"keep").write_text("unchanged")
    with pytest.raises(FileExistsError):
        plan_study(PROFILE,dest)
    assert (dest/"keep").read_text()=="unchanged"


def test_real_plan_cli_materializes_configs_without_training(tmp_path):
    dest=tmp_path/"cli"
    p=subprocess.run([sys.executable,"-m","uav_network","study","plan","--config",str(PROFILE),
                      "--output-dir",str(dest),"--variants","baseline","--seeds","44",
                      "--algorithms","matd3","--modes","pql"],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    m=json.loads((dest/"manifest.json").read_text())
    assert len(m["cases"])==1
    assert not (dest/m["cases"][0]["train_output"]).exists()


def test_tiny_real_training_and_final_evaluation_are_summarized(tmp_path):
    from uav_network.experiments.study import plan_study,run_study,summarize_study
    dest=tmp_path/"tiny"
    m=plan_study(PROFILE,dest,seeds=[44,100044],variants=["baseline"],
                 algorithms=["matd3"],modes=["base"],overrides=[
                     "runtime.device=cpu","runtime.replay_device=cpu","runtime.replay_cpu_fraction=0",
                     "runtime.num_envs=1","runtime.episodes=1","world.max_steps=2",
                     "runtime.eval_episodes=2","runtime.compile_models=false","tracking.enabled=false",
                     "tracking.evaluation_interval_steps=0","runtime.max_train_seconds=0",
                     "matd3.hidden_sizes=[8,8]","matd3.batch_size=2","matd3.replay_capacity=16",
                     "matd3.learning_starts=1","matd3.updates_per_step=1"])
    path=dest/"manifest.json"
    result=run_study(path)
    assert len(result["cases"])==2
    assert all(row["status"]=="completed" for row in result["cases"]),result
    summary=summarize_study(path)
    assert not summary["missing_cases"] and not summary["rejected_cases"]
    assert summary["groups"][0]["training_seed_count"]==2
    assert summary["groups"][0]["training_seeds"]==[44,100044]
    assert summary["groups"][0]["evaluation_episodes_per_seed"]==[2,2]
    case=m["cases"][0]
    train_path=dest/case["train_output"]/"summary.json"
    train=json.loads(train_path.read_text());train["episode_budget_complete"]=False
    train_path.write_text(json.dumps(train))
    summary=summarize_study(path)
    assert summary["groups"][0]["training_seed_count"]==1
    assert summary["rejected_cases"][0]["reason"]=="episode_budget_incomplete"


def test_summary_reports_seed_level_variance_and_rejects_wrong_provenance(tmp_path):
    from uav_network.experiments.study import plan_study,summarize_study
    from uav_network.experiments.comparison_metrics import evaluation_protocol
    dest=tmp_path/"summary"
    m=plan_study(PROFILE,dest,seeds=[44,100044],variants=["baseline"],
                 algorithms=["masac"],modes=["base"],overrides=["masac.hidden_sizes=[8,8]"])
    for case,value,step in zip(m["cases"],[10.,30.],[100,200]):
        c=load_config(dest/case["config_path"])
        tp=dest/case["train_output"];ep=dest/case["evaluation_output"]
        tp.mkdir(parents=True);ep.mkdir(parents=True)
        (tp/"resolved_config.yaml").write_text((dest/case["config_path"]).read_text())
        from uav_network.algorithms.checkpoints import make_learner,save_policy
        from uav_network.environment.observations import observation_dim,state_dim
        from uav_network.experiments.checkpoint_store import file_sha256
        model=make_learner(c,observation_dim(c),state_dim(c))
        checkpoint=tp/"final.pt"
        save_policy(model,checkpoint)
        (tp/"summary.json").write_text(json.dumps({
            "checkpoint":str(checkpoint),"episode_budget_complete":True,"episodes":20000,"global_step":step}))
        seeds=list(range(c.runtime.eval_seed,c.runtime.eval_seed+30))
        protocol=evaluation_protocol(c,seeds,{"next_seed":case["seed"]+24096,
                                               "global_step":step,"episodes":20000})
        fixture={"algorithm":"masac","episodes":30,"protocol":protocol,
                 "policy_provenance":{"checkpoint_sha256":file_sha256(checkpoint),
                     "config_sha256":case["config_hash"],"algorithm":"masac",
                     "training_mode":"base","amp_mode":"off"},
                 "mean_return":value,"mean_search_rate_percent":value,
                 "mean_delivery_rate_percent":value,"success_rate":0.,
                 "records":[{"seed":s,"return":value} for s in seeds]}
        (ep/"metrics.json").write_text(json.dumps(fixture))
    result=summarize_study(dest/"manifest.json")
    group=result["groups"][0]
    assert group["training_seed_count"]==2
    assert group["metrics"]["mean_return"]["mean"]==20.
    assert group["metrics"]["mean_return"]["std_over_training_seeds"]==pytest.approx(14.142135623730951)
    assert group["equal_environment_interaction_budget"] is False
    case=m["cases"][1];path=dest/case["evaluation_output"]/"metrics.json"
    data=json.loads(path.read_text());data["protocol"]["training_seed"]=44
    path.write_text(json.dumps(data))
    result=summarize_study(dest/"manifest.json")
    assert result["groups"][0]["training_seed_count"]==1
    assert result["groups"][0]["metrics"]["mean_return"]["std_over_training_seeds"] is None
    assert result["rejected_cases"][0]["reason"]=="invalid_provenance_or_config"


def test_new_profile_periodic_evaluator_completes_a_real_child(tmp_path):
    import time
    from uav_network.algorithms.checkpoints import make_learner
    from uav_network.environment.observations import observation_dim,state_dim
    from uav_network.experiments.async_evaluation import AsyncCheckpointEvaluator
    from uav_network.experiments.diagnostics import TrainingDiagnostics
    c=load_config(PROFILE,["runtime.device=cpu","runtime.num_envs=1",
                          "runtime.compile_models=false","tracking.enabled=false",
                          "world.max_steps=2","tracking.evaluation_interval_steps=2",
                          "tracking.periodic_evaluation_episodes=1",
                          "masac.hidden_sizes=[8,8]","sensor.coverage_grid_cells=8"])
    learner=make_learner(c,observation_dim(c),state_dim(c))
    manager=AsyncCheckpointEvaluator(c,tmp_path,None,TrainingDiagnostics(c,tmp_path))
    try:
        progress={"episodes":0,"global_step":2,"collection_steps":2,"next_seed":45}
        manager.tick(2,0,lambda:(learner.actors.state_dict(),0),progress)
        deadline=time.monotonic()+30
        while manager.active is not None and time.monotonic()<deadline:
            manager.poll(2);time.sleep(.02)
        assert manager.active is None
        assert manager.counts["completed"]==1,manager.counts
        result=manager.last_result
        assert result["episodes"]==1
        assert result["protocol"]["scenario"]["world"]["num_targets"]==20
        assert result["protocol"]["scenario"]["network"]["power_control"]=="link_budget"
        assert result["protocol"]["scenario"]["sensor"]["coverage_grid_cells"]==8
    finally:
        manager.close(2)


def test_case_config_fingerprint_detects_learner_parameter_drift(tmp_path):
    import yaml
    from uav_network.experiments.study import plan_study,summarize_study
    dest=tmp_path/"drift"
    m=plan_study(PROFILE,dest,seeds=[44],variants=["baseline"],algorithms=["masac"],modes=["base"])
    path=dest/m["cases"][0]["config_path"]
    data=yaml.safe_load(path.read_text());data["masac"]["learning_rate"] = .001
    path.write_text(yaml.safe_dump(data))
    summary=summarize_study(dest/"manifest.json")
    assert not summary["missing_cases"]
    assert summary["rejected_cases"][0]["reason"]=="invalid_provenance_or_config"


def test_final_evaluation_cannot_be_reused_for_another_training_mode(tmp_path):
    from uav_network.experiments.study import plan_study,run_study,summarize_study
    dest=tmp_path/"mode_swap"
    m=plan_study(PROFILE,dest,seeds=[44],variants=["baseline"],algorithms=["masac"],
                 modes=["base","pql"],overrides=[
                     "runtime.device=cpu","runtime.replay_device=cpu","runtime.replay_cpu_fraction=0",
                     "runtime.num_envs=1","runtime.episodes=1","world.max_steps=2",
                     "runtime.eval_episodes=2","runtime.compile_models=false","tracking.enabled=false",
                     "tracking.evaluation_interval_steps=0","runtime.max_train_seconds=0",
                     "masac.hidden_sizes=[8,8]","masac.batch_size=2","masac.replay_capacity=16",
                     "masac.learning_starts=1","masac.updates_per_step=1",
                     "pql.critic_sample_ratio=1","pql.warmup_collection_steps=1"])
    result=run_study(dest/"manifest.json")
    assert all(x["status"]=="completed" for x in result["cases"]),result
    paths=[dest/case["evaluation_output"]/"metrics.json" for case in m["cases"]]
    paths[1].write_bytes(paths[0].read_bytes())
    summary=summarize_study(dest/"manifest.json")
    assert summary["complete"] is False
    assert summary["rejected_cases"][0]["id"]==m["cases"][1]["id"]
    assert summary["rejected_cases"][0]["reason"]=="invalid_provenance_or_config"


def test_summary_exposes_the_actual_protocol_even_before_training(tmp_path):
    from uav_network.experiments.study import plan_study,summarize_study
    dest=tmp_path/"protocol"
    m=plan_study(PROFILE,dest,seeds=[44],variants=["baseline"],algorithms=["masac"],modes=["base"])
    summary=summarize_study(dest/"manifest.json")
    protocol=summary["groups"][0]["protocol"]
    assert protocol["scenario_hash"]==m["cases"][0]["scenario_hash"]
    assert protocol["amp_mode"]=="off"
    assert protocol["requested_training_episodes"]==20000
    assert protocol["max_steps"]==1200
    assert protocol["num_targets"]==20 and protocol["num_obstacles"]==15
    assert protocol["evaluation_episodes"]==30
