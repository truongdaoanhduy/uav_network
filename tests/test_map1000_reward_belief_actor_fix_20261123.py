"""RED/GREEN: MAP1000 NoOracle corrected scientific MDP, no truth leakage.

This is a NEW run: never rewrite previous MATD3 checkpoints or YAML.
"""
import math
from pathlib import Path
import pytest

OLD='configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml'
NEW='configs/matd3-nooracle-map1000-tx06-prior0021875-reward02-actorstable-20261123.yaml'


def test_calibrated_new_configuration_does_not_mutate_old_science():
    from uav_network.configuration import load_config
    a,b=load_config(OLD),load_config(NEW)
    assert a.sensor.belief_prior==.5
    assert a.reward.communication_efficiency==1
    assert a.matd3.output_normalization is False
    assert a.matd3.actor_output_regularization==0
    assert b.sensor.belief_prior == pytest.approx(35/1600)
    assert b.reward.communication_efficiency == pytest.approx(.2)
    assert b.matd3.output_normalization is True
    assert b.matd3.actor_output_regularization == pytest.approx(.001)
    assert b.world.motion_safety == a.world.motion_safety == 'reject'
    assert b.sensor.confirmation_mode == a.sensor.confirmation_mode == 'posterior_only'
    assert b.sensor.confirmation_threshold == a.sensor.confirmation_threshold == .99
    assert b.network.tx_power_max_w == a.network.tx_power_max_w == .6
    assert b.runtime.num_envs == a.runtime.num_envs == 4096
    assert b.matd3.batch_size == a.matd3.batch_size == 8192
    assert b.matd3.updates_per_step == a.matd3.updates_per_step == 4
    assert b.world.max_steps==a.world.max_steps==1500
    assert b.runtime.episodes==a.runtime.episodes==20000
    assert b.runtime.warm_start_checkpoint=='none'
    b.validate()


def test_truth_scoring_precision_metrics_and_undefined_empty_case():
    from uav_network.experiments.training_mission_metrics import terminal_paper_metrics
    exemplar={'search_rate_percent':80.,'delivery_rate_percent':50.,'success':False,
              'targets_found':28,'targets_delivered':17,
              'false_confirmed_cells':12,'false_reports_delivered':4}
    r=terminal_paper_metrics(exemplar)
    assert r['training/confirmation_precision_percent']==70
    assert r['training/gcs_report_precision_percent']==pytest.approx(100*17/21)
    assert r['training/false_confirmed_cells']==12
    assert r['training/false_reports_delivered']==4
    empty={**exemplar,'targets_found':0,'targets_delivered':0,
           'false_confirmed_cells':0,'false_reports_delivered':0}
    p=terminal_paper_metrics(empty)
    assert 'training/confirmation_precision_percent' not in p
    assert 'training/gcs_report_precision_percent' not in p


def test_false_confirmation_precision_validates_counter_nonnegative():
    from uav_network.experiments.training_mission_metrics import terminal_paper_metrics
    obj={'search_rate_percent':100,'delivery_rate_percent':100,'success':False,
         'targets_found':35,'targets_delivered':35,'false_confirmed_cells':-1,
         'false_reports_delivered':4}
    with pytest.raises(ValueError,match='false_confirmed_cells'):
        terminal_paper_metrics(obj)


def test_gpu_heldout_precision_is_truth_scoring_only():
    from uav_network.experiments.gpu_mission_metrics import gpu_episode_record,aggregate_gpu_mission_metrics
    from uav_network.configuration import load_config
    c=load_config(NEW)
    n=c.world.num_targets
    info={'targets_found':3,'targets_delivered':2,'success':False,'step':1500,
          'search_rate_percent':100*3/n,'delivery_rate_percent':100*2/n,
          'false_confirmed_cells':2,'false_reports_delivered':1,
          'report_diagnostics':{'first_discovery_s':[10.,20.,30.]+[None]*(n-3),
                                 'first_delivery_s':[12.,22.]+[None]*(n-2)},
          'network':{'enabled':True,'generated_packets':10,'received_packets':5,
                     'uav_communication_energy_j':4},
          'energy':{'propulsion_energy_j':100},'coverage_percent':50}
    rec=gpu_episode_record(info,c,seed=420,episode=1,episode_return=9)
    assert rec['comparison_metrics']['confirmation_precision_percent']==60
    assert rec['comparison_metrics']['gcs_report_precision_percent']==pytest.approx(200/3)
    agg=aggregate_gpu_mission_metrics([rec])['mission_metrics']
    assert agg['confirmation_precision_percent']==60
    assert agg['gcs_report_precision_percent']==pytest.approx(200/3)


def test_new_kaggle_notebook_seals_new_source_and_never_reuses_old_checkpoint():
    from scripts.build_map1000_reward_belief_actor_fix_ui_20261123 import build_notebook
    n=build_notebook()
    assert len(n.cells)==3
    assert n.metadata['science_config']==NEW
    assert n.metadata['new_mdp_requires_training_from_scratch'] is True
    assert n.metadata['reward_communication_efficiency']==.2
    assert n.metadata['belief_prior']==pytest.approx(35/1600)
    assert n.metadata['actor_output_regularization']==.001
    assert n.metadata['actor_motion_output_normalization'] is True
    assert 'TRAIN_EPISODES = 20000' in n.cells[1].source
    assert 'VALIDATION_EPISODES = 100' in n.cells[1].source
    assert 'FINAL_TEST_EPISODES = 1000' in n.cells[1].source
    assert 'longlonthiu/notebookfe799af211' in n.cells[1].source
    assert 'checkpoint' not in n.metadata.get('warm_start','none')


def test_only_four_science_and_actor_knobs_change_not_phy_mac_or_safety():
    import yaml
    old=yaml.safe_load(Path(OLD).read_text())
    new=yaml.safe_load(Path(NEW).read_text())
    expected={('sensor','belief_prior'),('reward','communication_efficiency'),
              ('matd3','actor_output_regularization'),('matd3','output_normalization')}
    changed={(group,key) for group in old if group in new and isinstance(old[group],dict)
             for key in old[group] if old[group][key]!=new[group][key]}
    assert changed==expected|{('tracking','tags')}  # New run tags, no simulation effect
    assert new['runtime']['warm_start_checkpoint']=='none'
    assert new['world']['motion_safety']=='reject'
    assert new['sensor']['confirmation_min_likelihood_ratio']==0
    assert new['sensor']['report_cell_hypotheses'] is True
    assert new['network']==old['network']


def test_normalized_actor_motion_avoids_full_tanh_saturation_without_changing_rf():
    import torch
    from uav_network.algorithms.models import DeterministicActor
    actor=DeterministicActor(8,7,[16,16],continuous_dim=4,
                              output_normalization=True,normalization_dims=3)
    with torch.no_grad():
        actor.continuous.weight.zero_()
        actor.continuous.bias.fill_(12.)
    observations=torch.ones(3,8)
    motion,destination=actor(observations,torch.ones((3,7),dtype=torch.bool))
    assert motion.shape==(3,4)
    assert torch.all(motion[:,:3].abs()<.8),motion
    assert torch.all(motion[:,:3].abs()>.7),motion
    assert torch.all(motion[:,3]>.99),'RF head must be independent from XYZ normalization'
    penalty=actor.output_regularization(observations).mean()*.001
    penalty.backward()
    assert float(actor.continuous.bias.grad.abs().sum())>0


def test_communication_penalty_reweight_reverses_bad_reward_order_offline():
    # Two actual per-episode-group means from W&B HF diagnostic 000034/000050.
    # This is an offline *diagnostic*, not a trained-policy ablation result.
    old_return,late_return=-880.26,-223.187
    old_communication,late_communication=-1050.969,-136.615
    def reweighted(total,communication,coefficient):
        return total+(coefficient-1.)*communication
    assert late_return>old_return  # Existing reward favors fewer true deliveries
    assert reweighted(old_return,old_communication,.2)>reweighted(late_return,late_communication,.2)


def test_new_bundle_includes_score_only_precision_module_and_science_yaml():
    from scripts.build_map1000_reward_belief_actor_fix_ui_20261123 import build_notebook
    import base64,ast,hashlib,io,re,tarfile
    n=build_notebook();s=n.cells[2].source
    m=re.search(r'payload = base64\.b85decode\(("[^"\n]+")\)',s)
    assert m
    data=base64.b85decode(ast.literal_eval(m.group(1)))
    assert hashlib.sha256(data).hexdigest()==n.metadata['source_sha256']
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as bundle:
        names=bundle.getnames()
        assert NEW in names
        assert OLD not in names,'The code archive should contain only selected science config'
        assert 'uav_network/experiments/report_quality_metrics.py' in names
        assert 'tests/test_cuda.py' in names
        assert 'tests/test_cuda_scenario_v2.py' in names
        assert 'tests/test_training_audit_regressions.py' in names
        import yaml
        inner=yaml.safe_load(bundle.extractfile(NEW).read())
        assert inner['sensor']['belief_prior']==35/1600
        assert inner['reward']['communication_efficiency']==.2
        assert inner['matd3']['output_normalization'] is True
        assert inner['matd3']['actor_output_regularization']==.001
    assert "SPEC[\"science_ablation\"]" in s
    assert 'GPUAUTOTUNED' in s


def test_new_embedded_source_cpu_matd3_train_checkpoint_and_heldout_eval(tmp_path):
    """Execute EXACT Kaggle embedded code, not repo source, through real Warp CPU."""
    import ast,base64,hashlib,io,os,re,subprocess,sys,tarfile
    from scripts.build_map1000_reward_belief_actor_fix_ui_20261123 import build_notebook
    n=build_notebook();s=n.cells[2].source
    m=re.search(r'payload = base64\.b85decode\(("[^"\n]+")\)',s)
    assert m
    raw=base64.b85decode(ast.literal_eval(m.group(1)))
    assert hashlib.sha256(raw).hexdigest()==n.metadata['source_sha256']
    root=tmp_path/'sealed_kaggle_source';root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        for member in t:
            assert member.isfile()
            path=root.joinpath(*Path(member.name).parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(t.extractfile(member).read())
    code='''from uav_network.configuration import load_config
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate
from pathlib import Path
import json
c=load_config(__CONFIG__)
assert c.sensor.belief_prior == 35/1600
assert c.reward.communication_efficiency == 0.2
assert c.matd3.output_normalization and c.matd3.actor_output_regularization == 0.001
assert c.world.motion_safety == "reject" and c.network.tx_power_max_w == 0.6
c.runtime.device="cpu";c.runtime.replay_device="cpu";c.runtime.num_envs=2
c.runtime.episodes=2;c.runtime.compile_models=False;c.runtime.fused_optimizers=False
c.world.max_steps=12;c.matd3.hidden_sizes=[32,32];c.matd3.batch_size=16
c.matd3.learning_starts=4;c.matd3.updates_per_step=1;c.matd3.replay_capacity=1024
c.tracking.enabled=False;c.tracking.deferred_gpu_eval_enabled=False
c.runtime.output_dir=__TRAIN__
c.validate()
r=train(c)
assert r["episode_budget_complete"] and r["episodes"]==2
assert Path(r["checkpoint"]).is_file()
e=evaluate(r["checkpoint"],episodes=2,seed=2000000044,output_dir=__EVAL__)
assert e["episodes"] == 2
print("NEW_EMBEDDED_MAP1000_MATD3_TRAIN_EVAL_OK",json.dumps({"episodes":r["episodes"],"tested":e["episodes"],"belief_prior":c.sensor.belief_prior,"reward_comm":c.reward.communication_efficiency}))
'''
    code=code.replace('__CONFIG__',repr(NEW)).replace('__TRAIN__',repr(str(tmp_path/'train'))).replace('__EVAL__',repr(str(tmp_path/'eval')))
    env={**os.environ,'PYTHONPATH':str(root),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1'}
    p=subprocess.run([sys.executable,'-c',code],cwd=root,env=env,
                     capture_output=True,text=True,timeout=85)
    assert p.returncode==0,(p.stdout[-2500:],p.stderr[-2500:])
    assert 'NEW_EMBEDDED_MAP1000_MATD3_TRAIN_EVAL_OK' in p.stdout
