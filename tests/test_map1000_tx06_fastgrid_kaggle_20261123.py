"""Regression: MAP1000 square, TX .6 W, 40x40 belief; FASTGRID unchanged."""
from pathlib import Path
import ast
import base64
import hashlib
import io
import re
import tarfile
import nbformat
import pytest
import yaml

NEW='configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml'
OLD='configs/matd3-nooracle-posterior-only-map800-h256x256-orthogonal-compile-tx08-hf-fullflow-20261123.yaml'


def test_only_requested_science_fields_change():
    from uav_network.configuration import load_config
    orig=load_config(OLD)
    c=load_config(NEW)
    c.validate()
    assert c.world.map_size_m==1000
    assert c.sensor.grid_cell_m==25
    assert c.world.map_size_m / c.sensor.grid_cell_m==40
    assert c.network.tx_power_min_w==.001
    assert c.network.tx_power_max_w==.6
    assert c.reward.communication_reference_tx_power_w==.4
    assert c.network.power_control=='learned'
    assert c.sensor.confirmation_mode=='posterior_only'
    assert c.sensor.confirmation_threshold==.99
    assert c.sensor.confirmation_min_likelihood_ratio==0
    assert c.world.num_uavs==6 and c.world.num_targets==35
    assert c.world.num_obstacles==25 and c.world.max_steps==1500
    assert c.runtime.num_envs==4096 and c.runtime.episodes==20000
    assert c.matd3.hidden_sizes==[256,256] and c.matd3.batch_size==8192
    assert c.matd3.updates_per_step==4
    assert c.tracking.deferred_gpu_eval_collection_stride==100
    assert c.tracking.periodic_evaluation_episodes==100
    assert c.runtime.replay_cpu_fraction==.75
    a=orig.to_dict();b=c.to_dict()
    for section,key in [('world','map_size_m'),('sensor','grid_cell_m'),('network','tx_power_max_w'),('runtime','output_dir'),('tracking','group'),('tracking','run_name')]:
        a[section].pop(key);b[section].pop(key)
    assert a==b,'Unexpected science/training parameter changed besides map/cell/TX and naming'


def test_generated_notebook_has_exact_config_and_sha():
    from scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 import build_notebook
    n=build_notebook()
    nbformat.validate(n)
    assert len(n.cells)==3
    assert len(nbformat.writes(n).encode())<850000
    assert len(n.cells[2].source.encode())<825000
    assert n.metadata['science_config']==NEW
    assert n.metadata['tx_power_max_w']==.6
    assert n.metadata['map_size_m']==1000
    assert n.metadata['belief_grid_cells_per_side']==40
    assert n.metadata['checkpoint_policy']=='periodic_only_every100_initial_final'
    intro=n.cells[0].source
    assert '1.000×1.000' in intro and '0,6 W' in intro
    assert '0,8 W' not in intro and '800×800' not in intro
    code=n.cells[1].source
    assert 'TX_POWER_MAX_W = .6' in code
    assert 'MAP_SIZE_M = 1000' in code
    assert 'GRID_CELL_M = 25' in code
    src=n.cells[2].source
    ast.parse(src)
    assert "world.map_size_m=1000" in src and "sensor.grid_cell_m=25" in src
    assert "network.tx_power_max_w=0.6" in src
    assert "world.map_size_m=800" not in src
    assert "sensor.grid_cell_m=20" not in src
    assert "network.tx_power_max_w=0.8" not in src
    assert 'UAV_PERIODIC_ACTORS_ONLY' in src
    assert 'UAV_BATCH_SCALAR_METRICS' in src
    assert 'UAV_FAST_GRAD_CLIP' in src
    assert 'UAV_FORENSIC_ENABLED' in src
    m=re.search(r'payload = base64\.b85decode\(("[^"]+")\)',src)
    assert m
    blob=base64.b85decode(ast.literal_eval(m.group(1)))
    assert hashlib.sha256(blob).hexdigest()==n.metadata['source_sha256']
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as ar:
        names=ar.getnames()
        assert NEW in names and OLD not in names
        assert all('tests/'+test in names for test in ('test_cuda.py','test_cuda_scenario_v2.py','test_training_audit_regressions.py'))
        cfg=yaml.safe_load(ar.extractfile(NEW).read())
        assert cfg['world']['map_size_m']==1000
        assert cfg['sensor']['grid_cell_m']==25
        assert cfg['network']['tx_power_max_w']==.6
        assert cfg['network']['tx_power_min_w']==.001


def test_real_cpu_simulation_with_new_square_and_power(tmp_path):
    from uav_network.configuration import load_config
    from uav_network.environment.vector import make_vector
    c=load_config(NEW)
    c.runtime.device='cpu'
    c.runtime.num_envs=2
    c.runtime.compile_models=False
    c.runtime.output_dir=str(tmp_path/'map1000')
    c.validate()
    vector=make_vector(c,device='cpu')
    try:
        obs,infos=vector.reset()
        assert obs['observation'].shape[0]==2
        assert int(vector.p.grid_n)==40
        for _ in range(5):
            act=vector.random_actions(obs['observation'])
            obs,reward,done,truncated,infos=vector.step({'continuous':act.continuous,'destination':act.destination})
            assert len(reward)==2
    finally:
        if hasattr(vector,'close'):vector.close()


def test_notebook_actual_kaggle_spec_has_no_legacy_overrides():
    from scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 import build_notebook
    n=build_notebook()
    tree=ast.parse(n.cells[2].source)
    spec=next(ast.literal_eval(node.value) for node in tree.body
              if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SPEC' for t in node.targets))
    assert spec['config']==NEW
    overrides=spec['overrides']
    assert 'world.map_size_m=1000' in overrides
    assert 'sensor.grid_cell_m=25' in overrides
    assert 'network.tx_power_max_w=0.6' in overrides
    assert 'world.map_size_m=800' not in overrides
    assert 'sensor.grid_cell_m=20' not in overrides
    assert 'network.tx_power_max_w=0.8' not in overrides
    assert 'MAP1000-TX06' in spec['runs']['matd3']['name']
    assert spec['dense_eval_schedule']['validation_scenarios_per_checkpoint']==100
    assert spec['dense_eval_schedule']['final_test_scenarios']==1000
    assert spec['gpus_per_run']==2
    assert spec['validate_all_snapshots'] is True
    assert 'UAV_PERIODIC_ACTORS_ONLY' in n.cells[2].source
    assert n.metadata['reference_commit']


def test_real_matd3_training_and_periodic_snapshots_with_new_config(tmp_path,monkeypatch):
    """Real Warp CPU train smoke, including replay, updates and checkpoint bank."""
    from uav_network.configuration import load_config
    from uav_network.experiments.training import train
    import json
    c=load_config(NEW)
    c.runtime.device='cpu'
    c.runtime.replay_device='cpu'
    c.runtime.num_envs=2
    c.runtime.episodes=6
    c.runtime.compile_models=False
    c.runtime.fused_optimizers=False
    c.runtime.output_dir=str(tmp_path/'tx06-map1000')
    c.world.max_steps=4
    c.matd3.hidden_sizes=[32,32]
    c.matd3.batch_size=8
    c.matd3.learning_starts=4
    c.matd3.replay_capacity=64
    c.matd3.updates_per_step=1
    c.tracking.enabled=False
    c.tracking.deferred_gpu_eval_enabled=True
    c.tracking.deferred_gpu_eval_keep_all=True
    c.tracking.deferred_gpu_eval_collection_stride=2
    c.validate()
    monkeypatch.setenv('UAV_PERIODIC_ACTORS_ONLY','true')
    monkeypatch.setenv('UAV_FAST_GRAD_CLIP','true')
    monkeypatch.setenv('UAV_BATCH_SCALAR_METRICS','true')
    result=train(c)
    assert result['episodes']==6
    manifest=json.loads((tmp_path/'tx06-map1000'/'deferred-gpu-eval'/'manifest.json').read_text())
    assert manifest['count']>=2
    assert all('training_metric_best_candidate' not in row['reasons'] for row in manifest['snapshots'])
    assert all(row['global_step']%2==0 for row in manifest['snapshots'])
