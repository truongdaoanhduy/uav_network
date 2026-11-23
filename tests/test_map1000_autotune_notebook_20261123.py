"""T4 autopilot: source integrity, exact science config, CPU full-flow bundle."""
from __future__ import annotations
import ast,base64,hashlib,io,os,re,subprocess,sys,tarfile,json
from pathlib import Path
import nbformat
import pytest
from scripts.build_map1000_tx06_autotune_fullflow_ui_20261123 import build_notebook as full_build
from scripts.build_map1000_tx06_t4_mac_replay_ab_ui_20261123 import build_notebook as pilot_build
from scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 import CONFIG


def get_archive(notebook, kind):
    body=notebook.cells[2].source
    label='payload = base64.b85decode' if kind=='full' else 'source=base64.b85decode'
    p=body.index(label)+len(label)
    match=re.search(r'\(\s*("[^"\n]+")\s*\)',body[p:])
    assert match,kind
    archive=base64.b85decode(ast.literal_eval(match.group(1)))
    assert hashlib.sha256(archive).hexdigest()==notebook.metadata['source_sha256']
    return archive


@pytest.mark.parametrize('kind,builder',[('full',full_build),('pilot',pilot_build)])
def test_both_pinned_3cell_notebooks_have_exact_source_and_science(kind,builder):
    nb=builder()
    nbformat.validate(nb)
    assert len(nb.cells)==3
    assert len(nbformat.writes(nb).encode())<850000
    assert nb.metadata['science_config']==CONFIG
    assert nb.metadata['reference_commit']
    assert '1000' in str(nb.metadata)
    tar=get_archive(nb,kind)
    with tarfile.open(fileobj=io.BytesIO(tar),mode='r:gz') as t:
        names=t.getnames()
        assert CONFIG in names
        assert 'uav_network/communication/gpu/mac.py' in names
        assert 'uav_network/communication/gpu/streaming.py' in names
        vec=t.extractfile('uav_network/environment/gpu/vector.py').read().decode()
        assert 'UAV_WARP_MAC_BLOCK_DIM' in vec
        assert 'kernel is packet_step' in vec
        assert 'block_dim=self._mac_block_dim' in vec
        import yaml
        c=yaml.safe_load(t.extractfile(CONFIG).read())
        assert c['world']['map_size_m']==1000 and c['world']['max_steps']==1500
        assert c['sensor']['grid_cell_m']==25
        assert c['sensor']['confirmation_mode']=='posterior_only'
        assert c['sensor']['confirmation_threshold']==.99
        assert c['sensor']['confirmation_min_likelihood_ratio']==0
        assert c['network']['tx_power_max_w']==.6
        assert c['matd3']['batch_size']==8192
        assert c['matd3']['updates_per_step']==4
        assert c['runtime']['num_envs']==4096 and c['runtime']['episodes']==20000
    script=nb.cells[2].source
    ast.parse(script)
    assert nb.metadata["reference_commit"]
    assert '--mac-block-dim' in script
    assert 'HYBRID_REPLAY_CUDA_AB_JSON' in script
    assert 'MAC' in script
    if kind=='full':
        assert nb.metadata['gpu_autotune_before_training'] is True
        assert nb.metadata['full_20000_train_val100_test1000_intact'] is True
        assert nb.metadata['validation_episodes_per_checkpoint']==100
        assert nb.metadata['test_episodes']==1000
        assert 'SPEC["gpu_hardware_autotune"]' in script
        assert 'MAP1000_TX06_AUTOTUNE_RESULT.json' in script
        assert 'one_session_train_eval_test' in script
        assert 'UAV_HYBRID_PINNED_STAGING' in script
        assert script.index('MAP1000_AUTOTUNE_DECISION') < script.index('spec_file = Path("/kaggle/working/job.json")')
    else:
        assert 'kaggle_secrets' not in script
        assert 'WANDB_API_KEY' in script # only to delete from benchmark child process
        assert 'one_session_train_eval_test' not in script
        assert 'MAP1000_TX06_T4_MAC_REPLAY_AB_RESULTS.json' in script


def test_embedded_autotune_kaggle_source_real_matd3_cpu_train_to_checkpoint_to_eval(tmp_path):
    n=full_build()
    archive=get_archive(n,'full')
    dest=tmp_path/'source'
    dest.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
        for member in t:
            assert member.isfile()
            path=dest.joinpath(*Path(member.name).parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(t.extractfile(member).read())
    runner='''from uav_network.configuration import load_config
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate
from uav_network.environment.observations import observation_version
from pathlib import Path
import json
c=load_config(__CFG__)
c.runtime.device='cpu';c.runtime.replay_device='cpu';c.runtime.num_envs=2
c.runtime.episodes=2;c.runtime.compile_models=False;c.runtime.fused_optimizers=False
c.world.max_steps=12;c.matd3.hidden_sizes=[32,32];c.matd3.batch_size=16
c.matd3.learning_starts=4;c.matd3.updates_per_step=1;c.matd3.replay_capacity=1024
c.tracking.enabled=False;c.tracking.deferred_gpu_eval_enabled=False
c.runtime.output_dir=__OUT__
c.validate()
assert observation_version(c)==11
assert c.world.map_size_m==1000 and c.network.tx_power_max_w==.6
result=train(c)
assert result['episodes']==2 and result['episode_budget_complete'],result
assert Path(result['checkpoint']).is_file()
t=evaluate(result['checkpoint'],episodes=2,seed=2000000044,output_dir=__EVAL__)
assert t['episodes']==2
print('KAGGLE_AUTOTUNE_EMBEDDED_CPU_FULLFLOW_OK',json.dumps({'trained':result['episodes'],'tested':t['episodes'],'posterior_only':True,'map':1000,'tx':.6}))
'''
    code=runner.replace('__CFG__',repr(CONFIG)).replace('__OUT__',repr(str(tmp_path/'train'))).replace('__EVAL__',repr(str(tmp_path/'eval')))
    env={**os.environ,'PYTHONPATH':str(dest),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1'}
    proc=subprocess.run([sys.executable,'-c',code],cwd=dest,env=env,
                        capture_output=True,text=True,timeout=100)
    assert proc.returncode==0,(proc.stdout[-1800:],proc.stderr[-2300:])
    assert 'KAGGLE_AUTOTUNE_EMBEDDED_CPU_FULLFLOW_OK' in proc.stdout
