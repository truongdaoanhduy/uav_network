"""Prevent stale Kaggle notebook ref and execute all generated Kaggle cells.

The public version 372917422 UI currently crashes while loading KernelViewer;
this confirms local source correctness but not the inaccessible Kaggle log.
"""
import json,os,re,subprocess,sys,types
from pathlib import Path
from unittest.mock import patch
import nbformat,pytest


def test_new_notebook_defaults_to_actual_kaggle_kernel_not_old_name():
    from scripts.build_map1000_tx06_autotune_fullflow_fe799_fix_20261123 import build_notebook
    nb=build_notebook()
    nbformat.validate(nb)
    assert len(nb.cells)==3
    assert len(nbformat.writes(nb).encode())<850_000
    cell=nb.cells[1].source
    assert "os.environ.get('KAGGLE_KERNEL_REF','longlonthiu/notebookfe799af211')" in cell
    assert 'notebookbbd0aec150' not in cell
    assert nb.metadata['target_kernel_ref']=='longlonthiu/notebookfe799af211'
    assert nb.metadata['source_sha256']


def test_full_real_kaggle_bootstrap_and_autotune_then_cloud_runner_mocked(tmp_path,monkeypatch):
    from scripts.build_map1000_tx06_autotune_fullflow_fe799_fix_20261123 import build_notebook
    import torch
    nb=build_notebook()
    class FakeSecrets:
        def get_secret(self,name):
            assert name in ('WANDB_API_KEY','HF_TOKEN')
            return 'SYNTHETIC_TEST_SECRET'
    secrets=types.ModuleType('kaggle_secrets')
    secrets.UserSecretsClient=FakeSecrets
    monkeypatch.setitem(sys.modules,'kaggle_secrets',secrets)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'Tesla T4 (simulated)')
    tmp=tmp_path
    root=tmp/'runtime_src'; job=tmp/'job.json'; autotune=tmp/'autotune.json'; report=tmp/'one-session-train-eval-test-report.json'
    cell3=nb.cells[2].source
    for old,new in (
        ('root = Path("/tmp/uav_network")',f'root = Path({str(root)!r})'),
        ('spec_file = Path("/kaggle/working/job.json")',f'spec_file = Path({str(job)!r})'),
        ('_tune_path=Path("/kaggle/working/MAP1000_TX06_AUTOTUNE_RESULT.json")',f'_tune_path=Path({str(autotune)!r})'),
        ("report=Path('/kaggle/working/one-session-train-eval-test-report.json')",f'report=Path({str(report)!r})'),
    ):
        assert cell3.count(old)==1,old
        cell3=cell3.replace(old,new)
    real_run=subprocess.run
    invoked=[]
    def intercept(args,**kwargs):
        args=list(map(str,args));invoked.append(args)
        if args[1:3]==['-m','venv']:
            py=root/'.venv/bin/python';py.parent.mkdir(parents=True,exist_ok=True);py.symlink_to(sys.executable)
            return types.SimpleNamespace(returncode=0)
        if args[1:3]==['-m','pip']:
            return types.SimpleNamespace(returncode=0)
        if '-c' in args:
            source=args[args.index('-c')+1]
            env={**os.environ,'PYTHONPATH':str(root),'TORCHINDUCTOR_COMPILE_THREADS':'1',
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
            return real_run([sys.executable,'-c',source],cwd=root,env=env,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45,check=True)
        if 'scripts.profile_nooracle_warp_phases_20261122' in args:
            block=int(args[args.index('--mac-block-dim')+1])
            score={0:1000,32:1100,64:1020,128:900}[block]
            obj={'world_state_sha256':'MATCHED_ALL_STATE','env_transitions_per_second':score,
                 'per_kernel_ms_total':{'packet_step':1.0}}
            return types.SimpleNamespace(returncode=0,stdout='UAV_WARP_PROFILE_JSON '+json.dumps(obj),stderr='')
        if 'scripts.benchmark_hybrid_replay_cuda_ab_20261123' in args:
            mode=args[args.index('--staging')+1]
            obj={'median_sample_seconds':.090 if mode=='on' else .10,
                 'all_sampled_tensor_sha256':'MATCHED','rng_state_sha256':'MATCHED',
                 'seen_bitmap_sha256':'MATCHED'}
            return types.SimpleNamespace(returncode=0,stdout='HYBRID_REPLAY_CUDA_AB_JSON '+json.dumps(obj),stderr='')
        if 'uav_network.cloud.remote' in args:
            payload=json.loads(job.read_text())
            assert payload['kernel_ref']=='longlonthiu/notebookfe799af211'
            assert payload['gpu_hardware_autotune']['mac_block_dim']==32
            assert payload['gpu_hardware_autotune']['replay_pinned'] is True
            return types.SimpleNamespace(returncode=0)
        if 'uav_network.experiments.one_session' in args:
            payload=json.loads(job.read_text())
            assert payload['config'].endswith('map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml')
            assert args[args.index('--test-batch-size')+1]=='500'
            report.write_text(json.dumps({'gpu_val_status':'complete','gpu_test_status':'complete'}))
            return types.SimpleNamespace(returncode=0)
        raise AssertionError('Unrecognized generated command: '+str(args[:7]))
    monkeypatch.setattr(subprocess,'run',intercept)
    with patch.dict(os.environ,dict(os.environ),clear=True):
        os.environ.pop('KAGGLE_KERNEL_REF',None)
        namespace={'__name__':'__main__'}
        exec(compile(nb.cells[1].source,'Kaggle_cell2','exec'),namespace)
        exec(compile(cell3,'Kaggle_cell3','exec'),namespace)
        assert namespace['SPEC']['kernel_ref']=='longlonthiu/notebookfe799af211'
        assert namespace['SPEC']['full_flow_protocol']['validation_episodes']==100
        assert namespace['SPEC']['full_flow_protocol']['test_episodes']==1000
        assert namespace['SPEC']['runs']['matd3']['id']
    assert report.exists()
    assert autotune.exists()
    assert (root/'uav_network'/'environment'/'gpu'/'vector.py').exists()
    assert len(invoked)>6


def test_fix_never_changes_scientific_bundle_or_gpu_20k_protocol():
    from scripts.build_map1000_tx06_autotune_fullflow_fe799_fix_20261123 import build_notebook
    from scripts.build_map1000_tx06_autotune_fullflow_ui_20261123 import build_notebook as old_build
    from scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 import CONFIG
    import ast
    old=old_build();fixed=build_notebook()
    assert fixed.metadata['source_sha256']==old.metadata['source_sha256']
    assert fixed.metadata['science_config']==old.metadata['science_config']==CONFIG
    assert fixed.metadata['reference_commit']==old.metadata['reference_commit']
    assert fixed.cells[2].source==old.cells[2].source
    assert fixed.cells[1].source.count('longlonthiu/notebookfe799af211')==1
    assert 'notebookbbd0aec150' not in fixed.cells[1].source
    for expected in ('TRAIN_EPISODES = 20000','ENVS = 4096','UPDATES_PER_VECTOR_STEP = 4',
                     'BATCH = 8192','VALIDATION_EPISODES = 100','FINAL_TEST_EPISODES = 1000',
                     'MAP_SIZE_M = 1000','GRID_CELL_M = 25','TX_POWER_MAX_W = .6'):
        assert expected in fixed.cells[1].source
    assert fixed.metadata['full_20000_train_val100_test1000_intact']
    assert 'MAP1000_AUTOTUNE_DECISION' in fixed.cells[2].source


def test_runtime_override_is_preserved_for_another_kernel(monkeypatch):
    from scripts.build_map1000_tx06_autotune_fullflow_fe799_fix_20261123 import build_notebook
    import torch,types
    n=build_notebook()
    m=types.ModuleType('kaggle_secrets')
    class Dummy:
        def get_secret(self,name):return 'SYNTHETIC_TEST_ONLY'
    m.UserSecretsClient=Dummy
    monkeypatch.setitem(sys.modules,'kaggle_secrets',m)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda idx:'T4 STUB')
    with patch.dict(os.environ,dict(os.environ),clear=True):
        os.environ['KAGGLE_KERNEL_REF']='exampleuser/alternative-kernel'
        ns={'__name__':'__main__'}
        exec(compile(n.cells[1].source,'Kaggle_reference_override','exec'),ns)
        assert ns['NOTEBOOK_REF']=='exampleuser/alternative-kernel'
