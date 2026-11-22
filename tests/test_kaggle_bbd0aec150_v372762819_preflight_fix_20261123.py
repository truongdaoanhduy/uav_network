"""Reproduce Kaggle scriptVersionId=372762819 NameError in notebook In[2].

Log: Kaggle pip install succeeded then preflight failed at
'c=load_config('+repr(CONFIG)+'); ': CONFIG was generator-scope only.
The same generated Kaggle code must be executed with ONLY runtime globals.
"""
from pathlib import Path
import subprocess
import sys
import nbformat

from scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 import (
    build_notebook,CONFIG,ROOT)


def test_kaggle_bootstrap_preflight_executes_using_only_runtime_cell_globals(monkeypatch):
    notebook=build_notebook()
    code=notebook.cells[2].source
    before_payload=code[:code.index('# Kaggle runs only this Python bootstrap; source is inside the payload.')]
    preflight=code[code.index('# Verify actual new scientific config before spending any GPU time'):code.index('spec_file = Path("/kaggle/working/job.json")')]
    # Recreate exactly what is in the user's notebook: the configuration cell
    # defines NOTEBOOK_REF etc, but does NOT define generator-local CONFIG.
    monkeypatch.setenv('KAGGLE_KERNEL_REF','longlonthiu/notebookbbd0aec150')
    ctx={'NOTEBOOK_REF':'longlonthiu/notebookbbd0aec150',
         'CHECKPOINTS_PER_GPU_BATCH':81,'MAX_WARP_EVAL_WORLDS_PER_GPU':8192,
         'FINAL_TEST_EPISODES':1000}
    isolated=(before_payload+'\n'
              'python='+repr(sys.executable)+'\n'
              'root=Path('+repr(str(ROOT))+')\n'
              +preflight)
    # Let the actual subprocess run the same science imports and assertions
    # against the repo, while blocking network, pip and remote training.
    exec(compile(isolated,'kaggle_cell3_runtime_preflight','exec'),ctx)
    assert ctx['SPEC']['config']==CONFIG
    assert ctx['SPEC']['kernel_ref']=='longlonthiu/notebookbbd0aec150'


def test_kaggle_source_cap_and_three_cell_contract_remain_valid():
    n=build_notebook()
    nbformat.validate(n)
    assert len(n.cells)==3
    assert len(nbformat.writes(n).encode('utf-8'))<850_000
    assert len(n.cells[2].source.encode('utf-8'))<825_000


def test_complete_3_cell_kaggle_bootstrap_reaches_remote_train_and_valtest_calls_without_nameerrors(
        tmp_path, monkeypatch):
    """Kaggle-like three-cell run; ONLY cloud/GPU/secret transports are simulated.

    Source SHA/extraction and the actual preflight imports execute on CPU.
    This catches cross-cell globals (e.g. generator-only CONFIG) that ast.parse
    cannot catch. No upload, credentials, full train or GPU allocation occurs.
    """
    import json,os,types
    import torch
    from pathlib import Path

    nb=build_notebook()
    class DummySecrets:
        def get_secret(self,name):
            assert name in {'WANDB_API_KEY','HF_TOKEN'}
            return 'SYNTHETIC_TEST_ONLY'
    module=types.ModuleType('kaggle_secrets')
    module.UserSecretsClient=DummySecrets
    monkeypatch.setitem(sys.modules,'kaggle_secrets',module)
    monkeypatch.setenv('KAGGLE_KERNEL_REF','longlonthiu/notebookbbd0aec150')
    monkeypatch.setattr(torch.cuda,'is_available',lambda: True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda: 2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'Tesla T4 (test stub)')

    rootsrc=tmp_path/'uav_network'
    job=tmp_path/'job.json'
    report=tmp_path/'one-session-train-eval-test-report.json'
    cell=nb.cells[2].source
    for old,new in [
        ('root = Path("/tmp/uav_network")',f'root = Path({str(rootsrc)!r})'),
        ('spec_file = Path("/kaggle/working/job.json")',f'spec_file = Path({str(job)!r})'),
        ("report=Path('/kaggle/working/one-session-train-eval-test-report.json')",f'report=Path({str(report)!r})'),
    ]:
        assert cell.count(old)==1,old
        cell=cell.replace(old,new)
    actual_run=subprocess.run
    calls=[]
    def local_fake_run(args,**kwargs):
        args=[str(x) for x in args]
        calls.append(args)
        if len(args)>2 and args[1:3]==['-m','venv']:
            py=rootsrc/'.venv/bin/python'
            py.parent.mkdir(parents=True,exist_ok=True)
            py.symlink_to(sys.executable)
            return types.SimpleNamespace(returncode=0)
        if len(args)>3 and args[1:3]==['-m','pip']:
            return types.SimpleNamespace(returncode=0)
        if '-c' in args:
            idx=args.index('-c')
            environ=dict(os.environ,PYTHONPATH=str(rootsrc),TORCHINDUCTOR_COMPILE_THREADS='1')
            return actual_run([sys.executable,'-c',args[idx+1]],env=environ,
                              cwd=rootsrc,check=True,timeout=30,
                              capture_output=True,text=True)
        if 'uav_network.cloud.remote' in args:
            assert job.exists() and json.loads(job.read_text())['config']==CONFIG
            return types.SimpleNamespace(returncode=0)
        if 'uav_network.experiments.one_session' in args:
            assert job.exists()
            assert '--policies-per-batch' in args
            assert args[args.index('--policies-per-batch')+1]=='81'
            assert args[args.index('--eval-world-pool-size')+1]=='8192'
            assert args[args.index('--test-batch-size')+1]=='500'
            report.write_text(json.dumps({'gpu_val_status':'complete','gpu_test_status':'complete'}))
            return types.SimpleNamespace(returncode=0)
        raise AssertionError('Unexpected cloud/bootstrap subprocess: '+repr(args[:6]))
    monkeypatch.setattr(subprocess,'run',local_fake_run)

    glob={'__name__':'__main__'}
    exec(compile(nb.cells[1].source,'kaggle_cell1','exec'),glob)
    exec(compile(cell,'kaggle_cell3','exec'),glob)
    assert len(calls)>=5
    assert rootsrc.joinpath(CONFIG).is_file()
    assert glob['SPEC']['config']==CONFIG
    assert glob['SPEC']['runs']['matd3']['id']!='resolved_at_runtime'
    assert report.exists()
    assert os.environ.get('UAV_DIAGNOSTIC_STORE')=='huggingface_public'
