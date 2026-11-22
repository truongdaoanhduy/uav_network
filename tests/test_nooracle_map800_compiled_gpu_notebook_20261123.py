"""Pinned Kaggle source bundle must contain the NEW 800m compiled orthogonal MDP."""
import ast,hashlib,json,os,subprocess,sys
from pathlib import Path
import nbformat

from scripts.build_nooracle_map800_orthogonal_compile_pilot_20261123 import build_notebook,OUT,pinned_files


def test_new_pilot_embeds_exact_user_changes_without_mutating_earlier_notebook():
    n=build_notebook();nbformat.validate(n)
    assert n.metadata['num_envs']==4096
    assert n.metadata['batch_size']==8192
    assert n.metadata['hidden_sizes']==[256,256]
    assert n.metadata['world_map_size_m']==800
    assert n.metadata['compile_models'] is True
    assert n.metadata['initialization']=='orthogonal'
    assert n.metadata['full_20000_training'] is False
    assert n.metadata['adaptive_altitude_new_mdp'] is True
    assert len(n.cells)==4
    assert 'torch.compile' in n.cells[0].source
    assert '800' in n.cells[0].source
    assert 'trial_nooracle_map800_orthogonal_compile_gpu_pilot_20261123' in n.cells[3].source
    assert 'VECTOR_STEPS=64' in n.cells[1].source
    assert 'bytes.fromhex("fd377a585a00")' in n.cells[2].source
    for c in n.cells[1:]:ast.parse(c.source)
    assert len(nbformat.writes(n).encode())<1_000_000


def test_saved_notebook_pinned_bundle_sha_matches_every_embedded_source_file():
    assert OUT.exists()
    nb=nbformat.read(OUT,as_version=4)
    built=build_notebook()
    assert nb.metadata['source_sha256']==built.metadata['source_sha256']
    files,_,_=pinned_files()
    assert len(files)>=549
    for name in ('configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-20261123.yaml',
                 'scripts/trial_nooracle_map800_orthogonal_compile_gpu_pilot_20261123.py'):
        assert files[name]==Path(name).read_bytes()
    assert nb.cells[2].source==built.cells[2].source


def test_real_compiled_cpu_training_from_kaggle_extracted_sources(tmp_path):
    n=build_notebook();bootstrap=n.cells[2].source
    preflight=bootstrap[:bootstrap.index('venv=root/".venv"')]
    preflight=preflight.replace('/kaggle/working/nooracle_map800_compiled_trial',str(tmp_path/'isolated'))
    ns={};exec(compile(preflight,'pinned_map800_source_extract','exec'),ns)
    root=ns['source_root']
    assert len(ns['files_sha'])>=549
    for name,digest in ns['files_sha'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    env={**os.environ,'PYTHONPATH':str(root),'TORCHINDUCTOR_COMPILE_THREADS':'1',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    cmd=[sys.executable,'-m','scripts.trial_nooracle_map800_orthogonal_compile_gpu_pilot_20261123',
         '--device','cpu','--output',str(tmp_path/'training'),'--envs','4','--steps','16',
         '--batch','16','--updates','1','--horizon','12','--learning-starts','16']
    p=subprocess.run(cmd,cwd=root,env=env,text=True,capture_output=True,timeout=100)
    assert p.returncode==0,(p.stdout[-1000:],p.stderr[-1200:])
    rows=[json.loads(line.split(' ',1)[1]) for line in p.stdout.splitlines()
          if line.startswith('MAP800_COMPILED_GPU_PILOT_JSON ')]
    assert len(rows)==1
    assert rows[0]['transitions']==64
    assert rows[0]['actual_learner_updates']>0
    assert rows[0]['model_architecture']==[256,256]
    assert rows[0]['compile_models'] is True
    assert rows[0]['initialization']=='orthogonal'
    assert rows[0]['completed_20k'] is False
