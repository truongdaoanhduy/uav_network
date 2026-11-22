"""The adaptive CUDA pilot must package exact source and stop before 20k."""
import ast,hashlib,nbformat
from pathlib import Path
from scripts.build_nooracle_adaptive_altitude_pilot_20261123 import build_notebook,OUT,FILES


def test_adaptive_gpu_notebook_packaging_is_self_contained_and_bounded():
    notebook=build_notebook()
    nbformat.validate(notebook)
    assert len(notebook.cells)==4
    assert all(x.cell_type=='code' for x in notebook.cells[1:])
    assert notebook.metadata['source_commit_basis']
    assert notebook.metadata['adaptive_altitude_new_mdp']
    assert notebook.metadata['full_20000_training'] is False
    assert notebook.metadata['num_envs']==4096
    assert notebook.metadata['batch_size']==8192
    assert notebook.metadata['hidden_sizes']==[256,512]
    assert len(FILES)>500
    for cell in notebook.cells[1:]:ast.parse(cell.source)
    text=notebook.cells[2].source
    assert 'mode="r:xz"' in text
    assert 'tarfile.open(fileobj=io.BytesIO(blob)' in text
    assert 'lzma.decompress(blob)' not in text
    assert 'Expected pinned TAR.XZ' in text
    runner=notebook.cells[3].source
    assert 'trial_adaptive_altitude_gpu_pilot_20261123' in runner
    assert "'--steps',str(VECTOR_STEPS)" in runner
    assert 'gpu_science_convergence_unverified' in runner
    assert 'CUDA_VISIBLE_DEVICES' in runner
    assert len(nbformat.writes(notebook).encode())<1_000_000


def test_saved_adaptive_gpu_notebook_matches_newly_built_after_export():
    assert OUT.exists()
    stored=nbformat.read(OUT,as_version=4)
    built=build_notebook()
    assert stored.cells[2].source==built.cells[2].source
    assert stored.metadata['source_sha256']==built.metadata['source_sha256']


def test_actual_embedded_kaggle_source_unpack_and_real_cpu_train(tmp_path):
    """Execute the real emitted notebook cell and train from its unpacked src."""
    import json,os,subprocess,sys
    n=nbformat.read(OUT,as_version=4)
    cell=n.cells[2].source
    assert cell.count('venv=root/".venv"')==1
    preflight=cell[:cell.index('venv=root/".venv"')]
    preflight=preflight.replace('/kaggle/working/nooracle_adaptive_altitude_trial',str(tmp_path/'kaggle'))
    scope={}
    exec(compile(preflight,'kaggle_adaptive_bootstrap','exec'),scope)
    source_root=scope['source_root']
    assert len(scope['files_sha'])>=500
    for name in ('uav_network/environment/gpu/evaluation_truth.py',
                 'uav_network/environment/gpu/bayesian.py',
                 'configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml',
                 'scripts/trial_adaptive_altitude_gpu_pilot_20261123.py',
                 'pyproject.toml','scripts/requirements-network.txt'):
        assert (source_root/name).is_file(),name
        assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==scope['files_sha'][name]
    env=os.environ.copy()
    env['PYTHONPATH']=str(source_root)
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    argv=[sys.executable,'-m','scripts.trial_adaptive_altitude_gpu_pilot_20261123',
          '--device','cpu','--output',str(tmp_path/'pilot_out'),
          '--envs','4','--steps','16','--batch','16','--updates','1',
          '--horizon','12','--learning-starts','16']
    process=subprocess.run(argv,cwd=source_root,env=env,capture_output=True,text=True,timeout=90)
    assert process.returncode==0,(process.stdout[-1200:],process.stderr[-1800:])
    rows=[json.loads(line.split(' ',1)[1]) for line in process.stdout.splitlines()
          if line.startswith('ADAPTIVE_ALTITUDE_GPU_PILOT_JSON ')]
    assert len(rows)==1
    a=rows[0]
    assert a['transitions']==64 and a['completed_episodes']==4
    assert a['actual_learner_updates']>0
    assert a['observation_version']==9 and a['completed_20k'] is False
    assert (tmp_path/'pilot_out'/'adaptive_pilot_report.json').is_file()
