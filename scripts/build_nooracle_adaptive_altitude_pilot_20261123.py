"""Build a self-contained, source-pinned, bounded Kaggle T4x2 pilot notebook.

Refuses to download code or use credentials at Kaggle runtime. Uses the
previously pinned 544-file source bundle with a tiny SHA-verified overlay of
new adaptive science and evaluation fixes. This is a NEW MDP, not a resume.
"""
from __future__ import annotations
import ast,base64,hashlib,io,lzma,re,tarfile
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK=ROOT/'notebooks/MATD3-NOORACLE-V9-CT095-H256x256-TRAIN-SPEED-AB-E4096-U4-B8192-T4X2-PILOT-20261123.ipynb'
NAME='MATD3-NOORACLE-ADAPTIVE-ALTITUDE-LR4-H256x512-E4096-U4-B8192-T4X2-BOUNDED-PILOT-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME
OVERLAY_FILES=(
    'uav_network/configuration_xyz.py',
    'uav_network/environment/bayesian.py',
    'uav_network/environment/gpu/bayesian.py',
    'uav_network/environment/gpu/state.py',
    'uav_network/environment/gpu/vector.py',
    'uav_network/environment/observations.py',
    'uav_network/environment/gpu/evaluation_truth.py',
    'uav_network/algorithms/hybrid_replay.py',
    'uav_network/experiments/training.py',
    'configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml',
    'scripts/trial_adaptive_altitude_gpu_pilot_20261123.py',
)
FILES=()


def pinned_files():
    nb=nbformat.read(SOURCE_NOTEBOOK,as_version=4)
    source=nb.cells[2].source
    match=re.search(r'blob=base64\.b85decode\((\'[A-Za-z0-9!#$%&()*+,-./:;<=>?@^_`{|}~]+\')\)',source)
    if not match:raise RuntimeError('Missing pinned original source archive')
    compressed=base64.b85decode(ast.literal_eval(match.group(1)))
    if hashlib.sha256(compressed).hexdigest()!=nb.metadata['source_sha256']:
        raise RuntimeError('Original baseline scientific source SHA changed')
    files={}
    with tarfile.open(fileobj=io.BytesIO(compressed),mode='r:xz') as archive:
        for member in archive:
            if not member.isfile() or '..' in Path(member.name).parts:
                raise RuntimeError('Unexpected archive entry '+member.name)
            files[member.name]=archive.extractfile(member).read()
    if len(files)<500:raise RuntimeError('Incomplete source archive')
    for name in OVERLAY_FILES:
        path=ROOT/name
        if not path.is_file():raise FileNotFoundError(path)
        files[name]=path.read_bytes()
    return files,nb.metadata['source_sha256'],nb.metadata['source_commit']


def build_notebook():
    files,base_hash,source_commit=pinned_files()
    # Repack the exact pinned source plus the known scientific delta. Nothing
    # from .git, .venv, credentials, local diagnostics or caches is embedded.
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as archive:
        for name,data in sorted(files.items()):
            info=tarfile.TarInfo(name)
            info.size=len(data);info.mode=0o644;info.mtime=0
            archive.addfile(info,io.BytesIO(data))
    payload=lzma.compress(buf.getvalue(),preset=7)
    source_sha=hashlib.sha256(payload).hexdigest()
    files_sha={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    token=base64.b85encode(payload).decode()
    intro='''# NoOracle V9 — Adaptive altitude likelihood-ratio MATD3 scientific CUDA smoke

**Different MDP from historical 18d756ef/e079eace and CT095.** UAV flight range remains 0–150 m. The camera footprint grows with altitude, but positive evidence decreases (Pd/Pf). A true positive sensor sample and Bayesian posterior >=0.99 plus E[yes|occupied]/E[yes|empty]>=4 permit confirmation without fixed 50m cut-off. All report content remains NoOracle cell IDs.

This notebook ONLY runs 64 vector steps on Kaggle T4×2 with E4096, U4, batch8192, H256×512 and 1M hybrid replay. It does NOT run 20,000 episodes, full validation, or Final Test. It does NOT establish convergence. There will likely be zero finished 1500-step episodes.

**Select Kaggle T4 x2 accelerator and enable Internet** (to install dependencies). No Kaggle, W&B or Hugging Face secrets are needed; no online run will be posted. Output: `/kaggle/working/nooracle_adaptive_altitude_gpu_pilot.json`. Source archive SHA-256 is embedded and verified before extracting. Both GPUs must be present; otherwise fail rather than silently train on CPU.'''
    setup='''import torch
assert torch.cuda.is_available() and torch.cuda.device_count()>=2, "Select Kaggle accelerator T4 x2"
VECTOR_STEPS=64
ENVS=4096
BATCH=8192
UPDATES=4
HORIZON=1500
LEARNING_STARTS=131072
print('ADAPTIVE_GPU0',torch.cuda.get_device_name(0))
print('ADAPTIVE_GPU1',torch.cuda.get_device_name(1))
print('ADAPTIVE_ALTITUDE_ONLY_BOUNDED_PILOT')'''
    bootstrap_lines=[
        'import base64,hashlib,io,os,sys,subprocess,shutil,tarfile',
        'from pathlib import Path,PurePosixPath',
        'sha=lambda b:hashlib.sha256(b).hexdigest()',
        'blob=base64.b85decode('+repr(token)+')',
        'assert sha(blob)=='+repr(source_sha),
        'assert blob[:6]==bytes.fromhex("fd377a585a00"), "Expected pinned TAR.XZ source archive"',
        'root=Path("/kaggle/working/nooracle_adaptive_altitude_trial")',
        'if root.exists():raise RuntimeError("Refuse to replace previous trial outputs")',
        'root.mkdir(parents=True)',
        'source_root=root/"source";source_root.mkdir()',
        'files_sha='+repr(files_sha),
        'with tarfile.open(fileobj=io.BytesIO(blob),mode="r:xz") as tf:',
        '    for member in tf:',
        '        rel=PurePosixPath(member.name)',
        '        if not member.isfile() or rel.is_absolute() or any(p in ("..",".","") for p in rel.parts) or member.size>2_000_000:',
        '            raise RuntimeError("Unsafe source archive entry")',
        '        destination=source_root.joinpath(*rel.parts)',
        '        destination.parent.mkdir(parents=True,exist_ok=True)',
        '        content=tf.extractfile(member).read()',
        '        assert sha(content)==files_sha[member.name], "Source file checksum mismatch: "+member.name',
        '        destination.write_bytes(content)',
        'assert len(files_sha)>=500',
        'venv=root/".venv"',
        'subprocess.run([sys.executable,"-m","venv","--system-site-packages","--without-pip",str(venv)],check=True)',
        'python=venv/"bin/python"',
        'subprocess.run([str(python),"-m","pip","install","-e",".[cloud]","-r","scripts/requirements-network.txt"],cwd=source_root,check=True)',
        'print("ADAPTIVE_SOURCE_PIN",sha(blob),"files",len(files_sha),flush=True)',
    ]
    runner='''import json,os,subprocess
from pathlib import Path
output=root/"training"
env=os.environ.copy()
env['PYTHONPATH']=str(source_root)
env['CUDA_VISIBLE_DEVICES']='0,1'
for key in ('WANDB_API_KEY','HF_TOKEN','HUGGINGFACE_HUB_TOKEN'):
    env.pop(key,None)
env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']=env['OPENBLAS_NUM_THREADS']='1'
cmd=[str(python),'-m','scripts.trial_adaptive_altitude_gpu_pilot_20261123',
     '--device','cuda','--output',str(output),
     '--envs',str(ENVS),'--steps',str(VECTOR_STEPS),
     '--batch',str(BATCH),'--updates',str(UPDATES),
     '--horizon',str(HORIZON),'--learning-starts',str(LEARNING_STARTS)]
print('ADAPTIVE_PILOT_LAUNCH',cmd[2], 'worlds',ENVS,'steps',VECTOR_STEPS,flush=True)
proc=subprocess.run(cmd,cwd=source_root,env=env,capture_output=True,text=True,timeout=1800)
rows=[json.loads(line.split(' ',1)[1]) for line in proc.stdout.splitlines()
      if line.startswith('ADAPTIVE_ALTITUDE_GPU_PILOT_JSON ')]
if proc.returncode or len(rows)!=1:
    print('STDOUT_TAIL',proc.stdout[-6000:],flush=True)
    print('STDERR_TAIL',proc.stderr[-5000:],flush=True)
    Path('/kaggle/working/nooracle_adaptive_altitude_gpu_error.json').write_text(
        json.dumps(dict(returncode=proc.returncode,stdout_tail=proc.stdout[-5000:],stderr_tail=proc.stderr[-5000:]),indent=2))
    raise RuntimeError('Adaptive altitude GPU training pilot failed')
result=rows[0]
assert result['transitions']==ENVS*VECTOR_STEPS
assert result['actual_learner_updates']>0
assert result['stop_reason']=='env_step_budget'
assert result['completed_20k'] is False
assert result['observation_version']==9
assert result['gpu_science_convergence_unverified'] is True
final=Path('/kaggle/working/nooracle_adaptive_altitude_gpu_pilot.json')
final.write_text(json.dumps(result,indent=2,allow_nan=False))
print('ADAPTIVE_GPU_REPORT',final,flush=True)
print('ADAPTIVE_GPU_SUMMARY',json.dumps(result,indent=2),flush=True)
'''
    for cell in (setup,'\n'.join(bootstrap_lines),runner):ast.parse(cell)
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro),
        nbformat.v4.new_code_cell(setup),nbformat.v4.new_code_cell('\n'.join(bootstrap_lines)),
        nbformat.v4.new_code_cell(runner)],metadata={
        'kernelspec':{'name':'python3','display_name':'Python 3'},
        'adaptive_altitude_new_mdp':True,'full_20000_training':False,
        'source_sha256':source_sha,'source_commit_basis':source_commit,
        'original_pinned_source_sha256':base_hash,
        'source_files_sha256':files_sha,
        'num_envs':4096,'batch_size':8192,'updates_per_step':4,
        'hidden_sizes':[256,512],'seed':44,'vector_steps':64})
    nbformat.validate(n)
    size=len(nbformat.writes(n).encode())
    if size>=1_000_000:
        raise ValueError('Kaggle source-pinned notebook too large: '+str(size))
    return n


def main():
    n=build_notebook()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUT)
    globals()['FILES']=tuple(n.metadata['source_files_sha256'])
    print('ADAPTIVE_NOTEBOOK_READY',OUT,'SIZE',OUT.stat().st_size,
          'SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest(),
          'SOURCE_SHA',n.metadata['source_sha256'],flush=True)


if __name__=='__main__':main()
else:
    # Exposes the number of files for tests without caching the large archive.
    FILES=tuple(pinned_files()[0])
