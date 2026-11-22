"""Self-contained pinned V9 vs optimized GPU0 Warp-4096 kernel profiling notebook.

Embeds the immutable original V9 Kaggle xz archive plus a SHA-verified
small overlay for changed performance modules (not a two-run training job).
"""
from __future__ import annotations
import ast
import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import re
import tarfile
import nbformat

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'notebooks/MATD3-EVAL100-ACTORBN-PLAINCRITIC-NOORACLE-V9-E4096-U4-B8192-H256x512-S44-KAGGLE-UI.ipynb'
OUTPUT=ROOT/'notebooks/MATD3-V9-TRAINING-CUDA-AB-4096-MAC-SPARSEFAST-V2-T4x2.ipynb'
CHANGED_MODULES=(
    'uav_network/communication/gpu/mac.py',
    'uav_network/communication/gpu/reports.py',
    'uav_network/communication/gpu/telemetry.py',
    'uav_network/environment/gpu/bayesian.py',
    'uav_network/environment/gpu/observations.py',
    'uav_network/environment/gpu/vector.py',
    'uav_network/communication/gpu/streaming.py',
)
BENCH_SCRIPTS=('scripts/profile_nooracle_warp_phases_20261122.py',
               'scripts/verify_v9_training_parity_20261122.py')


def load_original():
    nb=nbformat.read(REFERENCE,as_version=4)
    src=nb.cells[2].source
    b=re.search(r'payload\s*=\s*base64\.b85decode\("([^"\n]+)"\)',src)
    d=re.search(r'if hashlib\.sha256\(payload\)\.hexdigest\(\) != "([0-9a-f]{64})":',src)
    if not b or not d:raise ValueError('Cannot recover original V9 Kaggle source archive')
    blob=base64.b85decode(b.group(1))
    if hashlib.sha256(blob).hexdigest()!=d.group(1):raise ValueError('Pinned V9 source hash changed')
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:xz') as tf:
        original={x.name:tf.extractfile(x).read() for x in tf if x.isfile()}
    if any(k not in original for k in CHANGED_MODULES):raise ValueError('Original source missing module')
    return b.group(1),d.group(1)


def build():
    original_b85,original_sha=load_original()
    overlay={str(k):(ROOT/k).read_bytes() for k in CHANGED_MODULES+BENCH_SCRIPTS}
    overlay_manifest={k:hashlib.sha256(v).hexdigest() for k,v in overlay.items()}
    payload=json.dumps({k:base64.b85encode(v).decode() for k,v in overlay.items()},sort_keys=True).encode()
    overlay_b85=base64.b85encode(lzma.compress(payload,preset=9)).decode()
    marker='''# Compare unchanged V9 random-action Warp physics to the accelerated source.
# This is a KERNEL PROFILER, not a full MATD3 learner or 20k episode run.
ENVS = 4096
WARMUP_STEPS = 8
MEASURED_STEPS = 96
REPEATS = 3
GPU_DEVICE = 0
TRAJECTORY_PARITY_ENVS = 4
TRAJECTORY_PARITY_STEPS = 24

import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Choose Kaggle T4 x2"
assert ENVS == 4096 and REPEATS >= 2 and MEASURED_STEPS >= 8
for dev in range(2):print("CUDA",dev,torch.cuda.get_device_name(dev))
print("V9 PERF A/B: same source snapshot, same NoOracle scene and network, GPU0 serial trials")
'''
    runner=f'''import base64,hashlib,io,json,lzma,os
from pathlib import Path,PurePosixPath
import shutil,subprocess,sys,tarfile,time,statistics

root=Path('/kaggle/working/v9_training_ab_cpu_cuda')
root.mkdir(parents=True,exist_ok=True)
old=root/'original_1346ad47'
new=root/'optimized_v2_mac_session_sparse'
if old.exists() or new.exists():
    raise RuntimeError('A/B source directory already exists; start a fresh Kaggle session')

orig_blob=base64.b85decode({original_b85!r})
assert hashlib.sha256(orig_blob).hexdigest()=={original_sha!r}
old.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(orig_blob),mode='r:xz') as tf:
    written=0
    for item in tf:
        name=PurePosixPath(item.name)
        if (not item.isfile() or name.is_absolute() or '..' in name.parts
            or item.size>2_000_000):raise RuntimeError('Unsafe pinned source entry')
        written+=item.size
        if written>40_000_000:raise RuntimeError('Pinned source budget exceeded')
        dest=old.joinpath(*name.parts)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tf.extractfile(item) as inp,dest.open('wb') as out:shutil.copyfileobj(inp,out)

shutil.copytree(old,new)
encoded=json.loads(lzma.decompress(base64.b85decode({overlay_b85!r})))
manifest={overlay_manifest!r}
assert set(encoded)==set(manifest)
for name,body in encoded.items():
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:raise RuntimeError('Unsafe overlay path')
    data=base64.b85decode(body)
    if hashlib.sha256(data).hexdigest()!=manifest[name]:raise RuntimeError('Overlay checksum mismatch: '+name)
    # The profiler/parity scripts are shared instrumentation, not part of
    # the original V9 scientific simulation.
    dest=new.joinpath(*path.parts)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(data)
    if name.startswith('scripts/'):
        alt=old.joinpath(*path.parts)
        alt.parent.mkdir(parents=True,exist_ok=True)
        alt.write_bytes(data)

py=old/'.venv/bin/python'
subprocess.run([sys.executable,'-m','venv','--system-site-packages','--without-pip',str(old/'.venv')],check=True)
subprocess.run([str(py),'-m','pip','install','-e','.[cloud]','-r','scripts/requirements-network.txt'],cwd=old,check=True)
report={{'benchmark_kind':'V9_mac_v2_random_action_cuda_environment_only_not_training',
  'source_original_sha256':{original_sha!r},'overlay_sha256':manifest,
  'envs':ENVS,'steps':MEASURED_STEPS,'warmup':WARMUP_STEPS,'repeats':REPEATS,'trials':[]}}
json_path=root/'v9_gpu_kernel_ab.json'

# Sequential trials on GPU0 remove contention between the two variants.
# Alternate order to reduce temperature/cache-order bias.
for repeat in range(REPEATS):
    variants=['original','optimized'] if repeat%2==0 else ['optimized','original']
    for variant in variants:
        directory=old if variant=='original' else new
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=str(GPU_DEVICE)
        env['PYTHONPATH']=str(directory)
        cmd=[str(py),str(directory/'scripts/profile_nooracle_warp_phases_20261122.py'),
           '--config','configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml',
           '--device','cuda','--envs',str(ENVS),'--steps',str(MEASURED_STEPS),
           '--warmup',str(WARMUP_STEPS)]
        print('PROFILE_START',variant,repeat,'GPU',GPU_DEVICE,flush=True)
        t0=time.perf_counter()
        result=subprocess.run(cmd,cwd=directory,env=env,check=True,capture_output=True,text=True)
        raw=[x for x in result.stdout.splitlines() if x.startswith('UAV_WARP_PROFILE_JSON ')]
        if len(raw)!=1:raise RuntimeError('Cannot find truthful Warp per-kernel output')
        data=json.loads(raw[0].split(' ',1)[1]);data.update(variant=variant,repeat=repeat,
                  process_wall_seconds=time.perf_counter()-t0)
        report['trials'].append(data)
        json_path.write_text(json.dumps(report,indent=2))
        print('PROFILE_DONE',variant,'trans_per_s',round(data['env_transitions_per_second'],2),
              'wall_s',round(data['wall_seconds'],3),flush=True)

for variant in ('original','optimized'):
    values=[t['env_transitions_per_second'] for t in report['trials'] if t['variant']==variant]
    report[variant+'_median_transitions_per_s']=statistics.median(values)
base=report['original_median_transitions_per_s'];optimized=report['optimized_median_transitions_per_s']
report['median_improvement_percent']=100*(optimized/base-1)
report['trajectory_parity']='not_tested'
print('E4096_CUDA_MEDIAN_SPEEDUP_PERCENT',round(report['median_improvement_percent'],2),flush=True)

# Actual same-seed CUDA trajectory parity for separate V9 scientific modules.
hashes=[]
for variant,directory in (('original',old),('optimized',new)):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(GPU_DEVICE);env['PYTHONPATH']=str(directory)
    cmd=[str(py),str(directory/'scripts/verify_v9_training_parity_20261122.py'),
         '--device','cuda','--envs',str(TRAJECTORY_PARITY_ENVS),
         '--steps',str(TRAJECTORY_PARITY_STEPS)]
    result=subprocess.run(cmd,cwd=directory,env=env,capture_output=True,text=True,check=True)
    rows=[x for x in result.stdout.splitlines() if x.startswith('TRAJECTORY_PARITY_JSON ')]
    if len(rows)!=1:raise RuntimeError('CUDA trajectory hash missing: '+variant)
    hashes.append(json.loads(rows[0].split(' ',1)[1]))
report['trajectory_parity']= ('matched' if hashes[0]==hashes[1] else 'mismatch')
report['trajectory_hashes_by_variant']={{'original':hashes[0],'optimized':hashes[1]}}
json_path.write_text(json.dumps(report,indent=2))
print('CUDA_TRAJECTORY_PARITY',report['trajectory_parity'],flush=True)
if report['trajectory_parity']!='matched':raise RuntimeError('CUDA parity MISMATCH; do not treat benchmark as validated')
print('GPU_PROFILE_RESULT',json_path,flush=True)
'''
    ast.parse(marker);ast.parse(runner)
    nb=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('''# NoOracle V9 training bottleneck: A/B Warp CUDA profiling

**Đây là benchmark môi trường GPU, KHÔNG train lại 20k episodes.** Chọn Kaggle Accelerator **GPU T4x2**, bật Internet để cài dependencies. Không cần W&B/HF token. Ở GPU0, notebook lần lượt mô phỏng cùng V9 gốc và V9 tối ưu, mỗi lần **4096 worlds**, đo per-kernel CUDA time và simulated transitions/second. Nó kiểm tra đồng nhất trajectory GPU bằng checksum.

Source cũ được nhúng theo SHA256 chính xác từ notebook V9. Source tối ưu chỉ ghi đè những file đã thay đổi từ branch riêng, kiểm tra SHA256 từng file. Vẫn mô phỏng đầy đủ Bayesian sensor, bản đồ, PHY/MAC và report; KHÔNG chạy chế độ network-disabled. 3 lượt A/B trên cùng GPU giúp giảm sai lệch do so sánh hai GPU khác nhau. Kết quả: `/kaggle/working/v9_training_ab_cpu_cuda/v9_gpu_kernel_ab.json`.

**Chú ý:** đây là *environment-only* random-action benchmark, chưa đo riêng replay CPU75/GPU25, optimizer và actor inference của toàn bộ training. Dùng kết quả để xác nhận kernel nào chiếm thời gian T4; đừng tự suy ra mức tăng tốc end-to-end.
'''),nbformat.v4.new_code_cell(marker),nbformat.v4.new_code_cell(runner)],
        metadata={'kernelspec':{'display_name':'Python 3','name':'python3'},
                  'performance_kind':'V9_CUDA_environment_only_4096_MAC_V2_AB',
                  'original_v9_sha256':original_sha,
                  'optimized_module_sha256':overlay_manifest,
                  'has_train_commands':False,'scientific_environment_changed':False})
    nbformat.validate(nb)
    if len(nbformat.writes(nb).encode())>=1_000_000:
        raise RuntimeError('Notebook exceeds Kaggle code size guard')
    return nb


def main():
    n=build();OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUTPUT)
    print('CUDA_AB_NOTEBOOK',OUTPUT,'BYTES',OUTPUT.stat().st_size,
          'ORIGINAL_SHA256',n.metadata['original_v9_sha256'])

if __name__=='__main__':main()
