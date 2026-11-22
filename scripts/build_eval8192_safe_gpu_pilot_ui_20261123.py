"""Package the SHA-pinned NoOracle V9 code for a GPU-only, no-training trial."""
from pathlib import Path
import re, base64, hashlib, lzma, ast
import nbformat

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'notebooks/MATD3-NOORACLE-V9-CT095-H256x256-EVAL8192-CONTIGUOUS-TEST1000-CANDIDATE-T4X2.ipynb'
SMOKE=ROOT/'scripts/gpu_smoke_eval8192_no_training_20261123.py'
OUTPUT=ROOT/'notebooks/MATD3-NOORACLE-V9-CT095-H256x256-EVAL8192-GPU-SMOKE-NO-TRAIN-T4X2-20261123.ipynb'

def build_notebook():
    original=nbformat.read(SOURCE,as_version=4)
    source_code=original.cells[2].source
    m=re.search(r'payload\s*=\s*base64\.b85decode\("([^"\n]+)"\)',source_code)
    assert m, "Expected pinned payload"
    embedded=m.group(1)
    source_sha=original.metadata['source_sha256']
    assert hashlib.sha256(base64.b85decode(embedded)).hexdigest()==source_sha
    script=SMOKE.read_bytes()
    script_sha=hashlib.sha256(script).hexdigest()
    script_b85=base64.b85encode(lzma.compress(script,preset=9)).decode()
    m1="""# NoOracle V9 — GPU T4x2 Eval8192 (NO TRAINING)

Notebook này chỉ kiểm tra GPU. Không chạy train 20.000 episodes, không cập nhật W&B hay Hugging Face, không cần token.

Hai GPU đồng thời tạo mỗi GPU 8.192 Warp worlds; 81 Actor ngẫu nhiên cố định x 100 seed thật = 8.100 worlds, còn 92 world dư không tính metrics. Kịch bản NoOracle CT095, H256x256, PHY/MAC được bật.

Horizon chỉ 12 bước để smoke test. Kết quả này kiểm tra khả năng cấp phát CUDA và tính đúng flow, không đại diện cho tốc độ training 20.000 episodes hoặc độ chính xác model đã học.

Dùng Kaggle Accelerator GPU T4 x2 và bật Internet. Sau khi chạy, tải file eval8192_smoke_report.json và gửi lại để phân tích."""
    pre="""import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Choose GPU T4 x2"
SMOKE_STEPS=12
POLICIES_PER_GPU=81
SEEDS_PER_CHECKPOINT=100
PHYSICAL_WORLDS_PER_GPU=8192
assert POLICIES_PER_GPU*SEEDS_PER_CHECKPOINT==8100
assert PHYSICAL_WORLDS_PER_GPU-POLICIES_PER_GPU*SEEDS_PER_CHECKPOINT==92
for idx in range(2): print("GPU",idx,torch.cuda.get_device_name(idx))
print("SMOKE_ONLY_NO_TRAINING",flush=True)
"""
    unpack='''import base64,hashlib,io,lzma,os,sys,subprocess,tarfile,shutil
from pathlib import Path,PurePosixPath
blob=base64.b85decode("''' + embedded + '''")
assert hashlib.sha256(blob).hexdigest()=="''' + source_sha + '''"
root=Path("/kaggle/working/ct095_eval8192_trial_source")
if root.exists():raise RuntimeError("Run this trial in a fresh Kaggle session")
root.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(blob),mode="r:xz") as tf:
    total=0
    for member in tf:
        relative=PurePosixPath(member.name)
        if (not member.isfile() or relative.is_absolute() or
            any(part in (".","..","") for part in relative.parts) or member.size>2_000_000):
            raise RuntimeError("Unsafe source entry")
        total+=member.size
        if total>40_000_000:raise RuntimeError("Oversized source")
        dest=root.joinpath(*relative.parts)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tf.extractfile(member) as inp,dest.open("wb") as out:shutil.copyfileobj(inp,out)
smoke=lzma.decompress(base64.b85decode("''' + script_b85 + '''"))
assert hashlib.sha256(smoke).hexdigest()=="''' + script_sha + '''"
probe=root/"scripts/gpu_smoke_eval8192_no_training_20261123.py"
probe.parent.mkdir(parents=True,exist_ok=True)
probe.write_bytes(smoke)
py=root/".venv/bin/python"
subprocess.run([sys.executable,"-m","venv","--system-site-packages","--without-pip",
                str(root/".venv")],check=True)
subprocess.run([str(py),"-m","pip","install","-e",".[cloud]","-r",
                "scripts/requirements-network.txt"],cwd=root,check=True)
print("PINNED_SOURCE_SHA256",hashlib.sha256(blob).hexdigest(),flush=True)
print("PINNED_GPU_SMOKE_SCRIPT_SHA256",hashlib.sha256(smoke).hexdigest(),flush=True)
'''
    run="""import json,os,subprocess
from pathlib import Path
children=[]
for gpu in (0,1):
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    env["OMP_NUM_THREADS"]="1"
    env["MKL_NUM_THREADS"]="1"
    env["OPENBLAS_NUM_THREADS"]="1"
    env["PYTHONPATH"]=str(root)
    out=Path("/kaggle/working")/f"eval8192_smoke_gpu{gpu}.json"
    argv=[str(py),"-m","scripts.gpu_smoke_eval8192_no_training_20261123",
          "--device","cuda","--gpu-index",str(gpu),
          "--policies",str(POLICIES_PER_GPU),
          "--episodes",str(SEEDS_PER_CHECKPOINT),
          "--world-pool",str(PHYSICAL_WORLDS_PER_GPU),
          "--max-steps",str(SMOKE_STEPS),
          "--checkpoint-offset",str(81*gpu),
          "--output",str(out)]
    p=subprocess.Popen(argv,cwd=root,env=env,stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,text=True)
    children.append((gpu,p))
    print("GPU_SMOKE_STARTED",gpu,flush=True)
reports=[]
for gpu,p in children:
    try:
        stdout,stderr=p.communicate(timeout=2400)
    except subprocess.TimeoutExpired:
        p.kill()
        stdout,stderr=p.communicate()
        stderr+=" TIMEOUT_2400_S"
    rows=[json.loads(line.split(" ",1)[1]) for line in stdout.splitlines()
          if line.startswith("UAV_V9_EVAL8192_GPU_SMOKE_JSON ")]
    success=(p.returncode==0 and len(rows)==1)
    print("GPU_SMOKE_RESULT",gpu,"PASS" if success else "FAIL",flush=True)
    if not success:
        print("STDOUT_TAIL",gpu,stdout[-1800:],flush=True)
        print("STDERR_TAIL",gpu,stderr[-2800:],flush=True)
    reports.append({"gpu":gpu,"ok":success,"returncode":p.returncode,
                    "measurements":rows,"error_tail":"" if success else stderr[-2800:]})
result={"kind":"GPU_ONLY_NO_TRAINING","success":all(r["ok"] for r in reports),
        "source_sha256":hashlib.sha256(blob).hexdigest(),
        "worlds_per_gpu":8192,"real_policies_per_gpu":81,
        "seeds_per_policy":100,"smoke_max_steps":SMOKE_STEPS,
        "workers":reports}
out=Path("/kaggle/working/eval8192_smoke_report.json")
out.write_text(json.dumps(result,indent=2,allow_nan=False))
print("REPORT_SAVED",out,"BOTH_GPU_PASS",result["success"],flush=True)
if not result["success"]:
    raise RuntimeError("GPU smoke failed; do not start production training")
"""
    for code in (pre,unpack,run):ast.parse(code)
    notebook=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(m1),
        nbformat.v4.new_code_cell(pre),
        nbformat.v4.new_code_cell(unpack),
        nbformat.v4.new_code_cell(run)],metadata={
        'kernelspec':{'display_name':'Python 3','name':'python3'},
        'training_permitted':False,
        'smoke_only':True,
        'source_sha256':source_sha,
        'source_commit':original.metadata['reference_commit'],
        'smoke_script_sha256':script_sha,
        'gpu_world_pool_size':8192,
        'gpu_policies':81,
        'seeds_per_policy':100,
        'smoke_steps':12,
        'production_steps':1500,
        'production_final_test_episodes':1000,
    })
    nbformat.validate(notebook)
    assert len(nbformat.writes(notebook).encode())<1_000_000
    assert not any(x in run for x in ("one_session","train(","UAV_TRAINING_ONLY"))
    return notebook

if __name__=='__main__':
    n=build_notebook()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUTPUT)
    print("GPU_ONLY_NOTEBOOK",OUTPUT,"SIZE",OUTPUT.stat().st_size,
          "SOURCE_SHA",n.metadata["source_sha256"])
