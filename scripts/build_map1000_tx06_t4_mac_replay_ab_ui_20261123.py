"""Standalone 3-cell Kaggle T4x2 MAC blockdim + pinned replay performance pilot.

No W&B/HF secrets; never launches production train or uploads externally.
Source is SHA256-pinned and embedded; every result can be repeated from scratch.
"""
from __future__ import annotations
import base64,gzip,hashlib,json,subprocess,ast
from pathlib import Path
import nbformat
import scripts.build_nooracle_map800_posterioronly_tx08_fullflow_ui_20261123 as base
from scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 import (
    map1000_tx06_builder_context,CONFIG,ROOT)

NAME='MAP1000-TX06-T4X2-MACBLOCK-REPLAYPIN-AB-PILOT-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME
PROFILER=ROOT/'scripts/profile_nooracle_warp_phases_20261122.py'
REPLAY=ROOT/'scripts/benchmark_hybrid_replay_cuda_ab_20261123.py'


def pack(raw:bytes):return base64.b85encode(gzip.compress(raw,compresslevel=9,mtime=0)).decode()
def sha(raw:bytes):return hashlib.sha256(raw).hexdigest()


def build_notebook():
    with map1000_tx06_builder_context(): archive=base.minimal_runtime_source_archive()
    assert len(archive)<640_000
    profile_data=PROFILER.read_bytes();replay_data=REPLAY.read_bytes()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    intro='''# MAP1000/TX0.6 — T4×2 benchmark MAC launch & replay pinned staging

**Independent performance pilot, NOT production training.** This notebook does not
access W&B, Hugging Face or Kaggle Secrets; it does not replace an existing
checkpoint. It uses the EXACT MAP1000×1000/40×40/6-UAV/35-target/25-obstacle
Posterior-only + real Warp UavNetSim physics and unchanged source, seed44.

**Phases:** full scientific state SHA256 parity on GPU0 at E16 across MAC
CUDA block sizes 0/8/16/32/64/128; then paired E4096 environment-kernel
profiling (80 vector steps each, excluding compile warmup), then exact
B8192 replay CPU75%/GPU25% sampling A/B pinned staging OFF vs ON on GPU1.
A candidate is admissible ONLY if full state + replay minibatches/RNG match.

**Instructions:** Kaggle Accelerator = GPU T4×2, Internet ON. Run all 3 cells.
Upload `MAP1000_TX06_T4_MAC_REPLAY_AB_RESULTS.json` from /kaggle/working.
CUDA/T4 improvements remain unverified until results exist. Full 20,000-episode
Train→Val100/every Actor→best→heldout Test1000 belongs to a separate notebook.
'''
    pre='''import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "SELECT KAGGLE T4 X2 ACCELERATOR"
MAC_BLOCK_DIMS=[0,8,16,32,64,128]
ENVS=4096
WARMUP=8
STEPS=80
PARITY_ENVS=16
PARITY_STEPS=64
REPEATS=2
REPLAY_BATCH=8192
REPLAY_CAPACITY=16384
REPLAY_SAMPLES=6
assert ENVS==4096 and REPLAY_BATCH==8192
assert MAC_BLOCK_DIMS[0]==0 and all(v in (0,1,2,4,8,16,32,64,128,256) for v in MAC_BLOCK_DIMS)
assert 1<=REPEATS<=3 and 40<=STEPS<=250
print("KAGGLE_T4X2_MAC_REPLAY_AB",[torch.cuda.get_device_name(i) for i in range(2)],flush=True)
'''
    payload=base64.b85encode(archive).decode()
    # Source extraction is deliberately explicit, checksummed and path-guarded.
    code='''import os,sys,io,tarfile,base64,gzip,hashlib,json,subprocess,shutil,statistics,time
from pathlib import Path,PurePosixPath
sha=lambda b:hashlib.sha256(b).hexdigest()
source=base64.b85decode("'''+payload+'''")
assert sha(source)=="'''+sha(archive)+'''", "Pinned scientific source checksum mismatch"
root=Path("/kaggle/working/map1000_t4x2_mac_replay_ab")
if root.exists():raise RuntimeError("Fresh Kaggle working dir required; refuse overwrite")
root.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(source),mode="r:gz") as tf:
    total=0
    for member in tf:
        rel=PurePosixPath(member.name)
        if (not member.isfile() or rel.is_absolute() or any(k in ("..",".","") for k in rel.parts)
                or member.size>2_000_000):raise RuntimeError("Unsafe source member")
        total+=member.size
        if total>42_000_000:raise RuntimeError("Oversized source archive")
        target=root.joinpath(*rel.parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        with tf.extractfile(member) as f,target.open("wb") as w:shutil.copyfileobj(f,w)
'''
    for name,raw in [(PROFILER.name,profile_data),(REPLAY.name,replay_data)]:
        data=pack(raw)
        code+='''worker=gzip.decompress(base64.b85decode("'''+data+'''"))
assert sha(worker)=="'''+sha(raw)+'''"
(root/"scripts"/"'''+name+'''").write_bytes(worker)
'''
    code+='''
print("SCIENTIFIC_SOURCE_SHA256",sha(source),"SOURCE_GIT_COMMIT","'''+commit+'''",flush=True)
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","-q",
                "-e",".[cloud]","-r","scripts/requirements-network.txt"],
               cwd=root,check=True,timeout=900)
env=os.environ.copy()
env["PYTHONPATH"]=str(root)
env["CUDA_VISIBLE_DEVICES"]="0,1"
env["OMP_NUM_THREADS"]=env["MKL_NUM_THREADS"]=env["OPENBLAS_NUM_THREADS"]="1"
for key in ("WANDB_API_KEY","HF_TOKEN","HUGGINGFACE_HUB_TOKEN"):
    env.pop(key,None)
config="'''+CONFIG+'''"
report={"kind":"paired_MAP1000_T4x2_MAC_cuda_blockdim_and_hybrid_replay_pilot_not_training",
        "source_sha256":sha(source),"source_commit":"'''+commit+'''",
        "device_names":[torch.cuda.get_device_name(i) for i in range(2)],
        "science":{"map_m":1000,"grid":40,"tx_max_w":.6,"posterior_only":True,
                   "envs":ENVS,"replay_cpu_fraction":.75,"batch":REPLAY_BATCH},
        "mac_gpu_bitwise":[],"mac_performance":[],"replay_performance":[]}
out=Path("/kaggle/working/MAP1000_TX06_T4_MAC_REPLAY_AB_RESULTS.json")
def write():out.write_text(json.dumps(report,indent=2,allow_nan=False))
def run_bench(label,argv,timeout=1200):
    print("AB_START",label,flush=True)
    p=subprocess.run([sys.executable,*argv],cwd=root,env=env,capture_output=True,
                      text=True,timeout=timeout)
    if p.returncode:
        report["failure"]={"stage":label,"code":p.returncode,
                            "stderr_tail":p.stderr[-3800:],"stdout_tail":p.stdout[-2000:]}
        write();print("FAILED",label,p.stderr[-2000:],flush=True)
        raise RuntimeError("Benchmark subprocess failed: "+label)
    return p.stdout

def profile(dim,small=False):
    argv=["-m","scripts.profile_nooracle_warp_phases_20261122","--config",config,
          "--device","cuda","--envs",str(PARITY_ENVS if small else ENVS),
          "--warmup",str(4 if small else WARMUP),
          "--steps",str(PARITY_STEPS if small else STEPS),
          "--mac-block-dim",str(dim)]
    if small:argv.append("--state-digest")
    stdout=run_bench(f"mac_dim_{dim}_small_{small}",argv)
    rows=[json.loads(line.split(" ",1)[1]) for line in stdout.splitlines()
          if line.startswith("UAV_WARP_PROFILE_JSON ")]
    assert len(rows)==1,(dim,stdout[-1500:])
    return rows[0]

for dim in MAC_BLOCK_DIMS:
    row=profile(dim,small=True)
    report["mac_gpu_bitwise"].append({"mac_block_dim":dim,
       "state_sha256":row["world_state_sha256"],"envs":PARITY_ENVS,
       "steps":PARITY_STEPS})
    write()
assert len({x["state_sha256"] for x in report["mac_gpu_bitwise"]})==1, \
    "CUDA MAC scheduling changed full simulated state; reject ALL candidates"
print("MAC_ALL_167_WORLDSTATE_FIELDS_GPU_SHA256_MATCH",flush=True)

for repeat in range(REPEATS):
    order=MAC_BLOCK_DIMS if repeat%2==0 else list(reversed(MAC_BLOCK_DIMS))
    for dim in order:
        item=profile(dim)
        report["mac_performance"].append({"repeat":repeat,
            "mac_block_dim":dim,"world_state_parity":True,
            "env_transitions_per_second":item["env_transitions_per_second"],
            "packet_kernel_ms":item["per_kernel_ms_total"].get("packet_step"),
            "packet_kernel_pct":item["per_kernel_percent"].get("packet_step"),
            "wall_seconds":item["wall_seconds"]})
        write()
        print("MAC_PROFILE",dim,repeat,
              round(item["env_transitions_per_second"],2),"env_transitions_per_sec",flush=True)

for repeat in range(REPEATS):
    order=("off","on") if repeat%2==0 else ("on","off")
    for mode in order:
        argv=["-m","scripts.benchmark_hybrid_replay_cuda_ab_20261123",
            "--device","cuda:1","--staging",mode,"--batch",str(REPLAY_BATCH),
            "--capacity",str(REPLAY_CAPACITY),"--samples",str(REPLAY_SAMPLES)]
        stdout=run_bench(f"replay_{mode}_r{repeat}",argv)
        rows=[json.loads(x.split(" ",1)[1]) for x in stdout.splitlines()
              if x.startswith("HYBRID_REPLAY_CUDA_AB_JSON ")]
        assert len(rows)==1
        rec=rows[0];rec["repeat"]=repeat
        report["replay_performance"].append(rec)
        write()
        print("REPLAY_PROFILE",mode,repeat,
              round(rec["median_sample_seconds"],5),"sample_seconds",flush=True)
for k in ("all_sampled_tensor_sha256","rng_state_sha256","seen_bitmap_sha256"):
    assert len({x[k] for x in report["replay_performance"]})==1, \
        "Replay CUDA A/B failed data/RNG parity; reject pinned staging"

med={dim:statistics.median(x["env_transitions_per_second"]
                            for x in report["mac_performance"] if x["mac_block_dim"]==dim)
     for dim in MAC_BLOCK_DIMS}
base_tps=med[0]
best=max(med,key=lambda d:med[d])
report["mac_summary"]={"tps_median_by_block_dim":med,"baseline_tps":base_tps,
    "best_block_dim":best,"best_speedup_percent":100*(med[best]/base_tps-1),
    "candidate_recommended_only_if_gain_over_5pct":bool(best!=0 and med[best]>1.05*base_tps)}
replay={mode:statistics.median(x["median_sample_seconds"]
                               for x in report["replay_performance"] if x["staging"]==mode)
        for mode in ("off","on")}
report["replay_summary"]={"median_seconds":replay,
   "pinned_speedup_percent":100*(replay["off"]/replay["on"]-1),
   "pinned_recommended_only_if_gain_over_5pct":bool(replay["on"]<replay["off"]/1.05)}
# Conservative explicit gate: no automatic production promotion.
report["science_state_parity_verified_gpu"]=True
report["replay_sample_parity_verified_gpu"]=True
report["completed"]=True
write()
print("MAP1000_T4X2_MAC_REPLAY_AB_COMPLETE",out,flush=True)
print("MAC_TPS_MEDIAN",med,"REPLAY_MEDIAN_SECONDS",replay,flush=True)
'''
    ast.parse(pre);ast.parse(code)
    nb=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(intro),
        nbformat.v4.new_code_cell(pre),
        nbformat.v4.new_code_cell(code)],metadata={
          'kernelspec':{'name':'python3','display_name':'Python 3'},
          'source_sha256':sha(archive),'reference_commit':commit,'science_config':CONFIG,
          'gpu_t4x2_production_verified':False,'cuda_A_B_done':False,
          'report_output':'/kaggle/working/MAP1000_TX06_T4_MAC_REPLAY_AB_RESULTS.json',
          'model_and_scenario':{'world_m':1000,'grid_n':40,'tx_max_w':.6,'uavs':6,
                'targets':35,'obstacles':25,'envs':4096,'batch':8192,'replay_cpu_frac':.75}})
    nbformat.validate(nb)
    count=len(nbformat.writes(nb).encode())
    if count>=999000:raise ValueError(f'Notebook exceeds Kaggle source cap: {count}')
    return nb


def main():
    nb=build_notebook();OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,OUT)
    print('T4_MAC_REPLAY_AB_NOTEBOOK',OUT,'BYTES',OUT.stat().st_size,'BUNDLE_SHA',nb.metadata['source_sha256'])


if __name__=='__main__':main()
