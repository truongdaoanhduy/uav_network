"""Build standalone Kaggle paired real MATD3 bounded-training speed test."""
from __future__ import annotations
import ast,base64,hashlib,io,json,lzma,re,tarfile,subprocess
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
BASE_DIR=Path('/home/aduy/Documents/NCKH/uav_network_eval8192_perf_candidate_20261123')
REF=BASE_DIR/'notebooks/MATD3-NOORACLE-V9-CT095-H256x256-EVAL8192-CONTIGUOUS-TEST1000-CANDIDATE-T4X2.ipynb'
FILES=('uav_network/algorithms/hybrid_replay.py','uav_network/experiments/training.py')
WORKER=ROOT/'scripts/trial_real_matd3_train_gpu_worker_20261123.py'
OUT=ROOT/'notebooks/MATD3-NOORACLE-V9-CT095-H256x256-TRAIN-SPEED-AB-E4096-U4-B8192-T4X2-PILOT-20261123.ipynb'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def enc(raw):return base64.b85encode(lzma.compress(raw,preset=9)).decode()

def build_notebook():
    base=nbformat.read(REF,as_version=4)
    m=re.search(r'payload\s*=\s*base64\.b85decode\("([^"\n]+)"\)',base.cells[2].source)
    assert m,'Pinned baseline archive absent'
    token=m.group(1)
    source_bytes=base64.b85decode(token)
    source_sha=base.metadata['source_sha256']
    assert sha(source_bytes)==source_sha
    overlays={}
    original_hashes={}
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(source_bytes)),mode='r:') as tf:
        for file in FILES:
            raw=tf.extractfile(file).read()
            assert raw==Path(BASE_DIR,file).read_bytes(),file
            patched=Path(ROOT,file).read_bytes()
            assert raw!=patched,file
            original_hashes[file]=sha(raw)
            overlays[file]={'payload':enc(patched),'sha256':sha(patched)}
    work=WORKER.read_bytes()
    work_token=enc(work)
    work_sha=sha(work)
    intro="""# MATD3 NoOracle V9 — T4 x2 real TRAINING speed benchmark

Only a short, bounded A/B training benchmark. It does not run all 20,000 episodes.

Choose Kaggle Accelerator GPU T4 x2 and Internet ON. No W&B/HF tokens or uploads.
Both variants use the SAME scenario (6 UAV, 35 targets, 25 obstacles, 1000m map), Bayesian CT095, Warp full PHY/MAC, MATD3 hidden256x256, E4096, U4, B8192, 75pct CPU and 25pct GPU replay of 1M transitions, and seed44.

Every trial uses GPU0 as rollout/collector and GPU1 as MATD3 learner. The new version only skips replay seen-index sorting and optionally copies Actor weights directly from GPU1 to GPU0.
This notebook runs two paired A/B repetitions, 64 vector steps per training trial (not full training).
Output JSON: /kaggle/working/v9_train_speed_ab_report.json. Send that JSON for analysis.

This is a speed/logic test, not a new trained model; Eval and held-out Test are intentionally not run. If checksum differs, the candidate MUST NOT be promoted."""
    pre="""import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Select Kaggle T4 x2 accelerator!"
PAIRS=2
VECTOR_STEPS=64
ENVS=4096
BATCH=8192
UPDATES=4
LEARNING_STARTS=131072
HORIZON=1500
assert 1 <= PAIRS <= 3
assert 40 <= VECTOR_STEPS <= 256
assert (ENVS,BATCH,UPDATES,LEARNING_STARTS,HORIZON)==(4096,8192,4,131072,1500)
for i in range(2):print("TRAINING_GPU",i,torch.cuda.get_device_name(i))
print("TRAINING_AB_BOUNDED_NO_20K",flush=True)
"""
    parts=[
        "import base64,hashlib,io,lzma,os,sys,subprocess,shutil,tarfile",
        "from pathlib import Path,PurePosixPath",
        "sha=lambda b:hashlib.sha256(b).hexdigest()",
        "blob=base64.b85decode("+repr(token)+")",
        "assert sha(blob)=="+repr(source_sha),
        'root=Path("/kaggle/working/ct095_matd3_training_ab")',
        'if root.exists():raise RuntimeError("Use fresh Kaggle session; refuse to replace trial results")',
        'root.mkdir(parents=True)',
        'old_root=root/"source_baseline";new_root=root/"source_candidate"',
        "old_root.mkdir()",
        'with tarfile.open(fileobj=io.BytesIO(lzma.decompress(blob)),mode="r:xz") as tf:',
        "    total=0",
        "    for member in tf:",
        "        rel=PurePosixPath(member.name)",
        '        if not member.isfile() or rel.is_absolute() or any(p in ("..",".","") for p in rel.parts) or member.size>2_000_000:',
        '            raise RuntimeError("Unsafe embedded source")',
        "        total+=member.size",
        '        if total>40_000_000:raise RuntimeError("Embedded source too large")',
        "        path=old_root.joinpath(*rel.parts)",
        "        path.parent.mkdir(parents=True,exist_ok=True)",
        '        with tf.extractfile(member) as src,path.open("wb") as dst:shutil.copyfileobj(src,dst)',
        "shutil.copytree(old_root,new_root)",
        "original_hashes="+repr(original_hashes),
        "overlays="+repr(overlays),
        "for file,meta in overlays.items():",
        "    assert sha((old_root/file).read_bytes())==original_hashes[file]",
        '    candidate=lzma.decompress(base64.b85decode(meta["payload"]))',
        '    assert sha(candidate)==meta["sha256"]',
        "    (new_root/file).write_bytes(candidate)",
        "worker=lzma.decompress(base64.b85decode("+repr(work_token)+"))",
        "assert sha(worker)=="+repr(work_sha),
        "for folder in (old_root,new_root):",
        '    (folder/"scripts/trial_real_matd3_train_gpu_worker_20261123.py").write_bytes(worker)',
        'venv=root/".venv"',
        'subprocess.run([sys.executable,"-m","venv","--system-site-packages","--without-pip",str(venv)],check=True)',
        'python=venv/"bin/python"',
        'subprocess.run([str(python),"-m","pip","install","-e",".[cloud]","-r","scripts/requirements-network.txt"],cwd=old_root,check=True)',
        'print("BASELINE_SOURCE_SHA256",sha(blob),"CANDIDATE_SOURCE_SHA256",{k:v["sha256"] for k,v in overlays.items()},flush=True)'
    ]
    bootstrap="\n".join(parts)
    run="""import json,os,statistics,subprocess
from pathlib import Path

runs=[]
for repeat in range(PAIRS):
    order=("baseline","candidate") if repeat%2==0 else ("candidate","baseline")
    for variant in order:
        folder=old_root if variant=="baseline" else new_root
        output=root/("run_"+str(repeat)+"_"+variant)
        env=os.environ.copy()
        env["PYTHONPATH"]=str(folder)
        env["CUDA_VISIBLE_DEVICES"]="0,1"
        for key in ("WANDB_API_KEY","HF_TOKEN","HUGGINGFACE_HUB_TOKEN"):env.pop(key,None)
        env["OMP_NUM_THREADS"]=env["MKL_NUM_THREADS"]=env["OPENBLAS_NUM_THREADS"]="1"
        argv=[str(python),"-m","scripts.trial_real_matd3_train_gpu_worker_20261123",
              "--mode",variant,"--device","cuda","--output",str(output),
              "--steps",str(VECTOR_STEPS),"--envs",str(ENVS),
              "--batch",str(BATCH),"--updates",str(UPDATES),
              "--horizon",str(HORIZON),"--learning-starts",str(LEARNING_STARTS)]
        print("TRAIN_AB_START",repeat,variant,flush=True)
        try:
            proc=subprocess.run(argv,cwd=folder,env=env,text=True,capture_output=True,timeout=1800)
            rows=[json.loads(line.split(" ",1)[1]) for line in proc.stdout.splitlines()
                  if line.startswith("MATD3_REAL_TRAIN_GPU_WORKER_JSON ")]
            if proc.returncode!=0 or len(rows)!=1:
                print("TRAIN_AB_EXIT",proc.returncode,flush=True)
                print("STDOUT_TAIL",proc.stdout[-2400:],flush=True)
                print("STDERR_TAIL",proc.stderr[-4000:],flush=True)
                raise RuntimeError("GPU bounded training worker failed: "+variant)
            item=rows[0]
            item["repeat"]=repeat
            assert item["total_environment_transitions"]==VECTOR_STEPS*ENVS
            assert item["actual_learner_updates"]>0
            assert item["stop_reason"]=="env_step_budget"
            assert item["training_completed_20k"] is False
            runs.append(item)
            print("TRAIN_AB_COMPLETE",repeat,variant,"TPS",
                  item["train_joint_transitions_per_second"],"MODEL_HASH",
                  item["model_sha256"][:18],flush=True)
        except Exception as ex:
            Path("/kaggle/working/v9_train_speed_ab_error.json").write_text(
                json.dumps({"variant":variant,"repeat":repeat,"error":str(ex),"completed":runs},indent=2))
            raise
assert len(runs)==PAIRS*2
matched_science=len({x["science_sha256"] for x in runs})==1
matched_model=len({x["model_sha256"] for x in runs})==1
matched_episodes=len({x["episode_sha256"] for x in runs})==1
median={m:statistics.median(x["train_joint_transitions_per_second"]
                         for x in runs if x["variant"]==m)
        for m in ("baseline","candidate")}
result={
 "benchmark":"bounded_matd3_real_training_no_20k",
 "source_sha256":sha(blob),
 "changed_files_sha256":{k:v["sha256"] for k,v in overlays.items()},
 "pairs":PAIRS,"vector_steps":VECTOR_STEPS,
 "scientific_config_match":matched_science,
 "model_weights_match":matched_model,
 "episode_records_match":matched_episodes,
 "median_transitions_per_second":median,
 "candidate_speedup_percent":100*(median["candidate"]/median["baseline"]-1),
 "candidate_verified_beneficial":matched_science and matched_model and matched_episodes and median["candidate"]>median["baseline"],
 "runs":runs}
path=Path("/kaggle/working/v9_train_speed_ab_report.json")
path.write_text(json.dumps(result,indent=2,allow_nan=False))
print("TRAIN_AB_JSON",path,flush=True)
print("TRAIN_AB_SUMMARY",json.dumps({k:result[k] for k in
    ("scientific_config_match","model_weights_match","episode_records_match","median_transitions_per_second","candidate_speedup_percent","candidate_verified_beneficial")}),flush=True)
if not matched_science or not matched_model or not matched_episodes:
    raise RuntimeError("MATD3 training parity mismatch: do not apply candidate")
"""
    for s in (pre,bootstrap,run):ast.parse(s)
    nb=nbformat.v4.new_notebook(cells=[
      nbformat.v4.new_markdown_cell(intro),
      nbformat.v4.new_code_cell(pre),
      nbformat.v4.new_code_cell(bootstrap),
      nbformat.v4.new_code_cell(run)],metadata={
      "kernelspec":{"name":"python3","display_name":"Python 3"},
      "training_ab_bounded":True,"full_20000_training":False,
      "source_sha256":source_sha,"source_commit":base.metadata['reference_commit'],
      "candidate_patch_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "baseline_file_sha256":original_hashes,
      "candidate_file_sha256":{k:v['sha256'] for k,v in overlays.items()},
      "worker_sha256":work_sha,"envs":4096,"batch":8192,"updates":4,
      "replay_size":1000000,"replay_cpu_fraction":.75,
      "max_steps":1500,"seed":44,"pairs":2,"pilot_vector_steps":64})
    nbformat.validate(nb)
    assert len(nbformat.writes(nb).encode())<1_000_000
    assert "UAV_TRAIN_ENV_STEP_LIMIT" in work.decode()
    return nb

if __name__=="__main__":
    nb=build_notebook()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,OUT)
    print("TRAIN_GPU_AB_NOTEBOOK",OUT,"BYTES",OUT.stat().st_size,"SOURCE_SHA",nb.metadata["source_sha256"])
