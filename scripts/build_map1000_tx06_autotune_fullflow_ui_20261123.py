"""Three-cell SHA-pinned MAP1000/TX0.6 20k training with safe CUDA A/B autotune.

Runtime A/B never changes scientific config; if an optimization fails bitwise
or cannot beat baseline by >5%, always fall back to stock scheduler/staging.
Every Actor snapshot still gets Val100, and heldout final Test1000.
"""
from __future__ import annotations
import base64,gzip,hashlib,ast
from pathlib import Path
import nbformat
from scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 import (
    build_notebook as base_build,CONFIG,ROOT)

NAME='MATD3-MAP1000-TX06-AUTOTUNE-T4X2-TRAIN20000-VAL100-TEST1000-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME


def encode(source:bytes):
    return base64.b85encode(gzip.compress(source,compresslevel=9,mtime=0)).decode()


def build_notebook():
    n=base_build()
    n.cells[0].source+='''
### Hardware-only autotune (no scientific changes)
The notebook automatically measures CUDA `packet_step` block dimensions
0 / 32 / 64 / 128 and the **exact same B8192 hybrid replay** with pinned
staging OFF/ON on the selected T4x2 GPUs before starting 20k training.
Only matching state SHA256 / replay data / RNG and >5% median speed gains are
adopted. It falls back to the original path on slower/invalid candidates.
Train/Validation100 for every Actor/Best Actor/Test1000 and forensic logs stay.
**No T4 performance claim** is made until the notebook actually runs there.
'''
    n.cells[1].source+='''\n# Defaults: automatic conservative T4 tuning, no manual optimizer changes.
AUTO_TUNE_MAC_REPLAY=True
MAC_TUNE_CANDIDATES=[0,32,64,128]
MAC_TUNE_ENVS=4096
MAC_TUNE_WARMUP=6
MAC_TUNE_STEPS=48
MAC_TUNE_PARITY_ENVS=16
MAC_TUNE_PARITY_STEPS=48
MAC_TUNE_REPEATS=2
assert MAC_TUNE_CANDIDATES[0]==0 and all(x in (0,32,64,128) for x in MAC_TUNE_CANDIDATES)
assert MAC_TUNE_ENVS==4096 and MAC_TUNE_PARITY_ENVS==16
'''
    worker_files={
      'profile_nooracle_warp_phases_20261122.py':(ROOT/'scripts/profile_nooracle_warp_phases_20261122.py').read_bytes(),
      'benchmark_hybrid_replay_cuda_ab_20261123.py':(ROOT/'scripts/benchmark_hybrid_replay_cuda_ab_20261123.py').read_bytes(),
    }
    injections=['''# Conservative T4x2 speed autotune: this block is AFTER source extraction,
# venv install and scientific preflight, but BEFORE training and SPEC write.
# Trials exclude credentials and never run W&B or Hugging Face code.
import base64,gzip,hashlib,statistics
from pathlib import Path
for _key in ("UAV_WARP_MAC_BLOCK_DIM","UAV_HYBRID_PINNED_STAGING"):
    os.environ[_key]="0"
''']
    for filename,payload in worker_files.items():
        token=encode(payload);digest=hashlib.sha256(payload).hexdigest()
        injections.append('''_worker=gzip.decompress(base64.b85decode("'''+token+'''"))
assert hashlib.sha256(_worker).hexdigest()=="'''+digest+'''"
(root/"scripts"/"'''+filename+'''").write_bytes(_worker)
''')
    injections.append('''
_tune_report={"type":"safe_autotune_cuda_layout_and_replay_io_no_science_change",
  "source_sha256":SPEC.get("source_bundle_sha256"),"baseline_unchanged":True,
  "gpu_required":2,"science_config":SPEC["config"],
  "mac_parity":[],"mac_trials":[],"replay_trials":[],"errors":[],
  "chosen_mac_block_dim":0,"chosen_pinned_replay_staging":False}
_tune_path=Path("/kaggle/working/MAP1000_TX06_AUTOTUNE_RESULT.json")
def _flush_tune():
    _tune_path.write_text(json.dumps(_tune_report,indent=2,allow_nan=False))

def _probe_worker(label,args,marker,limit_s=360):
    _env=os.environ.copy()
    for _secret in ("WANDB_API_KEY","HF_TOKEN","HUGGINGFACE_HUB_TOKEN"):
        _env.pop(_secret,None)
    _env["PYTHONPATH"]=str(root)
    _env["OMP_NUM_THREADS"]=_env["MKL_NUM_THREADS"]=_env["OPENBLAS_NUM_THREADS"]="1"
    _env["UAV_WARP_MAC_BLOCK_DIM"]="0"
    _env["UAV_HYBRID_PINNED_STAGING"]="0"
    _proc=subprocess.run([str(python),*args],cwd=root,env=_env,
                         capture_output=True,text=True,timeout=limit_s)
    if _proc.returncode:
        raise RuntimeError(label+": "+_proc.stderr[-1400:]+" STDOUT "+_proc.stdout[-500:])
    _rows=[json.loads(line.split(" ",1)[1]) for line in _proc.stdout.splitlines()
           if line.startswith(marker+" ")]
    if len(_rows)!=1:raise RuntimeError("Missing result: "+label)
    return _rows[0]

def _mac_measure(dim,small):
    args=["-m","scripts.profile_nooracle_warp_phases_20261122",
          "--config",SPEC["config"],"--device","cuda",
          "--envs",str(MAC_TUNE_PARITY_ENVS if small else MAC_TUNE_ENVS),
          "--warmup",str(4 if small else MAC_TUNE_WARMUP),
          "--steps",str(MAC_TUNE_PARITY_STEPS if small else MAC_TUNE_STEPS),
          "--mac-block-dim",str(dim)]
    if small:args.append("--state-digest")
    return _probe_worker("MAC "+str(dim),args,"UAV_WARP_PROFILE_JSON")

if AUTO_TUNE_MAC_REPLAY:
    _valid=[0]
    try:
        _baseline=_mac_measure(0,True)
        _tune_report["mac_parity"].append({"dim":0,"sha256":_baseline["world_state_sha256"]})
        for _dim in MAC_TUNE_CANDIDATES[1:]:
            try:
                _rec=_mac_measure(_dim,True)
                _match=_rec["world_state_sha256"]==_baseline["world_state_sha256"]
                _tune_report["mac_parity"].append({"dim":_dim,"sha256":_rec["world_state_sha256"],"matched":_match})
                if _match:_valid.append(_dim)
            except Exception as _exc:
                _tune_report["errors"].append({"phase":"mac_parity","dim":_dim,"error":str(_exc)[:500]})
            _flush_tune()
        for _repeat in range(MAC_TUNE_REPEATS):
            _order=_valid if _repeat%2==0 else list(reversed(_valid))
            for _dim in _order:
                try:
                    _rec=_mac_measure(_dim,False)
                    _tune_report["mac_trials"].append({"dim":_dim,"repeat":_repeat,
                        "tps":_rec["env_transitions_per_second"],
                        "packet_ms":_rec["per_kernel_ms_total"].get("packet_step")})
                except Exception as _exc:
                    _tune_report["errors"].append({"phase":"mac_perf","dim":_dim,"error":str(_exc)[:500]})
                _flush_tune()
        _med={_dim:statistics.median([r["tps"] for r in _tune_report["mac_trials"] if r["dim"]==_dim])
              for _dim in _valid if sum(r["dim"]==_dim for r in _tune_report["mac_trials"])==MAC_TUNE_REPEATS}
        if 0 in _med:
            _best=max(_med,key=lambda k:_med[k])
            if _best!=0 and _med[_best]>1.05*_med[0]:
                _tune_report["chosen_mac_block_dim"]=_best
    except Exception as _exc:
        _tune_report["errors"].append({"phase":"mac_baseline","error":str(_exc)[:700]})
    _flush_tune()

    for _repeat in range(2):
        for _mode in (("off","on") if _repeat%2==0 else ("on","off")):
            try:
                _rec=_probe_worker("replay_"+_mode,
                    ["-m","scripts.benchmark_hybrid_replay_cuda_ab_20261123",
                     "--device","cuda:1","--staging",_mode,"--batch","8192",
                     "--capacity","16384","--warmup","2","--samples","5"],
                    "HYBRID_REPLAY_CUDA_AB_JSON")
                _tune_report["replay_trials"].append({"mode":_mode,"repeat":_repeat,
                    "seconds":_rec["median_sample_seconds"],
                    "batch_sha256":_rec["all_sampled_tensor_sha256"],
                    "rng_sha256":_rec["rng_state_sha256"],
                    "seen_sha256":_rec["seen_bitmap_sha256"]})
            except Exception as _exc:
                _tune_report["errors"].append({"phase":"replay","mode":_mode,"error":str(_exc)[:700]})
            _flush_tune()
    _recs=_tune_report["replay_trials"]
    if (sum(x["mode"]=="on" for x in _recs)==2 and
        sum(x["mode"]=="off" for x in _recs)==2 and
        all(len({x[k] for x in _recs})==1 for k in ("batch_sha256","rng_sha256","seen_sha256"))):
        _rmed={m:statistics.median(x["seconds"] for x in _recs if x["mode"]==m)
               for m in ("off","on")}
        if _rmed["on"]<_rmed["off"]/1.05:
            _tune_report["chosen_pinned_replay_staging"]=True
    _flush_tune()

os.environ["UAV_WARP_MAC_BLOCK_DIM"]=str(_tune_report["chosen_mac_block_dim"])
os.environ["UAV_HYBRID_PINNED_STAGING"]="1" if _tune_report["chosen_pinned_replay_staging"] else "0"
SPEC["gpu_hardware_autotune"]={"mac_block_dim":_tune_report["chosen_mac_block_dim"],
    "replay_pinned":_tune_report["chosen_pinned_replay_staging"],
    "source_sha256":_tune_report["source_sha256"],
    "full_report":"/kaggle/working/MAP1000_TX06_AUTOTUNE_RESULT.json"}
SPEC["runs"]["matd3"]["name"]+="-GPUAUTOTUNED"
_tune_report["completed"]=True
_flush_tune()
print("MAP1000_AUTOTUNE_DECISION",json.dumps(SPEC["gpu_hardware_autotune"]),flush=True)
''')
    blob=''.join(injections)
    ast.parse(blob)
    src=n.cells[2].source
    marker='spec_file = Path("/kaggle/working/job.json")'
    assert src.count(marker)==1
    src=src.replace(marker,blob+'\n'+marker,1)
    ast.parse(src)
    n.cells[2].source=src
    n.metadata.update({'gpu_autotune_before_training':True,
        'autotune_mac_block_candidates':[0,32,64,128],
        'autotune_replay_pinned_candidates':[False,True],
        'full_20000_train_val100_test1000_intact':True,
        't4_cuda_speed_verified':False})
    nbformat.validate(n)
    if len(nbformat.writes(n).encode())>=999_000:
        raise ValueError('Kaggle autotune+fullflow notebook over 1MB')
    return n


def main():
    n=build_notebook();OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUT)
    print('AUTOTUNE_FULLFLOW_NOTEBOOK',OUT,'BYTES',OUT.stat().st_size,
          'SOURCE_SHA256',n.metadata['source_sha256'])


if __name__=='__main__':main()
