"""Create an isolated private Kaggle GPU smoke job from a fixed Git SHA.

The source ZIP contains only approved, reviewed Python files; no tokens,
key files, prior notebooks, passwords or user data. The caller supplies an
authenticated Kaggle username, and uses the CLI with separate token env.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

PINNED_SOURCE_COMMIT="ee58650f389be72b547aa381ea68e417f4591a15"
SOURCE_PATHS=(
    "uav_network/__init__.py",
    "uav_network/environment/__init__.py",
    "uav_network/environment/gpu/zhang_xue_map_warp.py",
    "uav_network/communication/__init__.py",
    "uav_network/communication/gpu/__init__.py",
    "uav_network/communication/gpu/classic_v1_interference_warp.py",
    "uav_network/communication/gpu/zhang_xue_mac_retry_warp.py",
    "uav_network/communication/gpu/zhang_xue_persistent_queue_warp.py",
    "uav_network/communication/gpu/zhang_xue_payload_store_warp.py",
    "uav_network/communication/gpu/zhang_xue_origin_evidence_warp.py",
    "uav_network/communication/gpu/zhang_xue_distributed_routing_warp.py",
    "scripts/verify_zhang_xue_multihop_flow_v42.py",
)

RUNNER = r'''"""NVIDIA Warp E16/E1024/E4096 CUDA smoke, no model optimization.

Created from a fixed public Git commit and a strict Python module allowlist.
Never reads Kaggle Secrets or any account credentials inside this job.
"""
import base64
import io
import json
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback
import zipfile

SOURCE_SHA = "__SHA__"
BASE64_SOURCE = "__SOURCE__"
ROOT = Path("/kaggle/working/zhang-xue-v42-source")
OUTPUT = Path("/kaggle/working/zhang_xue_v43_cuda_probe.json")

def main():
    started = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(BASE64_SOURCE))) as zipped:
        for path in zipped.namelist():
            target = ROOT/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zipped.read(path))
    sys.path.insert(0, str(ROOT))
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            current=version("warp-lang")
        except PackageNotFoundError:
            current=None
        if current != "1.18.0":
            subprocess.check_call([
                sys.executable,"-m","pip","install",
                "--disable-pip-version-check","-q","warp-lang==1.18.0",
            ])
        import warp as wp
        wp.init()
        devices=wp.get_cuda_devices()
        assert devices, "Kaggle GPU runtime did not expose CUDA to Warp"
        from scripts.verify_zhang_xue_multihop_flow_v42 import (
            run_multihop_map_network_probe,
        )
        keys=(
            "app_data_received","e2e_ack_received","data_ttl_expired",
            "relay_data_count","relay_ack_count","receiver_ds_only_after_physical_data",
            "payload_origin_preserved","blackout_transmissions","queues_empty",
            "learned_route_0_to_2_via_1","learned_route_2_to_0_via_1",
        )
        baseline=run_multihop_map_network_probe(
            envs=16,device="cpu",seed=44,
        )
        gpu_runs={}
        for device,world_counts in (
            ("cuda:0",(16,1024,4096)),
            ("cuda:1",(16,1024)) if len(devices)>1 else ("cuda:1",()),
        ):
            for envs in world_counts:
                result=run_multihop_map_network_probe(
                    envs=envs,device=device,seed=44,
                )
                assert result["app_data_received"]==envs*3//4
                assert result["e2e_ack_received"]==envs*3//4
                assert result["data_ttl_expired"]==envs//4
                assert result["receiver_ds_only_after_physical_data"]
                assert result["queues_empty"]
                if envs==16:
                    for key in keys:
                        assert result[key]==baseline[key], (device,key)
                gpu_runs[f"{device}:E{envs}"] = {
                    k:result[k] for k in keys
                }
                gpu_runs[f"{device}:E{envs}"]["seconds"]=result["elapsed_seconds"]
                print("GPU_SMOKE_OK", device, envs,
                      result["app_data_received"],
                      result["e2e_ack_received"], flush=True)
        summary=dict(
            status="PASS",
            commit=SOURCE_SHA,
            warp_version="1.18.0",
            device_count=len(devices),
            devices=[str(d) for d in devices],
            runs=gpu_runs,
            elapsed_wall_seconds=time.time()-started,
            train_launched=False,
            full_gate5="NOT_PASSED",
        )
        OUTPUT.write_text(json.dumps(summary,indent=2))
        print("ZX_V43_CUDA_PROBE_PASS",json.dumps(summary),flush=True)
    except Exception as exc:
        summary=dict(
            status="FAIL",
            commit=SOURCE_SHA,
            error_type=type(exc).__name__,
            error_message=str(exc)[:450],
            elapsed_wall_seconds=time.time()-started,
            train_launched=False,
        )
        OUTPUT.write_text(json.dumps(summary,indent=2))
        traceback.print_exc()
        print("ZX_V43_CUDA_PROBE_FAIL",json.dumps(summary),flush=True)
        raise

if __name__=="__main__":
    main()
'''


def build_kaggle_cuda_smoke(repo:Path,*,owner:str,slug:str,dest:Path):
    repo=Path(repo)
    dest=Path(dest)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{2,64}",owner):
        raise ValueError("owner must be a valid Kaggle username")
    if not re.fullmatch(r"[a-z0-9-]{6,64}",slug):
        raise ValueError("slug must be lower-case kebab-case")
    bundle=io.BytesIO()
    entry_hashes={}
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in SOURCE_PATHS:
            blob=subprocess.check_output(
                ["git","show",f"{PINNED_SOURCE_COMMIT}:{path}"],
                cwd=repo,
            )
            if any(word in path.lower() for word in (
                "key.txt","credential","kaggle.json",".pem",
            )):
                raise ValueError("private file in source allowlist")
            archive.writestr(path,blob)
            entry_hashes[path]=hashlib.sha256(blob).hexdigest()
    archive_bytes=bundle.getvalue()
    encoded=base64.b64encode(archive_bytes).decode()
    script=RUNNER.replace("__SHA__",PINNED_SOURCE_COMMIT).replace(
        "__SOURCE__",encoded,
    )
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"kernel.py").write_text(script)
    metadata={
        "id":f"{owner}/{slug}",
        "title":"Zhang Xue v43 Warp CUDA network smoke 20261125",
        "code_file":"kernel.py",
        "language":"python",
        "kernel_type":"script",
        "is_private":True,
        "enable_gpu":True,
        "enable_internet":True,
        "dataset_sources":[],
        "competition_sources":[],
        "kernel_sources":[],
        "model_sources":[],
    }
    (dest/"kernel-metadata.json").write_text(
        json.dumps(metadata,indent=2)+"\n",
    )
    (dest/"source-manifest.json").write_text(
        json.dumps({
            "commit":PINNED_SOURCE_COMMIT,
            "source_hashes":entry_hashes,
            "bundle_sha256":hashlib.sha256(archive_bytes).hexdigest(),
            "credentials_included":False,
        },indent=2)+"\n",
    )
    return {
        "script":dest/"kernel.py",
        "metadata":dest/"kernel-metadata.json",
        "manifest":dest/"source-manifest.json",
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--owner",required=True)
    parser.add_argument("--slug",required=True)
    parser.add_argument("--dest",required=True,type=Path)
    args=parser.parse_args()
    result=build_kaggle_cuda_smoke(
        args.repo,owner=args.owner,slug=args.slug,dest=args.dest,
    )
    print(json.dumps({k:str(v) for k,v in result.items()},indent=2))


if __name__=="__main__":
    main()
