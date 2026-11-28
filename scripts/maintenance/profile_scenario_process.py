#!/usr/bin/env python3
"""Run a scenario command and sample device memory; no learner allocations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--log",type=Path,required=True)
    ap.add_argument("command",nargs=argparse.REMAINDER)
    a=ap.parse_args()
    cmd=a.command[1:] if a.command and a.command[0]=="--" else a.command
    if not cmd:
        ap.error("a command after -- is required")
    def memory():
        r=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
        return [int(v.strip()) for v in r.splitlines() if v.strip()]
    a.log.parent.mkdir(parents=True,exist_ok=True)
    baseline=memory();samples=[];start=time.monotonic()
    with a.log.open("w") as out:
        process=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
        while process.poll() is None:
            samples.append([round(time.monotonic()-start,3),*memory()])
            time.sleep(.25)
    peak=[max([baseline[i]]+[s[i+1] for s in samples]) for i in range(len(baseline))]
    result={"command":cmd,"exit_code":process.returncode,"elapsed_sec":round(time.monotonic()-start,3),
            "baseline_device_mib":baseline,"peak_device_mib":peak,
            "scope":"nvidia-smi whole-device memory, including baseline contexts; sampled peak, not allocator-exact or training VRAM",
            "samples":samples}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="samples"},indent=2))
    return process.returncode


if __name__=="__main__":
    raise SystemExit(main())
