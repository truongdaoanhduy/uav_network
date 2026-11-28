"""Emit an independent canonical state/output trace for V122-vs-V123 equality.

Run once with V122 on PYTHONPATH and once with V123; then compare outputs.
Avoid importing two env versions in one Python interpreter.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import yaml
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv

def digest(arr):
    x=np.ascontiguousarray(arr)
    return hashlib.sha256(x.tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--yaml',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[44,45]);p.add_argument('--steps',type=int,default=20)
    a=p.parse_args()
    cfg=EnvConfigV54(**yaml.safe_load(a.yaml.read_text()))
    assert not getattr(cfg,"enable_gimbal_azimuth_action",False)
    out=[]
    for seed in a.seeds:
        e=ZhangXue7NodePacketEnv(cfg);o,s,m=e.reset(seed=seed)
        out.append({'seed':seed,'step':0,'obs':digest(o),'state':digest(s),'mask':digest(m),
                    'position':digest(e.positions),'belief_pos':digest(e.net.ledger_pos),
                    'belief_neg':digest(e.net.ledger_neg),'targets':digest(e.true_targets)})
        for t in range(a.steps):
            # Non-oracle (fixed policy), bounded and deterministic across versions.
            cmd=np.tile(np.array([[12.0, -3.0]]),(e.actor_n,1));comm=np.ones(e.actor_n,dtype=np.int32)
            o,s,m,r,term,trunc,info=e.step(cmd,comm)
            out.append({'seed':seed,'step':t+1,'reward':float(r),'obs':digest(o),'state':digest(s),
                        'mask':digest(m),'position':digest(e.positions),
                        'velocity':digest(e.velocities),'battery':digest(e.battery_j),
                        'belief_pos':digest(e.net.ledger_pos),'belief_neg':digest(e.net.ledger_neg),
                        'gcs_tp':info['gcs_tp_confirmed'],'gcs_fp':info['gcs_fp_ever_confirmed'],
                        'terminated':term,'truncated':trunc})
            if term or trunc: break
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'records':len(out),'output':str(a.output),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
