#!/usr/bin/env python3
"""Bounded stage-by-stage Tesla T4 smoke/VRAM gate. Does NOT train or use Kaggle keys.

Run only after review on Kaggle with 2 attached T4 GPUs:
    python scripts/v58_t4_scale_gate.py --allow-experimental --devices cuda:0 cuda:1

Stops at first failed stage instead of proceeding toward E=2048 blindly.
Note: a passing smoke does not establish CPU-Warp full-event parity or full-MARL readiness.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / 'scripts' / 'v56_cuda_preflight.py'


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices', nargs='+', default=['cuda:0','cuda:1'])
    p.add_argument('--sizes', nargs='+', type=int, default=[32,256,1024,2048])
    p.add_argument('--timeout', type=int, default=300, help='Max seconds per stage; fail closed')
    p.add_argument('--allow-experimental', action='store_true')
    p.add_argument('--output', default='reports/v58_t4_scale_gate.json')
    a=p.parse_args()
    allowed={32,256,1024,2048}
    if not a.allow_experimental or any(n not in allowed for n in a.sizes) or a.sizes!=sorted(set(a.sizes)):
        p.error('Pass --allow-experimental and sorted unique sizes in 32,256,1024,2048')
    record={'status':'BLOCKED_UNTIL_ALL_STAGES_PASS','devices':a.devices,
            'sizes':a.sizes,'stages':[],'not_a_training_run':True,
            'full_event_parity_verified':False}
    for dev in a.devices:
        for n in a.sizes:
            cmd=[sys.executable,str(PREFLIGHT),'--device',dev,'--envs',str(n),'--allow-experimental']
            start=time.monotonic()
            try:
                r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True,timeout=a.timeout,check=False)
                # Warp may print build notes before the JSON dictionary.
                idx=r.stdout.find('{')
                data=json.loads(r.stdout[idx:]) if idx>=0 else {'raw_output':r.stdout[-1000:]}
                ok=r.returncode==0 and data.get('status')=='CUDA_WARP_SMOKE_PASSED_NOT_FULL_PARITY'
                row={'device':dev,'envs':n,'ok':ok,'seconds':round(time.monotonic()-start,3),
                     'exit_code':r.returncode,'data':data,'stderr_tail':r.stderr[-800:]}
            except subprocess.TimeoutExpired:
                row={'device':dev,'envs':n,'ok':False,'error':'TIMEOUT', 'seconds':round(time.monotonic()-start,3)}
            except Exception as e:
                row={'device':dev,'envs':n,'ok':False,'error':f'{type(e).__name__}: {e}'}
            record['stages'].append(row)
            if not row['ok']:
                break
        if not record['stages'][-1]['ok']:
            break
    if record['stages'] and all(x['ok'] for x in record['stages']) and len(record['stages']) == len(a.devices)*len(a.sizes):
        record['status']='CUDA_SMOKE_PASSED_BUT_PARITY_NOT_VERIFIED'
    output=ROOT/a.output
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2,ensure_ascii=False))
    return 0 if record['status'].startswith('CUDA_SMOKE_PASSED') else 2

if __name__=='__main__':
    raise SystemExit(main())
