#!/usr/bin/env python3
"""Scenario-only GPU allocation / step smoke test; no MATD3/MASAC training."""
import argparse,json,subprocess,sys,time,traceback,gc
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/zhang_xue/src')]
def mem():
    try:return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()[0].strip())
    except Exception:return None
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--scenario',type=Path,default=ROOT/'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml')
    ap.add_argument('--envs',default='64,256,1024,4096');ap.add_argument('--steps',type=int,default=2)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args(argv)
    import warp as wp
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params,WarpParallelMissionV80
    rows=[]
    for n in [int(x) for x in a.envs.split(',')]:
        row={'envs':n,'requested_steps':a.steps,'mem_mib_initial':mem(),'status':'PENDING'}
        start=time.monotonic()
        try:
            p=GPUWorldV80Params.from_yaml(a.scenario,envs=n,device=a.device,seed=44)
            w=WarpParallelMissionV80(p);wp.synchronize_device(a.device)
            row['initialization_sec']=round(time.monotonic()-start,4)
            row['mem_mib_after_init']=mem()
            actions=wp.zeros((n,6,2),dtype=wp.float32,device=a.device)
            destinations=wp.zeros((n,6),dtype=wp.int32,device=a.device)
            t1=time.monotonic()
            for i in range(a.steps):w.step(actions,destinations)
            wp.synchronize_device(a.device)
            row['stepping_sec']=round(time.monotonic()-t1,4)
            row['world_steps_per_sec']=round(n*a.steps/max(1e-9,row['stepping_sec']),2)
            row['mem_mib_after_steps']=mem()
            rw=w.buffers['reward'].numpy();fail=w.buffers['failed'].numpy()
            row['finite_rewards']=int(np.isfinite(rw).sum())
            row['safety_failed_worlds']=int(np.count_nonzero(fail))
            row['status']='PASS' if row['finite_rewards']==n and row['safety_failed_worlds']==0 else 'INVALID'
            del w,p,actions,destinations;gc.collect()
        except Exception as ex:
            row['status']='ERROR';row['exception']=repr(ex);row['traceback']=traceback.format_exc()[-3500:]
        rows.append(row);print(json.dumps(row),flush=True)
        if row['status']!='PASS':break
    out={'scope':'2-step Warp scenario allocation only, NOT RL training nor E4096x900 benchmark','device':a.device,'rows':rows,'status':'PASS' if len(rows)==len(a.envs.split(',')) and all(r['status']=='PASS' for r in rows) else 'FAIL'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    return 0 if out['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
