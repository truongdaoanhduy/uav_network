#!/usr/bin/env python3
"""Non-training privileged upper-bound feasibility probe for the V91 scenario.

Uses ground-truth target cells ONLY in the diagnostics script to plan
waypoints; never injects truth into actor observations. Not an RL baseline,
not a learning result, and not a guarantee of mission feasibility.
"""
from __future__ import annotations
import argparse, heapq, json, math, time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params,WarpParallelMissionV80

S=25.0
GRID=40
def grid_cell(pt):
    return tuple(max(0,min(GRID-1,int(x//S))) for x in pt[:2])
def center(cell):
    return np.array(((cell[0]+.5)*S,(cell[1]+.5)*S),dtype=np.float64)
def route(start,end,buildings):
    a=grid_cell(start);b=grid_cell(end)
    def passable(c):
        x,y=center(c)
        if min(x,y)<15 or max(x,y)>985:return False
        return all(math.hypot(x-bx,y-by)>r+12 for bx,by,r,_ in buildings)
    def heur(c):return math.hypot(c[0]-b[0],c[1]-b[1])
    # Start might not align with legal cell because safe margin too strong.
    h=[(heur(a),0,a)];dst={a:0.};par={};closed=set()
    while h:
        _,cost,node=heapq.heappop(h)
        if node in closed:continue
        closed.add(node)
        if node==b:
            path=[b]
            while path[-1]!=a:path.append(par[path[-1]])
            return [center(t) for t in path[-2::-1]]
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
            nxt=(node[0]+dx,node[1]+dy)
            if not (0<=nxt[0]<GRID and 0<=nxt[1]<GRID):continue
            if nxt!=b and not passable(nxt):continue
            if abs(dx)+abs(dy)==2 and not (passable((node[0]+dx,node[1])) and passable((node[0],node[1]+dy))):continue
            c=cost+math.hypot(dx,dy)
            if c+1e-8<dst.get(nxt,float("inf")):
                dst[nxt]=c;par[nxt]=node;heapq.heappush(h,(c+heur(nxt),c,nxt))
    return [np.asarray(end,dtype=np.float64)]

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--seeds",nargs="+",type=int,default=[44,2027,10044])
    pa.add_argument("--output",required=True,type=Path)
    args=pa.parse_args()
    rows=[]
    for seed in args.seeds:
        torch.manual_seed(seed)
        p=GPUWorldV80Params.from_yaml("research/zhang_xue/configs/scenario_v91_bayes_1km.yaml",envs=1,device="cuda:0",seed=seed)
        w=WarpParallelMissionV80(p)
        targets=np.flatnonzero(w.buffers["truth"].numpy()[0]).tolist()
        target_xy={cid:np.array(((cid%20+.5)*50,(cid//20+.5)*50),dtype=np.float64) for cid in targets}
        # Each UAV has distinct spatially local designated targets; no actor
        # gets these labels. Tests whether the WORLD permits a near-oracle route.
        starts=w.buffers["positions"].numpy()[0,:,:2]
        tasks={u:[] for u in range(6)}
        for cid in targets:
            u=min(range(6),key=lambda i:np.linalg.norm(starts[i]-target_xy[cid]))
            tasks[u].append(cid)
        routes={}
        for u in range(6):
            prev=starts[u].copy()
            for cid in sorted(tasks[u],key=lambda k:np.linalg.norm(target_xy[k]-starts[u])):
                for wpnt in route(prev,target_xy[cid],w.buildings):
                    routes.setdefault(u,[]).append((wpnt,None))
                routes[u].append((target_xy[cid],cid))
                prev=target_xy[cid]
        current={u:0 for u in range(6)}
        dwell={u:0 for u in range(6)}
        actions=torch.zeros((1,6,2),device="cuda:0",dtype=torch.float32)
        dest=wp.ones((1,6),dtype=wp.int32,device="cuda:0")
        t0=time.monotonic()
        trace=[]
        for i in range(900):
            # Host sync is intentional: oracle upper-bound diagnostic, not learner.
            pos=w.buffers["positions"].numpy()[0,:,:2]
            vel=w.buffers["velocities"].numpy()[0,:,:2]
            cmd=np.zeros((6,2),dtype=np.float32)
            for u in range(6):
                rr=routes.get(u,[])
                idx=current[u]
                if idx>=len(rr):continue
                goal,target=rr[idx]
                dx=goal-pos[u]
                distance=float(np.linalg.norm(dx))
                if distance<12.0:
                    if target is not None:
                        dwell[u]+=1
                        if dwell[u]<5:continue
                    current[u]=idx+1;dwell[u]=0
                    if current[u]>=len(rr):continue
                    goal,_=rr[current[u]]
                    dx=goal-pos[u];distance=float(np.linalg.norm(dx))
                if distance>1e-9:
                    desired=dx/distance*min(13.,distance*.72)
                    cmd[u]=np.clip(desired/p.v_max_mps,-1,1)
            actions[0]=torch.as_tensor(cmd,device="cuda:0")
            w.step(wp.from_torch(actions),dest)
            if i in (0,89,179,359,539,719,899) or int(w.buffers["done"].numpy()[0]):
                tp=int(w.buffers["gcs_tp"].numpy().sum())
                fp=int(w.buffers["gcs_fp_ever"].numpy().sum())
                reached=int(sum(current[u]>=len(routes.get(u,[])) for u in range(6)))
                trace.append({"step":i+1,"gcs_tp":tp,"gcs_fp_ever":fp,"route_completed_uavs":reached,
                    "safety_reason":int(w.buffers["safety_reason"].numpy()[0]),"done":int(w.buffers["done"].numpy()[0])})
            if int(w.buffers["done"].numpy()[0]):break
        end={
            "seed":seed,"policy":"oracle planner diagnostic, privileged truth outside observation, NOT RL",
            "steps":i+1,"gcs_true_targets":int(w.buffers["gcs_tp"].numpy().sum()),
            "gcs_false_ever":int(w.buffers["gcs_fp_ever"].numpy().sum()),
            "target_count":len(targets),"mission_complete":int(w.buffers["termination_reason"].numpy()[0])&4!=0,
            "safety_failed":int(w.buffers["failed"].numpy()[0]),
            "reason":int(w.buffers["termination_reason"].numpy()[0]),
            "battery_mean_pct":float(np.mean(w.buffers["battery"].numpy())/p.e_max_j*100.),
            "remaining_routes":{str(u):len(routes.get(u,[]))-current[u] for u in range(6)},
            "trace":trace,"elapsed_seconds":round(time.monotonic()-t0,3)}
        rows.append(end)
        print(json.dumps({k:v for k,v in end.items() if k not in ("trace","remaining_routes")}),flush=True)
    args.output.write_text(json.dumps({"scenario":"V91 Bayesian","purpose":"mission feasibility diagnostic with oracle planner","results":rows},indent=2)+"\n")

if __name__=="__main__":main()
