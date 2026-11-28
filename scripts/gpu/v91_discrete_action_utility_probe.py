#!/usr/bin/env python3
"""Scenario-only action meaning test: not a MARL baseline or tuning trial."""
import json,time
import numpy as np
import warp as wp
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params,WarpParallelMissionV80

result=[]
for label,disc in (("NO_SEND",0),("GCS",1),("PEER",2)):
    p=GPUWorldV80Params.from_yaml("research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml",envs=8,device="cuda:0",seed=44)
    w=WarpParallelMissionV80(p)
    act=wp.zeros((8,6,2),dtype=wp.float32,device="cuda:0")
    action=wp.full((8,6),disc,dtype=wp.int32,device="cuda:0")
    t=time.monotonic()
    for i in range(200):
        w.step(act,action)
    r={
      "mode":label,"discrete_action":disc,"envs":8,"steps":200,
      "gcs_true_confirmed":int(w.buffers["gcs_tp"].numpy().sum()),
      "gcs_false_ever":int(w.buffers["gcs_fp_ever"].numpy().sum()),
      "gcs_data_packets":int(w.network.data["stats_gcs"].numpy().sum()),
      "peer_data_packets":int(w.network.data["stats_peer"].numpy().sum()),
      "safety_failed_worlds":int(w.buffers["failed"].numpy().sum()),
      "overflow_worlds":int(w.network.data["overflow"].numpy().sum()),
      "elapsed_sec":round(time.monotonic()-t,3),
    }
    result.append(r)
    print(json.dumps(r),flush=True)
with open("/workspace/v91_discrete_action_ablation_stationary.json","w") as fp:
    json.dump({"policy":"stationary continuous XY; one compact-3 network action on all UAVs", "note":"Action semantics in this limited scenario only; NOT RL or learning usefulness.", "results":result},fp,indent=2)
