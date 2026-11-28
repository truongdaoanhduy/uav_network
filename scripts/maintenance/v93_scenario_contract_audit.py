#!/usr/bin/env python3
"""Recompute action, belief, reward and RF claims from active source/config."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"research/zhang_xue/src")]
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params
from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np,G0_1M
from zhang_xue.belief.dempster_shafer import counts_to_canonical_ds_stable


def audit(path):
    cfg=load_scenario(path)
    params=GPUWorldV80Params.from_scenario(cfg)
    builds,blackouts,br=scale_canonical_geometry(1000.,20)
    gains=[]
    legal=[]
    for y in range(20):
        for x in range(20):
            xx,yy=25.+50*x,25.+50*y
            p=np.array([[[xx,yy,50.],[50.,50.,15.]]],dtype=np.float32)
            g=compute_channel_gain_v67_np(p,builds,blackouts,br,20,50.,
                 obstacle_model=params.obstacle_model,per_building_loss_db=params.per_building_loss_db)
            gains.append(float(g[0,0,1]))
            legal.append(not any((xx-bx)**2+(yy-by)**2<=r*r for bx,by,r,h in builds))
    gain=np.array(gains);legal=np.array(legal)
    snr=10**(params.snr_threshold_db/10)
    rows=[]
    for delta_db in (0,6,12,18):
        noise=params.noise_power_w*10**(delta_db/10)
        direct=params.tx_power_w*gain>=noise*snr
        rows.append({"noise_increase_db":delta_db,"noise_dbm":10*math.log10(noise/.001),
                     "los_slant_range_m":math.sqrt(params.tx_power_w*G0_1M/(noise*snr)),
                     "direct_cells_all400":int(direct.sum()),
                     "direct_cells_legal":int((direct&legal).sum()),
                     "legal_cells":int(legal.sum()),
                     "distance_multiplier_at_pathloss_exponent2":10**(delta_db/20)})
    belief=[]
    for p,n in [(0,0),(1,0),(0,1),(1,1),(2,0),(2,1)]:
        ds=counts_to_canonical_ds_stable(p,n,params.ds_mf,params.ds_me)
        ub=.5**(p+n)
        belief.append({"p":p,"n":n,"ds_mass":ds,"bayes_coverage_score":ub,
                       "ds_first_step_coverage_reward":params.w_loc_team*(1-ds[2]),
                       "bayes_first_step_coverage_reward":params.w_loc_team*(1-ub),
                       "reward_scope":"potential drop from an initially unseen cell; not a full trajectory return"})
    root_path=Path(path)
    return {"status":"AUDIT_COMPLETE","scenario_sha256":hashlib.sha256(root_path.read_bytes()).hexdigest(),
            "effective_scenario":asdict(cfg),"effective_gpu_params":asdict(params),
            "action":{"continuous":["motion_x","motion_y"],"communication":{"0":"NO_SEND","1":"GCS application destination (routing may be multi-hop)","2":"PEER dissemination"},"tx_power_is_learned":cfg.enable_tx_power_control},
            "rf_sensitivity":rows,
            "rf_scope":"Single transmitter, no contention; geometry link availability does not prove relay rescue or mission PDR",
            "belief_reward_counterexamples":belief,
            "performance_not_correctness_gate":"Neither Bayes superiority nor full success on every heuristic seed is proof of simulator correctness",
            "unresolved_macro_validation":"No equivalent UavNetSim routing/payload/RNG/HELLO microprofile has been established",
            "scenario_frozen":False}


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario",type=Path,default=ROOT/"research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml")
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();r=audit(a.scenario)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n")
    print(json.dumps(r,indent=2,allow_nan=False))
