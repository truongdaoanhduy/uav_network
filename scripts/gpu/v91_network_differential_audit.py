#!/usr/bin/env python3
"""Reproducible V91 UavNetSim-v1 differential audit. Six *honestly scoped* gates.

PASS means a bounded contract actually tested. OPEN means unresolved calibration,
NOT silently counted as a passing IEEE 802.11 standard compliance test.
"""
from __future__ import annotations
import argparse, json, math, random, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"research/zhang_xue/src")]
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params,WarpParallelMissionV80
from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np
from zhang_xue.gpu.state_v67 import DIFS_US,SIFS_US,SLOT_US,CW_MIN,ACK_TIMEOUT_US,TX_MAC_ACK_US,PKT_TTL_US,HELLO_V89_TX_US

def g(label,status,detail,evidence=None):
    return {"gate":label,"status":status,"detail":detail,"evidence":evidence or {}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--upstream",type=Path,default=Path("/workspace/UavNetSim-v1-upstream-20261127"))
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    commit=subprocess.check_output(["git","-C",str(a.upstream),"rev-parse","HEAD"],text=True).strip()
    sys.path.insert(0,str(a.upstream))
    from utils import config as ref
    from zhang_xue.communication.common import scale_canonical_geometry

    gates=[]
    gates.append(g("01_DCF_SYMBOL_TIMING","PASS" if
        (DIFS_US,SIFS_US,SLOT_US,CW_MIN,ACK_TIMEOUT_US,TX_MAC_ACK_US)==
        (ref.DIFS_DURATION,ref.SIFS_DURATION,ref.SLOT_DURATION,ref.CW_MIN,ref.ACK_TIMEOUT,
         round(ref.ACK_PACKET_LENGTH/ref.BIT_RATE*1e6)) else "FAIL",
        "Validate DIFS, SIFS, slot, CW_MIN and ACK timing against pinned UavNetSim config",
        {"warp":[DIFS_US,SIFS_US,SLOT_US,CW_MIN,ACK_TIMEOUT_US,TX_MAC_ACK_US],
         "upstream":[ref.DIFS_DURATION,ref.SIFS_DURATION,ref.SLOT_DURATION,ref.CW_MIN,ref.ACK_TIMEOUT,round(ref.ACK_PACKET_LENGTH/ref.BIT_RATE*1e6)]}))
    # Upstream csma_ca.py calls randint(0,contention_window-1); Warp
    # mac_v67.py does % (cw+1). Distinct support by one slot at attempt 1.
    first=ref.CW_MIN
    gates.append(g("02_BACKOFF_UPSTREAM_DISTRIBUTION","FAIL",
        "Proven first-attempt contention support mismatch: upstream [0,CW_MIN-1], Warp [0,CW_MIN]. "
        "Identical seed cannot yield bit-exact event parity. Fix or document surrogate.",
        {"upstream_support":[0,first-1],"warp_support":[0,first],
         "upstream_file":"mac/csma_ca.py:68-71","warp_file":"gpu/kernels/mac_v67.py:31-34"}))

    import warp as wp
    params=GPUWorldV80Params.from_yaml(
        ROOT/"research/zhang_xue/configs/scenario_v91_bayes_1km.yaml",
        envs=4,device=a.device,seed=44)
    w=WarpParallelMissionV80(params)
    w._update_phy()
    pos=w.buffers["positions"].numpy()
    g_cuda=w.network.data["gain"].numpy()
    alive=w.network.data["radio_alive"].numpy()
    g_np=compute_channel_gain_v67_np(pos,w.buildings,w.blackouts,w.blackout_r,20,50.0,
             alive=alive,obstacle_model=params.obstacle_model,
             per_building_loss_db=params.per_building_loss_db)
    max_abs=float(np.max(np.abs(g_cuda-g_np)))
    max_rel=float(np.max(np.abs(g_cuda-g_np)/(np.maximum(np.abs(g_np),1e-14))))
    gates.append(g("03_PHY_NUMPY_CUDA_PARITY","PASS" if max_rel<3e-3 or max_abs<1e-9 else "FAIL",
       "GPU Warp gain must match this project's NumPy reference on current 3D buildings and blackout",
       {"max_abs":max_abs,"max_rel":max_rel,"worlds":4}))

    # Search a physically meaningful hidden-terminal configuration, not a fake label.
    # A and B cannot CCA each other but can both be decoded by GCS individually.
    builds, blackouts, br=scale_canonical_geometry(1000.0,20)
    rng=random.Random(44)
    found=None
    pos0=pos[:1].copy()
    for _ in range(3000):
        xy0=(rng.uniform(30,970),rng.uniform(30,970))
        xy1=(rng.uniform(30,970),rng.uniform(30,970))
        pos0[0,0,0:2]=xy0
        pos0[0,1,0:2]=xy1
        gains=compute_channel_gain_v67_np(pos0,builds,blackouts,br,20,50.0,
                obstacle_model=params.obstacle_model,per_building_loss_db=params.per_building_loss_db)
        received=float(params.tx_power_w*gains[0,0,1])
        snr0=float(params.tx_power_w*gains[0,0,6]/params.noise_power_w)
        snr1=float(params.tx_power_w*gains[0,1,6]/params.noise_power_w)
        if received<w.cca_w and snr0>w.snr_lin*2 and snr1>w.snr_lin*2:
            found={"uav0_xy":list(xy0),"uav1_xy":list(xy1),"cca_cross_w":received,
                   "cca_threshold_w":float(w.cca_w),"snr0":snr0,"snr1":snr1}
            break
    if found:
        from zhang_xue.gpu.state_v67 import wp as _wp
        t=_wp.to_torch(w.buffers["positions"])
        t[0,0,:2]=__import__("torch").tensor(found["uav0_xy"],device=t.device)
        t[0,1,:2]=__import__("torch").tensor(found["uav1_xy"],device=t.device)
        w._update_phy()
        gg=w.network.data["gain"].numpy()
        actual=float(params.tx_power_w*gg[0,0,1])
        assert actual<float(w.cca_w)
        gates.append(g("04_HIDDEN_TERMINAL_GEOMETRY","PASS","Valid hidden-terminal PHY topology verified on CUDA; MAC collision PDR remains uncalibrated.",found))
    else:
        gates.append(g("04_HIDDEN_TERMINAL_GEOMETRY","OPEN","No physically viable hidden-terminal topology found in 3000 reproducible trials. Do not claim validated."))

    # Actual GPU packet/TTL stress, not an idealized PHY shortcut.
    wp=__import__("warp")
    cont=wp.zeros((4,6,2),dtype=wp.float32,device=a.device)
    dest=wp.ones((4,6),dtype=wp.int32,device=a.device)
    before=w.network.data["stats_gcs"].numpy().copy()
    for _ in range(25): w.step(cont,dest)
    queue=w.network.data["q_count"].numpy()
    over=w.network.data["overflow"].numpy()
    pkts=w.network.data["stats_gcs"].numpy()
    qmax=int(queue.max())
    gates.append(g("05_PACKET_QUEUE_TTL_STRESS","PASS" if
                   qmax<=params.queue_capacity and np.isfinite(queue).all() and int((over>0).sum())==0
                   and int((pkts-before).sum())>0 else "FAIL",
                   "CUDA run validates bounded queues and actual GCS packet delivery, not exact upstream custody parity.",
                   {"steps":25,"worlds":4,"max_queue":qmax,"capacity":int(params.queue_capacity),
                    "overflow_worlds":int((over>0).sum()),"gcs_new_deliveries":int((pkts-before).sum()),
                    "packet_lifetime_us":PKT_TTL_US,
                    "warp_hello_us":HELLO_V89_TX_US,
                    "upstream_hello_us":ref.HELLO_PACKET_LENGTH/ref.BIT_RATE*1e6}))

    gates.append(g("06_UPSTREAM_32SEED_MACRO_PARITY","OPEN",
       "No matched 32-seed upstream SimPy vs Warp experiment with equivalent data payload, "
       "topology, routing and random process; Bianchi saturated single-hop model not "
       "valid for the current unsaturated multi-hop mission / hidden terminals.",
       {"upstream_commit":commit,"upstream_route":ref.ROUTING_PROTOCOL,
        "warp_route":"HELLO distance-vector with store-carry-forward",
        "upstream_payload_bits":ref.AVERAGE_PAYLOAD_LENGTH,
        "warp_evidence_payload":"8-byte origin-indexed entries plus packed header",
        "upstream_MAC":ref.MAC_PROTOCOL,
        "upstream_hello_airtime_us":ref.HELLO_PACKET_LENGTH/ref.BIT_RATE*1e6,
        "ref_arxiv":"https://arxiv.org/abs/2507.09852",
        "Bianchi":"https://ieeexplore.ieee.org/abstract/document/840210"}))
    result={"title":"V91 science-first six-gate network differential audit",
            "upstream":"https://github.com/Zihao-Felix-Zhou/UavNetSim",
            "upstream_commit":commit,"mode":"candidate calibrated surrogate, NOT exact UavNetSim parity",
            "gates":gates,"pass":sum(x["status"]=="PASS" for x in gates),
            "fail":sum(x["status"]=="FAIL" for x in gates),
            "open":sum(x["status"]=="OPEN" for x in gates),
            "scenario_frozen":False}
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
