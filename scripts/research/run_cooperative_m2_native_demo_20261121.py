"""Diagnostic-only 2D target belief mapping with actual 3D UAV flight and UavNetSim.

A hand-coded frontier/router policy is used for SMOKE TESTING, not an RL-trained
MASAC/MATD3 actor. Plots compare only GCS 2D target belief against
simulator target truth offline. Ground truth is NEVER passed to policy input.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from uav_network.configuration import Config
from uav_network.environment.env import Action
from uav_network.environment.m2_mission import CooperativeM2Env


def diagnostics_policy(env:CooperativeM2Env):
    """Observable frontier and contact heuristic — NOT learned policy."""
    n=env.config.world.num_uavs
    w=env.config.world
    a=np.zeros((n,env.config.continuous_dim),np.float32)
    destinations=np.zeros(n,np.int64)
    for i,position in enumerate(env.positions):
        node=env.nodes[i]
        unknown=np.flatnonzero(node.coverage==0)
        if len(unknown):
            # Use ONLY the node's local map and position; never oracle target coords.
            # Spread UAVs across the map by deterministic frontier IDs.
            stride=max(1,len(unknown)//max(n,1))
            shortlist=unknown[(i*stride)%len(unknown)::max(n,1)]
            if not len(shortlist):shortlist=unknown
            centers=np.array([env._center(int(cell)) for cell in shortlist])
            distance=np.linalg.norm(centers-position[:2],axis=1)
            goal=centers[np.argmin(distance)]
            desired_xy=goal-position[:2]
            length=np.linalg.norm(desired_xy)
            if length>1e-9:desired_xy=desired_xy/length*min(10.,length)
            a[i,:2]=np.clip((desired_xy-env.velocities[i,:2])/(w.max_accel_mps2*w.dt_s),-1,1)
        # Diagnostic policy keeps altitude. The simulator NEVER forces descent.
        gcs=w.num_uavs
        if env.radio.link_available(i,gcs,env.positions):
            destinations[i]=gcs
        else:
            eligible=[j for j in range(n) if j!=i and env.radio.link_available(i,j,env.positions)]
            if eligible:
                current=np.linalg.norm(env.positions[i]-env.scene.gcs)
                nearer=[j for j in eligible if np.linalg.norm(env.positions[j]-env.scene.gcs)<current]
                if nearer:
                    chosen=min(nearer,key=lambda j:np.linalg.norm(env.positions[j]-env.scene.gcs))
                    destinations[i]=chosen+1 if chosen<i else chosen
        a[i,-1]=1.0  # high TX power for the diagnostic only
    return Action(a,destinations)


def write_figure(env,output_dir,timeline):
    """Plot target belief only: the paper's F/E/U layers on a 2D ground grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n=env.grid_n
    local=env.nodes[0];gcs=env.gcs_map
    maps=[
      (env._truth_target.reshape(n,n).astype(float),"GROUND TRUTH targets — evaluator only","Greys",0,1),
      (local.target_masses[:,0].reshape(n,n),"UAV 0 target evidence F","viridis",0,1),
      (gcs.target_masses[:,0].reshape(n,n),"GCS target evidence F","viridis",0,1),
      (gcs.target_masses[:,1].reshape(n,n),"GCS empty evidence E","viridis",0,1),
      (gcs.target_masses[:,2].reshape(n,n),"GCS unknown mass U","magma",0,1),
      ((gcs.coverage>0).reshape(n,n).astype(float),"GCS received-evidence coverage","viridis",0,1)
    ]
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,(values,title,cmap,vmin,vmax) in zip(axes.flat,maps):
        im=ax.imshow(values,origin="lower",cmap=cmap,vmin=vmin,vmax=vmax,interpolation="nearest")
        ax.set_title(title,fontsize=11);ax.set_xlabel("grid x");ax.set_ylabel("grid y")
        fig.colorbar(im,ax=ax,fraction=.046,pad=.025)
    fig.suptitle(f"2D Target Belief Map • Native UavNetSim • {env.steps} steps • scripted (not trained)")
    fig.savefig(output_dir/"m2_target_belief_2d_groundtruth_vs_gcs.png",dpi=155)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    t=np.array([x["t"] for x in timeline])
    for name in ("gcs_coverage","target_brier","normalized_uncertainty","correct_gcs_delivery_recall"):
        ax.plot(t,[x[name] for x in timeline],label=name)
    ax.set_ylim(0,1);ax.set_xlabel("Mission step (s)")
    ax.set_ylabel("Normalized map/search metric")
    ax.grid(alpha=.2);ax.legend(loc='best');ax.set_title("GCS 2D target-belief reconstruction over actual radio arrivals")
    fig.savefig(output_dir/"m2_gcs_target_belief_progress.png",dpi=155)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=60)
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--output",type=Path,
        default=Path.home()/"Downloads"/"UAV_M2_TARGET_BELIEF_2D_1000M_20261121")
    args=parser.parse_args()
    if not 1<=args.steps<=1500:parser.error("--steps must be in 1..1500")
    args.output.mkdir(parents=True,exist_ok=True)
    c=Config()
    c.world.motion_dimensions=3;c.sensor.model="bayesian";c.world.motion_safety="cbf"
    c.world.num_uavs=6;c.world.num_targets=35;c.world.num_obstacles=25
    c.world.map_size_m=1000.;c.world.max_steps=args.steps
    c.sensor.grid_cell_m=25.;c.energy.enabled=True
    c.network.report_bytes=c.network.payload_bytes=1024
    c.network.peer_range_m=c.network.gcs_range_m=500.
    env=CooperativeM2Env(c,correlation_strength=0.,max_scan_cells=None,
                          map_buffer_bytes=65536)
    timeline=[]
    try:
        _,info=env.reset(seed=args.seed)
        for t in range(args.steps):
            obs,reward,done,_,info=env.step(diagnostics_policy(env))
            mm=info["gcs_map"]
            timeline.append({"t":t+1,
                "gcs_coverage":mm["coverage"],
                "target_brier":mm["target_brier"],
                "normalized_uncertainty":mm["mean_uncertainty"],
                "correct_gcs_delivery_recall":info["correct_gcs_delivery_recall"],
                "gcs_evidence_events":info["gcs_events"],
                "received_radio_packets":info["communication"]["received_packets"],
                "reward":float(reward)})
            if done:break
        write_figure(env,args.output,timeline)
        n=env.grid_n
        np.savez_compressed(args.output/"m2_target_belief_2d_evaluator_only.npz",
            truth_target=env._truth_target.reshape(n,n).astype(np.uint8),
            uav_target_masses=np.stack([m.target_masses.reshape(n,n,3) for m in env.nodes[:-1]]),
            gcs_target_masses=env.gcs_map.target_masses.reshape(n,n,3),
            gcs_coverage=env.gcs_map.coverage.reshape(n,n),
            gcs_last_observed=env.gcs_map.last_observed.reshape(n,n))
        import csv
        with (args.output/"m2_target_belief_2d_timeline.csv").open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=timeline[0].keys());w.writeheader();w.writerows(timeline)
        result={"status":"native_2d_target_belief_smoke_NOT_RL_training","seed":args.seed,
            "configuration":{"uavs":6,"targets":35,"obstacles_for_motion_and_occlusion_only":25,
                "map_m":1000,"grid_cell_m":25,"belief_map_layers":["F","E","U"],
                "correlated_sensor":False,
                "evidence_mode":"zhang_max_counts","confirmation_mode":"ds_threshold",
                "ds_positive_bpa":.8,"ds_negative_bpa":.8,
                "ds_detection_threshold":.95},
            "final":{key:val for key,val in info.items() if key in (
                "steps","gcs_map","gcs_events","correct_gcs_delivery_recall",
                "normalized_delivery_auc","gcs_verified_correct","gcs_verified_false",
                "false_report_rate","energy_total_j","energy_per_correct_delivery_j",
                "gcs_report_history_count")},
            "network":{"backend":info["communication"]["backend"],
                "generated_packets":info["communication"]["generated_packets"],
                "received_packets":info["communication"]["received_packets"],
                "packet_pdr_percent":info["communication"]["packet_pdr_percent"]},
            "timeline":timeline,
            "notes":["Scripted frontier/relay policy only; no RL convergence measured.",
                "Only the 2D target F/E/U belief map is built and synchronized; UAV physical flight remains 3D.",
                "True target coordinates are used only for sensing and offline evaluation.",
                "UavNetSim packet-level delivery; no global-map memory-copy shortcut."]}
        (args.output/"m2_demo_metrics.json").write_text(json.dumps(result,indent=2))
        print(json.dumps({"output":str(args.output),"steps":env.steps,
                "gcs_coverage":info["gcs_map"]["coverage"],
                "gcs_events":info["gcs_events"],
                "gcs_reported_true_targets":info["gcs_verified_correct"],
                "radio_packets":info["communication"]["received_packets"]},indent=2))
    finally:env.close()

if __name__=="__main__":main()
