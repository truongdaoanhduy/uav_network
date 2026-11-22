"""One bounded, real T4x2 MATD3 training run. Never trains to full 20k.

This module is embedded in the Kaggle A/B trial and imported from either
source tree, so the science and environment code are the only differences.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from uav_network.configuration import load_config
from uav_network.experiments.training import train


def scientific_config(config):
    c=config.to_dict()
    c["runtime"].pop("output_dir",None)
    c["tracking"].pop("group",None)
    c["tracking"].pop("run_name",None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest()


def trial(*,mode,output,steps,envs,batch,updates,device,horizon,learning_starts):
    os.environ["UAV_TRAIN_ENV_STEP_LIMIT"]=str(steps*envs)
    os.environ["UAV_TRAINING_ONLY"]="true"
    os.environ["UAV_DIAGNOSTIC_STORE"]="off"
    os.environ["UAV_FORENSIC_ENABLED"]="false"
    os.environ["UAV_DEFERRED_EXPORT_HF"]="false"
    os.environ["WANDB_MODE"]="disabled"
    os.environ["UAV_FAST_TRAINING"]="false"
    # Only the candidate enables the optional no-clone cross-GPU Actor copy.
    os.environ["UAV_DIRECT_GPU_COLLECTOR_COPY"]="true" if mode=="candidate" else "false"
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        os.environ[name]="1"
    if device=="cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count()<2:
            raise RuntimeError("Training A/B requires Kaggle T4 x2; never silently run CPU.")
    c=load_config('configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml')
    c.runtime.device=device;c.runtime.num_envs=envs
    c.runtime.output_dir=str(output);c.runtime.episodes=20000
    c.runtime.max_train_seconds=0
    c.matd3.batch_size=batch;c.matd3.updates_per_step=updates
    # CPU-only development smoke is memory-bounded; real GPU pilot is EXACT R1M.
    if device=="cpu":c.matd3.replay_capacity=max(512,batch*4)
    c.matd3.learning_starts=learning_starts
    c.world.max_steps=horizon
    c.tracking.enabled=False
    c.tracking.deferred_gpu_eval_enabled=False
    c.tracking.online_gpu_eval_enabled=False
    c.tracking.evaluation_interval=0
    c.tracking.evaluation_interval_steps=0
    c.tracking.periodic_evaluation_interval_episodes=0
    c.tracking.resumable_checkpoint_interval_seconds=0
    c.validate()
    assert (c.world.num_uavs,c.world.num_targets,c.world.num_obstacles)==(6,35,25)
    assert c.sensor.report_cell_hypotheses and c.sensor.confirmation_threshold==.95
    assert c.matd3.hidden_sizes==[256,256]
    assert c.runtime.replay_cpu_fraction==.75
    assert c.matd3.replay_capacity==(1_000_000 if device=="cuda" else max(512,batch*4))
    assert c.runtime.deterministic
    if output.exists():raise FileExistsError("Refuse to overwrite trial run directory: "+str(output))
    print("TRAIN_AB_WORKER_START",mode,"worlds",envs,"steps",steps,
          "updates",updates,"batch",batch,"seed",c.runtime.seed,"science",scientific_config(c),
          flush=True)
    started=time.perf_counter()
    summary=train(c)
    elapsed=time.perf_counter()-started
    if (summary["stop_reason"]!="env_step_budget" or summary["global_step"]!=steps*envs
        or summary["requested_episodes"]!=20000 or summary["episode_budget_complete"]):
        raise RuntimeError("Exceeded the bounded training budget; refusing success")
    weights=torch.load(summary["checkpoint"],map_location="cpu",weights_only=True)
    h=hashlib.sha256()
    for name,value in sorted(weights["model"].items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    episodes=Path(output,"episodes.jsonl")
    records=[json.loads(s) for s in episodes.read_text().splitlines() if s.strip()]
    for row in records:
        for key in tuple(row):
            if "time" in key.lower() and key!="mission_time_s":row.pop(key,None)
    metrics_hash=hashlib.sha256(json.dumps(records,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return {
        "kind":"bounded_real_matd3_train_t4x2_not_production",
        "variant":mode,"science_sha256":scientific_config(c),
        "envs":envs,"vector_steps":steps,"batch":batch,
        "updates_per_step":updates,"world_horizon":horizon,
        "total_environment_transitions":summary["global_step"],
        "actual_learner_updates":summary["updates"],
        "completed_episodes":summary["episodes"],
        "training_wall_seconds":elapsed,
        "train_core_seconds":summary["elapsed_s"],
        "train_joint_transitions_per_second":summary["joint_transitions_per_s"],
        "collector_wait_seconds":summary["collection_wait_s"],
        "learner_seconds":summary["learner_s"],
        "replay_sample_host_seconds":summary["replay_sample_host_s"],
        "optimizer_update_host_seconds":summary["optimizer_update_host_s"],
        "model_sha256":h.hexdigest(),"episode_sha256":metrics_hash,
        "checkpoint_path":str(summary["checkpoint"]),
        "stop_reason":summary["stop_reason"],"training_completed_20k":False,
        "requires_gpu_validation_for_speedup":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("baseline","candidate"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=64)
    p.add_argument("--envs",type=int,default=4096)
    p.add_argument("--batch",type=int,default=8192)
    p.add_argument("--updates",type=int,default=4)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--horizon",type=int,default=1500)
    p.add_argument("--learning-starts",type=int,default=131072)
    args=p.parse_args()
    if (args.steps<1 or args.envs<1 or args.batch<1 or args.updates<1
        or args.horizon<1 or args.learning_starts<0
        or args.steps*args.envs<=args.learning_starts):
        raise ValueError("Invalid training probe dimensions; must include real optimizer updates")
    data=trial(mode=args.mode,output=args.output,steps=args.steps,
               envs=args.envs,batch=args.batch,updates=args.updates,
               device=args.device,horizon=args.horizon,learning_starts=args.learning_starts)
    print("MATD3_REAL_TRAIN_GPU_WORKER_JSON "+json.dumps(data,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
