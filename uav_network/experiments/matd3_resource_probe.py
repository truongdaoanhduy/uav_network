"""Bounded, real two-GPU E1024/R1.5M MATD3 memory probe. NEVER training loop.

Real Warp rollout on GPU0, full hybrid replay allocation/physical writes on
CPU+GPU1, two optimizer steps for IND-NoLN, SHARED-NoLN, IND-LN on GPU1.
No W&B, no credentials, no checkpoint, no background process.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import time

import torch
from uav_network.algorithms.checkpoints import make_learner
from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay
from uav_network.algorithms.replay import transition_shapes
from uav_network.configuration import load_config
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.environment.vector import make_vector
from uav_network.experiments.resource_memory import training_memory_metrics, GIB
from uav_network.experiments.training import seed_runtime

PROFILE="configs/comparison-matd3-05-e1024-u4-b8192-r15m-ind-noln.yaml"
REPORT="/kaggle/working/matd3_resource_probe_1m5.json"
MIN_FREE_CUDA_GIB=3.0
MIN_FREE_HOST_GIB=3.0

def bytes_per_transition(config):
    shapes=transition_shapes((config.world.num_uavs, observation_dim(config)),
                             state_dim(config), config.continuous_dim)
    return sum(math.prod(shape)*(8 if name=="destination" else 4)
               for name,shape in shapes.items())

def write(report):
    target=Path(report["report_file"])
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix(".tmp")
    partial.write_text(json.dumps(report,indent=2,default=str)+"\n")
    partial.replace(target)

def snapshot(report,stage,**extra):
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    item={"stage":stage,"elapsed_seconds":round(time.monotonic()-report["started_monotonic"],3),
          **training_memory_metrics(),**extra}
    report["events"].append(item)
    write(report)
    print("PROBE",stage,"cuda0_peak",round(item.get("memory/cuda0_torch_peak_reserved_gib",0),2),
          "cuda1_peak",round(item.get("memory/cuda1_torch_peak_reserved_gib",0),2),
          "cgroup_headroom",round(item.get("memory/cgroup_headroom_now_gib",-1),2),
          flush=True)
    return item

def memory_guard(config,report,stage,*,include_full_replay=False):
    m=training_memory_metrics()
    freehost=m.get("memory/cgroup_headroom_now_gib",
                   m.get("memory/host_system_available_gib",0))
    freegpu=m.get("memory/cuda1_driver_free_now_gib",0)
    transition=bytes_per_transition(config)
    hot_rows=config.matd3.replay_capacity*(1-config.runtime.replay_cpu_fraction)
    cpu_rows=config.matd3.replay_capacity*config.runtime.replay_cpu_fraction
    if include_full_replay:
        required_cpu=cpu_rows*transition/GIB+MIN_FREE_HOST_GIB
        required_gpu=hot_rows*transition/GIB+MIN_FREE_CUDA_GIB
    else:
        required_cpu=MIN_FREE_HOST_GIB
        required_gpu=MIN_FREE_CUDA_GIB
    report["guards"].append({"stage":stage,"available_host_gib":freehost,
        "available_cuda1_gib":freegpu,"required_host_gib":required_cpu,
        "required_cuda1_gib":required_gpu})
    write(report)
    if freehost<required_cpu or freegpu<required_gpu:
        raise RuntimeError(
            "Unsafe memory headroom at "+stage+
            " host/gpu1 available "+f"{freehost:.2f}/{freegpu:.2f} GiB"+
            " required "+f"{required_cpu:.2f}/{required_gpu:.2f} GiB")

def run(report_file=REPORT):
    report={"report_file":report_file,"started_monotonic":time.monotonic(),
        "status":"running","purpose":"BOUNDED_HARDWARE_PROBE_NO_TRAINING",
        "stages":["check","warp","allocate","physically_touch_memory",
                  "collect_8_steps","two_updates_per_architecture","finish"],
        "events":[],"guards":[],"variants":[]}
    vec=replay=learner=None
    try:
        assert torch.cuda.is_available() and torch.cuda.device_count()==2, \
            "Requires Kaggle accelerator NvidiaTeslaT4 (two CUDA GPUs)"
        report["gpu_names"]=[torch.cuda.get_device_name(i) for i in range(2)]
        config=load_config(PROFILE)
        assert config.runtime.num_envs==1024
        assert config.matd3.batch_size==8192
        assert config.matd3.replay_capacity==1_500_000
        assert config.runtime.replay_cpu_fraction==.625
        assert not config.pql.enabled and not config.matd3.actor_layer_norm
        config.tracking.enabled=False
        config.runtime.device="cuda"
        torch.set_num_threads(1)
        seed_runtime(config)
        for i in range(2):
            torch.cuda.reset_peak_memory_stats(i)
        memory_guard(config,report,"before_simulation")
        snapshot(report,"initial")
        vec=make_vector(config,device=torch.device("cuda:0"))
        obs,_=vec.reset(seed=config.runtime.seed)
        snapshot(report,"warp_env_initialized",
                 env_memory_bytes=getattr(vec,"allocated_bytes",None))
        memory_guard(config,report,"before_replay_allocate",include_full_replay=True)
        replay=HybridMeasuredReplay(config.matd3.replay_capacity,
          (config.world.num_uavs,observation_dim(config)),state_dim(config),
          seed=config.runtime.seed,hot_device="cuda:1",
          cpu_fraction=config.runtime.replay_cpu_fraction,
          continuous_dim=config.continuous_dim)
        snapshot(report,"replay_allocated",
          replay_cpu_gib=round(replay.cpu_allocated_bytes/GIB,3),
          replay_gpu1_gib=round(replay.hot_allocated_bytes/GIB,3))
        # Make CUDA and Linux cgroup physically account allocated storage.
        for tensor in replay.cpu_data.values():tensor.zero_()
        for tensor in replay.hot_data.values():tensor.zero_()
        snapshot(report,"replay_physically_committed")
        memory_guard(config,report,"after_replay_touch")
        for t in range(8):
            actions=vec.random_actions(obs["observation"])
            nxt,reward,terminated,truncated,info=vec.step({
                 "continuous":actions.continuous,"destination":actions.destination})
            replay.add_batch(obs["observation"],obs["state"],actions,reward,
                             nxt["observation"],nxt["state"],terminated)
            obs=nxt
            if bool((terminated|truncated).any()):
                masks=terminated|truncated
                obs,_=vec.reset(seed=None,options={"reset_mask":masks.cpu().numpy()})
        assert len(replay)>=8192
        snapshot(report,"actual_8_vector_steps_collected",filled_rows=len(replay))
        for variant,shared,ln in [
            ("independent_no_ln",False,False),
            ("shared_no_ln",True,False),
            ("independent_ln",False,True)]:
            local=deepcopy(config)
            local.matd3.share_actor_parameters=shared
            local.matd3.share_critic_parameters=shared
            local.matd3.actor_layer_norm=ln
            learner=make_learner(local,observation_dim(local),state_dim(local),
                                 torch.device("cuda:1"))
            torch.cuda.reset_peak_memory_stats(1)
            started=time.monotonic()
            try:
                for update in range(2):
                    batch=replay.sample(8192)
                    result=learner.update(batch,collect_metrics=(update==1))
                    del batch
                torch.cuda.synchronize(1)
                event=snapshot(report,"two_updates_done_"+variant,
                    optimizer_elapsed_seconds=round(time.monotonic()-started,3),
                    actor_updates=learner.actor_updates,
                    actor_grad_norm=result.get("actor_grad_norm"),
                    critic_loss=result.get("critic_loss"))
                report["variants"].append({
                    "name":variant,"status":"ok","actor_updates":learner.actor_updates,
                    "cuda1_peak_reserved_gib":event.get("memory/cuda1_torch_peak_reserved_gib"),
                    "cgroup_headroom_gib":event.get("memory/cgroup_headroom_now_gib"),
                    "optimizer_elapsed_seconds":round(time.monotonic()-started,3)})
                write(report)
                memory_guard(local,report,"after_"+variant)
            finally:
                del learner
                learner=None
                gc.collect()
                torch.cuda.empty_cache()
        report["status"]="passed"
        snapshot(report,"finished")
    except Exception as e:
        report["status"]="failed_or_skipped"
        report["error_type"]=type(e).__name__
        report["error_message"]=str(e)[:500]
        try:snapshot(report,"failed_or_skipped")
        except Exception:write(report)
    finally:
        if vec is not None:vec.close()
        report.pop("started_monotonic",None)
        write(report)
        print("FINAL_REPORT",str(report_file),"STATUS",report["status"],flush=True)
    return 0 if report["status"]=="passed" else 2

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",default=REPORT)
    args=p.parse_args()
    raise SystemExit(run(args.report))

if __name__=="__main__":
    main()
