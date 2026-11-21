"""Bounded Kaggle hardware check for E2048 U8 B8192 R1M MATD3 runtime logs.

GPU0: Warp 2048 env + Actor inference + action/CBF telemetry.
GPU1: independent MATD3 6 actors, 12 Q, 8 optimizer updates, hybrid replay.
CPU: 75% cold replay. Never runs full training episodes, W&B or HF.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import traceback
import torch

from .action_cbf_monitor import ActionCBFMonitor
from .training_health import TrainingHealthMonitor
from .resource_memory import training_memory_metrics, reset_driver_watermarks, GIB
from .training import seed_runtime
from .pql_workers import inference_copy
from ..algorithms.matd3 import MATD3
from ..algorithms.hybrid_replay import HybridMeasuredReplay
from ..configuration import load_config
from ..environment.observations import observation_dim, state_dim
from ..environment.vector import make_vector


CONFIG="configs/comparison-matd3-07-e2048-u8-b8192-r1m-ind-noln.yaml"
OUT="/kaggle/working/matd3_e2048_u8_r1m_logging_probe.json"


def _save(report):
    path=Path(report["report_file"])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,default=str,ensure_ascii=False)+"\n")


def _sample(report, label):
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    reading=training_memory_metrics()
    report["snapshots"].append({"phase":label,"seconds":round(time.monotonic()-report["started"],3),**reading})
    _save(report)
    print("SNAPSHOT",label,
          "cuda0_driver_free",round(reading.get("memory/cuda0_driver_free_now_gib",-1),3),
          "cuda1_driver_free",round(reading.get("memory/cuda1_driver_free_now_gib",-1),3),
          "cuda1_peak_reserved",round(reading.get("memory/cuda1_torch_peak_reserved_gib",-1),3),
          "host_headroom",round(reading.get("memory/cgroup_headroom_now_gib",-1),3),
          flush=True)
    return reading


def _check_headroom(memory, required_cpu, required_gpu):
    cpu=memory.get("memory/cgroup_headroom_now_gib",memory.get("memory/host_system_available_gib",0))
    gpu=memory.get("memory/cuda1_driver_free_now_gib",0)
    if cpu < required_cpu+3.0 or gpu < required_gpu+3.0:
        raise RuntimeError(
            f"Safety guard: need host {required_cpu+3:.2f} GiB, GPU1 {required_gpu+3:.2f} GiB; "
            f"available {cpu:.2f}/{gpu:.2f} GiB")


def run(report_file=OUT):
    report={"report_file":report_file,"started":time.monotonic(),
      "status":"running","scope":"2-T4 BOUNDED NO LONG TRAINING",
      "config":CONFIG,"timestamp_utc":datetime.now(timezone.utc).isoformat(),
      "snapshots":[]}
    vector=replay=learner=collector=None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
            raise RuntimeError("Exact two NVIDIA GPUs required")
        config=load_config(CONFIG)
        assert config.runtime.num_envs==2048
        assert config.matd3.batch_size==8192
        assert config.matd3.replay_capacity==1_000_000
        assert config.matd3.updates_per_step==8
        assert not config.matd3.actor_layer_norm
        assert not config.matd3.share_actor_parameters
        assert not config.matd3.share_critic_parameters
        assert config.runtime.replay_cpu_fraction==.75
        assert not config.pql.enabled
        config.tracking.enabled=False
        torch.set_num_threads(1)
        seed_runtime(config)
        reset_driver_watermarks()
        report["hardware"]=[torch.cuda.get_device_name(x) for x in range(2)]
        for gpu in (0,1):torch.cuda.reset_peak_memory_stats(gpu)
        init=_sample(report,"start")
        vector=make_vector(config,device=torch.device("cuda:0"))
        obs,_=vector.reset(seed=config.runtime.seed)
        _sample(report,"warp2048")
        from .matd3_resource_probe import bytes_per_transition
        item=bytes_per_transition(config)
        cold=config.matd3.replay_capacity*.75*item/GIB
        hot=config.matd3.replay_capacity*.25*item/GIB
        _check_headroom(training_memory_metrics(),cold,hot)
        replay=HybridMeasuredReplay(config.matd3.replay_capacity,
            (config.world.num_uavs,observation_dim(config)),state_dim(config),
            seed=config.runtime.seed,hot_device="cuda:1",
            cpu_fraction=.75,continuous_dim=config.continuous_dim)
        # Commit real physical storage, not just virtual reservation.
        for t in replay.cpu_data.values():t.zero_()
        for t in replay.hot_data.values():t.zero_()
        _sample(report,"replay1m_physically_touched")
        learner=MATD3(config,observation_dim(config),state_dim(config),device="cuda:1")
        collector=inference_copy(learner,torch.device("cuda:0"))
        action_monitor=ActionCBFMonitor(config,Path(report_file).parent)
        health_monitor=TrainingHealthMonitor(config,Path(report_file).parent)
        # Exactly 4 vector steps x 2048 env = 8192 transitions.
        for step in range(4):
            before=obs
            action=collector.act_batch(before["observation"],step=2048*step,return_tensors=True)
            obs,reward,terminated,truncated,infos=vector.step(
                {"continuous":action.continuous,"destination":action.destination})
            replay.add_batch(before["observation"],before["state"],action,reward,
                             obs["observation"],obs["state"],terminated)
            if action_monitor.due((step+1)*2048):
                action_monitor.sample(vector,before["observation"],action,(step+1)*2048,phase="policy")
            if bool((terminated | truncated).any()):
                # 4 steps well before 1500 max, but keep the probe correct.
                obs,_=vector.reset(options={"reset_mask":(terminated|truncated).cpu().numpy()})
        assert len(replay)>=8192
        report["action_probe_fields"]=len(action_monitor.last_values)
        report["action_cbf_sample_file"]=str(action_monitor.path)
        _sample(report,"2048env_4steps")
        latest={}
        start=time.monotonic()
        for update in range(8):
            sampled=replay.sample(8192)
            latest=learner.update(sampled,collect_metrics=update==7)
            del sampled
        torch.cuda.synchronize(1)
        report["optimizer_seconds"]=round(time.monotonic()-start,3)
        report["critic_updates"]=learner.updates
        report["actor_updates"]=learner.actor_updates
        report["learner_metrics_logged"]=sorted(latest)
        res=_sample(report,"8_updates_backward_and_adam")
        health,alerts=health_monitor.capture(8192,action_monitor.last_values,latest,replay.stats(),res,
                                             updates=learner.updates,phase="policy")
        report["health_alerts"]=alerts
        report["health_log_file"]=str(health_monitor.path)
        report["health_metric_count"]=len(health)
        if res["memory/cuda1_driver_free_now_gib"]<2 or res.get("memory/cgroup_headroom_now_gib",3)<3:
            raise RuntimeError("Insufficient post-backward memory margin")
        report["status"]="passed"
        _sample(report,"completed")
    except Exception as exc:
        report["status"]="failed_or_skipped"
        report["error_type"]=type(exc).__name__
        report["error"]=str(exc)[:700]
        report["traceback"]=traceback.format_exc()[-4500:]
        try:_sample(report,"failed_or_skipped")
        except Exception:_save(report)
    finally:
        if vector is not None:vector.close()
        if collector is not None:del collector
        if learner is not None:del learner
        if replay is not None:del replay
        gc.collect()
        report.pop("started",None)
        _save(report)
        print("RESULT",report["status"],report_file,flush=True)
    return 0 if report["status"]=="passed" else 2


if __name__=="__main__":
    raise SystemExit(run())
