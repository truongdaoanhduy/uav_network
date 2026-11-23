"""Post-training (Validation + Final Test) wall-time updates to the same W&B run.

The learner owns its W&B SDK instance during training and closes it first.
This monitor starts only afterward; it updates *summary fields*, never
training-history step counters or algorithm metrics.
"""
from __future__ import annotations

from datetime import datetime,timezone
import math
import os
from pathlib import Path
import threading
import time

from .live_runtime_heartbeat import hms,total_job_seconds,format_runtime,format_episodes
from .eval_resource_probe import snapshot as resource_snapshot
import json


class PostTrainingRuntimeHeartbeat:
    def __init__(self,*,entity,project,run_id,output,interval=30,completed_episodes=0,target_episodes=20000,owner_run=None):
        if not 1<=interval<=300:
            raise ValueError("Invalid runtime monitor interval")
        self.path=f"{entity}/{project}/{run_id}"
        self.output=Path(output)
        self.interval=float(interval)
        self.stop_signal=threading.Event()
        self.thread=None
        self.last_error=None
        self.latest=None
        self.status="running"
        self.episodes= format_episodes(int(completed_episodes),int(target_episodes))
        self.owner_run=owner_run
        self.gpu_vram_peak_observed_gib={}
        self.resource_journal=self.output/"eval-resource-samples.jsonl"

    def _phase(self):
        if (self.output/"report.json").is_file():
            return "completed_gpu_evaluation"
        if (self.output/"best.pt").is_file():
            return "gpu_final_test"
        return "gpu_validation"

    def _payload(self,phase=None,status=None):
        now=time.perf_counter()
        total=total_job_seconds(now)
        d={
            "Runtime":format_runtime(total) if total is not None else None,
            "Episodes":self.episodes,
            "runtime_phase":phase or self._phase(),
            "runtime_status":status or self.status,
            "runtime_heartbeat_utc":datetime.now(timezone.utc).isoformat(),
            "runtime_session_source":"kaggle_notebook_monotonic_job_timer",
            "runtime_is_live":bool((status or self.status)=="running"),
        }
        if total is not None:
            d["runtime_session_seconds"]=round(total,3)
            d["runtime_session_hms"]=hms(total)
        if d["Runtime"] is None:
            d.pop("Runtime")
        return d

    def publish(self,*,phase=None,status=None):
        data=self._payload(phase=phase,status=status)
        measurements=resource_snapshot()
        data.update(measurements)
        # Sampled peaks are *not* true lifetime peaks; report as observed.
        for device in (0,1):
            key=f"eval_resources/gpu{device}_vram_used_gib"
            if key in measurements:
                self.gpu_vram_peak_observed_gib[device]=max(
                    self.gpu_vram_peak_observed_gib.get(device,0.0),measurements[key])
                data[f"eval_resources/gpu{device}_vram_peak_observed_gib"]=(
                    self.gpu_vram_peak_observed_gib[device])
        self.latest=data
        try:
            self.output.mkdir(parents=True,exist_ok=True)
            with self.resource_journal.open("a") as out:
                out.write(json.dumps(data,allow_nan=False)+"\n")
        except OSError:
            # File I/O must never interrupt running GPU evaluation.
            pass
        try:
            if self.owner_run is not None:
                self.owner_run.summary.update(data)
            else:
                import wandb
                api=wandb.Api(timeout=12)
                remote=api.run(self.path)
                remote.summary.update(data)
        except Exception as error:
            # Time tracking must never stop a real Final Test.
            self.last_error=type(error).__name__
        return data

    def _loop(self):
        self.publish()
        while not self.stop_signal.wait(self.interval):
            self.publish()

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Session heartbeat already started")
        self.thread=threading.Thread(target=self._loop,name="GPUFullFlowRuntimeHeartbeat",
                                     daemon=True)
        self.thread.start()
        return self

    def stop(self,*,failed=False,phase=None):
        self.stop_signal.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        self.status="failed" if failed else "finished"
        self.publish(phase=phase or ("gpu_evaluation_failed" if failed
                                      else "fullflow_complete"),status=self.status)
        return self.latest
