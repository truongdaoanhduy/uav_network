"""Small wall-clock heartbeat for W&B Runs tables, independent of episode updates.

Writes to the *summary only*, never to the training-history step counter.
The last received heartbeat remains meaningful after an abnormal kernel crash.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time


def hms(seconds):
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds < 0:
        return None
    total=int(seconds+0.5)
    hh,remaining=divmod(total,3600)
    mm,ss=divmod(remaining,60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def format_runtime(seconds):
    """The existing W&B Runs table Runtime column uses this summary string."""
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds < 0:
        return None
    sec=int(float(seconds)+0.5)
    hour,sec=divmod(sec,3600)
    minute,sec=divmod(sec,60)
    parts=[]
    if hour:parts.append(f"{hour}h")
    if minute:parts.append(f"{minute}m")
    if sec or not parts:parts.append(f"{sec}s")
    return " ".join(parts)


def format_episodes(completed, target):
    """Actual completed training episodes, not rollout iterations or estimates."""
    if (isinstance(completed,bool) or isinstance(target,bool)
            or not isinstance(completed,int) or not isinstance(target,int)
            or target < 1 or not 0 <= completed <= target):
        raise ValueError("Invalid actual completed episode count / budget")
    return f"{completed} / {target}"


def total_job_seconds(now):
    """Same monotonic clock across the Kaggle parent and training child."""
    try:
        start=float(os.environ["UAV_JOB_STARTED_AT"])
        if math.isfinite(start) and 0 <= now-start < 7*24*3600:
            return max(0.0, now-start)
    except (ValueError,KeyError,OverflowError):
        pass
    return None


def heartbeat_payload(*,now,started,phase,status,source="live_monotonic_heartbeat"):
    elapsed=max(0.0,float(now)-float(started))
    total=total_job_seconds(now)
    payload={
        "Runtime":format_runtime(total if total is not None else elapsed),
        "runtime_live_seconds":round(elapsed,3),
        "runtime_live_hms":hms(elapsed),
        "runtime_phase":str(phase),
        "runtime_status":str(status),
        "runtime_heartbeat_utc":datetime.now(timezone.utc).isoformat(),
        "runtime_source":source,
        "runtime_is_live":bool(status=="running"),
        "run_elapsed_seconds":round(elapsed,3),
        "run/elapsed_hms":hms(elapsed),
        "run/elapsed_hms_source":"live_monotonic_heartbeat",
    }
    if total is not None:
        payload["runtime_session_seconds"]=round(total,3)
        payload["runtime_session_hms"]=hms(total)
    return payload


class LiveRuntimeHeartbeat:
    """One daemon thread, one W&B summary patch every 30s, no step pollution."""

    def __init__(self,run,*,started,output,interval=30,initial_episodes=0,target_episodes=20000):
        if not 1<=interval<=300:
            raise ValueError("Invalid heartbeat interval")
        self.run=run
        self.started=float(started)
        self.output=Path(output)
        self.interval=float(interval)
        self.stop_signal=threading.Event()
        self.thread=None
        self.last_payload=None
        self.last_error=None
        self.lock=threading.Lock()
        format_episodes(int(initial_episodes),int(target_episodes))
        self.episodes_completed=int(initial_episodes)
        self.target_episodes=int(target_episodes)

    def record_episodes(self,completed):
        """In-memory only; never forces W&B I/O from the hot learner loop."""
        value=int(completed)
        with self.lock:
            format_episodes(value,self.target_episodes)
            self.episodes_completed=max(self.episodes_completed,value)

    def publish(self,phase="training",status="running"):
        with self.lock:
            values=heartbeat_payload(now=time.perf_counter(),started=self.started,
                                     phase=phase,status=status)
            values["Episodes"]=format_episodes(self.episodes_completed,self.target_episodes)
            self.last_payload=values
            # Keep local evidence if the W&B network gets disconnected. This
            # file is intentionally tiny; no training arrays are serialized.
            path=self.output/"runtime-heartbeat.json"
            try:
                temp=path.with_suffix(".json.tmp")
                temp.write_text(json.dumps(values,allow_nan=False))
                temp.replace(path)
            except OSError as error:
                self.last_error=type(error).__name__
            try:
                self.run.summary.update(values)
            except Exception as error:
                # Heartbeat must NEVER abort a healthy RL training run.
                self.last_error=type(error).__name__
        return values

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Heartbeat already started")
        self.publish()
        self.thread=threading.Thread(target=self._loop,name="WandbRuntimeHeartbeat",
                                     daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self.stop_signal.wait(self.interval):
            self.publish()

    def stop(self,failed=False):
        self.stop_signal.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        self.publish(phase="training_failed" if failed else "training_complete",
                     status="failed" if failed else "finished")
        return self.last_payload
