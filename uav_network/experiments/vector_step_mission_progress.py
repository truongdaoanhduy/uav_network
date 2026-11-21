"""Faithful per-vector-step mission progress, saved to HF and plotted on W&B.

Each record describes the **same post-step, pre-reset 4096 worlds** that
produced the current batch of team rewards. The progress percentages are
instantaneous in-episode state, NOT completed-episode success rates or AUC.

Only exact sufficient statistics (count, sum, sum of squares) leave GPU;
CUDA transfers are buffered in small groups to avoid per-vector-step sync.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "discovery_rate_percent",
    "gcs_delivery_rate_percent",
    "map_coverage_percent",
    "mean_battery_fraction",
    "mean_active_uavs",
)


def instantaneous_mission_progress(observations, visited, *, num_uavs, num_targets, num_obstacles,
                                  world_state=None, grid_n=None, cell_width=None):
    """Per-world vectors from the *actual* post-step observation and visited.

    Discovery = union of `known` targets across UAVs, / targets_count.
    Delivery = state.delivered bitset, / targets_count.
    Coverage = union of visited cells across UAVs, / grid_cells.
    This matches Warp `world_info` terminal values without forcing CPU copies.
    """
    if visited is None:
        raise ValueError("visited is required to measure real map coverage (never fabricate zero)")
    if world_state is not None:
        # The version-7 cell-report observation intentionally does not encode
        # truth-rated per-target found/delivered masks. Only a private telemetry
        # scorer may use simulator target XY for these real convergence plots.
        from .nooracle_telemetry import true_cell_progress
        info=true_cell_progress(world_state,int(grid_n),float(cell_width))
        local=observations["observation"]
        seen=visited.bool().any(dim=1)
        return {
            "discovery_rate_percent":info['found'].double().mean(dim=1)*100.,
            "gcs_delivery_rate_percent":info['delivered'].double().mean(dim=1)*100.,
            "map_coverage_percent":seen.double().mean(dim=1)*100.,
            "mean_battery_fraction":local[:,:,8].double().mean(dim=1),
            "mean_active_uavs":local[:,:,9].double().sum(dim=1),
        }
    local = observations["observation"]
    state = observations["state"]
    e, u, t, o = local.shape[0], int(num_uavs), int(num_targets), int(num_obstacles)
    if not t or not u or local.ndim != 3 or local.shape[1] != u or local.shape[2] < 10+t:
        raise ValueError("Invalid UAV local observations for progress extraction")
    offset=4*u + 2*t + 3*o + 2*u*t
    if state.ndim != 2 or state.shape[0] != e or state.shape[1] < offset+t:
        raise ValueError("Invalid centralized state for delivered-target progress")
    if visited.ndim != 3 or visited.shape[:2] != (e,u) or visited.shape[2] < 1:
        raise ValueError("visited must be [environments, UAVs, coverage_cells]")
    if hasattr(local,"detach"):
        import torch
        if not (hasattr(state,"detach") and hasattr(visited,"detach")):
            raise TypeError("Torch progress requires Torch local/state/visited on the same device")
        if state.device != local.device or visited.device != local.device:
            raise ValueError("Progress tensors must use the same device")
        with torch.no_grad():
            known=(local[:,:,10:10+t] > .5).any(dim=1)
            delivered=state[:,offset:offset+t] > .5
            seen=visited.bool().any(dim=1)
            return {
                "discovery_rate_percent":known.to(torch.float64).mean(dim=1)*100.,
                "gcs_delivery_rate_percent":delivered.to(torch.float64).mean(dim=1)*100.,
                "map_coverage_percent":seen.to(torch.float64).mean(dim=1)*100.,
                "mean_battery_fraction":local[:,:,8].to(torch.float64).mean(dim=1),
                "mean_active_uavs":local[:,:,9].to(torch.float64).sum(dim=1),
            }
    if hasattr(state,"detach") or hasattr(visited,"detach"):
        raise TypeError("Cannot combine numpy observations with Torch progress state")
    local=np.asarray(local)
    state=np.asarray(state)
    seen=np.asarray(visited).astype(bool).any(axis=1)
    return {
        "discovery_rate_percent":(local[:,:,10:10+t] > .5).any(axis=1).mean(axis=1)*100.,
        "gcs_delivery_rate_percent":(state[:,offset:offset+t] > .5).mean(axis=1)*100.,
        "map_coverage_percent":seen.mean(axis=1)*100.,
        "mean_battery_fraction":local[:,:,8].mean(axis=1,dtype=np.float64),
        "mean_active_uavs":local[:,:,9].sum(axis=1,dtype=np.float64),
    }


class PerVectorStepMissionProgressJournal:
    """Lossless-for-mean-and-SD progress archive plus live 1/2/4/8 W&B scalars."""
    FILENAME="training-step-progress.jsonl"

    def __init__(self,num_envs,output,num_uavs,num_targets,num_obstacles,
                 *,host_sync_every=4,plot_windows=(1,2,4,8)):
        self.num_envs=int(num_envs)
        self.num_uavs=int(num_uavs)
        self.num_targets=int(num_targets)
        self.num_obstacles=int(num_obstacles)
        self.host_sync_every=int(host_sync_every)
        self.plot_windows=tuple(sorted(set(int(v) for v in plot_windows)))
        if (self.num_envs<1 or self.host_sync_every<1 or
            not self.plot_windows or self.plot_windows[0]!=1 or
            any(v<1 for v in self.plot_windows)):
            raise ValueError("Invalid mission progress journal configuration")
        self.path=Path(output)/self.FILENAME
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._file=self.path.open("a",encoding="utf-8",buffering=65536)
        self._last_step=None
        self._pending=[]
        self._windows={v:[] for v in self.plot_windows if v>1}
        self._closed=False

    def observe(self,observations,global_step,*,visited,world_state=None,grid_n=None,cell_width=None):
        if self._closed:
            raise RuntimeError("Mission progress journal is closed")
        step=int(global_step)
        if step<self.num_envs or step%self.num_envs:
            raise ValueError("global_step must align with a full environment vector step")
        if self._last_step is not None and step!=self._last_step+self.num_envs:
            raise ValueError("Missing or duplicate vector step in mission progress")
        values=instantaneous_mission_progress(observations,visited,
            num_uavs=self.num_uavs,num_targets=self.num_targets,
            num_obstacles=self.num_obstacles,world_state=world_state,
            grid_n=grid_n,cell_width=cell_width)
        if any(int(v.shape[0])!=self.num_envs for v in values.values()):
            raise ValueError("Expected exactly one mission-progress value per environment")
        if hasattr(next(iter(values.values())),"detach"):
            import torch
            with torch.no_grad():
                tensor=torch.stack([values[name] for name in METRICS])
                # Per-world vectors stay on GPU; retain 5 sum/sum_sq pairs.
                sums=tensor.sum(dim=1)
                squares=tensor.square().sum(dim=1)
        else:
            tensor=np.stack([np.asarray(values[name],dtype=np.float64) for name in METRICS])
            sums=tensor.sum(axis=1,dtype=np.float64)
            squares=np.square(tensor).sum(axis=1,dtype=np.float64)
        self._last_step=step
        self._pending.append((step,sums,squares))
        return self.flush() if len(self._pending)>=self.host_sync_every else []

    def flush(self):
        if not self._pending:return []
        pending=self._pending
        self._pending=[]
        if hasattr(pending[0][1],"detach"):
            import torch
            with torch.no_grad():
                # One small transfer [4 steps, 2 aggregates, 5 metrics]
                packed=torch.stack([torch.stack((totals,squares))
                                    for _,totals,squares in pending]).detach().cpu().numpy()
        else:
            packed=np.stack([np.stack((totals,squares)) for _,totals,squares in pending])
        events=[]
        for i,(step,_,_) in enumerate(pending):
            raw={"schema_version":1,"global_step":step,
                 "vector_step":step//self.num_envs,"num_envs":self.num_envs,
                 "semantics":"post_step_pre_reset_live_episode_progress",
                 "metrics":{}}
            event={"global_step":step}
            current={}
            for index,name in enumerate(METRICS):
                total=float(packed[i,0,index])
                squared=float(packed[i,1,index])
                count=self.num_envs
                if not (math.isfinite(total) and math.isfinite(squared)):
                    raise ValueError(f"Nonfinite mission metric: {name}")
                mu=total/count
                sd=math.sqrt(max(0.,squared/count-mu*mu))
                raw["metrics"][name]={"count":count,"sum":total,"sum_sq":squared,
                                       "mean":mu,"std":sd}
                event[f"train_progress/{name}"]=mu
                event[f"train_progress/{name}_std"]=sd
                current[name]=(total,squared,count)
            self._file.write(json.dumps(raw,allow_nan=False,separators=(",",":"))+"\n")
            for width,window in self._windows.items():
                window.append(current)
                if len(window)==width:
                    for name in METRICS:
                        total=math.fsum(row[name][0] for row in window)
                        squared=math.fsum(row[name][1] for row in window)
                        count=sum(row[name][2] for row in window)
                        mu=total/count
                        event[f"train_progress/{name}_{width}"]=mu
                        event[f"train_progress/{name}_std_{width}"]=math.sqrt(
                            max(0.,squared/count-mu*mu))
                    window.clear()
            events.append(event)
        self._file.flush()
        return events

    def close(self):
        if self._closed:return []
        events=self.flush()
        self._file.close()
        self._closed=True
        return events


def pair_reward_and_progress_events(reward_events,progress_events):
    """Publish both observables in ONE W&B history row per actual vector step.

    In addition to a common X, the observations come from the same transition;
    this prevents accidental comparisons between different sampled worlds.
    """
    if len(reward_events)!=len(progress_events):
        raise ValueError("Reward/progress logger cadence mismatch")
    out=[]
    for reward,progress in zip(reward_events,progress_events):
        if reward["global_step"]!=progress["global_step"]:
            raise ValueError("Reward and mission progress have different global_step")
        if any(k in reward for k in progress if k!="global_step"):
            raise ValueError("Overlapping reward/progress names are forbidden")
        out.append({**reward,**{k:v for k,v in progress.items() if k!="global_step"}})
    return out
