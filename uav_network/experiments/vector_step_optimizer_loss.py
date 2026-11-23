"""Per-vector-step optimizer loss diagnostics from *real completed updates*.

No force-collect_metrics=True per optimizer update: this passive instrumentation
saves detached scalar objectives already computed by the algorithm. The GPU
keeps 1-step sum, sum-squared and count, with host transfer batched across
four vector steps. Delayed MATD3 actor updates are counted separately; a
missing actor update is NOT imputed as loss=0. A raw HF-backed row per vector
step can be regrouped into arbitrary nonoverlapping windows later.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

METRICS=(
    'critic_loss', 'actor_loss',
    'critic_grad_norm', 'actor_grad_norm',
    'td_error_abs_mean', 'alpha_loss',
)


class PerVectorStepOptimizerLossJournal:
    FILENAME='training-step-optimizer.jsonl'

    def __init__(self,num_envs,output,*,host_sync_every=4,plot_windows=(1,2,4,8)):
        self.num_envs=int(num_envs)
        self.host_sync_every=int(host_sync_every)
        self.plot_windows=tuple(sorted(set(int(w) for w in plot_windows)))
        if (self.num_envs<1 or self.host_sync_every<1 or not self.plot_windows
            or self.plot_windows[0]!=1):
            raise ValueError('Invalid dense optimizer journal configuration')
        self.path=Path(output)/self.FILENAME
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._file=self.path.open('a',encoding='utf-8',buffering=65536)
        self._last_step=None
        self._current_step=None
        self._current={}
        self._pending=[]
        self._windows={size:[] for size in self.plot_windows if size>1}
        self._closed=False

    def begin(self,global_step):
        if self._closed or self._current_step is not None:
            raise RuntimeError('Cannot begin another optimizer vector step')
        step=int(global_step)
        if step<self.num_envs or step%self.num_envs:
            raise ValueError('Optimizer global_step is not aligned to vector step')
        if self._last_step is not None and step!=self._last_step+self.num_envs:
            raise ValueError('Missing or duplicate optimizer vector step')
        self._current_step=step
        self._current={}

    def record(self,name,value):
        if self._current_step is None:
            raise RuntimeError('No active optimizer vector step')
        if name not in METRICS:
            raise ValueError('Unknown optimizer diagnostic metric: '+str(name))
        if hasattr(value,'detach'):
            if value.numel()!=1:
                raise ValueError('Only scalar optimizer metrics are supported')
            val=value.detach().reshape(())
        else:
            val=float(value)
            if not math.isfinite(val):
                raise ValueError('Nonfinite optimizer metric '+name)
        self._current.setdefault(name,[]).append(val)

    def end(self,global_step):
        step=int(global_step)
        if self._current_step!=step:
            raise ValueError('Optimizer end global_step differs from begin global_step')
        row={}
        for name,values in self._current.items():
            first=values[0]
            if hasattr(first,'detach'):
                import torch
                with torch.no_grad():
                    vector=torch.stack([v.detach().to(dtype=torch.float64) for v in values])
                    row[name]=(vector.sum(),vector.square().sum(),len(values))
            else:
                row[name]=(math.fsum(values),math.fsum(x*x for x in values),len(values))
        self._pending.append((step,row))
        self._current_step=None
        self._current={}
        self._last_step=step
        return self.flush() if len(self._pending)>=self.host_sync_every else []

    def flush(self):
        if self._current_step is not None:
            raise RuntimeError('Incomplete optimizer vector step cannot be flushed')
        if not self._pending:return []
        pending=self._pending
        self._pending=[]
        # One device->host transfer for ALL present CUDA scalar statistics.
        positions=[]
        scalars=[]
        for i,(_,metrics) in enumerate(pending):
            for name,values in metrics.items():
                for j in (0,1):
                    v=values[j]
                    if hasattr(v,'detach'):
                        positions.append((i,name,j))
                        scalars.append(v)
        floats={}
        if scalars:
            import torch
            with torch.no_grad():
                host=torch.stack(scalars).detach().cpu().tolist()
            floats=dict(zip(positions,host))
        events=[]
        for i,(step,metrics) in enumerate(pending):
            raw={'schema_version':1,'global_step':step,
                 'vector_step':step//self.num_envs,'num_envs':self.num_envs,
                 'semantics':'completed_optimizer_updates_at_collection_global_step',
                 'metrics':{}}
            event={'global_step':step}
            this_step={}
            for name in METRICS:
                value=metrics.get(name)
                if value is None:
                    total=sq=0.0;count=0
                else:
                    total=float(floats.get((i,name,0),value[0])
                                if not hasattr(value[0],'detach') else floats[(i,name,0)])
                    sq=float(floats.get((i,name,1),value[1])
                             if not hasattr(value[1],'detach') else floats[(i,name,1)])
                    count=value[2]
                    if not (math.isfinite(total) and math.isfinite(sq)):
                        raise ValueError('Nonfinite optimizer sample: '+name)
                raw['metrics'][name]={'count':count,'sum':total,'sum_sq':sq}
                this_step[name]=(total,sq,count)
                if count:
                    mean=total/count
                    event[f'train_loss/{name}']=mean
                    event[f'train_loss/{name}_std']=math.sqrt(max(0.,sq/count-mean*mean))
                    event[f'train_loss/{name}_updates']=count
            # Even pre-warmup has an explicitly measured update COUNT zero.
            event['train_loss/critic_updates']=this_step['critic_loss'][2]
            event['train_loss/actor_updates']=this_step['actor_loss'][2]
            self._file.write(json.dumps(raw,separators=(',',':'),allow_nan=False)+'\n')
            for width,window in self._windows.items():
                window.append(this_step)
                if len(window)!=width:continue
                for name in METRICS:
                    count=sum(item[name][2] for item in window)
                    if count:
                        total=math.fsum(item[name][0] for item in window)
                        sq=math.fsum(item[name][1] for item in window)
                        mean=total/count
                        event[f'train_loss/{name}_{width}']=mean
                        event[f'train_loss/{name}_std_{width}']=math.sqrt(
                            max(0.,sq/count-mean*mean))
                        event[f'train_loss/{name}_updates_{width}']=count
                window.clear()
            events.append(event)
        self._file.flush()
        return events

    def close(self):
        if self._closed:return []
        if self._current_step is not None:
            # Do not manufacture a completed optimizer batch after failure.
            self._current_step=None
            self._current={}
        events=self.flush()
        self._file.close()
        self._closed=True
        return events


class UnifiedTrainingStepJoiner:
    """Link reward+mission and optimizer records by their true global_step.

    Handles both non-overlapping and asynchronous collector/learner schedules:
    the optimizer may finish the preceding vector step one vector later.
    """
    def __init__(self):
        self._training={}
        self._loss={}
        self._last_published=None

    def _complete(self):
        out=[]
        for step in sorted(set(self._training)&set(self._loss)):
            if self._last_published is not None and step<=self._last_published:
                raise ValueError('Duplicate/out-of-order joined training global_step')
            reward=self._training.pop(step)
            loss=self._loss.pop(step)
            if set(reward)&set(loss)!={'global_step'}:
                raise ValueError('Overlapping or missing global_step in matched telemetry')
            out.append({**reward,**{k:v for k,v in loss.items() if k!='global_step'}})
            self._last_published=step
        return out

    def add_training(self,rows):
        for row in rows:
            step=int(row['global_step'])
            if step in self._training:
                raise ValueError('Duplicate reward/mission vector step')
            self._training[step]=dict(row)
        return self._complete()

    def add_loss(self,rows):
        for row in rows:
            step=int(row['global_step'])
            if step in self._loss:
                raise ValueError('Duplicate loss vector step')
            self._loss[step]=dict(row)
        return self._complete()

    def close(self):
        # A training crash may leave the last step with reward but no optimizer
        # update. Publish only the actual observations, never invent a zero loss.
        out=self._complete()
        for step in sorted(self._training):
            if self._last_published is not None and step<=self._last_published:
                raise ValueError('Out-of-order trailing vector telemetry')
            out.append(self._training[step])
            self._last_published=step
        self._training.clear()
        self._loss.clear()
        return out


def pair_step_observables(rewards,progress,losses):
    """Strict finite equal-step merge, for offline verification and tests."""
    if len(rewards)!=len(progress) or len(rewards)!=len(losses):
        raise ValueError('Vector step count differs between sources')
    result=[]
    for reward,mission,loss in zip(rewards,progress,losses):
        if not (reward['global_step']==mission['global_step']==loss['global_step']):
            raise ValueError('global_step differs across training observations')
        if (set(reward)&set(mission))!={'global_step'}:
            raise ValueError('Reward/mission metric key collision')
        if ((set(reward)|set(mission))&set(loss))!={'global_step'}:
            raise ValueError('Reward/loss metric key collision')
        result.append({**reward,**mission,**loss})
    return result
