"""Sample-faithful scalar aggregation for parallel RL collectors.

A vector step contributes one team reward per environment/world, regardless of
UAV count.  Never confuse these transition rewards with completed episode return.
The torch path retains scalar sums on-device until a W&B logging boundary.
"""
from __future__ import annotations

import math
import numpy as np


class DenseStepRewardWindow:
    """Population mean/std of every real transition in N synchronous worlds."""

    def __init__(self, num_envs: int, every_collections: int = 4):
        if num_envs < 1 or every_collections < 1:
            raise ValueError("num_envs and every_collections must both be positive")
        self.num_envs = int(num_envs)
        self.every_collections = int(every_collections)
        self._reset()

    def _reset(self):
        self._sum = None
        self._sum_sq = None
        self._count = 0
        self._collections = 0
        self._end_step = None

    def observe(self, rewards, global_step: int):
        if hasattr(rewards, "detach") and hasattr(rewards, "numel"):
            # torch tensors: one scalar CUDA reduction per collection, but no
            # host/device synchronization until an actual logging boundary.
            import torch
            with torch.no_grad():
                values = rewards.detach().reshape(-1).to(dtype=torch.float64)
                if values.numel() != self.num_envs:
                    raise ValueError("Expected one team reward per environment")
                batch_sum = values.sum()
                batch_sum_sq = values.square().sum()
        else:
            values = np.asarray(rewards, dtype=np.float64).reshape(-1)
            if values.size != self.num_envs:
                raise ValueError("Expected one team reward per environment")
            batch_sum = float(np.sum(values, dtype=np.float64))
            batch_sum_sq = float(np.dot(values, values))
        if self._end_step is not None and int(global_step) != self._end_step + self.num_envs:
            raise ValueError("Reward observations are not consecutive collector steps")
        self._sum = batch_sum if self._sum is None else self._sum + batch_sum
        self._sum_sq = batch_sum_sq if self._sum_sq is None else self._sum_sq + batch_sum_sq
        self._count += self.num_envs
        self._collections += 1
        self._end_step = int(global_step)
        if self._collections >= self.every_collections:
            return self.flush()
        return None

    def flush(self):
        if self._count == 0:
            return None
        mean = float(self._sum) / self._count
        variance = max(0., float(self._sum_sq) / self._count - mean * mean)
        result = {
            "global_step": self._end_step,
            "train/step_reward_mean": mean,
            "train/step_reward_std": math.sqrt(variance),
            "train/reward_window_transitions": self._count,
            "train/reward_window_collections": self._collections,
            "train/reward_window_start_step": self._end_step - self._count,
        }
        self._reset()
        return result


class EpisodeReturnWindow:
    """Mean/std across genuine completed episodes; X is completion index."""

    def __init__(self, every_episodes: int = 64):
        if every_episodes < 1:
            raise ValueError("every_episodes must be positive")
        self.every_episodes = int(every_episodes)
        self._returns = []
        self._last_episode = None
        self._end_step = None

    def observe(self, *, episode: int, global_step: int, episode_return: float):
        if self._last_episode is not None and episode != self._last_episode + 1:
            raise ValueError("Completed episode records must be sequential")
        self._last_episode = int(episode)
        self._end_step = int(global_step)
        self._returns.append(float(episode_return))
        if len(self._returns) >= self.every_episodes:
            return self.flush()
        return None

    def flush(self):
        if not self._returns:
            return None
        x = np.asarray(self._returns, dtype=np.float64)
        result = {
            "global_step": self._end_step,
            "episode/index": self._last_episode,
            "train/episode_return_mean": float(x.mean()),
            "train/episode_return_std": float(x.std(ddof=0)),
            "train/episode_window_count": len(self._returns),
        }
        self._returns.clear()
        return result


class PerVectorStepRewardJournal:
    """Preserve one sufficient-statistics record per real vector collection.

    Four CUDA scalar pairs can be transferred in one host synchronization.
    Their raw sum/count/sum-of-squares are sufficient to reconstruct exact
    nonoverlapping means/SDs for ANY later integer-sized vector-step window.
    This records no private observations, actions, targets or full GPU tensors.
    """

    FILENAME = "training-step-reward.jsonl"

    def __init__(self, num_envs, output, *, host_sync_every=4,
                 plot_windows=(1, 2, 4, 8)):
        import os
        from pathlib import Path
        self.num_envs = int(num_envs)
        self.host_sync_every = int(host_sync_every)
        self.plot_windows = tuple(sorted(set(int(x) for x in plot_windows)))
        if (self.num_envs < 1 or self.host_sync_every < 1
                or not self.plot_windows or self.plot_windows[0] != 1
                or any(x < 1 for x in self.plot_windows)):
            raise ValueError("Invalid parallel reward journal configuration")
        self.path = Path(output) / self.FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # New training run: create once; continuing one file after restart
        # appends only new real environment steps, never duplicates.
        self._file = self.path.open("a", encoding="utf-8", buffering=65536)
        self._pending = []
        self._windows = {n:[] for n in self.plot_windows if n > 1}
        self._last_step = None
        self._last_synced = 0
        self._closed = False

    def observe(self, rewards, global_step):
        if self._closed:
            raise RuntimeError("Reward journal already closed")
        x = int(global_step)
        if x < self.num_envs or x % self.num_envs:
            raise ValueError("global_step must be a cumulative full-vector count")
        if self._last_step is not None and x != self._last_step + self.num_envs:
            raise ValueError("Missing or duplicate vector step in reward journal")
        self._last_step = x
        if hasattr(rewards, "detach") and hasattr(rewards, "numel"):
            import torch
            with torch.no_grad():
                rewards = rewards.detach().reshape(-1).to(dtype=torch.float64)
                if rewards.numel() != self.num_envs:
                    raise ValueError("Need exactly one team reward per environment")
                # CUDA only: do not call .item(), .cpu(), or synchronize here.
                total = rewards.sum()
                square = rewards.square().sum()
        else:
            arr = np.asarray(rewards,dtype=np.float64).reshape(-1)
            if arr.size != self.num_envs:
                raise ValueError("Need exactly one team reward per environment")
            total = float(arr.sum(dtype=np.float64))
            square = float(np.dot(arr,arr))
        self._pending.append((x,total,square))
        return self.flush() if len(self._pending) >= self.host_sync_every else []

    def flush(self):
        if not self._pending:
            return []
        pending = self._pending
        self._pending = []
        if hasattr(pending[0][1], "detach"):
            import torch
            # One device->host transfer for ALL buffered CUDA reductions.
            # CPU receives only two numbers per vector step, not a reward array.
            with torch.no_grad():
                vals = torch.stack([torch.stack((sm, sq))
                                    for _,sm,sq in pending]).detach().cpu().numpy()
            rows = [(step,float(vals[i,0]),float(vals[i,1]))
                    for i,(step,_,_) in enumerate(pending)]
        else:
            rows = [(x,float(total),float(square)) for x,total,square in pending]
        events = []
        for step,total,square in rows:
            count = self.num_envs
            mu = total/count
            sd = math.sqrt(max(0., square/count-mu*mu))
            raw = {
                "schema_version":1, "global_step":step,
                "vector_step":step//count,
                "num_envs":count, "count":count,
                "reward_sum":total, "reward_sum_sq":square,
                "reward_mean":mu, "reward_std":sd,
            }
            self._file.write(__import__("json").dumps(raw,allow_nan=False,separators=(",",":"))+"\n")
            event={"global_step":step,
                   "train/step_reward_mean":mu,
                   "train/step_reward_std":sd,
                   "train/reward_window_transitions":count,
                   "train/reward_window_collections":1}
            for size, window in self._windows.items():
                window.append((total,square,count))
                if len(window) == size:
                    summed = math.fsum(x[0] for x in window)
                    squared = math.fsum(x[1] for x in window)
                    samples = sum(x[2] for x in window)
                    mean = summed/samples
                    event[f"train/step_reward_mean_{size}"] = mean
                    event[f"train/step_reward_std_{size}"] = math.sqrt(
                        max(0.,squared/samples-mean*mean))
                    event[f"train/reward_window_transitions_{size}"] = samples
                    window.clear()
            events.append(event)
        # The HF archive reader only takes complete newline-terminated rows.
        self._file.flush()
        self._last_synced = rows[-1][0]
        return events

    def close(self):
        if not self._closed:
            leftover = self.flush()
            self._file.close()
            self._closed = True
            return leftover
        return []


class CompletedMissionWindow:
    """Cohort mean/population SD of *completed* episode mission outcomes.

    Missing ratios are not silently replaced with zero; per-series valid_n
    explicitly records how many completed episodes have a denominator.
    For energy per delivery, two distinct estimands are preserved:
    - macro mean of per-episode ratios among valid episodes,
    - pooled sum of energy across ALL worlds / sum of delivered targets.
    The caller keeps every original episode in HF episodes.jsonl.
    """

    FIELD_NAMES = {
        'training/discovery_rate_percent': 'discovery_rate_percent',
        'training/gcs_delivery_rate_percent': 'gcs_delivery_rate_percent',
        'training/discovery_auc': 'discovery_auc',
        'training/delivery_auc': 'delivery_auc',
        'training/map_coverage_percent': 'map_coverage_percent',
        'training/energy_per_delivered_target_j': 'energy_per_delivered_target_j',
        'training/total_uav_energy_j': 'total_uav_energy_j',
        'training/gcs_delivery_time_capped_mean_s': 'gcs_delivery_time_capped_mean_s',
        'training/mission_success_rate_percent': 'mission_success_rate_percent',
        'episode/length': 'episode_length',
    }

    def __init__(self, every_episodes=64):
        if every_episodes < 1:
            raise ValueError('Cohort episode window must be positive')
        self.every_episodes = int(every_episodes)
        self._samples = []
        self._last_episode = None
        self._last_global_step = None

    def observe(self, *, episode, global_step, metrics, reward_components=None,
                targets_delivered=None):
        import numbers
        ep = int(episode)
        if self._last_episode is not None and ep != self._last_episode + 1:
            raise ValueError('Completed episode cohort records must be sequential')
        self._last_episode = ep
        self._last_global_step = int(global_step)
        # Store small numerical results, not observations or arbitrary payloads.
        row = {}
        for name,alias in self.FIELD_NAMES.items():
            v = metrics.get(name)
            if isinstance(v,numbers.Real) and math.isfinite(float(v)):
                row[alias] = float(v)
        energy = row.get('total_uav_energy_j')
        delivered = targets_delivered
        if isinstance(delivered, numbers.Integral) and delivered >= 0 and energy is not None:
            row['_pool_energy'] = energy
            row['_pool_delivered'] = int(delivered)
        if isinstance(reward_components, dict):
            for name,value in reward_components.items():
                # This key is a metric label; don't allow arbitrary paths.
                if not isinstance(name,str) or not name.replace('_','').isalnum():
                    continue
                if isinstance(value,numbers.Real) and math.isfinite(float(value)):
                    row['reward_component_'+name] = float(value)
        self._samples.append(row)
        return self.flush() if len(self._samples) >= self.every_episodes else None

    def flush(self):
        if not self._samples:
            return None
        rows=self._samples
        self._samples=[]
        outputs={
            'episode/index':self._last_episode,
            'global_step':self._last_global_step,
            'train_cohort/episode_count':len(rows),
            'train_cohort/requested_episode_count':self.every_episodes,
        }
        fields={field for row in rows for field in row if not field.startswith('_')}
        for field in sorted(fields):
            values=np.asarray([r[field] for r in rows if field in r],dtype=np.float64)
            if not values.size:
                continue
            outputs[f'train_cohort/{field}_mean']=float(values.mean())
            outputs[f'train_cohort/{field}_std']=float(values.std(ddof=0))
            outputs[f'train_cohort/valid_n_{field}']=int(values.size)
        valid=[r for r in rows if '_pool_energy' in r]
        denom=sum(r['_pool_delivered'] for r in valid)
        outputs['train_cohort/pooled_energy_episode_count']=len(valid)
        outputs['train_cohort/pooled_targets_delivered']=denom
        if denom:
            outputs['train_cohort/pooled_energy_per_delivered_target_j']=(
                math.fsum(r['_pool_energy'] for r in valid)/denom)
        return outputs
