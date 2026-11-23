"""Exactly one completed-episode-outcome point at a collector step IF any end.

Unlike live progress, AUC/episode return/final energy are only defined for
finished worlds. Aggregate all actual completions at a given global_step; do
NOT invent values on steps without completed episodes. The full individual
records remain in episodes.jsonl in the HF-compressed diagnostic archive.
"""
from __future__ import annotations

import math
import numbers

from .telemetry import training_tradeoff_metrics

FIELDS=(
    ('episode_return', 'return'),
    ('discovery_rate_percent', 'training/discovery_rate_percent'),
    ('gcs_delivery_rate_percent', 'training/gcs_delivery_rate_percent'),
    ('discovery_auc', 'training/discovery_auc'),
    ('delivery_auc', 'training/delivery_auc'),
    ('map_coverage_percent', 'training/map_coverage_percent'),
    ('mission_success_rate_percent', 'training/mission_success_rate_percent'),
    ('energy_per_delivered_target_j', 'training/energy_per_delivered_target_j'),
    ('episode_length', 'episode_length'),
)


def completed_episode_step_metrics(episodes,global_step):
    """Real mean/SD across completed episodes of ONE vector collection."""
    if not episodes:return {}
    x=int(global_step)
    data={name:[] for name,_ in FIELDS}
    for record in episodes:
        if int(record['global_step'])!=x:
            raise ValueError('Episode global_step must match its completed vector step')
        derived=training_tradeoff_metrics(record)
        for alias,key in FIELDS:
            value=(record.get(key) if key in ('return','episode_length')
                   else derived.get(key))
            if isinstance(value,numbers.Real) and not isinstance(value,bool) and math.isfinite(float(value)):
                data[alias].append(float(value))
    out={'global_step':x,'train_episode_step/episode_count':len(episodes)}
    for key,values in data.items():
        if not values:continue
        mean=math.fsum(values)/len(values)
        dev=math.fsum((v-mean)*(v-mean) for v in values)/len(values)
        out[f'train_episode_step/{key}_mean']=mean
        out[f'train_episode_step/{key}_std']=math.sqrt(max(0.,dev))
        out[f'train_episode_step/valid_n_{key}']=len(values)
    return out
