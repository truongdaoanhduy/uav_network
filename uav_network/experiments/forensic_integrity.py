"""Sampled diagnostics completeness manifest. Never silently infer missing = zero.

All of these paths are opt-in observers, not scientific losses or environment
mechanisms. The record audits whether expected observations were actually
stored after each sampled training step. Some failure records can be written
with empty metrics deliberately so users can distinguish missing vs zero.
"""
from __future__ import annotations

import json
from datetime import datetime,timezone
from pathlib import Path


class ForensicIntegrityJournal:
    def __init__(self,output,*,enabled=True):
        self.enabled=bool(enabled)
        self.path=Path(output)/'forensic-metric-integrity.jsonl'
        self.records=0
        if self.enabled:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.path.touch(exist_ok=True)

    def capture(self,*,step,action=None,mission=None,health=None,root_records=0,
                minibatch_records=0,reason='sample',errors=None):
        if not self.enabled:return {}
        action=action or {};mission=mission or {};health=health or {}
        status={'action':bool(action),'mission':bool(mission),'health':bool(health)}
        missing=[name for name,present in status.items() if not present]
        scores={'integrity/action_present':int(status['action']),
                'integrity/mission_present':int(status['mission']),
                'integrity/health_present':int(status['health']),
                'integrity/missing_action':int('action' in missing),
                'integrity/missing_mission':int('mission' in missing),
                'integrity/missing_health':int('health' in missing),
                'integrity/root_cause_records':int(root_records),
                'integrity/minibatch_records':int(minibatch_records),
                'integrity/has_error':int(bool(errors))}
        paths={
            'root_cause':'root-cause-timeline.jsonl',
            'minibatch':'forensic-minibatches.jsonl',
            'action_safety':'action-cbf-samples.jsonl',
            'mission':'mission-cause-snapshots.jsonl',
            'health':'training-health-snapshots.jsonl',
            'layerwise':'learner-layer-snapshots.jsonl',
            'world_trace':'forensic-transitions.jsonl',
            'step_reward':'training-step-reward.jsonl',
            'step_progress':'training-step-progress.jsonl',
            'step_optimizer':'training-step-optimizer.jsonl',
            'episode':'episodes.jsonl',
            'events':'training-events.jsonl',
        }
        sizes={label:(self.path.parent/rel).stat().st_size
               if (self.path.parent/rel).is_file() else None
               for label,rel in paths.items()}
        record={'schema':1,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                'global_step':int(step),'reason':str(reason),
                'record_scope':'at_sampling_cadence_missing_is_not_zero',
                'presence':status,'missing':missing,
                'sample_metric_counts':{'action':len(action),'mission':len(mission),'health':len(health)},
                'path_sizes_bytes':sizes,'errors':errors or [],
                'metrics':scores}
        with self.path.open('a') as out:
            out.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
        self.records+=1
        return scores
