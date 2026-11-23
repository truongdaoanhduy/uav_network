"""Save timestamped MATD3 per-block gradients locally for HF public .tar.gz.

Diagnostics are per sparse *actual optimizer update*, not each collection
step. Never interpret absence of Actor gradients during delayed TD3 updates
as vanishing gradients; the journal records explicit policy_updated flags.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path


class LayerwiseGradientJournal:
    def __init__(self, output, algorithm):
        self.algorithm=algorithm
        self.path=Path(output)/"learner-layer-snapshots.jsonl"
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.touch(exist_ok=True)
        self.records=0
        self.last_identity=None

    def capture(self,metrics,*,global_step,critic_updates=None,actor_updates=None,
                source="base",allow_duplicate=False):
        keys=("layerprobe/","gradprobe/","masac_probe/",
              "actor_motion_","actor_rf_","actor_first_encoder_",
              "critic_loss","actor_loss","critic_grad_norm","actor_grad_norm",
              "policy_updated","td_error_abs_mean","q_abs_max",
              "actor_mask_only_silent_fraction")
        # Full gradient dictionary can exceed 128 keys; generic json_safe()
        # intentionally truncates large maps. Here the only accepted values
        # are scalar learner measurements, so serialize ALL layer keys without
        # truncation while never including arbitrary strings/credentials.
        data={}
        for key,value in metrics.items():
            name=str(key)
            if not name.startswith(keys):
                continue
            if not isinstance(value,Real):
                continue
            scalar=float(value)
            data[name]=scalar if math.isfinite(scalar) else (
                "NaN" if math.isnan(scalar) else "Inf" if scalar>0 else "-Inf")
        if not data:
            return False
        update=(None if critic_updates is None else int(critic_updates))
        actor=(None if actor_updates is None else int(actor_updates))
        identity=(source,update,actor)
        if not allow_duplicate and self.last_identity==identity:
            return False
        self.last_identity=identity
        record={
            "schema":1,
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "algorithm":self.algorithm,
            "source":source,
            "global_step":int(global_step),
            "critic_updates":update,
            "actor_updates":actor,
            "policy_updated_in_metric":data.get("policy_updated"),
            "metric_scope":"one sampled actual optimizer update; tensor parameter "
                           "stats are exact, activation/output gradients sampled",
            "temporal_alignment":"same_optimizer_update" if source=="base" else
                                 "last_available_async_snapshot_may_be_stale",
            "metrics":data,
            "metric_count":len(data),
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
        self.records+=1
        return True

    def summary(self):
        return {"records":self.records,"path":str(self.path),
                "scope":"sparse actual optimizer updates, not every minibatch",
                "has_block_level_gradient_evidence":self.algorithm=="matd3"}
