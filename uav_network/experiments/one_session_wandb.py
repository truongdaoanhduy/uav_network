"""Single W&B run owner across post-training GPU validation and final test.

The learner closes its W&B SDK instance first. The one-session supervisor
resumes that exact run ID, holds it open throughout validation + test, streams
the *actual* per-checkpoint evaluation results as atomic JSON files arrive,
and closes it only after the whole flow is done (or genuinely fails).
GPU worker processes never open the same W&B run in this mode.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .gpu_mission_metrics import (
    gpu_eval_wandb_payload,
    periodic_gpu_eval_wandb_payload,
)


class EvalConvergenceStreamer:
    """Append one genuine evaluation record per checkpoint on its training X.

    Progress is recorded only after each GPU worker atomically renames its
    evaluation JSON. Rows are logged in *checkpoint order* even when GPU 1
    finishes index 1 before GPU 0 finishes index 0.
    """
    def __init__(self, run, snapshots, output, *, validation_seed_first,
                 validation_episodes, existing_steps=()):
        self.run=run
        self.snapshots=list(snapshots)
        self.output=Path(output)
        self.seed=int(validation_seed_first)
        self.episodes=int(validation_episodes)
        if self.episodes<1:
            raise ValueError("Expected positive evaluation episodes")
        self.index=0
        self.existing_steps={int(x) for x in existing_steps}
        self.seen_steps=set()
        self.run.define_metric("gpu_eval_periodic/checkpoint_global_step")
        self.run.define_metric(
            "gpu_eval_periodic/*",
            step_metric="gpu_eval_periodic/checkpoint_global_step",
            step_sync=False)
        self.run.define_metric("gpu_eval/checkpoint_global_step")
        self.run.define_metric("gpu_eval/*",
            step_metric="gpu_eval/checkpoint_global_step",
            step_sync=False)

    @property
    def finished(self):
        return self.index == len(self.snapshots)

    def poll(self):
        advanced=0
        while self.index < len(self.snapshots):
            path=self.output/f"eval_{self.index:04d}.json"
            if not path.is_file():
                break
            item=json.loads(path.read_text())
            expected=self.snapshots[self.index]
            source=item.get("source",{})
            provenance=item.get("policy_provenance",{})
            if provenance.get("checkpoint_sha256")!=expected["sha256"]:
                raise ValueError("GPU checkpoint hash mismatch: "+str(path))
            if (int(source.get("global_step",-1))!=int(expected["global_step"]) or
                    int(source.get("episodes",-1))!=int(expected["episodes"])):
                raise ValueError("GPU result checkpoint metadata mismatch: "+str(path))
            records=item.get("records",[])
            seeds=[x.get("seed") for x in records]
            bank=list(range(self.seed,self.seed+self.episodes))
            if len(records)!=self.episodes or seeds != bank or item.get("episodes")!=self.episodes:
                raise ValueError("GPU evaluation result missing/duplicate fixed seed bank: "+str(path))
            # Return must match actual records, not an invented smoothed score.
            real_returns=[float(x["return"]) for x in records]
            if not all(math.isfinite(x) for x in real_returns):
                raise ValueError("Nonfinite evaluation return: "+str(path))
            observed=sum(real_returns)/len(real_returns)
            if not math.isclose(float(item["mean_return"]), observed, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("GPU Eval mean return differs from raw 100 episodes: "+str(path))
            step=int(source["global_step"])
            if step in self.seen_steps:
                raise ValueError("Duplicate snapshot global_step")
            self.seen_steps.add(step)
            if step not in self.existing_steps:
                old=gpu_eval_wandb_payload(item)
                row={key:value for key,value in old.items() if key!="global_step"}
                reasons=expected.get("reasons",["periodic"])
                if "periodic" in reasons or "final" in reasons:
                    periodic=periodic_gpu_eval_wandb_payload(old)
                    periodic.pop("global_step",None)
                    row.update(periodic)
                row["gpu_eval/batch_policies"]=item.get("gpu_eval_batch_policies",1)
                row["gpu_eval/batch_time_s"]=item.get("gpu_eval_batch_time_s",0.0)
                row["gpu_eval/gpu_peak_allocated_mib"]=item.get("gpu_eval_gpu_peak_allocated_mib",0.0)
                if not all(math.isfinite(float(v)) for v in row.values() if isinstance(v,(int,float))):
                    raise ValueError("Nonfinite evaluation metric")
                # W&B implicit history step grows monotonically. Checkpoint
                # global step is an independent user-defined x-axis, not a
                # rewound training history step.
                self.run.log(row)
                self.existing_steps.add(step)
            self.index+=1
            advanced+=1
        if advanced:
            self.run.summary.update({
                "fullflow/validation_logged_checkpoints":len(self.seen_steps),
                "fullflow/validation_required_checkpoints":len(self.snapshots),
                "fullflow/validation_last_checkpoint_global_step":int(self.snapshots[self.index-1]["global_step"]),
            })
        return advanced


def previously_logged_steps(api, path):
    """Best-effort retry safety for incomplete previous Kaggle sessions."""
    try:
        remote=api.run(path)
        axis="gpu_eval_periodic/checkpoint_global_step"
        return {
            int(row[axis]) for row in remote.scan_history(keys=[axis],page_size=500)
            if isinstance(row.get(axis),(int,float))
        }
    except Exception:
        return set()


def log_final_test(run, final, *, best_global_step):
    """Final Test is ONE held-out result, never a convergence checkpoint."""
    if int(final["episodes"])<=0:
        raise ValueError("Final Test has no evaluated episodes")
    summary={
        "gpu_final/episodes":int(final["episodes"]),
        "gpu_final/mean_return":float(final["mean_return"]),
        "gpu_final/validation_best_global_step":int(best_global_step),
    }
    for key,value in final["mission_metrics"].items():
        if value is not None:
            if not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError("Invalid Final Test mission metric "+key)
            summary["gpu_final/"+key]=float(value)
    run.summary.update(summary)
    run.define_metric("gpu_final/validation_best_global_step")
    run.define_metric("gpu_final/*",
        step_metric="gpu_final/validation_best_global_step",step_sync=False)
    run.log(summary)
    return summary
