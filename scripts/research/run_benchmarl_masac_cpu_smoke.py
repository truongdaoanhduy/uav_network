"""Reproduce the isolated *upstream* BenchMARL continuous MASAC VMAS smoke.

Requires an independent Python environment with torchrl~=0.10, tensordict~=0.10,
VMAS and Hydra, and the pinned upstream BenchMARL clone. Does NOT run UAV code,
Kaggle, W&B or HF. Outputs only short local CPU experiment logs.
Usage:
 PYTHONPATH=. outputs/reference_source_audit_20261108/venv/bin/python \
   scripts/research/run_benchmarl_masac_cpu_smoke.py
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",type=Path,
                        default=Path("outputs/reference_source_audit_20261108/BenchMARL"))
    parser.add_argument("--output",type=Path,
                        default=Path("outputs/reference_source_audit_20261108/benchmarl_reproduction"))
    parser.add_argument("--frames",type=int,default=192)
    parser.add_argument("--seed",type=int,default=44)
    parser.add_argument("--data-out",type=Path,
                        default=Path("docs/methods/evidence/reference_benchmarl_masac_smoke_2026-11-08.json"))
    args=parser.parse_args()
    if not args.upstream.is_dir() or args.frames<64 or args.frames%32:
        parser.error("upstream clone must exist; frames >=64 and multiple of 32")
    out=args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/"hydra").mkdir(exist_ok=True)
    os.environ["WANDB_MODE"]="disabled"
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    import torch
    torch.set_num_threads(1)
    import gymnasium as gym
    sys.modules.setdefault("gym",gym) # VMAS 1.5 requires legacy 'gym' imports.
    sys.modules.setdefault("gym.spaces",gym.spaces)
    sys.path.insert(0,str(args.upstream.resolve()))
    from benchmarl.run import hydra_experiment

    saved_argv=sys.argv[:]
    sys.argv=[
        "benchmarl.run","algorithm=masac","task=vmas/balance",
        f"seed={args.seed}","task.max_steps=20",
        f"experiment.max_n_frames={args.frames}",
        "experiment.off_policy_collected_frames_per_batch=32",
        "experiment.off_policy_n_envs_per_worker=2",
        "experiment.off_policy_n_optimizer_steps=2",
        "experiment.off_policy_train_batch_size=16",
        "experiment.off_policy_memory_size=256",
        "experiment.off_policy_init_random_frames=0",
        "experiment.evaluation=false","experiment.render=false",
        "experiment.create_json=false","experiment.loggers=[csv]",
        "experiment.checkpoint_at_end=false",
        "experiment.save_folder="+str(out),
        "hydra.run.dir="+str(out/"hydra"),
        "hydra.job.chdir=false",
    ]
    try:
        hydra_experiment()
    finally:
        sys.argv=saved_argv
    values={}
    for metric in [
        "counters_total_frames","train_agents_alpha",
        "train_agents_loss_actor","train_agents_loss_qvalue",
        "train_agents_loss_alpha","collection_reward_episode_reward_mean",
    ]:
        files=list(out.rglob(metric+".csv"))
        if not files:
            raise RuntimeError(f"Missing training metric CSV for {metric}")
        with files[0].open(newline="") as f:
            rows=[list(map(float,line)) for line in csv.reader(f)]
        if not rows:
            raise RuntimeError("Empty metric "+metric)
        values[metric]={"first":rows[0][-1],"last":rows[-1][-1],"rows":len(rows)}
    if values["counters_total_frames"]["last"] != args.frames:
        raise RuntimeError("Unexpected short run did not finish all requested frames")
    commit=subprocess.check_output(["git","-C",str(args.upstream),
                                    "rev-parse","HEAD"],text=True).strip()
    result={
        "source":"facebookresearch/BenchMARL",
        "upstream_commit":commit,
        "algorithm":"MASAC (upstream TorchRL continuous version)",
        "task":"VMAS balance (not UAV SAR)",
        "frames":args.frames,
        "seed":args.seed,
        "training_success":True,
        "runtime":"CPU only; independent Python environment",
        "scope":"Tiny executable training smoke only, not convergence/reward comparison",
        "metrics":values,
    }
    args.data_out.parent.mkdir(parents=True,exist_ok=True)
    args.data_out.write_text(json.dumps(result,indent=2)+"\n")
    print("SMOKE_EVIDENCE",json.dumps(result,indent=2))


if __name__=="__main__":
    main()
