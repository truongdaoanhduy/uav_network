"""Explicit GPU production-training entrypoint with deep HF forensic evidence.

By default this is DRY-RUN ONLY, without network or credentials access. Pass
--execute on a 2-GPU Kaggle host after preparing credentials to start the
20,000-episode TRAIN phase. GPU Validation and Final Test are a separate
one_session handoff (do not claim complete from this entrypoint alone).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from uav_network.configuration import load_config

CONFIG='configs/matd3-nooracle-lr4-map800-h256x256-orthogonal-compile-tx08-learned-hf-20261123.yaml'


FORENSIC_PROFILE={
    'UAV_DIAGNOSTIC_STORE':'huggingface_public',
    'UAV_DIAGNOSTIC_HF_REPO':'duy95/uav-search-diagnostics',
    'UAV_HF_VERIFY_PUBLIC_READBACK':'true',
    'UAV_FORENSIC_ENABLED':'true',
    'UAV_ROOT_CAUSE_AUDIT':'true',  # Full MATD3 early minibatch/gradient timeline on HF
    'UAV_FORENSIC_WORLD_IDS':'0,1',
    # Preserve full pre/post state and raw action/power for two representative
    # worlds every 16 vector steps; dense early updates are automatic.
    'UAV_FORENSIC_STRIDE':'16',
    'UAV_FORENSIC_FULL_BELIEF':'true',
    'UAV_FORENSIC_MAX_RAW_MIB':'512',
    'UAV_DEFERRED_EXPORT_HF':'true',
    'UAV_TRAINING_ONLY':'true',
    'UAV_FAST_TRAINING':'false',
    'UAV_WANDB_COMPACT_LOGS':'true',
    'UAV_WANDB_PLOTS_ONLY':'false',  # Retain RF/relay/energy scalars every 32nd episode,
    'UAV_WANDB_EPISODE_LOG_EVERY':'32',
    'UAV_TRAIN_EPISODE_RETURN_LOG_EVERY':'32',
    'WANDB_MODE':'online',
}


def forensic_environment(base=None):
    result=dict(os.environ if base is None else base)
    result.update(FORENSIC_PROFILE)
    return result


def training_command():
    return [sys.executable,'-m','uav_network.experiments.cli','train','--config',CONFIG]


def validate_production(config,cuda_count):
    if cuda_count<2:
        raise RuntimeError('This production training requires two CUDA GPUs; refusing CPU fallback')
    if (config.runtime.algorithm!='matd3' or config.network.tx_power_max_w!=.8
            or config.network.power_control!='learned'
            or config.reward.communication_reference_tx_power_w!=.4
            or config.world.map_size_m!=800 or config.matd3.hidden_sizes!=[256,256]
            or not config.runtime.compile_models or not config.tracking.enabled):
        raise ValueError('Loaded configuration differs from the pinned TX08 learned-power science')
    if Path(config.runtime.output_dir).exists():
        raise FileExistsError('Refusing to overwrite any previous training output: '+config.runtime.output_dir)
    return config


def main(argv=None):
    parser=argparse.ArgumentParser(description='MAP800 TX08 learned-power HF-forensic training; dry-run by default')
    parser.add_argument('--execute',action='store_true',help='Explicitly start 20k-episode GPU train')
    args=parser.parse_args(argv)
    c=load_config(CONFIG)
    report=dict(kind='tx08_deep_hf_training_preflight',dry_run=not args.execute,
        config=CONFIG,envs=c.runtime.num_envs,batch=c.matd3.batch_size,
        episodes=c.runtime.episodes,world_m=c.world.map_size_m,
        tx_min_w=c.network.tx_power_min_w,tx_max_w=c.network.tx_power_max_w,
        power_selection=c.network.power_control,
        communication_reference_tx_power_w=c.reward.communication_reference_tx_power_w,
        dataset='duy95/uav-search-diagnostics',
        forensic_worlds=FORENSIC_PROFILE['UAV_FORENSIC_WORLD_IDS'],
        forensic_stride_vector_steps=int(FORENSIC_PROFILE['UAV_FORENSIC_STRIDE']),
        command=training_command(),post_training_eval_test='separate_gpu_one_session_handoff')
    print('TX08_FORENSIC_TRAINING_PROFILE '+json.dumps(report,sort_keys=True),flush=True)
    if not args.execute:return report
    import torch
    validate_production(c,torch.cuda.device_count())
    subprocess.run(training_command(),check=True,env=forensic_environment())
    return report


if __name__=='__main__':main()
