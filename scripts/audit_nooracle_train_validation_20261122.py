"""CPU fullflow smoke, including actual MATD3 optimization and 100-seed validation.

This is NOT a replacement for the Kaggle E4096/T4x2 20k-episode experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from uav_network.configuration import load_config, Config, _apply
from uav_network.experiments.training import train
from uav_network.experiments.deferred_gpu_batched import evaluate_group


def one(index, seed, root):
    c=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=2;c.runtime.episodes=4;c.world.max_steps=20
    c.runtime.seed=seed;c.matd3.learning_starts=4;c.matd3.batch_size=16
    c.matd3.replay_capacity=1024;c.matd3.updates_per_step=4
    c.runtime.output_dir=str(root/f'run_{index:02d}_seed{seed}')
    c.runtime.checkpoint_interval=2;c.runtime.log_interval_steps=8
    c.tracking.enabled=False
    c.tracking.deferred_gpu_eval_collection_stride=4
    c.tracking.online_gpu_eval_enabled=False
    c.runtime.max_train_seconds=0
    assert c.network.enabled and c.sensor.report_cell_hypotheses
    r=train(c)
    assert r['episodes']==4 and r['global_step']==80
    assert r['updates']>0 and r['episode_budget_complete'] is True
    manifest_path=Path(r['deferred_gpu_evaluation']['manifest'])
    meta=json.loads(manifest_path.read_text())
    assert meta['count']>=2
    snapshot_root=manifest_path.parent
    snapshots=[]
    for row in (meta['snapshots'][0],meta['snapshots'][-1]):
        policy=torch.load(snapshot_root/row['file'],map_location='cpu',weights_only=True)
        snapshots.append(policy)
    restored=Config();_apply(restored,snapshots[0]['config']);restored.validate()
    evaluations=evaluate_group(restored,snapshots,device='cpu',seed=1000000044,episodes=100)
    assert len(evaluations)==2 and all(len(x)==100 for x in evaluations)
    assert evaluations[0][0]['seed']==evaluations[1][0]['seed']==1000000044
    assert evaluations[0][-1]['seed']==evaluations[1][-1]['seed']==1000000143
    assert all(rec['targets_delivered']<=rec['targets_found'] for rows in evaluations for rec in rows)
    digest=hashlib.sha256()
    for k,v in sorted(snapshots[-1]['actors'].items()):
        digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return {'round':index,'training_seed':seed,'episodes':r['episodes'],
            'optimizer_updates':r['updates'],
            'actor_updates':meta['snapshots'][-1]['policy_version'],
            'snapshots':meta['count'],'checkpoint_weights_sha256':digest.hexdigest(),
            'validation_episodes_per_policy':100,
            'validation_mean_search_percent':[
                round(sum(x['search_rate_percent'] for x in rows)/len(rows),4) for rows in evaluations],
            'validation_mean_delivery_percent':[
                round(sum(x['delivery_rate_percent'] for x in rows)/len(rows),4) for rows in evaluations]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/tmp/uav_v7_cpu_train_validation_smoke_20261122.json'));a=p.parse_args()
    root=a.output.with_suffix('');root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,seed in enumerate((44,44,45),1):
        row=one(index,seed,root);rows.append(row)
        print('CPU_ACTORBN_MATD3_TRAIN_VAL100_PASS',json.dumps(row),flush=True)
        a.output.write_text(json.dumps({'rows':rows,'all_passed':len(rows)==3},indent=2))
    assert rows[0]['checkpoint_weights_sha256']==rows[1]['checkpoint_weights_sha256'], 'fixed-seed training model is not reproducible'
    print('ACTOR_WEIGHTS_FIXED_SEED_BITWISE_REPRODUCIBLE',rows[0]['checkpoint_weights_sha256'],flush=True)

if __name__=='__main__':main()
