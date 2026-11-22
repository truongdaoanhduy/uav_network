"""Actual source actor snapshot -> batched CPU validation -> Best -> held-out Test.

CPU smoke is deliberately bounded; production remains GPU100 x 81 and Test1000.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from uav_network.configuration import load_config
from uav_network.experiments.deferred_eval_snapshots import DeferredSnapshotBank
from uav_network.experiments.deferred_gpu_evaluation import (
    GPUActorOnlyPolicy, worker, publish_metrics, evaluate_snapshot,
)
from uav_network.experiments.gpu_eval_layout import plan_test


def test_real_checkpoint_bank_validation_best_final_test_and_safe_resume(tmp_path):
    torch.set_num_threads(1)
    c=load_config('configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml')
    c.runtime.device='cpu'
    c.runtime.num_envs=2
    c.world.max_steps=9
    c.tracking.periodic_evaluation_episodes=3
    c.tracking.periodic_evaluation_seed=1000000044
    c.runtime.eval_episodes=4
    assert c.tracking.deferred_gpu_eval_keep_all
    c.validate()

    bank=DeferredSnapshotBank(c,tmp_path/'train')
    for i in range(7):
        torch.manual_seed(22011+i)
        actor=GPUActorOnlyPolicy(c,'cpu')
        ok=bank.capture(actor.actors.state_dict(),i+1,
                        episodes=(i+1)*2,global_step=(i+1)*256,
                        collection_steps=(i+1)*100,next_seed=44+2*(i+1))
        assert ok
    manifest=bank.finalize()
    assert manifest['count']==7
    assert all(Path(bank.folder,row['file']).is_file() for row in manifest['snapshots'])
    output=tmp_path/'gpu_eval'
    count0=worker(bank.folder,output,0,2,'cpu',policies_per_batch=6,
                  world_pool_size=20,checkpoint_assignment='contiguous_batches')
    count1=worker(bank.folder,output,1,2,'cpu',policies_per_batch=6,
                  world_pool_size=20,checkpoint_assignment='contiguous_batches')
    assert (count0,count1)==(6,1)
    jsons=[output/f'eval_{i:04d}.json' for i in range(7)]
    assert all(x.is_file() for x in jsons)
    results=[json.loads(x.read_text()) for x in jsons]
    assert [x['source']['global_step'] for x in results]==[(i+1)*256 for i in range(7)]
    for i, r in enumerate(results):
        assert r['episodes']==3
        assert [row['seed'] for row in r['records']]==list(range(1000000044,1000000047))
        assert r['policy_provenance']['checkpoint_sha256']==manifest['snapshots'][i]['sha256']
        assert r['source']['policy_version']==i+1

    original=[x.read_bytes() for x in jsons]
    assert worker(bank.folder,output,0,2,'cpu',policies_per_batch=6,
                  world_pool_size=20,checkpoint_assignment='contiguous_batches')==6
    assert worker(bank.folder,output,1,2,'cpu',policies_per_batch=6,
                  world_pool_size=20,checkpoint_assignment='contiguous_batches')==1
    assert [x.read_bytes() for x in jsons]==original

    report=publish_metrics(bank.folder,results,external_wandb_owner=True)
    best=Path(report['best_checkpoint'])
    assert report['validation_completed']==7 and best.is_file()
    assert (best.parent/'best.json').is_file()
    assert (best.parent/'validation_results.csv').is_file()
    assert (best.parent/'validation_results.json').is_file()
    best_meta=json.loads((best.parent/'best.json').read_text())
    score=(best_meta['score'],best_meta['global_step'])
    expected=max(((r['success_rate'],r['mean_delivery_rate_percent'],
                  r['mean_search_rate_percent'],r['mean_return']),
                 r['checkpoint_global_step']) for r in results)
    assert tuple(score[0])==expected[0]
    assert score[1]==expected[1]

    assert plan_test(1000,2,500)==(((0,500),),((500,1000),))
    test=evaluate_snapshot(best,device='cpu',seed=2000000044,episodes=4)
    assert len(test['records'])==4
    assert [item['seed'] for item in test['records']]==list(range(2000000044,2000000048))
    assert test['protocol']['training_seed_overlap'] is False
    assert test['protocol']['validation_seed_overlap'] is False
    assert best_meta['validation_seeds']==list(range(1000000044,1000000047))
