"""Evaluation should use both T4 GPUs even when <81 policy checkpoints exist."""
from collections import Counter
import pytest
from uav_network.experiments.gpu_eval_layout import plan_validation,plan_test
from uav_network.experiments.one_session import select_training_snapshots_for_validation


@pytest.mark.parametrize('checkpoints', [1,2,10,72,80,81,82,161,162,163,200,324,325,600])
def test_balanced_contiguous_8192_world_batches_use_both_gpus_if_possible(checkpoints):
    batches=[plan_validation(checkpoints,100,8192,gpu_index=i,gpu_count=2,
                             assignment='contiguous_adaptive') for i in (0,1)]
    ids=[[cp for batch in group for cp in batch.checkpoint_indices] for group in batches]
    assert sorted(ids[0]+ids[1])==list(range(checkpoints))
    assert len(ids[0])+len(ids[1])==checkpoints
    assert all(batch.worlds<=8100 for group in batches for batch in group)
    if checkpoints>=2:
        assert ids[0] and ids[1]
    if checkpoints<=162 and checkpoints>=2:
        assert abs(len(ids[0])-len(ids[1]))<=1
        assert ids[0]==list(range(0,len(ids[0])))
        assert ids[1]==list(range(len(ids[0]),checkpoints))
    if checkpoints>=163:
        assert batches[0][0].checkpoint_indices==tuple(range(81))
        assert batches[1][0].checkpoint_indices==tuple(range(81,162))


def test_test1000_is_500_independent_episodes_per_gpu():
    assert plan_test(1000,2,8192)==(((0,500),),((500,1000),))


def test_explicit_all_snapshots_bypasses_shortlist_completely():
    source_manifest={'count':5,'snapshots':[{'global_step':i*409600} for i in range(5)]}
    spec={'validate_all_snapshots':True,'gpu_eval_candidate_limit':2,
          'validation_step_targets':[0]}
    m,folder,mode=select_training_snapshots_for_validation(
        spec,source_manifest,'/tmp/deferred-gpu-eval','/tmp/train')
    assert m is source_manifest
    assert str(folder)=='/tmp/deferred-gpu-eval'
    assert mode=='all_collected_actor_snapshots'
    assert len(m['snapshots'])==5
