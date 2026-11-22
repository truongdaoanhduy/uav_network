"""Regression for exact full-pool contiguous Eval on two GPUs with filler worlds."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy

import pytest

from uav_network.experiments.gpu_eval_layout import plan_validation


def test_8192_contiguous_batches_exact_checkpoint_range():
    g0 = plan_validation(200, 100, 8192, gpu_index=0, gpu_count=2,
                         assignment='contiguous_batches')
    g1 = plan_validation(200, 100, 8192, gpu_index=1, gpu_count=2,
                         assignment='contiguous_batches')
    assert [b.checkpoint_indices for b in g0] == [tuple(range(0,81)), tuple(range(162,200))]
    assert [b.checkpoint_indices for b in g1] == [tuple(range(81,162))]
    assert g0[0].checkpoint_ranges[0] == (0,0,100)
    assert g0[0].checkpoint_ranges[-1] == (80,8000,8100)
    assert max(b.worlds for b in g0+g1)==8100
    assert Counter(i for batch in g0+g1 for i in batch.checkpoint_indices) == Counter(range(200))


def test_contiguous_round_robin_batch_coverage_many_partial_sizes():
    for count in (0,1,2,80,81,82,161,162,163,324,325,600):
        for gpu_count in (1,2):
            groups = [plan_validation(count,100,8192,gpu_index=g,gpu_count=gpu_count,
                                      assignment='contiguous_batches') for g in range(gpu_count)]
            rows = [i for batches in groups for batch in batches for i in batch.checkpoint_indices]
            assert sorted(rows)==list(range(count))
            assert len(rows)==len(set(rows))
            assert all(b.worlds <=8100 for batches in groups for b in batches)


def test_stable_baseline_strided_assignment_unchanged():
    assert [b.checkpoint_indices for b in plan_validation(6,100,200,gpu_index=0,gpu_count=2)] == [(0,2),(4,)]
    with pytest.raises(ValueError):
        plan_validation(1,100,8192,assignment='unsupported')


def test_extra_pool_worlds_do_not_change_true_episode_records():
    from uav_network.configuration import load_config
    from uav_network.environment.gpu.vector import WarpVectorEnv
    from uav_network.experiments.deferred_gpu_evaluation import GPUActorOnlyPolicy
    from uav_network.experiments.deferred_gpu_batched import evaluate_group

    c=load_config('configs/matd3-nooracle-v9-ct095-h256x256-test1000-20261123.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=4
    c.world.max_steps=6
    actor=GPUActorOnlyPolicy(c,'cpu')
    payload={'config':c.to_dict(),'purpose':'evaluation_only',
             'actors':deepcopy(actor.actors.state_dict())}
    baseline=evaluate_group(c,[payload,payload],device='cpu',seed=1000000044,episodes=2)
    large_config=deepcopy(c);large_config.runtime.num_envs=6
    vector=WarpVectorEnv(large_config,device='cpu')
    try:
        padded=evaluate_group(c,[payload,payload],device='cpu',seed=1000000044,
                              episodes=2,vector=vector)
    finally:
        vector.close()
    assert [[r['seed'] for r in group] for group in padded] == [
        [1000000044,1000000045],[1000000044,1000000045]]
    assert padded==baseline


def test_candidate_notebook_passes_8192_pool_and_contiguous_assignment():
    from scripts.build_matd3_v9_eval8192_candidate_ui_20261123 import build_notebook
    nb=build_notebook()
    code=nb.cells[2].source
    preflight=nb.cells[1].source
    assert 'CHECKPOINTS_PER_GPU_BATCH = 81' in preflight
    assert 'MAX_WARP_EVAL_WORLDS_PER_GPU = 8192' in preflight
    assert 'MAX_WARP_TEST_WORLDS_PER_GPU = 500' in preflight
    assert '"--eval-world-pool-size", str(MAX_WARP_EVAL_WORLDS_PER_GPU)' in code
    assert '"--checkpoint-assignment", "contiguous_batches"' in code
    assert 'assert CHECKPOINTS_PER_GPU_BATCH == 81' in code
    assert 'assert CHECKPOINTS_PER_GPU_BATCH == 2' not in code
    assert nb.metadata['protocol']['validation_episodes']==100
    assert nb.metadata['protocol']['test_episodes']==1000



def test_parent_and_child_cli_expose_explicit_contiguous_assignment():
    import inspect
    from uav_network.experiments import one_session, deferred_gpu_evaluation
    assert '"--checkpoint-assignment", checkpoint_assignment' in inspect.getsource(one_session.run)
    assert '"--checkpoint-assignment", args.checkpoint_assignment' in inspect.getsource(deferred_gpu_evaluation.main)
    assert 'checkpoint_assignment="strided"' in inspect.getsource(deferred_gpu_evaluation.worker)
