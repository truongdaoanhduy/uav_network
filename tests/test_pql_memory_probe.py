"""Exercise actual asynchronous workers on tiny CPU tensors before the dual-T4 probe."""
import importlib
from threading import current_thread
from unittest.mock import patch

import pytest
import torch

from test_pql_system import pql_config, transition
from test_split_updates import replay_batch
from uav_network.experiments.pql_system import PQLSystem


def probe():
    import importlib.util
    assert importlib.util.find_spec("scripts.probe_masac_pql_memory"), "PQL probe is missing"
    return importlib.import_module("scripts.probe_masac_pql_memory")


def test_async_workers_apply_independent_scaled_steps_and_restore_helpers():
    p = probe()
    import uav_network.algorithms.masac as masac
    torch.set_num_threads(1)
    config = pql_config(mode="async")
    # CPU BF16 has different exp/cat autocast rules; this checks thread/scaler ownership.
    # Actual FP16 vectorized execution is verified only by the real T4 experiment.
    config.pql.warmup_collection_steps = 3
    config.pql.critic_sample_ratio = 2
    engine = PQLSystem(config)
    original_helpers = masac.step_optimizer, masac.step_actor_optimizers
    original_updates = engine.critic.update_critic, engine.policy.update_actor
    evidence = {}
    try:
        batch = replay_batch(engine.critic)
        with p.PQLProbeExecution(engine, evidence, compile_enabled=False,
                                 device_type="cpu") as execution:
            for _ in range(4):
                engine.submit(transition(batch))
            engine.flush()
        assert engine.critic.updates == engine.policy.actor_updates == 4
        assert all(value == {"attempted": 4, "applied": 4, "skipped": 0}
                   for value in execution.amp.counts.values())
        assert {row["thread"] for row in execution.rows} == {"pql-critic", "pql-policy"}
        assert all(row["autocast_enabled"] for row in execution.rows)
        assert all(row["thread"] != current_thread().name for row in execution.rows)
        assert len(execution.rows) == 8
        assert (masac.step_optimizer, masac.step_actor_optimizers) == original_helpers
        assert (engine.critic.update_critic, engine.policy.update_actor) == original_updates
        assert engine.pending_collections == engine.pending_policies == 0
    finally:
        engine.close()


def test_pql_profile_matches_base_probe_budget_without_allocating():
    p = probe()
    args = p.parser().parse_args(["--capacity", "1000000"])
    config, override = p.resolve_pql_config(args)
    assert config.pql.enabled and config.pql.mode == "async"
    assert (config.runtime.num_envs, config.masac.batch_size,
            config.masac.replay_capacity, config.world.max_steps) == (2048,4096,1000000,1800)
    assert config.pql.warmup_collection_steps == 3
    assert config.pql.critic_sample_ratio == 8
    assert config.runtime.fused_optimizers and not config.runtime.compile_models
    assert override["original_pql_warmup_collections"] == 32


def test_no_cuda_rejected_before_any_large_allocation(tmp_path):
    p = probe()
    args = p.parser().parse_args(["--capacity","1000000","--output",str(tmp_path)])
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.empty", side_effect=AssertionError("No CPU fallback allocation")
    ):
        result = p.run_probe(args)
    assert result["status"] == "failed"
    assert result["phase"] == "hardware_guard"
    assert "exactly two T4" in result["error"]


def test_cleanup_restores_after_close_reports_error_once_workers_stopped():
    p = probe()
    import uav_network.algorithms.masac as masac
    engine = PQLSystem(pql_config(mode="async"))
    original_helpers = masac.step_optimizer, masac.step_actor_optimizers
    original_close = engine.close
    def close_with_error():
        original_close()
        raise RuntimeError("close report error")
    try:
        with patch.object(engine,"close",side_effect=close_with_error):
            with pytest.raises(RuntimeError,match="close report error"):
                with p.PQLProbeExecution(engine,{},compile_enabled=False,device_type="cpu"):
                    pass
        assert all(not thread.is_alive() for thread in engine.threads)
        assert (masac.step_optimizer,masac.step_actor_optimizers) == original_helpers
    finally:
        # Keep the test process isolated even when the regression reproduces.
        original_close()
        masac.step_optimizer,masac.step_actor_optimizers = original_helpers


def test_compile_startup_tracks_both_worker_warmups_without_serializing_steady_work():
    p = probe()
    assert hasattr(p,"CompileStartupGate"), "Compile startup gate is missing"
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event
    gate = p.CompileStartupGate()
    first_entered, release_first, second_attempted, second_entered = (Event() for _ in range(4))
    def first_cold():
        with gate.enter("critic") as serialized:
            assert serialized
            first_entered.set()
            assert release_first.wait(timeout=2)
    def second_cold():
        second_attempted.set()
        with gate.enter("policy") as serialized:
            assert serialized
            second_entered.set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_cold)
        assert first_entered.wait(timeout=2)
        second = pool.submit(second_cold)
        assert second_attempted.wait(timeout=2)
        try:
            assert not second_entered.wait(timeout=.1)
        finally:
            release_first.set()
        first.result()
        second.result()
    barrier = Barrier(2)
    def steady(group):
        with gate.enter(group) as serialized:
            assert not serialized
            barrier.wait(timeout=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(steady,group) for group in ("critic","policy")]
        for job in jobs:
            job.result()
    assert gate.warmed == {"critic","policy"}


def test_probe_reports_both_hybrid_tiers_and_keeps_plain_storage_fields():
    from uav_network.algorithms.hybrid_replay import HybridMeasuredReplay
    from uav_network.algorithms.replay import TensorReplayBuffer
    import ast
    import inspect
    from types import SimpleNamespace
    module = probe()
    tree = ast.parse(inspect.getsource(module.run_probe))
    value = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                         and t.value.id == "report" and isinstance(t.slice, ast.Constant)
                         and t.slice.value == "replay" for t in n.targets))
    def storage_for(replay):
        report = eval(compile(ast.Expression(value), module.__file__, "eval"),
                      dict(vars(module), engine=SimpleNamespace(replay=replay),
                           args=SimpleNamespace(capacity=8)))
        return report["storage"]
    replay = HybridMeasuredReplay(8, (2,3), 4, hot_device="cpu", cpu_fraction=.75)
    storage = storage_for(replay)
    assert storage["cpu/observation"]["bytes"] == 144
    assert storage["hot/observation"]["bytes"] == 48
    assert sum(item["bytes"] for item in storage.values()) == 1024
    plain = storage_for(TensorReplayBuffer(8, (2,3), 4))
    assert "observation" in plain and "cpu/observation" not in plain
    assert plain["destination"]["bytes"] == 128
