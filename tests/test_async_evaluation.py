"""Real children prove evaluation isolation, bounded scheduling and failure cleanup."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from uav_network.configuration import Config
from uav_network.algorithms.checkpoints import load_policy, make_learner
from uav_network.environment.observations import observation_dim, state_dim


def settings():
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 1
    c.world.max_steps = 2
    c.network.enabled = False
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.num_envs = 1
    c.runtime.torch_threads = 1
    c.tracking.evaluation_interval = 0
    c.tracking.evaluation_interval_steps = 3
    c.tracking.periodic_evaluation_episodes = 2
    c.tracking.evaluation_timeout_seconds = 30
    for a in (c.masac, c.matd3):
        a.hidden_sizes, a.batch_size, a.replay_capacity = [8, 8], 2, 16
    return c


def progress(step=4):
    return {"episodes": 0, "global_step": step, "collection_steps": step,
            "next_seed": 45, "records": []}


def rngs():
    return random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def assert_same_rng(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0]
    np.testing.assert_array_equal(a[1][1], b[1][1])
    assert a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def make_manager(c, folder, tracker=None, initial_step=0):
    from uav_network.experiments.async_evaluation import AsyncCheckpointEvaluator
    from uav_network.experiments.diagnostics import TrainingDiagnostics
    return AsyncCheckpointEvaluator(c, folder, tracker, TrainingDiagnostics(c, folder), initial_step)


def wait_idle(manager, step=99, timeout=30):
    deadline = time.monotonic() + timeout
    while manager.active is not None and time.monotonic() < deadline:
        manager.poll(step)
        time.sleep(.02)
    assert manager.active is None, "Evaluation child did not finish"


@pytest.mark.parametrize("algorithm,pql", [("masac", False), ("matd3", False), ("masac", True), ("matd3", True)])
def test_actor_snapshot_has_exact_source_and_no_parent_rng_draws(tmp_path, algorithm, pql):
    from uav_network.experiments.evaluation_snapshots import save_evaluation_snapshot
    from uav_network.experiments.pql_system import PQLSystem
    c = settings()
    c.runtime.algorithm, c.pql.enabled = algorithm, pql
    c.pql.mode = "synchronous"
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    engine = PQLSystem(c) if pql else None
    learner = engine.policy if pql else make_learner(c, observation_dim(c), state_dim(c))
    source, version = engine.policy_mailbox.read() if pql else ({"actors": learner.actors.state_dict()}, 7)
    before = rngs()
    path = save_evaluation_snapshot(c, tmp_path, source["actors"], version, progress())
    assert_same_rng(before, rngs())
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["purpose"] == "evaluation_only"
    assert saved["source"]["policy_version"] == version
    assert saved["progress"]["global_step"] == 4
    assert "model" not in saved and "training" not in saved
    assert "records" not in saved["progress"]
    frozen = deepcopy(saved["actors"])
    with torch.no_grad():
        next(learner.actors.parameters()).add_(10.)
    for name in frozen:
        torch.testing.assert_close(torch.load(path, weights_only=True)["actors"][name], frozen[name], rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_policy(path)
    if engine:
        engine.close()


def test_real_evaluation_child_leaves_rng_immutable_logs_checkpoint_axis_at_arrival(tmp_path):
    c = settings()
    learner = make_learner(c, observation_dim(c), state_dim(c))
    logs = []
    tracker = SimpleNamespace(diagnostic=lambda metrics, step: logs.append((step, metrics)),
                              run=SimpleNamespace(save=lambda *args, **kwargs: None))
    manager = make_manager(c, tmp_path, tracker)
    before = rngs()
    manager.tick(4, 0, lambda: (learner.actors.state_dict(), 7), progress())
    checkpoint = next((tmp_path / "evaluations").glob("*.pt"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with torch.no_grad():
        next(learner.actors.parameters()).zero_()
    wait_idle(manager)
    manager.close(99)
    assert_same_rng(before, rngs())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    report = manager.last_result
    assert report["episodes"] == 2
    assert [r["seed"] for r in report["records"]] == [1000000044, 1000000045]
    assert report["source"]["checkpoint_sha256"] == digest
    assert report["source"]["global_step"] == 4
    assert report["source"]["scheduled_step"] == 3
    assert report["source"]["policy_version"] == 7
    assert report["protocol"]["training_seed_overlap"] is False
    assert report["protocol"]["validation_seed_overlap"] is True
    arriving = [(step, values) for step, values in logs if "evaluation/mean_return" in values]
    assert arriving[0][0] == 99
    assert arriving[0][1]["evaluation/checkpoint_global_step"] == 4
    assert arriving[0][1]["eval/mean_return"] == report["mean_return"]
    assert "network/generated_packets" in report["pooled"]


def test_schedule_advances_without_completed_episode_and_busy_work_is_not_queued(tmp_path):
    c = settings()
    learner = make_learner(c, observation_dim(c), state_dim(c))
    manager = make_manager(c, tmp_path, initial_step=4)
    calls = []
    source = lambda: (calls.append(1) or learner.actors.state_dict(), 1)
    # Real sleeping child: collection-facing calls return immediately.
    with patch.object(manager, "command", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]):
        begin = time.monotonic()
        manager.tick(5, 0, source, progress(5))
        assert manager.active is None
        manager.tick(7, 0, source, progress(7))
        child = manager.active["process"]
        manager.tick(10, 0, source, progress(10))
        manager.tick(10, 0, source, progress(10))
        assert time.monotonic() - begin < 2.
        assert calls == [1]
        assert manager.counts["skipped_busy"] == 1
        assert manager.next_step == 12
        manager.close(10, drain_seconds=0)
        assert child.poll() is not None
        assert manager.active is None
        assert manager.counts["cancelled"] == 1


@pytest.mark.parametrize("code,expected", [
    ("raise SystemExit(7)", "failed"),
    ("pass", "invalid_result"),
    ("from pathlib import Path; Path(__import__('sys').argv[1]).mkdir(parents=True); "
     "(Path(__import__('sys').argv[1])/'metrics.json').write_text('invalid')", "invalid_result"),
])
def test_nonzero_missing_and_malformed_results_are_visible_nonfatal(tmp_path, code, expected):
    c = settings()
    learner = make_learner(c, observation_dim(c), state_dim(c))
    manager = make_manager(c, tmp_path)
    def command(checkpoint, folder, episodes, seed):
        return [sys.executable, "-c", code, str(folder)]
    with patch.object(manager, "command", side_effect=command):
        manager.tick(4, 0, lambda: (learner.actors.state_dict(), 1), progress())
        wait_idle(manager)
    manager.close(99)
    assert manager.counts[expected] == 1
    assert manager.last_result is None
    events = [json.loads(line) for line in manager.events_path.read_text().splitlines()]
    assert events[-1]["status"] == expected


def test_timeout_terminates_ignoring_sigterm_child_and_reaps_with_bounded_close(tmp_path):
    c = settings()
    c.tracking.evaluation_timeout_seconds = 1
    learner = make_learner(c, observation_dim(c), state_dim(c))
    manager = make_manager(c, tmp_path)
    code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    with patch.object(manager, "command", return_value=[sys.executable, "-c", code]):
        manager.tick(4, 0, lambda: (learner.actors.state_dict(), 1), progress())
        child = manager.active["process"]
        time.sleep(.1)
        # Make elapsed time deterministic; the child has installed its signal handler.
        manager.active["started"] -= 2.
        begin = time.monotonic()
        manager.poll(9)
        assert time.monotonic() - begin < 2.
    assert child.poll() is not None
    assert manager.active is None
    assert manager.counts["timed_out"] == 1
    manager.close(9, drain_seconds=0)


def test_no_new_evaluation_near_deadline_and_spawn_failure_is_nonfatal(tmp_path):
    c = settings()
    manager = make_manager(c, tmp_path)
    manager.tick(4, 0, lambda: (_ for _ in ()).throw(AssertionError("must not snapshot")), progress(), allow_submit=False)
    assert manager.counts["skipped_deadline"] == 1
    learner = make_learner(c, observation_dim(c), state_dim(c))
    with patch("uav_network.experiments.async_evaluation.subprocess.Popen", side_effect=OSError("spawn failed")):
        manager.tick(7, 0, lambda: (learner.actors.state_dict(), 1), progress(7))
    assert manager.counts["failed"] == 1
    assert manager.active is None
    manager.close(9)


@pytest.mark.parametrize("malformation", ["protocol", "nonfinite", "count"])
def test_semantically_incomplete_success_payload_is_a_nonfatal_invalid_result(tmp_path, malformation):
    c = settings()
    learner = make_learner(c, observation_dim(c), state_dim(c))
    manager = make_manager(c, tmp_path)
    payload = {"mean_return": 0., "success_rate": 0., "mean_search_rate_percent": 0.,
               "mean_delivery_rate_percent": 0., "episodes": 2,
               "protocol": {"scenario_hash": "a" * 64},
               "records": [{"seed": 1000000044}, {"seed": 1000000045}]}
    if malformation == "protocol":
        payload["protocol"] = {}
    elif malformation == "nonfinite":
        payload["mean_return"] = float("nan")
    else:
        payload["records"] = []
    code = ("import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.mkdir(parents=True); "
            "(p/'metrics.json').write_text(sys.argv[2])")
    def command(checkpoint, folder, episodes, seed):
        return [sys.executable, "-c", code, str(folder), json.dumps(payload)]
    with patch.object(manager, "command", side_effect=command):
        manager.tick(4, 0, lambda: (learner.actors.state_dict(), 1), progress())
        wait_idle(manager)
    assert manager.counts["invalid_result"] == 1
    assert manager.last_result is None
    manager.close(9)


@pytest.mark.parametrize("stop_reason", ["timed_out", "cancelled"])
def test_unreaped_child_remains_reserved_until_nonblocking_reap(tmp_path, stop_reason):
    """A delayed OS reap must never permit a second evaluator to start."""
    c = settings()
    learner = make_learner(c, observation_dim(c), state_dim(c))
    manager = make_manager(c, tmp_path)
    with patch.object(manager, "command", return_value=[sys.executable, "-c", "import time; time.sleep(60)"]):
        manager.tick(4, 0, lambda: (learner.actors.state_dict(), 1), progress())
        child = manager.active["process"]
        try:
            with patch.object(manager, "_stop", return_value=False):
                if stop_reason == "timed_out":
                    manager.active["started"] -= 31.
                    manager.poll(5)
                else:
                    manager.close(5, drain_seconds=0)
            assert manager.active is not None
            assert manager.active["process"] is child
            assert child.poll() is None
            assert manager.counts[stop_reason] == 1
            manager.tick(7, 0, lambda: (_ for _ in ()).throw(AssertionError("must not replace unreaped child")), progress(7))
            assert manager.counts["submitted"] == 1
            assert manager.counts[stop_reason] == 1
        finally:
            manager._stop(child)
    manager.poll(9)
    assert manager.active is None
    assert manager.counts["reaped"] == 1
    manager.close(9, drain_seconds=0)
