"""Small real CPU AMP contracts; CUDA evidence needs an actual GPU run."""
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from threading import Barrier, Event, Thread
import pytest
import torch


def module():
    assert find_spec("uav_network.algorithms.precision") is not None, "Production mixed precision support is missing"
    return import_module("uav_network.algorithms.precision")


def precision(groups=("critic", "actor", "alpha"), mode="fp16"):
    result = module().MixedPrecision(mode, "cpu", groups)
    if result.enabled:
        state = result.state_dict()
        for scaler in state["scalers"].values():
            scaler["scale"], scaler["growth_interval"] = 128.0, 2
        result.load_state_dict(state)
    return result


def same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            same(value, reference)
    else:
        assert actual == expected


def test_off_preserves_existing_autocast_and_accepts_legacy_state():
    execution = precision(mode="off")
    assert not execution.enabled
    linear = torch.nn.Linear(2, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16), execution.autocast():
        assert torch.is_autocast_enabled("cpu")
        assert linear(torch.ones(1, 2)).dtype == torch.bfloat16
    execution.load_state_dict(None)
    module().validate_precision_state(None, "off", ("critic", "actor", "alpha"))


def test_real_fp16_forward_keeps_gradients_after_no_grad_forward():
    execution = precision(groups=("actor",))
    linear = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(linear.parameters())
    with execution.autocast():
        assert not torch.is_autocast_cache_enabled()
        with torch.no_grad():
            assert linear(torch.ones(1, 2)).dtype == torch.float16
        loss = linear(torch.ones(1, 2)).float().square().sum()
    assert loss.requires_grad
    execution.step_actors([loss], [optimizer], [linear.parameters()], 10)
    assert optimizer.state


def test_unscale_precedes_clip_and_backward_disables_autocast():
    execution = precision(groups=("critic",))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    observed = []
    parameter.register_hook(lambda gradient: observed.append(torch.is_autocast_enabled("cpu")))
    with execution.autocast():
        norm = execution.step("critic", parameter * 100, optimizer, [parameter], 2)
    assert norm.item() == pytest.approx(100)
    assert parameter.item() == pytest.approx(-1)
    assert observed == [False]
    assert execution.state_dict()["counts"]["critic"] == dict(attempted=1, applied=1, skipped=0)


def test_actor_overflow_skips_all_adam_states_and_resets_growth():
    execution = precision()
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizers = [torch.optim.Adam([p], lr=.01) for p in parameters]
    groups = [[p] for p in parameters]
    execution.step_actors([p * 2 for p in parameters], optimizers, groups, 1)
    values, adam = [p.detach().clone() for p in parameters], [deepcopy(o.state_dict()) for o in optimizers]
    before = execution.state_dict()
    norms = execution.step_actors([parameters[0] * 2, parameters[1] * 1e38], optimizers, groups, 1)
    assert not torch.isfinite(torch.stack(norms)).all()
    for p, value, optimizer, old_adam in zip(parameters, values, optimizers, adam):
        same(p, value)
        same(optimizer.state_dict(), old_adam)
    state = execution.state_dict()
    assert state["counts"]["actor"] == dict(attempted=2, applied=1, skipped=1)
    assert state["scalers"]["actor"]["scale"] == 64
    assert state["scalers"]["actor"]["_growth_tracker"] == 0
    for group in ("critic", "alpha"):
        assert state["scalers"][group] == before["scalers"][group]
        assert state["counts"][group] == before["counts"][group]
    execution.step_actors([p * 2 for p in parameters], optimizers, groups, 1)
    assert execution.state_dict()["scalers"]["actor"]["scale"] == 64
    assert execution.state_dict()["scalers"]["actor"]["_growth_tracker"] == 1


def test_finite_gradient_norm_overflow_backs_off_and_resets_growth():
    execution = precision(groups=("critic",))
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter])
    execution.step("critic", parameter.sum(), optimizer, [parameter], 1)
    previous, adam = parameter.detach().clone(), deepcopy(optimizer.state_dict())
    norm = execution.step("critic", (parameter * 1e20).sum(), optimizer, [parameter], 1)
    assert not torch.isfinite(norm)
    same(parameter, previous)
    same(optimizer.state_dict(), adam)
    state = execution.state_dict()
    assert state["scalers"]["critic"]["scale"] == 64
    assert state["scalers"]["critic"]["_growth_tracker"] == 0
    assert state["counts"]["critic"]["skipped"] == 1


@pytest.mark.parametrize("checked", [True, False])
def test_nonfinite_loss_with_finite_derivative_never_steps(checked):
    execution = precision(groups=("critic",))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter])
    with pytest.raises(FloatingPointError, match="Non-finite"):
        execution.step("critic", parameter * 2 + float("inf"), optimizer, [parameter], 1, checked)
    assert parameter.item() == 1
    assert not optimizer.state


def test_nonfinite_actor_loss_rejects_all_steps():
    execution = precision(groups=("actor",))
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizers = [torch.optim.Adam([p]) for p in parameters]
    with pytest.raises(FloatingPointError, match="Non-finite"):
        execution.step_actors([parameters[0] * 2, parameters[1] * 2 + float("inf")],
                              optimizers, [[p] for p in parameters], 1, False)
    assert all(p.item() == 1 for p in parameters)
    assert all(not o.state for o in optimizers)


def test_state_resume_reproduces_next_adam_step_exactly():
    execution = precision(groups=("critic", "actor"))
    parameters = [torch.nn.Parameter(torch.tensor(v)) for v in (1., 2., 3.)]
    optimizers = [torch.optim.Adam([p], lr=.01) for p in parameters]
    execution.step("critic", parameters[0].square(), optimizers[0], [parameters[0]], .5)
    execution.step_actors([p.square() for p in parameters[1:]], optimizers[1:], [[p] for p in parameters[1:]], .5)
    execution.step_actors([parameters[1] * 2, parameters[2] * 1e38], optimizers[1:], [[p] for p in parameters[1:]], .5)
    saved = deepcopy(execution.state_dict())
    replacement = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    replacement_adam = [torch.optim.Adam([p], lr=.01) for p in replacement]
    resumed = precision(groups=("critic", "actor"))
    resumed.load_state_dict(saved)
    for optimizer, reference in zip(replacement_adam, optimizers):
        optimizer.load_state_dict(deepcopy(reference.state_dict()))
    for current, values, adam in ((execution, parameters, optimizers), (resumed, replacement, replacement_adam)):
        current.step("critic", values[0].square(), adam[0], [values[0]], .5)
        current.step_actors([p.square() for p in values[1:]], adam[1:], [[p] for p in values[1:]], .5)
    for p, reference, optimizer, old_optimizer in zip(replacement, parameters, replacement_adam, optimizers):
        same(p, reference)
        same(optimizer.state_dict(), old_optimizer.state_dict())
    assert resumed.state_dict() == execution.state_dict()
    saved["counts"]["critic"]["attempted"] = 999
    assert resumed.state_dict()["counts"]["critic"]["attempted"] == 2


@pytest.mark.parametrize("corruption", ["missing", "mode", "group", "scale_nan", "scale_zero", "growth_factor",
    "backoff_factor", "growth_interval", "growth_tracker", "count_negative", "count_bool", "count_sum", "count_group"])
def test_corrupt_or_missing_state_is_rejected(corruption):
    execution = precision(groups=("critic", "actor"))
    state = execution.state_dict()
    if corruption == "missing": state = None
    elif corruption == "mode": state["mode"] = "off"
    elif corruption == "group": del state["scalers"]["actor"]
    elif corruption == "scale_nan": state["scalers"]["critic"]["scale"] = float("nan")
    elif corruption == "scale_zero": state["scalers"]["critic"]["scale"] = 0
    elif corruption == "growth_factor": state["scalers"]["critic"]["growth_factor"] = 1
    elif corruption == "backoff_factor": state["scalers"]["critic"]["backoff_factor"] = 1
    elif corruption == "growth_interval": state["scalers"]["critic"]["growth_interval"] = 0
    elif corruption == "growth_tracker": state["scalers"]["critic"]["_growth_tracker"] = 2
    elif corruption == "count_negative": state["counts"]["critic"]["attempted"] = -1
    elif corruption == "count_bool": state["counts"]["critic"]["attempted"] = False
    elif corruption == "count_sum": state["counts"]["critic"]["attempted"] = 1
    elif corruption == "count_group": del state["counts"]["actor"]
    with pytest.raises(ValueError, match="precision|Precision|scaler|Scaler"):
        module().validate_precision_state(state, "fp16", ("critic", "actor"))
    with pytest.raises(ValueError): execution.load_state_dict(state)


def test_invalid_mode_and_unknown_group_are_rejected():
    with pytest.raises(ValueError, match="mode"): module().MixedPrecision("bf16", "cpu", ("critic",))
    execution = precision(groups=("critic",))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter])
    with pytest.raises(ValueError, match="group"):
        execution.step("alpha", parameter.square(), optimizer, [parameter], 1)
    assert not optimizer.state


def test_startup_serializes_initial_workers_then_allows_overlap():
    gate = module().CompileStartupGate()
    first_inside, release, trying, second_inside = (Event() for _ in range(4))
    errors = []
    def first():
        try:
            with gate.enter("critic") as serialized:
                assert serialized
                first_inside.set()
                assert release.wait(2)
        except BaseException as error: errors.append(error)
    def second():
        try:
            assert first_inside.wait(2)
            trying.set()
            with gate.enter("policy") as serialized:
                assert serialized
                second_inside.set()
        except BaseException as error: errors.append(error)
    threads = [Thread(target=first), Thread(target=second)]
    for thread in threads: thread.start()
    assert trying.wait(2)
    assert not second_inside.wait(.05)
    release.set()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert not errors
    assert gate.warmed == {"critic", "policy"}
    rendezvous = Barrier(2)
    def steady(group):
        try:
            with gate.enter(group) as serialized:
                assert not serialized
                rendezvous.wait(2)
        except BaseException as error: errors.append(error)
    threads = [Thread(target=steady, args=(g,)) for g in ("critic", "policy")]
    for thread in threads: thread.start()
    for thread in threads:
        thread.join(3)
        assert not thread.is_alive()
    assert not errors


def test_failed_startup_does_not_mark_worker_warm():
    gate = module().CompileStartupGate()
    with pytest.raises(RuntimeError, match="failed"):
        with gate.enter("critic"): raise RuntimeError("failed")
    assert not gate.warmed


def test_decorator_enters_worker_autocast_and_shared_gate():
    api = module()
    class Learner:
        def __init__(self):
            self.precision = precision(groups=("critic", "actor"))
            self._compile_startup_gate = api.CompileStartupGate()
        @api.autocast_update
        def update_critic(self, value, collect_metrics=True):
            return value, collect_metrics, torch.is_autocast_enabled("cpu")
        @api.autocast_update
        def update_actor(self, value, collect_metrics=True):
            return value, collect_metrics, torch.is_autocast_enabled("cpu")
    learner = Learner()
    assert learner.update_critic(7, collect_metrics=False) == (7, False, True)
    assert learner.update_actor(9) == (9, True, True)
    assert learner._compile_startup_gate.warmed == {"critic", "policy"}
    assert learner.update_actor.__name__ == "update_actor"
    learner.precision = precision(mode="off")
    del learner._compile_startup_gate
    assert learner.update_actor(1) == (1, True, False)


def test_metrics_report_group_steps_and_decorator_records_real_autocast():
    api = module()
    execution = precision(groups=("critic",))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter])
    execution.step("critic", parameter.square(), optimizer, [parameter], 1)
    metrics = execution.metrics()
    assert metrics == {"amp/enabled": 1, "amp/critic/attempted": 1, "amp/critic/applied": 1,
                       "amp/critic/skipped": 0, "amp/critic/scale": 128.0}
    class Learner:
        def __init__(self):
            self.precision = execution
            self.device = torch.device("cpu")
        @api.autocast_update
        def update_critic(self, collect_metrics=True):
            return {"critic_loss": .25} if collect_metrics else {}
    learner = Learner()
    assert learner.update_critic() == {"critic_loss": .25, **metrics, "amp/autocast_fp16": 1}
    assert learner.update_critic(False) == {}
    learner.precision = precision(mode="off", groups=("critic",))
    assert learner.update_critic() == {"critic_loss": .25}


def test_fp32_step_and_grouped_actors_match_existing_optimizer_helpers():
    from uav_network.algorithms.optimization import step_actor_optimizers, step_optimizer
    execution = precision(mode="off", groups=("critic", "actor"))
    actual = [torch.nn.Parameter(torch.tensor(value)) for value in (1., 2., 3.)]
    expected = [torch.nn.Parameter(p.detach().clone()) for p in actual]
    actual_adam = [torch.optim.Adam([p], lr=.01) for p in actual]
    expected_adam = [torch.optim.Adam([p], lr=.01) for p in expected]
    expected_norm = step_optimizer(expected[0].square(), expected_adam[0], [expected[0]], .5)
    actual_norm = execution.step("critic", actual[0].square(), actual_adam[0], [actual[0]], .5)
    same(actual_norm, expected_norm)
    expected_norms = step_actor_optimizers([p.square() for p in expected[1:]], expected_adam[1:],
                                          [[p] for p in expected[1:]], .5)
    actual_norms = execution.step_actors([p.square() for p in actual[1:]], actual_adam[1:],
                                        [[p] for p in actual[1:]], .5)
    same(actual_norms, expected_norms)
    for p, reference, optimizer, old_optimizer in zip(actual, expected, actual_adam, expected_adam):
        same(p, reference)
        same(optimizer.state_dict(), old_optimizer.state_dict())
    module().validate_precision_state(execution.state_dict(), "off", ("critic", "actor"))


def test_alpha_scaler_advances_without_affecting_critic_or_actor():
    execution = precision()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter])
    before = execution.state_dict()
    for _ in range(2):
        execution.step("alpha", parameter.square(), optimizer, [parameter], 1)
    after = execution.state_dict()
    assert after["scalers"]["alpha"]["scale"] == 256
    assert after["counts"]["alpha"] == dict(attempted=2, applied=2, skipped=0)
    for group in ("critic", "actor"):
        assert after["scalers"][group] == before["scalers"][group]
        assert after["counts"][group] == before["counts"][group]


def test_update_autocast_enters_a_real_worker_thread():
    api = module()
    class Learner:
        def __init__(self):
            self.precision = precision(groups=("critic",))
            self.device = torch.device("cpu")
            self.linear = torch.nn.Linear(2, 1)
        @api.autocast_update
        def update_critic(self):
            return {"fp16_forward": int(self.linear(torch.ones(1, 2)).dtype == torch.float16)}
    learner, rows, errors = Learner(), [], []
    def run():
        try: rows.append(learner.update_critic())
        except BaseException as error: errors.append(error)
    assert not torch.is_autocast_enabled("cpu")
    worker = Thread(target=run)
    worker.start()
    worker.join(3)
    assert not worker.is_alive()
    assert not errors
    assert rows[0]["fp16_forward"] == rows[0]["amp/autocast_fp16"] == 1
    assert not torch.is_autocast_enabled("cpu")
