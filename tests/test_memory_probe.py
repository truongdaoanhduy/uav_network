"""Small CPU contracts; CUDA evidence comes only from an actual dual-T4 run."""
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def probe_autocast(module, device_type="cuda", dtype=torch.float16):
    # Execute the real script's context expression. CPU BF16 only substitutes
    # hardware/precision; cache_enabled is taken unchanged from the callsite.
    import ast
    tree = ast.parse(Path(module.__file__).read_text())
    contexts = [item.context_expr for node in ast.walk(tree) if isinstance(node, ast.With)
                for item in node.items if isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and isinstance(item.context_expr.func.value, ast.Name)
                and item.context_expr.func.value.id == "torch"
                and item.context_expr.func.attr == "autocast"
                and item.context_expr.args
                and isinstance(item.context_expr.args[0], ast.Constant)
                and item.context_expr.args[0].value == "cuda"]
    assert len(contexts) == 1
    def adapted(requested_device, *, dtype, **kwargs):
        assert requested_device == "cuda" and dtype == torch.float16
        return torch.autocast(device_type, dtype=selected_dtype, **kwargs)
    selected_dtype = dtype
    return eval(compile(ast.fix_missing_locations(ast.Expression(contexts[0])),
                        module.__file__, "eval"),
                {"torch": SimpleNamespace(autocast=adapted, float16=torch.float16)})


def test_base_probe_autocast_preserves_actor_gradients_after_target_pass():
    from test_split_updates import small_learner, replay_batch
    torch.manual_seed(44)
    p = probe(); learner = small_learner(vectorized=True)
    batch = replay_batch(learner)
    amp = p.ProbeAMP(learner, device_type="cpu", init_scale=128)
    with amp, probe_autocast(p, device_type="cpu", dtype=torch.bfloat16):
        learner.update(batch)
    assert learner.actor_updates == 1
    assert amp.counts["actor"]["applied"] == 1


def probe():
    assert Path("scripts/probe_masac_memory.py").is_file(), "Standalone memory probe is missing"
    return importlib.import_module("scripts.probe_masac_memory")


def test_replay_bytes_matches_original_storage_without_large_allocation():
    p = probe()
    from uav_network.algorithms.replay import TensorReplayBuffer
    small = TensorReplayBuffer(3, (6, 163), 643)
    assert p.replay_bytes(3, (6, 163), 643) == small.allocated_bytes
    assert p.replay_bytes(1_000_000, (6, 163), 643) == 13_096_000_000
    assert p.replay_bytes(2_000_000, (6, 163), 643) == 26_192_000_000
    with pytest.raises(ValueError):
        p.replay_bytes(0, (6, 163), 643)


def test_no_cuda_rejected_before_tensor_allocation_or_wandb(tmp_path):
    p = probe()
    args = p.parser().parse_args(["--capacity", "2000000", "--output", str(tmp_path)])
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.empty", side_effect=AssertionError("Must not allocate on CPU")
    ), patch.dict("os.environ", {}, clear=True):
        result = p.run_probe(args)
    assert result["status"] == "failed"
    assert result["phase"] == "hardware_guard"
    assert "exactly two T4" in result["error"]
    assert (tmp_path / "result.json").is_file()


def test_hardware_guard_rejects_other_board_and_gpu_counts():
    p = probe()
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=1
    ):
        with pytest.raises(RuntimeError, match="exactly two T4"):
            p.require_two_t4()
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=2
    ), patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="Tesla P100")):
        with pytest.raises(RuntimeError, match="exactly two T4"):
            p.require_two_t4()


def test_scaled_gradients_are_unscaled_before_clipping():
    p = probe()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    learner = SimpleNamespace(critic_optimizer=optimizer)
    adapter = p.ProbeAMP(learner, device_type="cpu", init_scale=128.0)
    norm = adapter.single_step(parameter * 100, optimizer, [parameter], 2.0)
    assert norm.item() == pytest.approx(100.0)
    assert parameter.item() == pytest.approx(-1.0)
    assert adapter.counts["critic"] == {"attempted": 1, "applied": 1, "skipped": 0}


def test_actor_overflow_skips_every_optimizer_and_scale_is_independent():
    p = probe()
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizers = [torch.optim.Adam([parameter], lr=.01) for parameter in parameters]
    adapter = p.ProbeAMP(SimpleNamespace(), device_type="cpu", init_scale=128.0)
    critic_scale = adapter.scalers["critic"].get_scale()
    adapter.actor_steps([parameters[0] * 2, parameters[1] * 1e38],
                        optimizers, [[parameter] for parameter in parameters], 1.0)
    assert [parameter.item() for parameter in parameters] == [1.0, 1.0]
    assert all(not optimizer.state for optimizer in optimizers)
    assert adapter.counts["actor"] == {"attempted": 1, "applied": 0, "skipped": 1}
    assert adapter.scalers["actor"].get_scale() < 128
    assert adapter.scalers["critic"].get_scale() == critic_scale
    adapter.actor_steps([parameter * 2 for parameter in parameters],
                        optimizers, [[parameter] for parameter in parameters], 1.0)
    assert all(parameter.item() < 1 for parameter in parameters)
    assert all(optimizer.state for optimizer in optimizers)
    assert adapter.counts["actor"] == {"attempted": 2, "applied": 1, "skipped": 1}


def test_probe_patches_restore_on_exception():
    p = probe()
    import uav_network.algorithms.masac as masac
    original = (masac.step_optimizer, masac.step_actor_optimizers)
    adapter = p.ProbeAMP(SimpleNamespace(), device_type="cpu")
    with pytest.raises(ValueError, match="stop"):
        with adapter:
            assert masac.step_optimizer != original[0]
            assert masac.step_actor_optimizers != original[1]
            raise ValueError("stop")
    assert (masac.step_optimizer, masac.step_actor_optimizers) == original


def test_parser_preserves_bounded_requested_defaults():
    args = probe().parser().parse_args(["--capacity", "1000000"])
    assert (args.envs, args.batch, args.rounds, args.updates_per_round) == (2048, 4096, 4, 8)
    assert args.compile and args.fused and args.amp == "fp16"


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() != 2,
                    reason="Requires real dual CUDA; CPU tests do not verify FP16/Inductor")
def test_cuda_probe_adapter_executes_fp16_compiled_critic_and_real_steps():
    p = probe()
    from test_split_updates import small_learner, replay_batch
    learner = small_learner(vectorized=True).to("cuda:1")
    learner.device = torch.device("cuda:1")
    batch = {key: value.to("cuda:1") for key, value in replay_batch(learner).items()}
    evidence = {}
    with p.ProbeCompiler(learner, evidence), p.ProbeAMP(learner, init_scale=16) as amp:
        with probe_autocast(p):
            learner.update(batch)
    assert all(amp.counts[group]["applied"] == 1 for group in ("critic", "actor", "alpha"))
    assert evidence["critic"]["fp16_outputs"] > 0
    assert evidence["actors"]["fp16_outputs"] > 0
    assert evidence["dynamo"]["stats"]["unique_graphs"] > 0
    assert evidence["inductor_generated_kernels"] > 0


def test_adapter_runs_backward_outside_autocast():
    p = probe()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    observed_autocast = []
    parameter.register_hook(lambda grad: observed_autocast.append(torch.is_autocast_enabled("cpu")))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    adapter = p.ProbeAMP(SimpleNamespace(critic_optimizer=optimizer), device_type="cpu", init_scale=128)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        adapter.single_step(parameter * 2, optimizer, [parameter], 10)
    assert observed_autocast == [False]


def test_cpu_adapter_preserves_original_masac_equations_and_compiler_patch_restore():
    from copy import deepcopy
    from test_split_updates import small_learner, replay_batch
    import uav_network.algorithms.masac as module
    p = probe()
    learner = small_learner(vectorized=True)
    original = deepcopy(learner)
    batch = replay_batch(learner)
    torch.manual_seed(500)
    expected = original.update(batch)
    forwards = [actor.encoder.forward for actor in learner.actors]
    critic_values = module.critic_values
    evidence = {}
    torch.manual_seed(500)
    with p.ProbeCompiler(learner, evidence, enabled=False), p.ProbeAMP(
        learner, device_type="cpu", init_scale=128
    ) as amp:
        actual = learner.update(batch)
    assert actual == expected
    for key, value in original.state_dict().items():
        torch.testing.assert_close(learner.state_dict()[key], value, rtol=0, atol=0)
    assert all(amp.counts[name]["applied"] == 1 for name in ("critic", "actor", "alpha"))
    assert module.critic_values is critic_values
    assert [actor.encoder.forward for actor in learner.actors] == forwards
    assert evidence["critic"]["completed_calls"] == 3
    assert evidence["critic"]["fp16_outputs"] == 0


def test_nonfinite_unscaled_loss_with_finite_derivative_never_steps():
    p = probe()
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizers = [torch.optim.Adam([parameter], lr=.01) for parameter in parameters]
    adapter = p.ProbeAMP(SimpleNamespace(critic_optimizer=optimizers[0]),
                         device_type="cpu", init_scale=128)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        adapter.single_step(parameters[0] * 2 + float("inf"), optimizers[0], [parameters[0]], 10)
    assert parameters[0].item() == 1
    assert not optimizers[0].state
    with pytest.raises(FloatingPointError, match="Non-finite"):
        adapter.actor_steps([parameters[0] * 2, parameters[1] * 2 + float("inf")],
                            optimizers, [[parameter] for parameter in parameters], 10)
    assert [parameter.item() for parameter in parameters] == [1, 1]
    assert all(not optimizer.state for optimizer in optimizers)


def test_default_resolved_probe_config_validates_without_compiling_inside_vmap():
    p = probe()
    args = p.parser().parse_args(["--capacity", "1000000"])
    assert hasattr(p, "resolve_config"), "CPU-testable default configuration resolver is missing"
    config, warmup = p.resolve_config(args)
    assert config.validate() is config
    assert args.compile is True
    assert config.runtime.compile_models is False
    assert config.runtime.vectorize_critics is True
    assert config.runtime.fused_optimizers is True
    assert (config.runtime.num_envs, config.masac.batch_size, config.world.max_steps) == (2048, 4096, 1800)
    assert warmup["original_learning_starts"] == 131072
    assert warmup["diagnostic_learning_starts"] == 4096


def finalize_case(tmp_path, status="passed", indices=(0, 1), save_error=False,
                  close_error=False, summary_error=False, bare_torch_uuid=False):
    p = probe()
    assert hasattr(p, "finalize_probe"), "Testable finalization boundary is missing"
    import json
    events = []
    report = {"status": status, "phase": "complete" if status == "passed" else "replay_allocation",
              "stages": [], "compiler": {}, "gpu_resources": {"gpus": [
                  {"physical_index": index, "uuid": f"GPU-{index}", "sample_count": 1,
                   "physical_memory_used_bytes_max": 1024, "power": {"status": "unavailable"}}
                  for index in indices]}}
    report["hardware"] = [{"index": logical, "uuid": str(physical) if bare_torch_uuid else f"GPU-{physical}"}
                          for logical, physical in enumerate(indices)]
    class Summary:
        def update(self, values):
            events.append(("summary", values["status"]))
            if summary_error:
                raise OSError("summary failed")
    class Run:
        summary = Summary()
        def save(self, *args, **kwargs):
            events.append(("save",))
            if save_error:
                raise OSError("upload failed")
        def finish(self, exit_code):
            events.append(("finish", exit_code))
    class Vector:
        def close(self):
            events.append(("close",))
            if close_error:
                raise RuntimeError("close failed")
    def persist():
        (tmp_path / "result.json").write_text(json.dumps(report))
    persist()
    p.finalize_probe(report, tmp_path, Run(), Vector(), None, 0, persist)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved == report
    return report, events


def test_success_requires_physical_memory_on_both_boards_and_power_is_optional(tmp_path):
    report, events = finalize_case(tmp_path, indices=(0,))
    assert report["status"] == "failed"
    assert report["probe_outcome"] == "passed"
    assert report["measurement_complete"] is False
    assert report["measurement"]["expected_physical_indices"] == [0, 1]
    assert report["measurement"]["missing_physical_indices"] == [1]
    assert events[-1] == ("finish", 1)
    report, events = finalize_case(tmp_path, indices=(0, 1))
    assert report["status"] == "passed"
    assert report["measurement_complete"] is True
    assert events[-1] == ("finish", 0)


def test_incomplete_measurement_retains_oom_capacity_outcome(tmp_path):
    report, events = finalize_case(tmp_path, status="oom", indices=())
    assert report["status"] == report["probe_outcome"] == "oom"
    assert report["measurement_complete"] is False
    assert events[-1] == ("finish", 1)


@pytest.mark.parametrize("boundary", ["upload", "summary"])
def test_upload_failure_still_finishes_nonzero_and_retains_probe_outcome(tmp_path, boundary):
    report, events = finalize_case(tmp_path, save_error=boundary == "upload",
                                   summary_error=boundary == "summary")
    assert report["status"] == "failed"
    assert report["probe_outcome"] == "passed"
    assert f"{boundary} failed" in report["tracking_finalize_error"]
    assert events[-1] == ("finish", 1)


def test_environment_close_failure_is_final_before_summary_upload(tmp_path):
    report, events = finalize_case(tmp_path, close_error=True)
    assert report["status"] == "failed"
    assert report["probe_outcome"] == "passed"
    assert report["environment_close_error"] == "close failed"
    assert events[0] == ("close",)
    assert ("summary", "failed") in events
    assert events[-1] == ("finish", 1)


def test_logical_cuda_devices_map_to_physical_uuid_after_reordering(tmp_path):
    report, _ = finalize_case(tmp_path, indices=(1, 0))
    assert report["measurement_complete"] is True
    assert report["cuda_physical_mapping"] == [
        {"logical_index": 0, "uuid": "GPU-1", "physical_index": 1},
        {"logical_index": 1, "uuid": "GPU-0", "physical_index": 0}]


def test_bare_torch_uuid_maps_to_prefixed_physical_uuid(tmp_path):
    report, _ = finalize_case(tmp_path, indices=(1, 0), bare_torch_uuid=True)
    assert [row["physical_index"] for row in report["cuda_physical_mapping"]] == [1, 0]
    assert [row["uuid"] for row in report["cuda_physical_mapping"]] == ["1", "0"]


@pytest.mark.parametrize("requested_fused,actual_fused", [(True, True), (True, False), (False, False)])
def test_success_evidence_requires_requested_fused_and_allows_no_fused(requested_fused, actual_fused):
    p = probe()
    assert hasattr(p, "verify_evidence"), "Testable success-evidence gate is missing"
    args = p.parser().parse_args(["--capacity", "1000000"] + ([] if requested_fused else ["--no-fused"]))
    report = {"compiler": {"probe_compile_executed": True,
                           "actors": {"fp16_outputs": 1}, "critic": {"fp16_outputs": 1}},
              "amp": {"counts": {name: {"applied": 1} for name in ("critic", "actor", "alpha")}},
              "fused": {"all_param_groups_fused": actual_fused}}
    if requested_fused and not actual_fused:
        with pytest.raises(RuntimeError, match="Requested fused"):
            p.verify_evidence(args, report)
    else:
        evidence = p.verify_evidence(args, report)
        assert evidence == {"fp16_forward": True, "compiled_graphs": True,
                            "all_optimizer_groups_applied": True,
                            "fused_optimizer_steps": requested_fused and actual_fused}


@pytest.mark.parametrize("missing", ["compile", "actor_fp16", "critic_call", "critic_step", "actor_step", "alpha_step"])
def test_success_evidence_preserves_compile_fp16_and_applied_step_gates(missing):
    p = probe()
    args = p.parser().parse_args(["--capacity", "1000000"])
    report = {"compiler": {"probe_compile_executed": True,
                           "actors": {"fp16_outputs": 1},
                           "critic": {"completed_calls": 1, "fp16_outputs": 0,
                                      "output_dtypes": ["torch.float32"]}},
              "amp": {"counts": {name: {"applied": 1} for name in ("critic", "actor", "alpha")}},
              "fused": {"all_param_groups_fused": True}}
    if missing == "compile":
        report["compiler"]["probe_compile_executed"] = False
    elif missing == "actor_fp16":
        report["compiler"]["actors"]["fp16_outputs"] = 0
    elif missing == "critic_call":
        report["compiler"]["critic"]["completed_calls"] = 0
    else:
        report["amp"]["counts"][missing.removesuffix("_step")]["applied"] = 0
    with pytest.raises(RuntimeError, match="compilation|FP16"):
        p.verify_evidence(args, report)


def test_fp32_critic_output_is_valid_with_direct_fp16_actor_and_completed_critic_evidence():
    p = probe()
    args = p.parser().parse_args(["--capacity", "1000000"])
    report = {"compiler": {"probe_compile_executed": True,
                           "actors": {"fp16_outputs": 7},
                           "critic": {"completed_calls": 3, "fp16_outputs": 0,
                                      "output_dtypes": ["torch.float32"]}},
              "amp": {"counts": {name: {"applied": 1} for name in ("critic", "actor", "alpha")}},
              "fused": {"all_param_groups_fused": True}}
    assert p.verify_evidence(args, report)["fp16_forward"] is True
