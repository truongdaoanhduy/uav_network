"""Small CPU contracts; CUDA evidence comes only from an actual dual-T4 run."""
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


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
    adapter.actor_steps([parameters[0] * 2, parameters[1] * float("inf")],
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
        with torch.autocast("cuda", dtype=torch.float16):
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
