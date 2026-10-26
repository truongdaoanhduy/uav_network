"""Sparse checkpoints must exclude unused replay storage and retain exact resume."""
import io
from types import SimpleNamespace

import pytest
import torch

from uav_network.experiments.pql_replay import MeasuredReplay


def replay_with_rows(capacity, device, rows):
    replay = MeasuredReplay(capacity, (2, 3), 4, seed=44, device=device)
    for offset in range(0, rows, 4):
        count = min(4, rows - offset)
        observation = torch.full((count, 2, 3), float(offset), device=device)
        state = torch.full((count, 4), float(offset), device=device)
        action = SimpleNamespace(continuous=torch.zeros((count, 2, 3), device=device),
                                 destination=torch.zeros((count, 2), dtype=torch.long, device=device))
        replay.add_batch(observation, state, action, torch.full((count,), float(offset), device=device),
                         observation + 1, state + 1, torch.zeros(count, device=device))
    if rows:
        replay.sample(min(3, len(replay)))
    return replay


def available(device):
    if device.startswith("cuda") and (not torch.cuda.is_available()
                                     or int(device.split(":")[1]) >= torch.cuda.device_count()):
        pytest.skip("requested real CUDA device unavailable")


def serialize(replay):
    stream = io.BytesIO()
    torch.save(replay.state_dict(), stream)
    return stream.getvalue()


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
@pytest.mark.parametrize("rows", [0, 3])
def test_sparse_checkpoint_size_does_not_grow_with_unused_capacity(device, rows):
    available(device)
    small = serialize(replay_with_rows(16, device, rows))
    large = serialize(replay_with_rows(2048, device, rows))
    # Both contain the same 0/3 valid transitions; capacity metadata needs only bytes.
    assert len(large) <= len(small) + 2048, (len(small), len(large))


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
@pytest.mark.parametrize("rows", [3, 20])
def test_serialized_replay_preserves_next_samples_and_measurements(device, rows):
    available(device)
    original = replay_with_rows(16, device, rows)
    saved = torch.load(io.BytesIO(serialize(original)), map_location="cpu", weights_only=True)
    restored = MeasuredReplay(16, (2, 3), 4, seed=99, device=device)
    restored.load_state_dict(saved)
    assert restored.index == original.index
    assert restored.size == original.size
    torch.testing.assert_close(restored.rng.get_state(), original.rng.get_state(), rtol=0, atol=0)
    for name in original.data:
        torch.testing.assert_close(restored.data[name][:original.size], original.data[name][:original.size],
                                   rtol=0, atol=0)
    for name in ("born", "seen"):
        torch.testing.assert_close(getattr(restored, name)[:original.size],
                                   getattr(original, name)[:original.size], rtol=0, atol=0)
    assert restored.stats() == original.stats()
    for _ in range(2):
        expected, actual = original.sample(3), restored.sample(3)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert restored.stats() == original.stats()
