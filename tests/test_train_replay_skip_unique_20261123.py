"""Network admission invariants and branch-neutral loop-invariant scheduling."""
from pathlib import Path

def test_replay_seen_repeated_indices_are_idempotent_without_sorting():
    from uav_network.algorithms import hybrid_replay
    import inspect
    source=inspect.getsource(hybrid_replay.HybridMeasuredReplay.sample)
    assert 'self.seen[indices] = True' in source
    assert 'indices.unique()' not in source
