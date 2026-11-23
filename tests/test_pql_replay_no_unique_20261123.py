"""Replay marking preserves exact sampling and ages when duplicate indices occur."""
import torch
from uav_network.experiments.pql_replay import MeasuredReplay

def test_direct_replay_seen_marking_matches_unique_reference():
    b = MeasuredReplay(17, (2, 4), 8, 44, "cpu", continuous_dim=4)
    n = 13
    o = torch.arange(n*2*4, dtype=torch.float32).view(n,2,4)
    s = torch.arange(n*8, dtype=torch.float32).view(n,8)
    action = type("Action", (), {"continuous": torch.zeros((n,2,4)),
                                  "destination": torch.zeros((n,2),dtype=torch.long)})()
    b.add_batch(o,s,action,torch.ones(n),o,s,torch.zeros(n))
    state_before = b.rng.get_state().clone()
    expected_gen = torch.Generator(device="cpu")
    expected_gen.set_state(state_before)
    requested = 12
    sampled_indices = torch.randint(b.size, (requested,), generator=expected_gen)
    expected_seen = torch.zeros(17, dtype=torch.bool)
    expected_seen[sampled_indices.unique()] = True
    batch = b.sample(requested)
    torch.testing.assert_close(b.seen,expected_seen)
    torch.testing.assert_close(batch["state"],b.data["state"][sampled_indices])
    torch.testing.assert_close(batch["observation"],b.data["observation"][sampled_indices])
    assert b.sample_uses == requested
    assert b.collections == 1
    assert b.stats()["replay/sample_age_mean_collections"] == 0
