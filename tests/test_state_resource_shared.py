"""Shared fallback certificates must be counted once, not once per world."""
from uav_network.configuration import load_config
from uav_network.environment.gpu.state import allocate_state, allocated_bytes
from uav_network.experiments.state_resources import estimate_warp_state_bytes


def test_estimate_matches_real_one_and_three_world_allocations():
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    estimate = estimate_warp_state_bytes(c, (1, 3))
    for count in (1, 3):
        state, _ = allocate_state(c, num_envs=count, seeds=list(range(44,44+count)), certify=True)
        assert estimate["estimated_bytes"][str(count)] == allocated_bytes(state)
