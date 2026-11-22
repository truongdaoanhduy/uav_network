"""Keep natural append-only report slots safe for sparse-step optimization."""
import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.environment.gpu.bayesian import confirm_observed_cells
from uav_network.environment.gpu.vector import WarpVectorEnv


def test_nooracle_candidate_allocator_keeps_unique_contiguous_prefix():
    cfg = load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    cfg.runtime.num_envs = 1
    cfg.runtime.device = 'cpu'
    env = WarpVectorEnv(cfg, device='cpu')
    try:
        env.reset(seed=[44])
        assert env.p.report_slots == 140
        cells = wp.to_torch(env.s.report_cell)
        cells.fill_(-1)
        counts = wp.to_torch(env.s.confirmed_count)
        candidates = wp.to_torch(env.s.confirmed_candidates)
        counts.zero_()
        counts[0, 0] = 2
        candidates[0, 0, :2] = torch.tensor([7, 9])
        counts[0, 1] = 2
        candidates[0, 1, :2] = torch.tensor([9, 11])
        env._launch(confirm_observed_cells, 1, env.s, env.p, env.s.terminated, 0)
        assert cells[0, :3].tolist() == [7, 9, 11]
        assert bool((cells[0, 3:] == -1).all())
        counts.zero_()
        counts[0, 0] = 2
        candidates[0, 0, :2] = torch.tensor([11, 13])
        env._launch(confirm_observed_cells, 1, env.s, env.p, env.s.terminated, 0)
        assert cells[0, :4].tolist() == [7, 9, 11, 13]
        assert bool((cells[0, 4:] == -1).all())
    finally:
        env.close()
