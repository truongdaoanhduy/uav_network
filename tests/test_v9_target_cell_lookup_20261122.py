"""Exact V9 target-cell index: performance optimization cannot alter sensing."""
import numpy as np
import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv


def _config():
    c=load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    c.runtime.device='cpu';c.runtime.num_envs=3
    return c


def assert_lookup(env):
    p=env.p
    assert p.cell_reports==1
    lookup=wp.to_torch(env.target_cell_lookup).numpy()
    targets=wp.to_torch(env.s.targets).numpy()
    assert lookup.shape==(env.num_envs,p.grid_n*p.grid_n)
    for e in range(env.num_envs):
        visited=[]
        for t in range(p.targets_count):
            gx=min(int(targets[e,t,0]/p.cell_width),p.grid_n-1)
            gy=min(int(targets[e,t,1]/p.cell_width),p.grid_n-1)
            cell=gy*p.grid_n+gx
            assert int(lookup[e,cell])==t
            visited.append(cell)
        assert len(set(visited))==len(visited)
        assert int((lookup[e]>=0).sum())==p.targets_count


def test_lookup_maps_each_real_target_to_exact_observed_cell():
    cfg=_config();env=WarpVectorEnv(cfg,device='cpu')
    try:
        env.reset(seed=[44,145,246]);assert_lookup(env)
        env.reset(seed=[900,901,902]);assert_lookup(env)
    finally:
        env.close()


def test_lookup_respects_partial_reset_mask_and_keeps_other_worlds():
    cfg=_config();env=WarpVectorEnv(cfg,device='cpu')
    try:
        env.reset(seed=[44,145,246]);before=wp.to_torch(env.target_cell_lookup).clone()
        env.reset(seed=[904,905,906],options={'reset_mask':[False,True,False]})
        assert_lookup(env)
        current=wp.to_torch(env.target_cell_lookup)
        assert torch.equal(before[0],current[0])
        assert torch.equal(before[2],current[2])
    finally:env.close()


def test_lookup_cost_estimate_bounded_at_original_grid_size():
    cfg=_config()
    assert cfg.world.map_size_m == 1000
    assert cfg.sensor.grid_cell_m == 25
    assert cfg.world.num_targets == 35
    assert cfg.sensor.report_cell_hypotheses


def test_cache_is_not_in_checkpoint_scientific_state():
    cfg=_config();env=WarpVectorEnv(cfg,device='cpu')
    try:
        env.reset(seed=[44,145,246])
        saved=env.snapshot_state()
        assert "target_cell_lookup" not in saved["arrays"]
        first=wp.to_torch(env.target_cell_lookup).clone()
        env.restore_state(saved)
        assert torch.equal(wp.to_torch(env.target_cell_lookup),first)
        assert_lookup(env)
    finally:
        env.close()
