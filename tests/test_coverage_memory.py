"""Coverage is local persistent sensing memory, including exact collector resume."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from uav_network.configuration import Config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.observations import observation_dim, state_dim, local_observations


def memory_config(cells=4):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 1, 1
    c.world.max_steps = 12
    c.sensor.altitude_m = 300.
    c.sensor.coverage_grid_cells = cells
    c.network.enabled = False
    c.runtime.num_envs = 2
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.environment_backend = "warp_packet"
    return c


SCENE = {"positions": [[125.,125.],[875.,875.]], "targets": [[500.,600.]],
         "obstacles": [[250.,125.,30.]]}


def test_native_coverage_remembers_only_local_visible_cells():
    c = memory_config()
    env = SwarmSearchEnv(c)
    try:
        obs, _ = env.reset(seed=44, options=SCENE)
        assert env.coverage.shape == (2,16)
        assert env.coverage[0,0] == 1
        assert env.coverage[0,1] == 0  # Cell behind the obstacle.
        assert env.coverage[0,4] == 1  # Unoccluded cell within FOV.
        assert env.coverage[0,15] == 0 and env.coverage[1,15] == 1
        np.testing.assert_array_equal(obs[0,-19:-3], env.coverage[0])
        before = env.coverage[1].copy()
        env.positions[0] = [625.,125.]
        env._sense()
        assert env.coverage[0,0] == 1 and env.coverage[0,2] == 1
        np.testing.assert_array_equal(env.coverage[1],before)
        env.battery_j[0] = 0
        before = env.coverage[0].copy()
        env.positions[0] = [125.,875.]
        env._sense()
        np.testing.assert_array_equal(env.coverage[0],before)
    finally:
        env.close()


def test_opt_in_memory_changes_only_declared_dimensions():
    c = memory_config(0)
    before = observation_dim(c), state_dim(c)
    c.sensor.coverage_grid_cells = 8
    assert observation_dim(c) == before[0]+64
    assert state_dim(c) == before[1]+128
    assert Config().sensor.coverage_grid_cells == 0


def test_native_and_warp_memory_and_observation_match():
    wp = pytest.importorskip("warp")
    from uav_network.environment.gpu.state import allocate_state
    from uav_network.environment.gpu.geometry import sense, scan_coverage
    from uav_network.environment.gpu.observations import observe, centralized
    c = memory_config()
    env = SwarmSearchEnv(c)
    try:
        obs, _ = env.reset(seed=44, options=SCENE)
        s,p = allocate_state(c,num_envs=1,device="cpu")
        for name in ("positions","targets","obstacles"):
            getattr(s,name).assign(np.asarray([SCENE[name]],dtype=float))
        wp.launch(sense,dim=(1,2,1),inputs=[s,p],device="cpu")
        wp.launch(scan_coverage,dim=(1,2,16),inputs=[s,p],device="cpu")
        wp.launch(observe,dim=(1,2),inputs=[s,p],device="cpu")
        wp.launch(centralized,dim=1,inputs=[s,p],device="cpu")
        np.testing.assert_array_equal(s.coverage.numpy()[0],env.coverage)
        np.testing.assert_allclose(s.observation.numpy()[0],obs,rtol=0,atol=1e-7)
        np.testing.assert_allclose(s.state.numpy()[0],env.state(),rtol=0,atol=1e-7)
    finally:
        env.close()


def test_warp_partial_reset_keeps_other_worlds_memory():
    pytest.importorskip("warp")
    from uav_network.environment.gpu.vector import WarpVectorEnv
    c = memory_config()
    env = WarpVectorEnv(c,device="cpu")
    try:
        env.reset(seed=[44,100044])
        torch.as_tensor(env.s.coverage.numpy())  # The buffer is part of world state.
        import warp as wp
        memory = wp.to_torch(env.s.coverage)
        memory[0].fill_(1)
        memory[1,0,0] = 1
        other = memory[1].clone()
        env.reset(seed=[200044,None],options={"reset_mask":[True,False]})
        assert memory[0].sum() < 32
        torch.testing.assert_close(memory[1],other,rtol=0,atol=0)
    finally:
        env.close()


def test_memory_snapshot_continues_identically():
    pytest.importorskip("warp")
    from uav_network.environment.gpu.vector import WarpVectorEnv
    c = memory_config()
    left,right = WarpVectorEnv(c,device="cpu"),WarpVectorEnv(c,device="cpu")
    try:
        obs,_ = left.reset(seed=[44,100044])
        for _ in range(3):
            obs,*_ = left.step(vars(left.random_actions(obs["observation"])))
        saved = left.snapshot_state()
        assert saved["arrays"]["coverage"].numel() == 64
        restored = right.restore_state(saved)
        for _ in range(2):
            a = left.random_actions(obs["observation"])
            b = right.random_actions(restored["observation"])
            torch.testing.assert_close(a.continuous,b.continuous,rtol=0,atol=0)
            obs,ra,da,*_ = left.step(vars(a))
            restored,rb,db,*_ = right.step(vars(b))
            torch.testing.assert_close(obs["observation"],restored["observation"],rtol=0,atol=0)
            torch.testing.assert_close(ra,rb,rtol=0,atol=0)
            torch.testing.assert_close(da,db)
    finally:
        left.close();right.close()


def test_legacy_snapshot_missing_inactive_new_fields_remains_loadable():
    pytest.importorskip("warp")
    from uav_network.environment.gpu.vector import WarpVectorEnv
    from uav_network.experiments.environment_checkpoints import validate_environment
    c = memory_config(0)
    left,right = WarpVectorEnv(c,device="cpu"),WarpVectorEnv(c,device="cpu")
    try:
        obs,_ = left.reset(seed=[44,45])
        snapshot = left.snapshot_state()
        snapshot["arrays"].pop("coverage",None)
        for key in ("power_control","link_budget_margin_db"):
            snapshot["config"]["network"].pop(key,None)
        snapshot["config"]["sensor"].pop("coverage_grid_cells",None)
        progress={"collection_steps":0,"next_seed":46}
        validate_environment({"version":1,"world":snapshot,"returns":torch.zeros(2,dtype=torch.float64),
                              "lengths":torch.zeros(2,dtype=torch.int64),"seeds":torch.tensor([44,45]),
                              "next_seed":46},c,progress)
        restored=right.restore_state(snapshot)
        torch.testing.assert_close(obs["observation"],restored["observation"],rtol=0,atol=0)
        changed=deepcopy(c);changed.network.power_control="link_budget"
        incompatible=WarpVectorEnv(changed,device="cpu")
        try:
            with pytest.raises(ValueError,match="config"):
                incompatible.restore_state(snapshot)
        finally:
            incompatible.close()
    finally:
        left.close();right.close()
