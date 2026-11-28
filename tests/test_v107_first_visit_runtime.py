"""Actual CPU and Warp reward checks: one reward per locally observed cell."""
import numpy as np
import pytest
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.environment.core import ZhangXue7NodePacketEnv

SC = "research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml"

def test_cpu_first_visit_reward():
    cfg = load_scenario(SC)
    env = ZhangXue7NodePacketEnv(cfg, seed=44)
    env.reset(seed=44)
    a = np.zeros((6, 2), dtype=np.float32)
    d = np.zeros(6, dtype=np.int64)
    *_, info = env.step(a, d)
    assert info["r_loc_team"] > 0
    *_, info = env.step(a, d)
    assert info["r_loc_team"] == 0

@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_first_visit_reward(device):
    import warp as wp
    if device == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
    p = GPUWorldV80Params.from_yaml(SC, envs=2, device=device, seed=44)
    w = WarpParallelMissionV80(p)
    a = wp.zeros((2,6,2), dtype=wp.float32, device=device)
    d = wp.zeros((2,6), dtype=wp.int32, device=device)
    w.step(a,d)
    assert np.all(w.buffers["reward_components"].numpy()[:,0] > 0)
    w.step(a,d)
    np.testing.assert_array_equal(w.buffers["reward_components"].numpy()[:,0], 0)
