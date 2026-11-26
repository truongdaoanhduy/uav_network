"""V89 deterministic reward-accounting contract, independent of RL algorithms."""
import pytest
import numpy as np
import torch
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.gpu.state_v67 import wp

@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_reward_components_are_auditable_and_sum_exactly(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    p=GPUWorldV80Params.from_yaml(
        "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml",
        envs=2, device=device, seed=44
    )
    world=WarpParallelMissionV80(p)
    assert "reward_components" in world.buffers
    a=wp.zeros((2,6,2),dtype=wp.float32,device=device)
    d=wp.zeros((2,6),dtype=wp.int32,device=device)
    for i in range(3):
        world.step(a,d)
        total=world.buffers["reward"].numpy().astype(np.float64)
        parts=world.buffers["reward_components"].numpy().astype(np.float64)
        assert parts.shape==(2,8)
        assert np.isfinite(parts).all()
        np.testing.assert_allclose(parts.sum(axis=1),total,rtol=1e-5,atol=1e-4)
        assert np.all(parts[:,0]>=0)
        assert np.all(parts[:,1]>=0)
        assert np.all(parts[:,2:]<=0)
    world.reset(np.array([1,0],dtype=np.uint8))
    c=world.buffers["reward_components"].numpy()
    np.testing.assert_array_equal(c[0],np.zeros(8))
