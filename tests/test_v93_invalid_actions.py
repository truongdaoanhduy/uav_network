"""Nonfinite controller outputs must not corrupt persistent flight state."""
import numpy as np
import pytest
import warp as wp
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_nonfinite_motion_fails_closed_before_physics_commit(bad, device):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    p = GPUWorldV80Params.from_yaml("research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml", envs=1, device=device)
    w = WarpParallelMissionV80(p)
    before = w.buffers["positions"].numpy().copy()
    a = np.zeros((1,6,2),dtype=np.float32)
    a[0,0,0] = bad
    w.step(wp.array(a,dtype=wp.float32,device=device), wp.zeros((1,6),dtype=wp.int32,device=device))
    assert np.isfinite(w.buffers["positions"].numpy()).all()
    np.testing.assert_array_equal(w.buffers["positions"].numpy(), before)
    assert int(w.buffers["failed"].numpy()[0]) == 1
    assert int(w.buffers["done"].numpy()[0]) == 1


def test_invalid_communication_action_cannot_be_silently_accepted():
    p = GPUWorldV80Params.from_yaml("research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml", envs=1, device="cpu")
    w = WarpParallelMissionV80(p)
    d = np.zeros((1,6),dtype=np.int32)
    d[0,0] = 3
    w.step(wp.zeros((1,6,2),dtype=wp.float32,device="cpu"), wp.array(d,dtype=wp.int32,device="cpu"))
    assert int(w.buffers["failed"].numpy()[0]) == 1
    assert int(w.buffers["done"].numpy()[0]) == 1

