"""Actual Warp CPU worlds must implement the same XYZ camera as Native."""
from copy import deepcopy
import numpy as np
import pytest
import torch
import warp as wp
from test_xyz_sensing import xyz_config
from uav_network.environment.env import SwarmSearchEnv,Action
from uav_network.environment.gpu.vector import WarpVectorEnv


def cfg(count=1):
    c=xyz_config(n=2,targets=2,obstacles=1)
    c.runtime.num_envs=count;c.runtime.device="cpu";c.runtime.environment_backend="warp_packet"
    return c


def test_warp_native_xyz_camera_and_motion_agree():
    c=cfg();native=SwarmSearchEnv(c);gpu=WarpVectorEnv(c,device="cpu")
    try:
        no,_=native.reset(seed=44);go,_=gpu.reset(seed=44)
        np.testing.assert_allclose(gpu.s.altitudes.numpy()[0],native.positions[:,2])
        np.testing.assert_allclose(gpu.s.belief.numpy()[0].reshape(native.belief.shape),native.belief,atol=1e-10)
        np.testing.assert_allclose(go["observation"][0].numpy(),no,atol=1e-6)
        rng=np.random.default_rng(7)
        for _ in range(4):
            a=Action(rng.uniform(-.3,.3,(2,4)).astype(np.float32),np.zeros(2,dtype=np.int64))
            no,nr,_,_,_=native.step(a)
            go,gr,_,_,_=gpu.step(dict(continuous=a.continuous[None],destination=a.destination[None]))
            np.testing.assert_allclose(gpu.s.positions.numpy()[0],native.positions[:,:2],atol=1e-7)
            np.testing.assert_allclose(gpu.s.altitudes.numpy()[0],native.positions[:,2],atol=1e-7)
            np.testing.assert_allclose(gpu.s.belief.numpy()[0].reshape(native.belief.shape),native.belief,atol=1e-9)
            np.testing.assert_allclose(go["observation"][0].numpy(),no,atol=1e-6)
            assert float(gr[0])==pytest.approx(nr,abs=1e-5)
    finally:native.close();gpu.close()


def test_masked_xyz_reset_leaves_other_world_untouched():
    v=WarpVectorEnv(cfg(2),"cpu")
    try:
        v.reset(seed=44);before=v.snapshot_state()
        v.reset(seed=[99,None],options={"reset_mask":[True,False]})
        after=v.snapshot_state()
        for name in ("belief","visited","positive","altitudes","vertical_velocity","sensor_tick","sensor_counts"):
            assert torch.equal(before["arrays"][name][1],after["arrays"][name][1]),name
    finally:v.close()


def test_xyz_resume_preserves_next_sensor_draw_and_rejects_nan_atomically():
    v=WarpVectorEnv(cfg(),"cpu")
    try:
        v.reset(seed=44);saved=v.snapshot_state()
        action=dict(continuous=np.zeros((1,2,4),np.float32),destination=np.zeros((1,2),np.int64))
        expected=v.step(action)[0]["observation"].clone()
        belief=wp.to_torch(v.s.belief).clone()
        v.restore_state(saved);actual=v.step(action)[0]["observation"]
        assert torch.equal(actual,expected)
        assert torch.equal(wp.to_torch(v.s.belief),belief)
        bad=deepcopy(saved);bad["arrays"]["altitudes"][0,0]=float("nan")
        before=v.snapshot_state()
        with pytest.raises(ValueError):v.restore_state(bad)
        for name,val in before["arrays"].items():assert torch.equal(wp.to_torch(getattr(v.s,name)).cpu(),val),name
    finally:v.close()
