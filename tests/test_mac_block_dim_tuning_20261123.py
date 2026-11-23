"""Pure scheduling tune does not alter action/science/network semantics."""
import os,inspect
import pytest
from uav_network.configuration import load_config
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.communication.gpu.mac import packet_step
CFG='configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml'


def test_launch_block_dim_is_explicitly_opt_in_and_cuda_only(monkeypatch):
    src=inspect.getsource(WarpVectorEnv._launch)
    assert 'UAV_WARP_MAC_BLOCK_DIM' in inspect.getsource(WarpVectorEnv.__init__)
    assert 'kernel is packet_step' in src
    assert 'block_dim=self._mac_block_dim' in src
    assert 'device.type == "cuda"' in src


def test_launch_invalid_mac_block_dim_rejected_at_constructor(monkeypatch):
    c=load_config(CFG);c.runtime.num_envs=1;c.runtime.device='cpu'
    for value in ('abc','-3','3','4096'):
        monkeypatch.setenv('UAV_WARP_MAC_BLOCK_DIM',value)
        with pytest.raises(ValueError,match='UAV_WARP_MAC_BLOCK_DIM'):
            WarpVectorEnv(c,device='cpu')
