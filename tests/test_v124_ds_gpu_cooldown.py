"""GPU/CPU Warp microtest for cooldown on the REAL sensing kernel, no RL policy."""
from dataclasses import replace
import numpy as np
import pytest

wp=pytest.importorskip('warp')
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

CFG='research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml'

@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_real_kernel_local_ds_counts_cooldown(device):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    p=GPUWorldV80Params.from_scenario(
        replace(load_scenario(CFG),sensor_revisit_cooldown_steps=3),envs=1,device=device,seed=44)
    w=WarpParallelMissionV80(p)
    a=wp.zeros((1,6,2),dtype=wp.float32,device=device)
    d=wp.array(np.ones((1,6),dtype=np.int32),dtype=wp.int32,device=device)
    counts=[]
    for t in range(4):
        w.step(a,d)
        own_p=w.network.data['ledger_pos'].numpy()[0]
        own_n=w.network.data['ledger_neg'].numpy()[0]
        count=sum(int(own_p[u,u].sum())+int(own_n[u,u].sum()) for u in range(6))
        counts.append(count)
    assert counts[0]>0
    assert counts[1]==counts[0] and counts[2]==counts[0]
    assert counts[3]>counts[2]
