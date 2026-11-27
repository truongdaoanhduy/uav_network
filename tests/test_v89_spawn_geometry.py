"""V89 scenario initialization: every UAV starts collision-free with braking reserve."""
import math
import numpy as np
import pytest
import torch
from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_spawn_has_two_sided_braking_margin_and_no_obstacle(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg=GPUWorldV80Params.from_yaml(
        "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml",
        envs=3,device=device,seed=44
    )
    world=WarpParallelMissionV80(cfg)
    xyz=world.buffers["positions"].numpy()[:,:6,:]
    assert np.all(xyz[...,2] == 50.0)
    # Worst-case two opposing UAVs flying at v_max and braking with a_max:
    # both need v^2/(2a), plus min physical separation.
    min_dist=cfg.uav_safe_distance_m + cfg.v_max_mps**2 / cfg.a_max_mps2
    buildings,blackouts,blackout_r=scale_canonical_geometry(1000.0,20)
    for positions in xyz:
        for u in range(6):
            x,y,_=positions[u]
            assert 10+min_dist/2 < x < 990-min_dist/2
            assert 10+min_dist/2 < y < 990-min_dist/2
            for bx,by,rad,height in buildings:
                assert math.hypot(x-bx,y-by) > rad + 2.0
            cx=int(x//50);cy=int(y//50)
            assert all(abs(cx-bx)+abs(cy-by)>blackout_r for bx,by in blackouts)
            for v in range(u+1,6):
                d=float(np.linalg.norm(positions[u,:2]-positions[v,:2]))
                assert d >= min_dist, f"spawn pair={u},{v} {d:.3f}m < {min_dist:.3f}m"
