"""V89 P0 contract: GCS transmit mask means a reachable, unexpired local route."""
import pytest
import torch

from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.gpu.kernels.mission_v79_reset_obs import _observe_core_v79
from zhang_xue.gpu.state_v67 import wp

@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_gcs_mask_does_not_enable_unroutable_destination(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not present")
    p = GPUWorldV80Params.from_yaml(
        "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml",
        envs=1,device=device,seed=44
    )
    world = WarpParallelMissionV80(p)
    n, w = world.network, world.buffers
    # Hold local valid radio/dirty evidence fixed; route state is the only variable.
    assert w["positions"].numpy()[0,0,0] == pytest.approx(90.0)
    dat = n.data
    wp.to_torch(dat["radio_alive"])[0,0] = 1
    wp.to_torch(dat["dirty_dest"])[0,0,6] = 1
    wp.to_torch(dat["route_hop"])[0,0,6] = -1
    wp.to_torch(dat["route_expiry"])[0,0,6] = 0

    def observe():
        wp.launch(_observe_core_v79, dim=(1,6),
          inputs=[n.state,n.dims,w["positions"],w["velocities"],w["battery"],
                  world.blackout_wp,len(world.blackouts),world.blackout_r,
                  w["obs"],w["mask"],float(p.e_max_j),float(p.queue_capacity),float(p.v_max_mps)],
          device=device)
        return w["obs"].numpy()[0,0,:], w["mask"].numpy()[0,0,:]

    obs, mask = observe()
    assert obs[5] == 1.0
    assert obs[10] == 0.0
    assert int(mask[1]) == 0, "GCS action mask erroneously accepts no valid route"
    from zhang_xue.gpu.kernels.inject_v67 import resolve_compact3_v67
    actions = wp.zeros((1,6), dtype=wp.int32, device=device)
    wp.to_torch(actions)[0,0] = 1
    def resolve():
        wp.launch(resolve_compact3_v67, dim=(1,6),
                  inputs=[n.state,n.dims,actions,w["pos_xy"],n._dest_buf],
                  device=device)
        return int(n._dest_buf.numpy()[0,0])
    assert resolve() == -1, "Resolver must fail closed for invalid route"

    wp.to_torch(dat["route_hop"])[0,0,6] = 6
    wp.to_torch(dat["route_expiry"])[0,0,6] = 1_000_000
    obs, mask = observe()
    assert obs[10] == 1.0
    assert int(mask[1]) == 1
    assert resolve() == 6
