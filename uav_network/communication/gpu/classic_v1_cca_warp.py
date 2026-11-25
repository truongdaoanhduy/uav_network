"""Actual NVIDIA Warp CCA occupancy check matched to classic UavNetSim-v1.

Upstream utils/util_function.py::check_channel_availability:
A sender cannot access the medium if ANY other drone's channel_state is
occupied within strictly less than config.SENSING_RANGE (750m).
This upstream function does not inspect 802.11 channel IDs at this stage.
"""

import numpy as np
import warp as wp


@wp.kernel
def _classic_v1_carrier_sense(
    xyz: wp.array(dtype=wp.vec3f, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    out_free: wp.array(dtype=wp.int32, ndim=2),
    node_count: int,
    sensing_range_squared: float,
):
    e, u = wp.tid()
    me = xyz[e,u]
    free = wp.int32(1)
    for j in range(node_count):
        if j==u or active[e,j]==0:
            continue
        other=xyz[e,j]
        dx=me[0]-other[0]
        dy=me[1]-other[1]
        dz=me[2]-other[2]
        squared=dx*dx+dy*dy+dz*dz
        # Strict <, exactly as classic v1 check_channel_availability.
        if squared<sensing_range_squared:
            free=0
    out_free[e,u]=free


def compute_cca_free(
    positions_xyz,
    on_air_transmitters,
    *,
    sensing_range_m=750.,
    device="cpu",
) -> np.ndarray:
    """Compute 802.11b carrier-free status for all UAVs in E independent worlds.

    Source matching: the already-occupied upstream channel_states keys,
    ignoring own channel occupancy. This function is a **snapshot rule**;
    it does not schedule DIFS, CW backoff, packet starts, ACK or retries.
    """
    xyz=np.asarray(positions_xyz,dtype=np.float32)
    active=np.asarray(on_air_transmitters)
    if xyz.ndim!=3 or xyz.shape[2]!=3 or xyz.shape[0]<1 or xyz.shape[1]<1:
        raise ValueError("positions_xyz must be [E,N,3] with nonzero E,N")
    if active.shape!=xyz.shape[:2]:
        raise ValueError("on_air_transmitters must have shape [E,N]")
    if not np.isfinite(xyz).all() or not np.isfinite(sensing_range_m) or sensing_range_m<=0:
        raise ValueError("Non-finite coordinates or invalid sensing range")
    if np.any((active!=0)&(active!=1)):
        raise ValueError("on_air_transmitters must be binary")
    n=int(xyz.shape[1])
    wp.init()
    xyz_w=wp.array(xyz,dtype=wp.vec3f,device=device)
    active_w=wp.array(active.astype(np.int32),dtype=wp.int32,device=device)
    free_w=wp.zeros((xyz.shape[0],n),dtype=wp.int32,device=device)
    wp.launch(
        _classic_v1_carrier_sense,dim=(xyz.shape[0],n),
        inputs=[xyz_w,active_w,free_w,n,float(sensing_range_m)**2],
        device=device,
    )
    return free_w.numpy()
