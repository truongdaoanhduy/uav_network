"""Batched NVIDIA Warp Friis SNR kernel pinned to classic UavNetSim-v1 PHY.

This computes free-space per-link SNR without concurrent interferers,
matching classic v1 phy/large_scale_fading.py::general_path_loss.
NOT an end-to-end PHY/MAC or UavNetSim-v1 replacement; GPU is optional.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _friis_pair_snr(
    positions: wp.array(dtype=wp.vec3f),
    snr_out: wp.array(dtype=wp.float32),
    n: int,
    tx_power_w: float,
    carrier_frequency_hz: float,
    noise_w: float,
):
    tid=wp.tid()
    a=tid//n
    b=tid%n
    xyz_a=positions[a]
    xyz_b=positions[b]
    dx=xyz_a[0]-xyz_b[0]
    dy=xyz_a[1]-xyz_b[1]
    dz=xyz_a[2]-xyz_b[2]
    dist_squared=dx*dx+dy*dy+dz*dz
    loss=wp.float32(1.0)
    if dist_squared>wp.float32(0.0):
        d=wp.sqrt(dist_squared)
        term=wp.float32(300000000.0)/(
            wp.float32(12.566370614359172)*carrier_frequency_hz*d
        )
        loss=term*term
    snr=tx_power_w*loss/noise_w
    snr_out[tid]=wp.float32(4.3429448190325175)*wp.log(snr)


def compute_classic_v1_friis_snr(
    positions_xyz,
    tx_power_w: float = 0.5,
    carrier_frequency_hz: float = 2.4e9,
    noise_w: float = 4.0e-11,
    device: str = "cpu",
) -> np.ndarray:
    """Return [N,N] FP32 dB SNR matrix via a real Warp kernel."""
    pos=np.asarray(positions_xyz,dtype=np.float32)
    if pos.ndim!=2 or pos.shape[1]!=3 or not np.isfinite(pos).all():
        raise ValueError("positions_xyz must be finite [N,3]")
    if pos.shape[0]<1 or tx_power_w<=0 or carrier_frequency_hz<=0 or noise_w<=0:
        raise ValueError("Invalid SNR kernel parameters")
    wp.init()
    n=int(pos.shape[0])
    pos_wp=wp.array(pos,dtype=wp.vec3f,device=device)
    out_wp=wp.zeros((n*n,),dtype=wp.float32,device=device)
    wp.launch(
        _friis_pair_snr,dim=n*n,
        inputs=[pos_wp,out_wp,n,tx_power_w,carrier_frequency_hz,noise_w],
        device=device,
    )
    return out_wp.numpy().reshape(n,n)
