"""FP32 Warp receiver-SINR with simultaneous co-channel UAV transmissions.

Matches the interference geometry and adjacent-channel rule used by classic
UavNetSim-v1, *given* pre-scheduled frame starts and durations. This is a
piece of a future collision-aware Warp surrogate, NOT a CSMA/CA or retry
scheduler and NOT full SimPy↔Warp equivalence.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.func
def _friis_gain(pos_tx: wp.vec3f, pos_rx: wp.vec3f, carrier_hz: float) -> float:
    dx=pos_tx[0]-pos_rx[0]
    dy=pos_tx[1]-pos_rx[1]
    dz=pos_tx[2]-pos_rx[2]
    d2=dx*dx+dy*dy+dz*dz
    if d2<=0.0:
        return 1.0
    d=wp.sqrt(d2)
    term=300000000.0/(12.566370614359172*carrier_hz*d)
    return term*term


@wp.kernel
def _batched_link_sinr(
    positions: wp.array(dtype=wp.vec3f, ndim=2),
    tx_src: wp.array(dtype=wp.int32, ndim=2),
    rx_dst: wp.array(dtype=wp.int32, ndim=2),
    channels: wp.array(dtype=wp.int32, ndim=2),
    start_s: wp.array(dtype=wp.float32, ndim=2),
    duration_s: wp.array(dtype=wp.float32, ndim=2),
    valid: wp.array(dtype=wp.int32, ndim=2),
    out_sinr_db: wp.array(dtype=wp.float32, ndim=2),
    out_success: wp.array(dtype=wp.int32, ndim=2),
    n_links: int,
    tx_power_w: float,
    carrier_hz: float,
    noise_w: float,
    threshold_db: float,
):
    e,i=wp.tid()
    if valid[e,i]==0:
        out_sinr_db[e,i]=-1.0e9
        out_success[e,i]=0
        return
    src=tx_src[e,i]
    dst=rx_dst[e,i]
    rx_pos=positions[e,dst]
    signal=tx_power_w*_friis_gain(positions[e,src],rx_pos,carrier_hz)
    i_start=start_s[e,i]
    i_end=i_start+duration_s[e,i]
    interference=wp.float32(0.0)
    for j in range(n_links):
        if j==i or valid[e,j]==0:
            continue
        # Classic v1 ChannelAssigner: adjacent 802.11b channels interfere
        # iff abs(channel_id1 - channel_id2) < 5.
        channel_diff=wp.abs(channels[e,i]-channels[e,j])
        if channel_diff>=5:
            continue
        j_start=start_s[e,j]
        j_end=j_start+duration_s[e,j]
        overlap=i_start<j_end and j_start<i_end
        if overlap:
            interferer=tx_src[e,j]
            interference+=tx_power_w*_friis_gain(
                positions[e,interferer],rx_pos,carrier_hz
            )
    sinr_lin=signal/(noise_w+interference)
    db=4.3429448190325175*wp.log(sinr_lin)
    out_sinr_db[e,i]=db
    if db>=threshold_db:
        out_success[e,i]=1
    else:
        out_success[e,i]=0


def compute_simultaneous_link_sinr(
    positions_xyz,
    transmitter_ids,
    receiver_ids,
    channels,
    start_s,
    duration_s,
    valid,
    *,
    tx_power_w: float = .5,
    carrier_frequency_hz: float = 2.4e9,
    noise_power_w: float = 4e-11,
    snr_threshold_db: float = 6.,
    device: str = "cpu",
):
    """Run a batch of independently scheduled E worlds, up to M TXs each.

    Inputs: positions [E,N,3], other arrays [E,M]. Only PHY SINR is
    computed; **no** carrier sensing, MAC scheduler, ACKs, timeouts, queue
    lengths or retransmissions are represented.
    """
    positions=np.asarray(positions_xyz,dtype=np.float32)
    args=[np.asarray(x) for x in (
        transmitter_ids,receiver_ids,channels,start_s,duration_s,valid
    )]
    tx,rx,ch,starts,durations,mask=args
    if positions.ndim!=3 or positions.shape[2]!=3:
        raise ValueError("positions_xyz must be [E,N,3]")
    if not np.isfinite(positions).all():
        raise ValueError("Non-finite position")
    if tx.ndim!=2 or tx.shape!=rx.shape or any(a.shape!=tx.shape for a in (ch,starts,durations,mask)):
        raise ValueError("Transmission arrays must have identical [E,M] shapes")
    e,n,_=positions.shape
    m=tx.shape[1]
    if e!=tx.shape[0] or e<1 or n<2 or m<1:
        raise ValueError("World and transmission shapes mismatch")
    if any(not np.isfinite(a).all() for a in (starts,durations)):
        raise ValueError("Non-finite time input")
    if np.any(tx<0) or np.any(tx>=n) or np.any(rx<0) or np.any(rx>=n):
        raise ValueError("Invalid source/destination index")
    if np.any((ch<1)|(ch>14)) or np.any((mask!=0)&(mask!=1)):
        raise ValueError("Invalid 802.11 channel or valid mask")
    if np.any(starts<0) or np.any(durations<=0):
        raise ValueError("Start time must be nonnegative and duration positive")
    if any(not np.isfinite(v) or v<=0 for v in (
        tx_power_w,carrier_frequency_hz,noise_power_w,
    )) or not np.isfinite(snr_threshold_db):
        raise ValueError("Invalid PHY parameters")
    wp.init()
    pos_w=wp.array(positions,dtype=wp.vec3f,device=device)
    tx_w=wp.array(tx.astype(np.int32),dtype=wp.int32,device=device)
    rx_w=wp.array(rx.astype(np.int32),dtype=wp.int32,device=device)
    ch_w=wp.array(ch.astype(np.int32),dtype=wp.int32,device=device)
    st_w=wp.array(starts.astype(np.float32),dtype=wp.float32,device=device)
    du_w=wp.array(durations.astype(np.float32),dtype=wp.float32,device=device)
    mask_w=wp.array(mask.astype(np.int32),dtype=wp.int32,device=device)
    sinr_w=wp.zeros((e,m),dtype=wp.float32,device=device)
    success_w=wp.zeros((e,m),dtype=wp.int32,device=device)
    wp.launch(
        _batched_link_sinr,dim=(e,m),
        inputs=[pos_w,tx_w,rx_w,ch_w,st_w,du_w,mask_w,sinr_w,success_w,
                m,tx_power_w,carrier_frequency_hz,noise_power_w,snr_threshold_db],
        device=device,
    )
    return {"sinr_db":sinr_w.numpy(),"success":success_w.numpy()}
