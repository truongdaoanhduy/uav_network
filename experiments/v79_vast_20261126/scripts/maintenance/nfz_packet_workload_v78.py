#!/usr/bin/env python3
"""Gate2: reproducible real packet/evidence delivery around a two-cylinder NFZ.

The PHY changes with building obstruction, and application data genuinely traverses
UAV0 -> UAV1 -> GCS. Includes CPU reference and optional Warp CPU/CUDA.
The MACs are distinct implementations; endpoint contracts are verified, not full
per-event equality or trained-policy equivalence.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/zhang_xue/src')]
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.communication.common import link_path_gain
from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np
from scripts.gpu.packet_workload import run_packet_workload

# Two buildings block UAV0->GCS, but both legs via UAV1 have clear 3D LOS.
POSITIONS=np.asarray([[[100.,100.,50.],[220.,180.,50.],[400.,100.,15.]]],dtype=np.float32)
NFZ=((200.,100.,28.,85.),(300.,100.,28.,85.))
MODES=('binary_nlos','multi_obstacle','rf_opaque')
NOISE_W=4.0e-11
TX_POWER_W=0.1


def check_physics(mode: str, loss_db: float) -> dict:
    power=np.asarray([[TX_POWER_W*link_path_gain(POSITIONS[0,i],POSITIONS[0,j],NFZ,(),
        obstacle_model=mode,per_building_loss_db=loss_db) if i!=j else 0.0 for j in range(3)]
        for i in range(3)],dtype=np.float64)
    power_np=TX_POWER_W*compute_channel_gain_v67_np(POSITIONS,NFZ,(),0,20,50.0,
            obstacle_model=mode,per_building_loss_db=loss_db)[0]
    np.testing.assert_allclose(power,power_np,rtol=3e-4,atol=1e-16)
    snr_threshold=10**(6.0/10.0)
    if not (power[0,2] < NOISE_W*snr_threshold and
            power[0,1] > NOISE_W*snr_threshold and power[1,2] > NOISE_W*snr_threshold):
        raise AssertionError(f'NFZ route geometry not appropriate: mode={mode}, power={power}')
    return {'direct_u0_gcs_rx_power_w':float(power[0,2]),
            'u0_peer_rx_power_w':float(power[0,1]),
            'peer_gcs_rx_power_w':float(power[1,2]),
            'loss_db':float(loss_db),'reference_max_abs_difference':float(np.max(abs(power-power_np)))}


def cpu_experiment(mode: str, loss_db: float, seed: int) -> dict:
    n=SevenRadioCsmaCaNetwork(num_uavs=2,grid_w=20,tx_power_w=TX_POWER_W,
        noise_power_w=NOISE_W,queue_cap=64,buildings=NFZ,blackout_centers=[],
        aoi_ewma_end_to_end=True,enable_data_mule_forwarding=True,
        rf_obstacle_model=mode,rf_per_building_loss_db=loss_db,seed=seed)
    n.reset(POSITIONS[0].astype(np.float64),episode_id=seed)
    assert n.run_until_us(1_000_000), 'CPU HELLO/DV discovery event overflow'
    n.record_local_observation(0,42,True,1)
    first=n.get_dirty_entries_non_oracle(0,1)
    assert first and n.enqueue_app_data(0,1,first), 'Failed to enqueue peer evidence'
    assert n.run_until_us(1_500_000), 'CPU first-hop event overflow'
    if int(n.ledger_pos[1,0,42])!=256:
        raise AssertionError('UAV1 did not obtain UAV0 evidence')
    later=n.get_dirty_entries_non_oracle(1,2)
    assert later and n.enqueue_app_data(1,2,later), 'Failed to enqueue GCS evidence'
    assert n.run_until_us(2_000_000), 'CPU second-hop event overflow'
    assert int(n.ledger_pos[2,0,42])==256 and int(n.gcs_latest_u_s[42])==1
    assert n.stats['e2e_ack_rx']>=2 and n.stats['data_delivered_gcs']>=1
    return {'endpoint':'PASS','gcs_evidence_count_uq88':int(n.ledger_pos[2,0,42]),
            'gcs_generation_s':int(n.gcs_latest_u_s[42]), 'network_stats':dict(n.stats)}


def warp_experiment(mode: str, loss_db: float, device: str, envs: int, seed: int) -> dict:
    import warp as wp
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    if device.startswith('cuda') and not wp.get_cuda_devices():
        return {'status':'BLOCKED_NO_CUDA','gpu_executed':False}
    n=WarpCompressedRadioNetworkV67(envs,num_uavs=2,grid_w=20,device=device,seed=seed,
                                    q_cap=64,enable_peer=True,enable_data_mule=True)
    pos=np.broadcast_to(POSITIONS,(envs,3,3)).copy()
    gain_ref=compute_channel_gain_v67_np(pos,NFZ,(),0,20,50.0,
                  obstacle_model=mode,per_building_loss_db=loss_db)
    # Exactly match CPU's noise and TX power; Warp defaults are intentionally higher noise.
    # Explicitly pass step_kwargs to avoid mixing two independent PHY ablations.
    n.update_phy_from_positions(pos,NFZ,(),0,50.0,obstacle_model=mode,per_building_loss_db=loss_db)
    np.testing.assert_allclose(n.data['gain'].numpy(),gain_ref,rtol=2e-4,atol=1e-15)
    kwargs={'tx_power_w':TX_POWER_W,'noise_w':NOISE_W}
    snap=run_packet_workload(n,step_kwargs=kwargs,buildings=NFZ,
         cell_size_m=50.0,positions=pos,obstacle_model=mode,per_building_loss_db=loss_db)
    return {'status':'PASS','gpu_executed':device.startswith('cuda'),
            'device':device,'num_envs':envs,
            'min_gcs_delivery_events':int(snap['stats_gcs'].min()),
            'min_peer_delivery_events':int(snap['stats_peer'].min()),
            'max_event_count':int(snap['event_count'].max()),
            'max_phy_reference_abs_difference':float(np.max(abs(n.data['gain'].numpy()-gain_ref))),
            'application_provenance_verified':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['none','cpu','cuda:0','cuda:1'],default='none')
    p.add_argument('--envs',type=int,default=2)
    p.add_argument('--seed',type=int,default=44)
    p.add_argument('--loss-db',type=float,default=20.0)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.envs<1 or not math.isfinite(a.loss_db) or a.loss_db<0: p.error('invalid envs/loss-db')
    result={'gate':'G2_PACKET_NFZ_ROUTE_ACTUAL_EVENTS','mode_results':{},'seed':a.seed,
            'warning':'Endpoints/PHY, not statistically verified MAC-event parity'}
    for mode in MODES:
        obj={'phy':check_physics(mode,a.loss_db),
             'cpu':cpu_experiment(mode,a.loss_db,a.seed)}
        if a.device != 'none': obj['warp']=warp_experiment(mode,a.loss_db,a.device,a.envs,a.seed)
        result['mode_results'][mode]=obj
    blocked=any(obj.get('warp',{}).get('status','PASS')!='PASS' for obj in result['mode_results'].values())
    result['status']='BLOCKED' if blocked else 'PASS'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    return 3 if blocked else 0

if __name__=='__main__': raise SystemExit(main())
