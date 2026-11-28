#!/usr/bin/env python3
"""True randomized CSMA hidden-terminal experiment with obstacle-aware NFZ.

U0 and U1 both reach GCS by unobstructed LoS; they cannot hear each other's
CCA across one 3D building. Compare to geometry with the building removed.
Uses the *same* seeds and traffic offered to both scenarios, no artificial
forced simultaneous transmission. PDR can remain 100% thanks to MAC retries.
The measured causal quantity is excess contention/retries from blocked sensing.
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
from zhang_xue.communication.common import link_path_gain
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork

POSITIONS=np.asarray([[380.,500.,50.],[620.,500.,50.],[500.,800.,15.]],dtype=np.float64)
BUILDING=((500.,500.,60.,80.),)
NOISE_W=4.38957709293155e-13
TX_POWER_W=0.1
CCA_W=10**((-82.-30.)/10.)
SNR_TH=10**(6./10.)

def geometry_checks():
    def power(i,j,buildings):return TX_POWER_W*link_path_gain(POSITIONS[i],POSITIONS[j],buildings,())
    cross_blocked=power(0,1,BUILDING)
    cross_clear=power(0,1,())
    hops=[power(i,2,BUILDING) for i in range(2)]
    assert cross_blocked<CCA_W<cross_clear
    assert min(hops)>NOISE_W*SNR_TH
    return {'cca_threshold_w':CCA_W,'cross_blocked_w':cross_blocked,
            'cross_clear_w':cross_clear,'toward_gcs_w':hops,
            'physical_hidden_terminal_verified':True}

def simulate(seed:int,building:bool):
    n=SevenRadioCsmaCaNetwork(num_uavs=2,grid_w=20,tx_power_w=TX_POWER_W,
        noise_power_w=NOISE_W,queue_cap=64,
        buildings=BUILDING if building else (),blackout_centers=[],
        aoi_ewma_end_to_end=True,enable_data_mule_forwarding=True,
        seed=int(seed))
    n.reset(POSITIONS,episode_id=int(seed))
    if not n.run_until_us(1_000_000):raise RuntimeError('HELLO discovery event overflow')
    for u in range(2):
        n.record_local_observation(u,42+u,True,1)
        entries=n.get_dirty_entries_non_oracle(u,2)
        if not entries or not n.enqueue_app_data(u,2,entries):
            raise RuntimeError('missing original GCS packet')
    if not n.run_until_us(2_000_000):raise RuntimeError('MAC event overflow')
    return {'sinr_drops':n.stats['sinr_drops'],
            'mac_ack_timeout':n.stats['mac_ack_timeout'],
            'mac_ack_ok':n.stats['mac_ack_ok'],
            'gcs_packets':n.stats['data_delivered_gcs'],
            'origin0_delivered':bool(n.ledger_pos[2,0,42]),
            'origin1_delivered':bool(n.ledger_pos[2,1,43])}

def run(seeds:int,offset:int):
    if not (5<=seeds<=1000):raise ValueError('seeds out of safe range [5,1000]')
    geom=geometry_checks()
    baseline=[];obstructed=[]
    for seed in range(offset,offset+seeds):
        baseline.append(simulate(seed,False))
        obstructed.append(simulate(seed,True))
    def summarize(data):
        return {k:{'mean':float(np.mean([d[k] for d in data])),
                   'min':int(np.min([d[k] for d in data])),
                   'max':int(np.max([d[k] for d in data]))}
                for k in ('sinr_drops','mac_ack_timeout','mac_ack_ok','gcs_packets')}
    a=summarize(baseline);b=summarize(obstructed)
    if not(b['sinr_drops']['mean']>a['sinr_drops']['mean'] and
           b['mac_ack_timeout']['mean']>a['mac_ack_timeout']['mean']):
        raise AssertionError('NFZ did not increase collisions/retries over same-seed baseline')
    if not all(x['origin0_delivered'] and x['origin1_delivered'] for x in baseline+obstructed):
        raise AssertionError('Lost data in selected low-load hidden terminal experiment')
    return {'status':'PASS','seeds':seeds,'offset':offset,'geometry':geom,
            'clear_los':a,'rf_obstacle_nfz':b,
            'paired_mean_excess_sinr_drops':b['sinr_drops']['mean']-a['sinr_drops']['mean'],
            'paired_mean_excess_mac_ack_timeout':b['mac_ack_timeout']['mean']-a['mac_ack_timeout']['mean'],
            'warning':'CPU randomized CSMA microbenchmark; not a CPU-Warp MAC event parity proof or all-load PDR claim'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--seeds',type=int,default=64)
 p.add_argument('--offset',type=int,default=100)
 p.add_argument('--output',type=Path,default=None)
 a=p.parse_args()
 d=run(a.seeds,a.offset)
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps(d,indent=2))
