"""Manual Warp CPU event-level checks; run on GPU too after CPU results agree.

Requires warp-lang; not included in basic pytest because event compile can take >1min.
"""
import sys
from pathlib import Path
import numpy as np
import warp as wp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from v56_warp_7radio_mac import NetworkBuffers,SHAPES
from v56_warp_reset_kernels import reset_all
from v56_warp_injection_kernels import _add_observation,_inject
from v56_warp_event_kernels import _step


def make_worlds(envs,seed=44,device='cpu'):
    s=NetworkBuffers()
    for name,(shape,dtype) in SHAPES.items():
        typ={np.dtype(np.int32):wp.int32,np.dtype(np.uint32):wp.uint32,
             np.dtype(np.uint16):wp.uint16,np.dtype(np.float32):wp.float32}[np.dtype(dtype)]
        setattr(s,name,wp.zeros((envs,*shape) if shape!=(1,) else (envs,),dtype=typ,device=device))
    reset_all(s,np.ones(envs,dtype=np.uint8),envs,seed,device)
    return s

def step(s,duration_us,envs=2,device='cpu'):
    wp.launch(_step,dim=envs,inputs=[s,duration_us,4096,.5,2e-11,4e-11,3.9810717],device=device)


def run(device='cpu'):
    e=2;s=make_worlds(e,device=device)
    g=np.zeros((e,7,7),np.float32)
    # world 0: ONLY 0--1--6 reachable; world 1: disconnected.
    for a,b in ((0,1),(1,6)):
        g[0,a,b]=1e-8;g[0,b,a]=1e-8
    wp.copy(s.gain,wp.array(g,dtype=wp.float32,device=device))
    step(s,1_000_000,device=device)
    step(s,600_000,device=device)
    routes=s.route_hop.numpy()
    print('ROUTES',routes[0,0,6],routes[1,0,6],flush=True)
    assert routes[0,0,6]==1 and routes[1,0,6]==-1
    cells=np.full((e,6),-1,np.int32);cells[:,0]=12
    positives=np.zeros_like(cells);positives[:,0]=1
    wp.launch(_add_observation,dim=(e,6),inputs=[s,wp.array(cells,dtype=wp.int32,device=device),
        wp.array(positives,dtype=wp.int32,device=device),2],device=device)
    actions=np.zeros((e,6),np.int32);actions[:,0]=6
    wp.launch(_inject,dim=(e,6),inputs=[s,wp.array(actions,dtype=wp.int32,device=device),
        wp.array(np.zeros(e,np.int32),dtype=wp.int32,device=device)],device=device)
    step(s,1_000_000,device=device)
    obs=s.ledger_pos.numpy()
    gcs=s.stats_gcs.numpy();ack=s.stats_e2e.numpy();mac=s.stats_mac_ack.numpy();overflow=s.overflow.numpy()
    print('2HOP',gcs,ack,mac,overflow,flush=True)
    assert gcs[0]>=1 and gcs[1]==0
    assert ack[0]>=1 and mac[0]>=4
    assert obs[0,6,0,12]>=1 and obs[0,1,0,12]==0
    assert overflow.tolist()==[0,0]
    reset_all(s,np.array([1,0],np.uint8),e,44,device)
    obs2=s.ledger_pos.numpy()
    assert obs2[0].sum()==0 and obs2[1,0,0,12]==1
    print('PASS two-hop relay, E2E ACK, no premature fusion, reset isolation',flush=True)
    # Clean up Warp-owned array fields before interpreter shutdown.
    wp.synchronize_device(device)
    del s
    import gc
    gc.collect()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu',help='cpu or cuda:0 (requires CUDA driver)')
    run(parser.parse_args().device)
