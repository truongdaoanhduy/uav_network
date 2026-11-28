import json, pathlib, numpy as np, warp as wp
from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
from zhang_xue.gpu.state_v67 import SimDimsV67,NetworkBuffersV67
from zhang_xue.gpu.kernels.mac_v67 import _rand_slot_v67

@wp.kernel
def count_slots(s:NetworkBuffersV67, d:SimDimsV67, h:wp.array(dtype=wp.int32), retry:int, loops:int):
    tid=wp.tid()
    e=tid//d.n_uav
    r=tid%d.n_uav
    for k in range(loops):
        slot=_rand_slot_v67(s,d,e,r,retry)
        wp.atomic_add(h,slot,1)

rows=[]
for retry in (0,2):
    for inclusive in (0,1):
        net=WarpCompressedRadioNetworkV67(num_envs=128,num_uavs=6,grid_w=2,q_cap=8,enable_data_mule=False,enable_peer=False,enable_peer_telemetry=False,cw_inclusive_upper=inclusive,seed=20261127,device='cuda:0')
        h=wp.zeros(1024,dtype=wp.int32,device='cuda:0')
        wp.launch(count_slots,dim=128*6,inputs=[net.state,net.dims,h,retry,256],device='cuda:0')
        x=h.numpy(); cw=min(1023,((net.dims.cw_min+1)<<retry)-1); span=cw+1 if inclusive else cw
        assert x.sum()==128*6*256 and np.sum(x[span:])==0 and x[span-1]>0
        rows.append({'retry':retry,'cw':cw,'inclusive':inclusive,'span':span,'draws':int(x.sum()),'out_of_support':int(np.sum(x[span:])),'last_slot_count':int(x[span-1]),'chi_square':float((((x[:span]-x.sum()/span)**2)/(x.sum()/span)).sum())})
out={'status':'PASS','scope':'direct CUDA Warp kernel slot support only, NOT UavNetSim parity','runs':rows}
print(json.dumps(out,indent=2))
pathlib.Path('/workspace/v96_final_evidence_20261127/v97_s12a_backoff_gpu.json').write_text(json.dumps(out,indent=2)+'\n')
