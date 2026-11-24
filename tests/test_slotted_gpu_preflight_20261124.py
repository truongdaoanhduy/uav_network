"""GPU-only preflight for raw Warp kernels, not an end-to-end GPU simulator proof.

This test must run on the Kaggle T4 instance. A skip outside CUDA is NOT a pass.
"""
import numpy as np
import pytest

wp = pytest.importorskip('warp')
from uav_network.communication.gpu.phy_sinr_fp32 import compute_phy_matrices_fp32
from uav_network.communication.gpu.mac_slotted_fp32 import mac_slotted_step_fp32
from uav_network.environment.gpu.ds_aoi_belief_fp32 import ds_sense_and_energy_step_fp32


def test_cuda_real_kernel_step_jit_and_gcs_true_scoring():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('NO CUDA GPU: full GPU preflight has not been performed')
    dev='cuda:0'
    def a(x, dtype): return wp.array(np.asarray(x), dtype=dtype, device=dev)
    def iz(shape): return wp.zeros(shape, dtype=wp.int32, device=dev)
    def fz(shape): return wp.zeros(shape, dtype=wp.float32, device=dev)
    n=2
    pos=np.zeros((n,7,3),dtype=np.float32)
    pos[:,:6]=[60.,60.,50.]
    pos[:,6]=[40.,40.,10.]
    nodes=a(pos,wp.vec3f)
    obstacles=a(np.zeros((n,22,4),np.float32),wp.vec4f)
    dist,rx=fz((n,7,7)),fz((n,7,7))
    los,cca=iz((n,7,7)),iz((n,7,7))
    wp.launch(compute_phy_matrices_fp32,dim=n,inputs=[nodes,obstacles,0,dist,los,rx,cca],device=dev)
    assert float(rx.numpy()[0,0,6])>0
    qcount=iz((n,6));q_ids=iz((n,6,32));q_ts=fz((n,6,32));q_retries=iz((n,6,32));q_hops=iz((n,6,32));q_visit=iz((n,6,32))
    truths=np.full((n,400),-1,dtype=np.int32)
    truths[:,0]=11
    target=a(truths,wp.int32)
    reported=iz((n,400));pos_cnt=iz((n,400));neg_cnt=iz((n,400));mf=fz((n,400));mu=a(np.ones((n,400),np.float32),wp.float32)
    battery=a(np.full((n,6),150000,np.float32),wp.float32)
    alive=a(np.ones((n,6),np.int32),wp.int32)
    t=fz(1);rand=a(np.arange(1,n+1,dtype=np.uint32),wp.uint32)
    wp.launch(ds_sense_and_energy_step_fp32,dim=n,inputs=[a(pos[:,:6].copy(),wp.vec3f),a(np.zeros((n,6,3),np.float32),wp.vec3f),battery,alive,target,pos_cnt,neg_cnt,mf,mu,reported,qcount,q_ids,q_ts,q_retries,q_hops,q_visit,t,rand,fz(n),fz(n),60.,1.],device=dev)
    # Verify DS actually writes to the GPU buffers.
    assert int((pos_cnt.numpy()+neg_cnt.numpy()).sum())>0
    # Force an observed report (not a hidden truth oracle) to exercise MAC.
    queue=a(np.array([[1,0,0,0,0,0],[1,0,0,0,0,0]],np.int32),wp.int32)
    ids=np.zeros((n,6,32),np.int32);ids[:,0,0]=0
    gcs=iz((n,400));true_new=iz(n);false_new=iz(n);multi=iz(n);col=iz(n);hd=iz(n);drops=iz(n);ttl=iz(n);score=fz(n)
    wp.launch(mac_slotted_step_fp32,dim=n,inputs=[a(np.array([[6,0,0,0,0,0]]*n,np.int32),wp.int32),rx,cca,alive,t,queue,a(ids,wp.int32),q_ts,q_retries,q_hops,q_visit,gcs,target,rand,score,true_new,false_new,multi,col,hd,drops,ttl,0.02],device=dev)
    assert int(true_new.numpy().sum())==n
