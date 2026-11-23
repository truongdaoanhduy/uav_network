"""Native 2D correlated-DS projection vs Warp GPU/CPU device kernel.

This ONLY tests mass projection of already deduplicated evidence LLR support.
It is not a complete Warp multi-hop GCS implementation.
"""
import numpy as np
import pytest
import warp as wp
from uav_network.communication.m2_protocol import stable_mass_from_llr_totals
from uav_network.environment.gpu.m2_mass_projector import m2_project_mass

@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if wp.is_cuda_available() else []))
def test_m2_mass_projector_exact_native_parity_random_conflict_and_extreme_evidence(device):
    wp.init()
    rng=np.random.default_rng(20261121)
    shape=(4,7,1600) # M2 2D ground map 1000/25=40 x 40, 7=6 UAV + GCS
    a=np.asarray(rng.uniform(0,40,shape),dtype=np.float32)
    b=np.asarray(rng.uniform(0,40,shape),dtype=np.float32)
    a[0,0,:12]=np.array([0.,0.,1e-7,1e-4,1.,10.,200.,1000.,1000.,1e-6,10000.,10000.],np.float32)
    b[0,0,:12]=np.array([0.,1e-6,0.,1e-4,10.,1.,200.,1000.,1e-6,1000.,10000.,5.],np.float32)
    aa=wp.array(a,device=device)
    bb=wp.array(b,device=device)
    f=wp.zeros(shape,dtype=wp.float32,device=device)
    e=wp.zeros(shape,dtype=wp.float32,device=device)
    u=wp.zeros(shape,dtype=wp.float32,device=device)
    wp.launch(m2_project_mass,dim=shape,inputs=[aa,bb,f,e,u],device=device)
    actual=np.stack([f.numpy(),e.numpy(),u.numpy()],axis=-1)
    expected=np.empty_like(actual)
    for idx in np.ndindex(shape):expected[idx]=stable_mass_from_llr_totals(float(a[idx]),float(b[idx]))
    assert np.isfinite(actual).all()
    assert np.all(actual>=0)
    assert np.allclose(actual.sum(-1),1.,atol=1e-6)
    np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=3e-6)

@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if wp.is_cuda_available() else []))
def test_empty_map_stays_100_percent_unknown(device):
    shape=(2,7,1600)
    p=wp.zeros(shape,dtype=wp.float32,device=device)
    n=wp.zeros(shape,dtype=wp.float32,device=device)
    f=wp.zeros(shape,dtype=wp.float32,device=device)
    e=wp.zeros(shape,dtype=wp.float32,device=device)
    u=wp.zeros(shape,dtype=wp.float32,device=device)
    wp.launch(m2_project_mass,dim=shape,inputs=[p,n,f,e,u],device=device)
    assert np.all(f.numpy()==0) and np.all(e.numpy()==0) and np.all(u.numpy()==1)

@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if wp.is_cuda_available() else []))
def test_real_m2_conditioned_unique_event_streams_project_to_same_masses(device):
    """GPU projector consumes *conditional* LLR support from the real M2 map."""
    from uav_network.communication.m2_protocol import CooperativeWorldMap
    from uav_network.environment.correlated_mapping import Event
    nodes=[CooperativeWorldMap(12) for _ in range(7)]
    # Each node independently fuses the events it actually receives.
    originals=[Event(f'node0-{t}',t%6,t,t%12,'target',bool(t%3),.92,.08,.63)
               for t in range(1,73)]
    for i,node in enumerate(nodes):
        readings=originals if i==6 else originals[:(12+i*10)]
        for event in reversed(readings):
            node.ingest(event,arrived_at=100)
        # repeated ID is deduplicated
        if readings:assert node.ingest(readings[0],arrived_at=100) is False
    shape=(1,7,12)
    positive=np.zeros(shape,np.float32)
    negative=np.zeros(shape,np.float32)
    for ni,node in enumerate(nodes):
        for cell,stream in node._streams.items():
            positive[0,ni,cell]=stream[1]
            negative[0,ni,cell]=stream[2]
    pos=wp.array(positive,device=device);neg=wp.array(negative,device=device)
    f=wp.zeros(shape,dtype=wp.float32,device=device)
    e=wp.zeros(shape,dtype=wp.float32,device=device)
    u=wp.zeros(shape,dtype=wp.float32,device=device)
    wp.launch(m2_project_mass,dim=shape,inputs=[pos,neg,f,e,u],device=device)
    projected=np.stack([f.numpy(),e.numpy(),u.numpy()],axis=-1)[0]
    reference=np.stack([node.target_masses for node in nodes])
    np.testing.assert_allclose(projected,reference,atol=5e-6,rtol=5e-6)

@pytest.mark.parametrize('device',['cpu']+(['cuda:0'] if wp.is_cuda_available() else []))
def test_sparse_warp_support_delta_accumulation_matches_unique_native_events(device):
    """GCS/UAV cell support can be updated from event deltas without dense scans.

    Event ID de-duplication and conditional LLR are deliberately performed by
    the owner/source protocol before GPU sparse apply; there is no target truth.
    """
    from uav_network.environment.gpu.m2_mass_projector import m2_apply_sparse_llr
    shape=(2,7,1600)
    rng=np.random.default_rng(2026)
    n=700
    worlds=rng.integers(0,2,n,dtype=np.int32)
    nodes=rng.integers(0,7,n,dtype=np.int32)
    cells=rng.integers(0,1600,n,dtype=np.int32)
    signed=rng.uniform(-2.,2.,n).astype(np.float32)
    # Interleaved positive/negative and many repeats of one cell.
    worlds[:30]=1;nodes[:30]=6;cells[:30]=1
    pos=np.zeros(shape,dtype=np.float32);neg=np.zeros(shape,dtype=np.float32)
    for i in range(n):
        key=(worlds[i],nodes[i],cells[i])
        if signed[i]>=0:pos[key]+=signed[i]
        else:neg[key]-=signed[i]
    wp_pos=wp.zeros(shape,dtype=wp.float32,device=device)
    wp_neg=wp.zeros(shape,dtype=wp.float32,device=device)
    wp.launch(m2_apply_sparse_llr,dim=n,inputs=[
        wp_pos,wp_neg,wp.array(worlds,device=device),wp.array(nodes,device=device),
        wp.array(cells,device=device),wp.array(signed,device=device)],device=device)
    np.testing.assert_allclose(wp_pos.numpy(),pos,rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(wp_neg.numpy(),neg,rtol=2e-6,atol=2e-6)
    f=wp.zeros(shape,dtype=wp.float32,device=device)
    e=wp.zeros(shape,dtype=wp.float32,device=device)
    u=wp.zeros(shape,dtype=wp.float32,device=device)
    wp.launch(m2_project_mass,dim=shape,inputs=[wp_pos,wp_neg,f,e,u],device=device)
    assert np.isfinite(f.numpy()).all()
    np.testing.assert_allclose(f.numpy()+e.numpy()+u.numpy(),1,rtol=0,atol=1e-6)
