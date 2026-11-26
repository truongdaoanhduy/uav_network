"""V89 actor peer observations have a strict RF HELLO provenance (no oracle)."""
import numpy as np
import pytest
import torch
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80
from zhang_xue.gpu.state_v67 import wp

CONFIG="research/zhang_xue/configs/scenario_v89_static_1km.yaml"

@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_hello_provenance_and_peer_oracle_leakage(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    p=GPUWorldV80Params.from_yaml(CONFIG,envs=2,device=device,seed=44)
    w=WarpParallelMissionV80(p)
    assert tuple(w.buffers["obs"].shape)==(2,6,549)
    packet_times=w.network.data["hello_rx_at_us"].numpy()
    now=w.network.data["now"].numpy()
    valid=[(rx,peer) for rx in range(6) for peer in range(6)
           if peer!=rx and packet_times[0,rx,peer]>=0 and now[0]-packet_times[0,rx,peer]<2_000_000]
    assert valid, "Physical RF warmup must deliver at least one HELLO"
    w._observe()
    initial=w.buffers["obs"].numpy()
    rx,peer=valid[0]
    assert initial[0,rx,514:549:7].sum() != 0 or initial[0,rx,518:549:7].sum() > 0
    before=initial[0,rx,514:549].copy()
    # Move a hidden peer true world position but neither advance MAC nor deliver
    # another HELLO: receiver's local observation MUST NOT track it.
    positions=wp.to_torch(w.buffers["positions"])
    positions[0,peer,0]+=130.0
    positions[0,peer,1]+=20.0
    w._observe()
    after=w.buffers["obs"].numpy()[0,rx,514:549].copy()
    np.testing.assert_allclose(before,after,rtol=0,atol=1e-6)
    # Expire cached HELLO at receiver; missing valid flag must be 0.
    received=wp.to_torch(w.network.data["hello_rx_at_us"])
    received[0,rx,:]=-1
    w._observe()
    missing=w.buffers["obs"].numpy()[0,rx,514:549]
    assert (missing==0.0).all()

def test_hello_blackout_does_not_create_oracle_peer_state():
    p=GPUWorldV80Params.from_yaml(CONFIG,envs=1,device="cpu",seed=11)
    w=WarpParallelMissionV80(p)
    n=w.network
    rx,peer=0,1
    cache=wp.to_torch(n.data["hello_rx_at_us"])
    cache[0,rx,peer]=-1
    # Blackout in PHY is not a global-peer-position lookup fallback.
    for _ in range(3):
        n.data["gain"].numpy()  # No radio writeback from reading.
        w._observe()
        raw=w.buffers["obs"].numpy()
        assert not np.isnan(raw).any()
        assert cache[0,rx,peer].item()==-1
