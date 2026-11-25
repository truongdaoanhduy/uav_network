"""Compare a real upstream CsmaCa.mac_send process to a Warp freeze kernel."""
from __future__ import annotations
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
pytest.importorskip("warp")

from uav_network.communication.classic_v1_simpy import _load_upstream
from uav_network.communication.gpu.classic_v1_freeze_warp import (
    compute_backoff_freeze_schedule,
)

PIN=Path(os.environ.get(
    "UAV_CLASSIC_V1_ROOT",
    "/home/aduy/Documents/NCKH/UavNetSim_v1_classic_PIN_64bc3a24",
))


def native_start_us(slots, busy_intervals,attempt=1):
    """Execute the actual pinned upstream mac_send generator, not a rewrite."""
    if not (PIN/"mac/csma_ca.py").is_file():
        pytest.skip("Pinned classic UavNetSim-v1 not available")
    imp=_load_upstream(PIN)
    simpy=imp["simpy"]
    env=simpy.Environment()
    sim=SimpleNamespace(
        env=env,seed=44,
        channel_states={
            u:simpy.Resource(env,capacity=1) for u in (0,1)
        },drones=[],
    )
    sender=SimpleNamespace(
        identifier=0,
        simulator=sim,env=env,
        coords=np.array([0.,0.,100.]),
        mac_process_finish={},
        mac_process_dict={},
    )
    carrier=SimpleNamespace(
        identifier=1,coords=np.array([400.,0.,100.]),
    )
    sim.drones=[sender,carrier]
    mac=imp["CsmaCa"](sender)
    sender.mac_protocol=mac
    mac.enable_ack=False
    sent=[]
    mac.phy=SimpleNamespace(
        broadcast=lambda packet:sent.append(env.now),
    )

    class Rng:
        def randint(self,lo,hi):
            upper=(32*(2**(attempt-1))-1)-1
            assert (lo,hi)==(0,upper)
            assert 0<=slots<=upper
            return slots

    mac.rng_mac=Rng()
    packet=SimpleNamespace(
        packet_id=55,number_retransmission_attempt={0:attempt},
        transmission_mode=1,packet_length=80,
        increase_ttl=lambda:None,
    )
    for start,end in busy_intervals:
        def occupy(s=start,t=end):
            yield env.timeout(s)
            with sim.channel_states[1].request() as resource:
                yield resource
                yield env.timeout(t-s)
        env.process(occupy())
    key="mac_send0_55"
    sender.mac_process_finish[key]=0
    sender.mac_process_dict[key]=env.process(mac.mac_send(packet))
    env.run(until=30000)
    assert len(sent)==1
    return int(sent[0])


@pytest.mark.parametrize("slots,intervals",[
    (0,()),
    (5,()),
    (20,()),
    (20,((200,300),)),
    (20,((20,120),)),
    (20,((100,160),(280,350))),
    (30,((0,500),)),
    (30,((40,130),(320,450))),
])
def test_warp_matches_real_upstream_csma_when_external_carrier_is_busy(slots,intervals):
    starts=np.zeros((1,1,4),np.int32)
    ends=np.zeros_like(starts)
    for idx,(begin,finish) in enumerate(intervals):
        starts[0,0,idx]=begin
        ends[0,0,idx]=finish
    result=compute_backoff_freeze_schedule(
        np.array([[slots]],np.int32),
        starts,ends,
        np.array([[len(intervals)]],np.int32),
        device="cpu",
    )
    actual=native_start_us(slots,intervals)
    got=int(result["first_tx_us"][0,0])
    assert abs(got-actual)<=20,(
        f"native mac_send={actual}us, Warp={got}us "
        f"for slots={slots}, carrier intervals={intervals}"
    )
    assert result["interruption_count"][0,0]>=0
    assert result["first_tx_us"].dtype==np.int32


def test_e4096_independent_source_csma_backoff_worlds():
    envs=4096
    slots=(np.arange(envs)%31).astype(np.int32).reshape(-1,1)
    starts=np.zeros((envs,1,4),np.int32)
    ends=np.zeros_like(starts)
    starts[::2,0,0]=200
    ends[::2,0,0]=300
    count=np.zeros((envs,1),np.int32)
    count[::2,0]=1
    output=compute_backoff_freeze_schedule(
        slots,starts,ends,count,device="cpu",
    )
    expected=50+20*slots[:,0]
    assert np.array_equal(output["first_tx_us"][1::2,0],expected[1::2])
    assert np.all(output["first_tx_us"][::2,0]>=expected[::2])
    assert np.all(output["interruption_count"][1::2,0]==0)
    assert output["first_tx_us"].shape==(envs,1)


def test_native_fifth_retry_busy_application_ack_freezes_over_500us():
    # Native seed45 last attempt: UAV0 MAC backoff begins at 40036us,
    # draw is 456 slots at attempt#5. Relay UAV1 transmits an APP
    # E2E ACK at 48530..49020us, inside the UAV0 backoff window.
    # Native DATA starts at 49756us, 550us after the prior Warp
    # standalone no-traffic projection 49206us.
    attempt=5
    slots=456
    local_begin=48530-40036
    local_finish=49020-40036
    assert local_begin==8494 and local_finish==8984
    actual=native_start_us(
        slots,((local_begin,local_finish),),attempt=attempt,
    )
    result=compute_backoff_freeze_schedule(
        np.array([[slots]],np.int32),
        np.array([[[local_begin,0,0,0]]],np.int32),
        np.array([[[local_finish,0,0,0]]],np.int32),
        np.array([[1]],np.int32),
        transmission_attempts=np.array([[5]],np.int32),
        device="cpu",
    )
    predicted=int(result["first_tx_us"][0,0])
    assert abs(actual-predicted)<=20
    assert actual-(50+20*slots)>=500
    assert result["interruption_count"][0,0]==1


def test_mixed_4096_native_attempt_windows_support_cw_doubling():
    e=4096
    attempts=((np.arange(e)%5)+1).astype(np.int32).reshape(e,1)
    cw=(32*(2**(attempts-1))-1).astype(np.int32)
    draws=((np.arange(e).reshape(e,1)*7)%cw).astype(np.int32)
    starts=np.zeros((e,1,4),np.int32)
    ends=np.zeros_like(starts)
    nbusy=np.zeros((e,1),np.int32)
    result=compute_backoff_freeze_schedule(
        draws,starts,ends,nbusy,
        transmission_attempts=attempts,device="cpu",
    )
    assert np.array_equal(result["first_tx_us"],50+draws*20)
