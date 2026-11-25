"""Network-after-map: initial Warp CSMA frame-scheduling integration gates.

Known limits: first MAC attempt only; retries, queues and E2E ACK remain
mandatory before full Gate5 and before any training.
"""
import numpy as np
import pytest
pytest.importorskip("warp")

from uav_network.communication.gpu.zhang_xue_mac_first_attempt_warp import (
    schedule_initial_csma_frame_attempts,
)


def _base_worlds():
    xyz=np.full((3,6,3),2000,np.float32)
    xyz[:,:,2]=100
    xyz[:,0,:]=[0,0,100]
    xyz[:,1,:]=[400,0,100]
    xyz[:,2,:]=[800,0,100]
    xyz[2,2,:]=[200,0,100]  # source 0 and 2 can sense one another
    dest=np.full((3,6),-1,np.int32)
    dest[:,[0,2]]=1
    bytes_=np.zeros((3,6),np.int32)
    bytes_[:,[0,2]]=566
    channels=np.ones((3,6),np.int32)
    channels[1,2]=6
    backoff=np.zeros((3,6),np.int32)
    backoff[2,2]=5
    return xyz,dest,bytes_,channels,backoff


def test_first_attempt_hidden_terminal_collision_orthogonal_and_cca_deferral():
    xyz,dest,size,ch,slots=_base_worlds()
    result=schedule_initial_csma_frame_attempts(
        xyz,dest,size,ch,slots,device="cpu",
    )
    assert result["success"].shape==(3,6)
    np.testing.assert_array_equal(result["success"][:,[0,2]],[
        [0,0], # 800m hidden terminals collide at 400m relay
        [1,1], # nonoverlapping channels 1 and 6
        [1,1], # sources can sense, staggered CCA defers one transmission
    ])
    assert result["start_us"][2,2]>result["end_us"][2,0]
    assert np.all(result["start_us"][:,:]>=0)


def test_4096_independent_worlds_one_attempt_integrity():
    e=4096
    xyz=np.full((e,6,3),1500.,np.float32)
    xyz[:,:,2]=100
    xyz[:,0,:]=[0,0,100]
    xyz[:,1,:]=[400,0,100]
    dest=np.full((e,6),-1,np.int32)
    dest[:,0]=1
    size=np.zeros((e,6),np.int32)
    size[:,0]=566
    ch=np.ones((e,6),np.int32)
    slots=np.zeros((e,6),np.int32)
    result=schedule_initial_csma_frame_attempts(xyz,dest,size,ch,slots,device="cpu")
    assert result["success"].shape==(e,6)
    assert int(result["success"][:,0].sum())==e
    assert int(result["success"][:,1:].sum())==0
    np.testing.assert_array_equal(result["start_us"][:,0],50.)


def test_invalid_action_or_contention_window_rejected():
    xyz,dest,size,ch,slots=_base_worlds()
    with pytest.raises(ValueError):
        schedule_initial_csma_frame_attempts(xyz,dest,size,ch,np.full_like(slots,31))
    with pytest.raises(ValueError):
        schedule_initial_csma_frame_attempts(xyz,np.full_like(dest,7),size,ch,slots)
