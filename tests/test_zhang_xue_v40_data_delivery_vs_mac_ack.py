"""DATA reaching a receiver is not the same as MAC ACK reaching the sender."""
import numpy as np
import pytest
import os
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1]
_UPSTREAM=Path(os.environ.get("UAV_CLASSIC_V1_ROOT",_ROOT/"outputs/reference_sources/UavNetSim-v1-classic"))
if not (_UPSTREAM/"entities/drone.py").is_file():
    pytest.skip("Pinned native UavNetSim-v1 source unavailable",allow_module_level=True)
pytest.importorskip("warp")
from uav_network.communication.gpu.zhang_xue_mac_retry_warp import (
    simulate_one_hop_retry_batch,
)
from scripts.verify_zhang_xue_network_v40 import run_parity


def test_seed54_data_can_arrive_even_when_mac_ack_is_lost():
    p=np.full((1,6,3),2000.,np.float32)
    p[:,:,2]=100.
    p[:,0,:]=[0.,0.,100.]
    p[:,1,:]=[400.,0.,100.]
    p[:,2,:]=[800.,0.,100.]
    dst=np.full((1,6),-1,np.int32)
    dst[0,[0,2]]=1
    sizes=np.zeros((1,6),np.int32)
    sizes[0,[0,2]]=566
    slots=np.zeros((1,6,5),np.int32)
    slots[0,0]=[2,59,83,242,235]
    slots[0,2]=[5,35,27,207,464]
    got=simulate_one_hop_retry_batch(
        p,dst,sizes,np.ones((1,6),np.int32),slots,
        np.full((1,6),10e6,np.float32),
    )
    assert got["data_received"][0,0]==1
    assert got["mac_acked"][0,0]==0
    assert got["ack_collided"][0,0].sum()>=1
    assert got["dropped"][0,0]==1


def test_32_seed_true_application_data_pdr_matches_native_trace():
    got=run_parity(start_seed=44,num_seeds=32)
    assert got["pdr_seed_mismatch_count"]==0
    assert got["native_metric_matches_physical_data_delivery"]
    assert got["data_tx_attempt_total_native"]==got["data_tx_attempt_total_warp"]
