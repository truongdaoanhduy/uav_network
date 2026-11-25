"""Gate5E PHY CCA check: parity with the original UavNetSim-v1 750m sensing rule.

This does NOT simulate CSMA backoff, ACK or retransmissions. See
UavNetSim-v1/utils/util_function.py::check_channel_availability.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip("warp")
from uav_network.communication.gpu.classic_v1_cca_warp import compute_cca_free


def test_hidden_terminal_is_not_visible_to_other_sender():
    xyz=np.array([[[0,0,100],[400,0,100],[800,0,100]]],np.float32)
    active=np.array([[1,0,0]],np.int32)
    cca=compute_cca_free(xyz,active,sensing_range_m=750,device="cpu")
    np.testing.assert_array_equal(cca,[[1,0,1]])


def test_exact_750m_boundary_is_available_as_in_original_source():
    xyz=np.array([[[0,0,100],[750,0,100],[749.9,0,100]]],np.float32)
    active=np.array([[1,0,0]],np.int32)
    cca=compute_cca_free(xyz,active,sensing_range_m=750,device="cpu")
    np.testing.assert_array_equal(cca,[[1,1,0]])
    np.testing.assert_array_equal(
        compute_cca_free(xyz,np.zeros((1,3),np.int32),device="cpu"),
        [[1,1,1]]
    )


def test_cca_kernel_exactly_matches_pinned_classic_v1_source():
    root=Path(__file__).resolve().parents[1]
    v1=Path(os.environ.get("UAV_CLASSIC_V1_ROOT",root/"outputs/reference_sources/UavNetSim-v1-classic"))
    if not (v1/"utils/util_function.py").is_file():
        pytest.skip("Pin classic v1 with scripts/fetch_uavnetsim_v1_classic.sh")
    env={**os.environ,"UAV_CLASSIC_V1_ROOT":str(v1)}
    env["PYTHONPATH"]=str(root)+os.pathsep+env.get("PYTHONPATH","")
    with tempfile.TemporaryDirectory(prefix="uav_classic_cca_") as temp:
        proc=subprocess.run(
            [sys.executable,"-m","uav_network.communication.classic_v1_simpy",
             "--case","warp_cca","--json"],
            cwd=temp,env=env,text=True,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,timeout=45,
        )
    assert proc.returncode==0,proc.stderr[-1500:]
    result=json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["source_sha"]=="64bc3a2424f2976418731ee94f9a8fd3e8810b22"
    assert result["original_source_function"]=="utils.util_function.check_channel_availability"
    assert result["worlds"]>=128 and result["nodes_per_world"]==6
    assert result["cca_mismatch_count"]==0
    assert result["num_receiver_checks"]==result["worlds"]*6
    assert result["full_warp_mac_parity_status"]=="NOT_TESTED"
