"""Gate5F-DIFS/CW subgate: source-faithful CSMA backoff arithmetic on Warp.

NOT a complete MAC state machine: actual scheduling, busy-channel backoff
freeze, MAC ACK and retransmission remain unimplemented.
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
from uav_network.communication.gpu.classic_v1_backoff_warp import (
    compute_backoff_windows,
)


def test_80211b_cw_growth_and_difs_slot_timing():
    attempts=np.array([[1,2,3,4,5]],np.int32)
    draws=np.array([[0,30,62,126,510]],np.int32)
    got=compute_backoff_windows(attempts,draws,device="cpu")
    np.testing.assert_array_equal(got["contention_window"],[[31,63,127,255,511]])
    np.testing.assert_array_equal(
        got["backoff_us"],[draws[0]*20])
    np.testing.assert_array_equal(
        got["wait_difs_plus_backoff_us"],[draws[0]*20+50])


def test_draw_must_be_in_native_randint_range():
    with pytest.raises(ValueError):
        compute_backoff_windows(
            np.array([[1]],np.int32),np.array([[31]],np.int32),device="cpu")
    with pytest.raises(ValueError):
        compute_backoff_windows(
            np.array([[0]],np.int32),np.array([[0]],np.int32),device="cpu")
    with pytest.raises(ValueError):
        compute_backoff_windows(
            np.array([[6]],np.int32),np.array([[0]],np.int32),device="cpu")


def test_warp_backoff_math_matches_pinned_v1_csma_source():
    root=Path(__file__).resolve().parents[1]
    v1=Path(os.environ.get("UAV_CLASSIC_V1_ROOT",root/"outputs/reference_sources/UavNetSim-v1-classic"))
    if not (v1/"mac/csma_ca.py").is_file():
        pytest.skip("Fetch pinned classic v1 source first")
    env={**os.environ,"UAV_CLASSIC_V1_ROOT":str(v1)}
    env["PYTHONPATH"]=str(root)+os.pathsep+env.get("PYTHONPATH","")
    with tempfile.TemporaryDirectory(prefix="uav_v1_backoff_") as tmp:
        proc=subprocess.run(
            [sys.executable,"-m","uav_network.communication.classic_v1_simpy",
             "--case","warp_backoff","--json"],
            env=env,cwd=tmp,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            timeout=45,
        )
    assert proc.returncode==0,proc.stderr[-1600:]
    actual=json.loads(proc.stdout.strip().splitlines()[-1])
    assert actual["source_sha"]=="64bc3a2424f2976418731ee94f9a8fd3e8810b22"
    assert actual["mac_source"]=="mac.csma_ca.CsmaCa.mac_send"
    assert actual["worlds"]>=128 and actual["agents_per_world"]==6
    assert actual["backoff_mismatch_count"]==0
    assert actual["full_mac_backoff_freeze_and_ack_parity"]=="NOT_TESTED"
