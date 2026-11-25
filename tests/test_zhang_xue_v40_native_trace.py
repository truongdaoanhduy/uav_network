"""Source-faithful trace visibility: native SimPy DATA and MAC events."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
V1=Path(os.environ.get("UAV_CLASSIC_V1_ROOT",ROOT/"outputs/reference_sources/UavNetSim-v1-classic"))

def native_trace():
    if not (V1/"entities/drone.py").is_file():
        pytest.skip("Pinned upstream UavNetSim-v1 not available")
    env={**os.environ,"UAV_CLASSIC_V1_ROOT":str(V1),
         "PYTHONPATH":str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH","")}
    proc=subprocess.run(
        [sys.executable,"-m","uav_network.communication.classic_v1_simpy",
         "--case","collision_trace","--json"],
        env=env,text=True,capture_output=True,timeout=90,
    )
    assert proc.returncode==0,proc.stderr[-1500:]
    return json.loads(proc.stdout.strip().splitlines()[-1])

def test_native_trace_reports_exact_packet_events_not_reconstructed_times():
    trace=native_trace()
    assert trace["source_sha"]=="64bc3a2424f2976418731ee94f9a8fd3e8810b22"
    assert trace["scope"]=="actual_upstream_mac_events"
    assert len(trace["seeds"])==6
    for physical,metric in zip(
        trace["native_delivered_ids_by_seed"],
        trace["metric_delivered_ids_by_seed"],
    ):
        # BUG REPRODUCER: per-destination packet delivered but metric
        # is still zero because ApplicationDataPacket is not SearchDataPacket.
        assert physical==metric
    assert any(trace["native_delivered_ids_by_seed"])
    assert len(trace["events_by_seed"])==6
    for seed,events in zip(trace["seeds"],trace["events_by_seed"]):
        data_tx=[x for x in events if x["type"]=="data_tx"]
        assert data_tx,seed
        assert all(x["packet_id"] in (1,2) for x in data_tx)
        assert all(x["time_us"]>=25000. for x in data_tx)
        assert all(x["uav_id"] in (0,2) for x in data_tx)
        assert sum(x["type"]=="data_tx" for x in events)<=10
        assert sum(x["type"]=="data_ack_timeout" for x in events)>=2


def test_native_data_random_draws_and_first_mac_attempt_times_are_visible():
    trace=native_trace()
    assert len(trace["mac_draws_by_seed"])==6
    for events,draws in zip(trace["events_by_seed"],trace["mac_draws_by_seed"]):
        assert draws
        for u in (0,2):
            first_tx=next(ev for ev in events
                          if ev["type"]=="data_tx" and ev["uav_id"]==u)
            matching=[d for d in draws if d["uav_id"]==u]
            assert matching
            first=matching[0]
            assert first["time_us"]>=25000.
            assert first["lo"]==0 and first["hi"]==30
            assert 0<=first["slot"]<=30
            # Native feed_packet and CCA may delay the start;
            # strictly check it cannot start before DIFS + draw.
            assert first_tx["time_us"]-first["time_us"]>=50+20*first["slot"]
