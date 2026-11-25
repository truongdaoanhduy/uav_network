"""Isolated process tests for real pinned classic UavNetSim-v1 SimPy PHY/MAC.

Requires UAV_CLASSIC_V1_ROOT or the fetched vendor checkout.
Cannot run in same interpreter as new Sionna UavNetSim due global utils/phy imports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pytest


ROOT=Path(__file__).resolve().parents[1]
V1_PATH=Path(os.environ.get(
    "UAV_CLASSIC_V1_ROOT",
    ROOT/"outputs/reference_sources/UavNetSim-v1-classic",
))


def run_case(name):
    if not (V1_PATH/"entities/drone.py").is_file():
        pytest.skip("Classic UavNetSim-v1 source not fetched; run scripts/fetch_uavnetsim_v1_classic.sh")
    env={**os.environ, "UAV_CLASSIC_V1_ROOT":str(V1_PATH)}
    env["PYTHONPATH"]=str(ROOT)+os.pathsep+env.get("PYTHONPATH","")
    with tempfile.TemporaryDirectory(prefix="uav_classic_v1_test_") as tmp:
        proc=subprocess.run(
            [sys.executable,"-m","uav_network.communication.classic_v1_simpy",
             "--case",name,"--json"],
            cwd=tmp,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            timeout=50,
        )
    assert proc.returncode==0, (
        f"simpy case {name} exited {proc.returncode}\n"
        f"stdout tail:{proc.stdout[-1800:]}\nstderr tail:{proc.stderr[-1800:]}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_real_v1_phy_mac_two_hop_application_and_e2e_ack():
    got=run_case("line")
    assert got["engine"]=="uavnetsim_v1_real_simpy_csma"
    assert got["v1_commit"]=="64bc3a2424f2976418731ee94f9a8fd3e8810b22"
    assert got["app_data_delivered"]==1
    assert got["app_e2e_ack_delivered"]>=1
    assert got["hop_count"]==2
    assert got["receiver_own_origin_count"]>=1
    assert got["native_data_pdr"]<=1.0
    assert got["default_poisson_process_count"]==0
    assert got["default_mobility_process_count"]==0


def test_real_v1_radio_blackout_recovers_and_keeps_receive_alive():
    got=run_case("blackout")
    assert got["engine"]=="uavnetsim_v1_real_simpy_csma"
    assert got["blackout_data_delivered"]==0
    assert got["post_restore_data_delivered"]>=1
    assert got["radio_receive_process_alive"] is True


def test_real_v1_gate5_comparison_reports_nonidentical_mac_times():
    got=run_case("gate5")
    assert got["engine"]=="uavnetsim_v1_real_simpy_csma"
    assert got["reference_engine"]=="mini_discrete_event"
    assert 0 <= got["simpy_pdr"] <= 1
    assert 0 <= got["reference_pdr"] <= 1
    assert got["simpy_mac_real"] is True
    assert got["gate5_status"] in (
        "PARTIAL_SIMPY_REFERENCE_PASS","PARTIAL_SIMPY_REFERENCE_FAIL",
    )
    assert got["full_warp_parity_status"]=="NOT_TESTED"
    assert got["simpy_native_data_generated"]==1
    assert got["simpy_native_data_arrived"]==1
    assert got["simpy_hops"]==2
    assert got["reference_hops"]==2
    assert isinstance(got["reference_data_latency_ms"],float)
    assert isinstance(got["simpy_data_latency_ms"],float)
    assert isinstance(got["latency_difference_ms"],float)
    assert abs(got["latency_difference_ms"]) < 0.5

def test_real_classic_v1_backend_runs_actor_env_replay_update():
    """End-to-end real SimPy backend through vanilla MATD3 on CPU only."""
    if not (V1_PATH/"entities/drone.py").is_file():
        pytest.skip("Pinned classic v1 checkout unavailable")
    env={**os.environ,"UAV_CLASSIC_V1_ROOT":str(V1_PATH)}
    env["PYTHONPATH"]=str(ROOT)+os.pathsep+env.get("PYTHONPATH","")
    with tempfile.TemporaryDirectory(prefix="uav_real_v1_training_smoke_") as tmp:
        proc=subprocess.run(
            [
                sys.executable,"-m","uav_network.experiments.train_zhang_xue_pilot",
                "--network-backend","classic_simpy_v1",
                "--algorithm","matd3",
                "--episodes","1","--horizon-steps","3",
                "--learning-starts","2","--batch-size","2",
                "--eval-episodes","1","--hidden-size","16","--seed","44",
            ],
            cwd=tmp,env=env,text=True,stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,timeout=95,
        )
    assert proc.returncode==0, proc.stderr[-2400:]
    data=json.loads(proc.stdout[proc.stdout.find("{"):])
    assert data["backend"]=="cpu_uavnetsim_v1_simpy_csma"
    assert data["updates"]>0
    assert data["total_steps"]==3
    assert 0<=data["mean_eval_pdr"]<=1
    assert data["actor_parameter_delta_norm"]>0

def test_real_v1_actor_neighbours_require_received_hello_not_global_position():
    obs=run_case("observations")
    assert obs["local_peer_offset_before_hello"]==[0.,0.,0.]
    assert any(abs(v)>0 for v in obs["local_peer_offset_after_hello"])
    assert obs["received_hello_from_peer"] is True


def test_real_v1_friis_snr_matches_new_warp_pairwise_cpu_kernel():
    got=run_case("warp_phy")
    assert got["source_sha"]=="64bc3a2424f2976418731ee94f9a8fd3e8810b22"
    assert got["device"]=="cpu"
    assert got["num_nodes"]==6
    assert got["max_abs_snr_error_db"]<0.001
    assert got["connectivity_mismatch_count"]==0
    assert got["full_warp_mac_parity_status"]=="NOT_TESTED"


def test_real_v1_hidden_terminal_contention_exposes_reference_limitations():
    got=run_case("collision")
    assert got["source_sha"]=="64bc3a2424f2976418731ee94f9a8fd3e8810b22"
    assert got["num_seeds"]>=5
    assert len(got["native_pdr_by_seed"])==got["num_seeds"]
    assert len(got["reference_pdr_by_seed"])==got["num_seeds"]
    assert all(0.<=v<=1. for v in got["native_pdr_by_seed"])
    assert all(abs(v-1.0)<1e-12 for v in got["reference_pdr_by_seed"])
    assert got["native_total_collisions"]>0
    assert got["full_warp_mac_parity_status"]=="NOT_TESTED"
