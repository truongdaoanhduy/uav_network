"""Real 1km randomized 10-target scenes with 6 moving UAS and actual Warp MAC."""
import numpy as np
import pytest
pytest.importorskip("warp")
from scripts.verify_zhang_xue_v45_sampled_scene_network import (
    run_sampled_scenario_network_probe,
)


@pytest.mark.parametrize(("envs","steps"),[(16,200),(128,20)])
def test_sampled_scene_motion_evidence_network_reward_only_truth(envs,steps):
    r=run_sampled_scenario_network_probe(
        envs=envs,steps=steps,seed=44,device="cpu",
    )
    assert r["worlds"]==envs and r["steps"]==steps
    assert r["scene_map_size_m"]==1000
    assert r["uavs"]==6
    assert r["targets_per_world"]==10
    assert r["no_fly_disks_per_world"]==2
    assert r["blackout_diamonds_per_world"]==2
    assert r["target_layout_valid"]
    assert r["flight_positions_inside_world"]
    assert r["flight_did_not_penetrate_no_fly"]
    assert r["safe_inter_uav_separation"]
    assert r["real_mac_hello_attempts"]>0
    assert r["all_direct_routes_from_received_hello_only"]
    assert r["source_evidence_observations"]>0
    assert r["app_data_generated"]>0
    assert r["app_data_physical_receipts"]>0
    assert r["app_e2e_ack_valid"]>0
    assert r["app_e2e_ack_valid"]<=r["app_data_physical_receipts"]
    assert r["app_ack_enqueues_attempted"]==r["app_data_physical_receipts"]
    assert (r["app_ack_enqueued"]+r["app_ack_queue_rejected"]==
            r["app_ack_enqueues_attempted"])
    assert (r["app_e2e_ack_valid"]<=
            r["app_ack_physical_receipts"]<=r["app_ack_enqueued"])
    assert r["real_mac_hello_received"]<=r["real_mac_hello_attempts"]
    assert r["duplicate_reward_count"]==0
    assert r["inflight_over_one"]==0
    assert r["e2e_watermarks_advanced"]==r["app_e2e_ack_valid"]
    assert r["uavs_communicating_during_blackout"]==0
    assert r["receiver_evidence_before_physical_receipt"]==0
    assert r["reward_target_truth_only"]
    assert np.isfinite(r["cumulative_truth_reward_mean"])
    assert r["true_target_confirmations"]<=10*envs
    assert r["false_confirmations"]<=390*envs
    assert r["map_coverage_mean"]>0.02
    assert r["training_launched"] is False
    assert r["full_network_gate5"]=="BLOCKED"


def test_E1024_sampled_scene_short_cuda_scale_cpu_preflight():
    r=run_sampled_scenario_network_probe(envs=1024,steps=4,seed=53,device="cpu")
    assert r["worlds"]==1024
    assert r["target_layout_valid"]
    assert r["flight_positions_inside_world"]
    assert r["safe_inter_uav_separation"]
    assert r["app_data_generated"]>0
    assert r["receiver_evidence_before_physical_receipt"]==0
