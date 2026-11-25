"""Six concurrently transmitting UAVs: MAC, queue, DS and real E2E watermark."""
import pytest
pytest.importorskip("warp")
from scripts.verify_zhang_xue_v44_six_uav_contention import (
    run_six_uav_contention_probe,
)


@pytest.mark.parametrize(("envs","steps"),[(16,200),(256,12)])
def test_six_source_concurrent_network_sustains_correct_state(envs,steps):
    result=run_six_uav_contention_probe(
        envs=envs,steps=steps,seed=44,device="cpu",
    )
    assert result["worlds"]==envs
    assert result["step_count"]==steps
    assert result["source_uavs"]==6
    assert result["hello_receipts"]==12*envs
    assert result["initial_routes_local_hello_only"]
    assert result["app_data_generated"]>=envs*5
    assert result["app_data_received"]>0
    assert result["real_e2e_ack_received"]>0
    assert result["real_e2e_ack_received"]<=result["app_data_received"]
    assert result["e2e_watermarks_advanced"]==result["real_e2e_ack_received"]
    assert result["unexpected_reward_count"]==0
    assert result["remote_evidence_before_data_count"]==0
    assert result["remote_evidence_leak_count"]==0
    assert result["blackout_transmission_count"]==0
    assert result["queue_inflight_beyond_one_count"]==0
    assert result["max_queued_per_uav"]<=4
    assert result["no_false_mac_ack_as_e2e_ack"]
    assert result["multiuser_contention_exercised"]
    assert result["training_launched"] is False
    assert result["gate5_status"]=="NOT_PASSED"


def test_E4096_six_uav_contention_short_stress():
    result=run_six_uav_contention_probe(envs=4096,steps=4,seed=47,device="cpu")
    assert result["app_data_generated"]>=4096*5
    assert result["app_data_received"]>0
    assert result["queue_inflight_beyond_one_count"]==0
    assert result["remote_evidence_leak_count"]==0
    assert result["blackout_transmission_count"]==0
