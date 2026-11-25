"""Six-UAV discovered chain: 5 hops DATA and 5 hops application E2E ACK."""
import pytest
pytest.importorskip("warp")
from scripts.verify_zhang_xue_v48_six_uav_multihop import (
    run_six_uav_multihop_probe,
)


@pytest.mark.parametrize("worlds", [16,1024])
def test_5_hop_source_to_destination_and_reverse_e2e_ack(worlds):
    result=run_six_uav_multihop_probe(worlds=worlds,seed=44,device="cpu")
    assert result["worlds"]==worlds
    assert result["uavs"]==6
    assert result["hello_frames_physically_received"]==10*worlds
    assert result["route_advertisements_received"]>=8*worlds
    assert result["source_route_only_after_dv"]
    assert result["app_data_generated"]==worlds
    assert result["app_data_received_at_ultimate"]==3*worlds//4
    assert result["data_forwarded_hops"]>=4*(3*worlds//4)
    assert result["application_ack_returned"]==3*worlds//4
    assert result["e2e_watermark_updated"]==3*worlds//4
    assert result["radio_blackout_holds_no_send"]
    assert result["data_ttl_expired_in_relay"]==worlds//4
    assert result["receiver_DS_only_after_data"]
    assert result["source_provenance_and_original_ttl_preserved"]
    assert result["no_packet_forwarding_loops"]
    assert result["no_duplicate_app_reward"]
    assert result["queues_empty_at_end"]
    assert not result["training_launched"]
    assert result["gate5"]=="BLOCKED"


def test_4096_5_hop_chain_world_independence():
    r=run_six_uav_multihop_probe(worlds=4096,seed=53,device="cpu")
    assert r["app_data_received_at_ultimate"]==3072
    assert r["application_ack_returned"]==3072
    assert r["data_ttl_expired_in_relay"]==1024
    assert r["no_packet_forwarding_loops"]
    assert r["receiver_DS_only_after_data"]
