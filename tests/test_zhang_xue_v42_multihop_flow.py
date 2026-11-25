"""Full controlled HELLO→DV→2-hop DATA→2-hop E2E ACK→receiver-local DS."""
import pytest
pytest.importorskip("warp")
from scripts.verify_zhang_xue_multihop_flow_v42 import run_multihop_map_network_probe


@pytest.mark.parametrize("envs",[16,1024])
def test_multihop_bootstrap_blackout_expiration_and_ack(envs):
    r=run_multihop_map_network_probe(envs=envs,device="cpu",seed=44)
    assert r["worlds"]==envs
    assert r["hello_mac_success"]>0
    assert r["distance_vector_mac_success"]>0
    assert r["no_route_0_to_2_before_advert"]
    assert r["learned_route_0_to_2_via_1"]
    assert r["learned_route_2_to_0_via_1"]
    assert r["app_data_generated"]==envs
    assert r["app_data_received"]==3*envs//4
    assert r["e2e_ack_received"]==3*envs//4
    assert r["data_ttl_expired"]==envs//4
    assert r["relay_data_count"]==3*envs//4
    assert r["relay_ack_count"]==3*envs//4
    assert r["receiver_ds_only_after_physical_data"]
    assert r["payload_origin_preserved"]
    assert r["duplicate_reward_count"]==0
    assert r["training_started"] is False
    assert r["network_full_gate5"]=="BLOCKED"
