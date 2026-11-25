"""Persistent multi-timestep queue→one-hop MAC→routed app ACK smoke."""
import pytest
pytest.importorskip("warp")
from scripts.verify_zhang_xue_queue_mac_v41 import run_queued_onehop_e2e_probe


@pytest.mark.parametrize("envs", [64,4096])
def test_queue_blackout_until_recovery_or_expiry_and_application_ack(envs):
    r=run_queued_onehop_e2e_probe(envs=envs,seed=44,device="cpu")
    assert r["envs"]==envs
    assert r["data_generated"]==envs
    assert r["data_received"]==3*envs//4
    assert r["e2e_ack_generated"]==3*envs//4
    assert r["e2e_ack_received"]==3*envs//4
    assert r["mac_ack_success"]==2*(3*envs//4)
    assert r["queue_expired"]==envs//4
    assert r["queues_empty_after_final_step"]
    assert r["no_blackout_transmissions"]
    assert r["no_duplicate_e2e_ack_reward"]
    assert r["truth_not_serialized"]
    assert not r["training_launched"]
    assert r["full_gate5"]=="NOT_PASSED"
