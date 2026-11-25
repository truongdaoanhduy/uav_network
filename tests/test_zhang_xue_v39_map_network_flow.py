"""Small integrated CPU map-to-Warp MAC-to-receiver DS evidence gating."""
import pytest
pytest.importorskip("warp")
from scripts.verify_zhang_xue_map_to_network_v39 import run_map_network_probe

@pytest.mark.parametrize("seed", [44, 1337])
def test_integrated_map_network_gate_is_not_training(seed):
    r=run_map_network_probe(envs=16,seed=seed)
    assert r["envs"]==16
    assert r["scene_targets_per_env"]==10
    assert r["packet_generated"]==16*6
    assert r["total_own_evidence_counts"]>0
    assert r["frame_max_bytes"]<=570
    assert r["source_owned_receiver_events"]==r["first_network_data_delivered"]
    assert r["receiver_remote_nonzero_origin_cells"]==r["application_entries_total"]
    assert r["receiver_max_merge_replay_idempotent"]
    assert r["receiver_ds_finite_and_normalized"]
    assert r["truth_not_serialized"]
    assert r["delivery_barrier_verified"]
    assert r["full_network_gate5"]=="NOT_PASSED"
    assert r["no_training"]
