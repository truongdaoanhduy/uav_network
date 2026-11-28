"""Paired packet endpoints must agree without claiming event-by-event parity."""
import pytest

from scripts.gpu import packet_workload


@pytest.mark.parametrize('uavs', [1, 6])
def test_cpu_and_warp_exchange_the_same_original_evidence(uavs):
    pytest.importorskip('warp')
    assert hasattr(packet_workload, 'compare_contract_fixture')
    result = packet_workload.compare_contract_fixture(uavs)
    assert result['status'] == 'APPLICATION_CONTRACT_PASS_NOT_EVENT_PARITY'
    assert result['cpu_gcs_counts'] == result['warp_gcs_counts'] == [256, 0]
    assert result['cpu_latest_generation_s'] == result['warp_latest_generation_s'] == 1
    assert result['event_parity_verified'] is False
    assert result['cpu_e2e_ack_rx'] >= (2 if uavs > 1 else 1)
    assert result['warp_e2e_ack_rx'] >= (2 if uavs > 1 else 1)
