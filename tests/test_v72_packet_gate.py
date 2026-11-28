"""Run the actual shared packet workload on Warp's CPU backend."""
import numpy as np
import pytest

from scripts.gpu import warp_packet_cuda_gate as gate


@pytest.mark.parametrize('uavs,grid', [(1, 20), (6, 20), (1, 1)])
def test_shared_gate_delivers_real_evidence_and_acks(uavs, grid):
    pytest.importorskip('warp')
    assert hasattr(gate, 'run_packet_workload'), 'CPU and CUDA need the same real packet workload'
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    net = WarpCompressedRadioNetworkV67(2, num_uavs=uavs, grid_w=grid, device='cpu', seed=44)
    snapshot = gate.run_packet_workload(net)
    assert np.all(snapshot['overflow'] == 0)
    assert np.all(snapshot['stats_gcs'] >= 1)
    assert np.all(snapshot['stats_e2e'] >= (2 if uavs > 1 else 1))
    cid = min(42, grid * grid - 1)
    assert np.all(net.data['ledger_pos'].numpy()[:, uavs, 0, cid] == 256)
    assert np.all(net.data['gcs_latest'].numpy()[:, cid] == 1)


def test_cpu_packet_gate_explicitly_labels_backend_and_scope():
    pytest.importorskip('warp')
    assert hasattr(gate, 'exercise_cpu'), 'The CPU gate must be separately labeled from CUDA'
    result, code = gate.exercise_cpu(2, num_uavs=1, grid_w=20)
    assert code == 0
    assert result['status'] == 'WARP_CPU_NETWORK_EXECUTED_NOT_CUDA_OR_RL'
    assert result['cuda_executed'] is False
    assert result['full_rl_training'] is False


def test_cpu_gate_does_not_claim_cuda_without_hardware():
    import torch
    if torch.cuda.is_available():
        pytest.skip('This assertion covers the no-CUDA host')
    result, code = gate.exercise(2, 'cuda:0')
    assert code == 3
    assert result['status'] == 'BLOCKED_NO_CUDA_HOST'
