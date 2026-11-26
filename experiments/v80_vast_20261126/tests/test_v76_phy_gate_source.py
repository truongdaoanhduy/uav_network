"""Verify GPU/CPU PHY gate is explicit and does not mislabel simulator equivalence."""
from scripts.gpu.warp_phy_cuda_gate import run_phy_gate
from zhang_xue.gpu.runtime_v67 import HAS_WARP


def test_gate_returns_true_blocker_without_warp():
    if not HAS_WARP:
        assert run_phy_gate('cpu')['status'] == 'BLOCKED_NO_WARP'


def test_gate_reports_not_full_rl_source():
    from inspect import getsource
    s = getsource(run_phy_gate)
    assert 'PHY_REFERENCE_PASS_NOT_FULL_RL' in s
    assert 'full_gpu_rl_training' in s
    assert 'np.testing.assert_allclose' in s
