"""V119 dynamic C38 motor ABI parity: actor input disk projection before accel."""
from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip('warp')
from zhang_xue.gpu.kernels.mission_v80_shield import _candidate_v80
from zhang_xue.learners.actor_adapter import scale_normalized_xy_to_mps
from zhang_xue.mobility.flight_safety_energy import clamp_velocity_with_accel


def test_cpu_warp_candidate_parity_nonzero_prev_velocity():
    # At (1,1) and v_prev=(5,0), V118 CPU and Warp disagree on the
    # acceleration direction even though both end up under speed limit.
    controls = np.array([[1., 1.], [1., 0.], [0.6, 0.8],
                         [-1., 1.], [-1., -1.], [0., 0.]], dtype=np.float32)
    prev = np.array([[5., 0.], [0., 4.], [4., 3.],
                     [-2., 5.], [7., -1.], [0., 0.]], dtype=np.float32)
    commands = scale_normalized_xy_to_mps(controls, v_max_mps=15.0)
    expected = np.array([clamp_velocity_with_accel(prev[u], commands[u],
                    dt_s=1.0, v_max_mps=15., a_max_mps2=5.) for u in range(6)])
    for dev in ['cpu'] + (['cuda:0'] if wp.is_cuda_available() else []):
        a = wp.array(controls[None], dtype=wp.float32, device=dev)
        v = wp.array(prev[None], dtype=wp.float32, device=dev)
        cand = wp.zeros((1, 6, 2), dtype=wp.float32, device=dev)
        wp.launch(_candidate_v80, dim=(1, 6), inputs=[a, v, cand, 1., 15., 5.], device=dev)
        wp.synchronize_device(dev)
        actual = cand.numpy()[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
        assert np.all(np.linalg.norm(actual, axis=1) <= 15.0 + 1e-6)


def test_v118_old_per_axis_scaling_did_not_equal_cpu_disk():
    p = np.array([5., 0.])
    a = np.array([1., 1.])
    from zhang_xue.mobility.flight_safety_energy import clamp_velocity_with_accel
    old_gpu = clamp_velocity_with_accel(p, a * 15., v_max_mps=15., a_max_mps2=5.)
    cpu = clamp_velocity_with_accel(p, a * (15. / np.sqrt(2)), v_max_mps=15., a_max_mps2=5.)
    assert np.linalg.norm(old_gpu - cpu) > 0.1  # dynamic counterexample
