"""v7.0 RED->GREEN regressions for risks hidden by CPU-only v6.9 tests."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from zhang_xue.gpu.parity_v67_ref import NumPySoARefV67
from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67

PKG = Path(__file__).resolve().parents[1] / "research/zhang_xue/src/zhang_xue/gpu"


def _source(path: str) -> str:
    return (PKG / path).read_text(encoding="utf-8")


def test_packet_payload_uses_sender_saved_before_rx_lock_clear():
    mac = _source("kernels/mac_v67.py")
    events = _source("kernels/events_v67.py")
    assert "tx_node: int" in mac
    section = mac.split("def _deliver_data_v67(", 1)[1].split("\n    @wp.func", 1)[0]
    assert "int(s.rx_lock[e, rx])" not in section, "rx_lock is cleared before delivery"
    assert "_deliver_data_v67(s, dims, e, rx, tx, slot, src, seq)" in events


def test_gpu_receive_threshold_default_matches_cpu_snr_decode():
    default = inspect.signature(WarpCompressedRadioNetworkV67).parameters["rx_lock_on_cca"].default
    assert default is False


def test_partial_reset_does_not_keep_stale_evidence_or_network_state():
    ref = NumPySoARefV67(num_envs=1, num_uavs=2, grid_w=20)
    ref.sense_batch(np.array([[10, -1]]), np.array([[1, 0]]), t_s=3)
    ref.data["next_seq"][0] = 567
    ref.data["stats_gcs"][0] = 12
    ref.data["route_hop"][0, 0, 1] = 1
    ref.data["pending_word1"][0, 0, 0, 0] = 812
    ref.reset()
    assert int(ref.data["ledger_pos"][0, 0, 0, 10]) == 0
    assert int(ref.data["ledger_gen"][0, 0, 0, 10]) == -1
    assert int(ref.data["ack_origin_gen"][0, 0, 1, 10]) == -1
    assert int(ref.data["stats_gcs"][0]) == 0
    assert int(ref.data["next_seq"][0]) == 1
    assert int(ref.data["route_hop"][0, 0, 1]) == -1
    assert int(ref.data["pending_word1"][0, 0, 0, 0]) == 0


def test_kernel_reset_module_defines_numpy_used_by_reset_all():
    tree = ast.parse(_source("kernels/reset_v67.py"))
    aliases = [alias for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    assert any(a.name == "numpy" and a.asname == "np" for a in aliases)


def test_pending_slot_is_held_until_ack_or_expiry():
    from zhang_xue.gpu.runtime_v67 import pending_slot_available
    assert pending_slot_available(0, 10, 20) is True
    assert pending_slot_available(1, 10, 20) is False
    assert pending_slot_available(1, 20, 20) is True
    code = _source("kernels/inject_v67.py")
    assert "if s.pending_valid[e, u, p_idx] != wp.uint8(0):" in code
    assert "s.pending_expiry[e, u, p_idx]" in code


def test_warp_runtime_validates_phy_array_shapes_before_launch():
    from zhang_xue.gpu.runtime_v67 import normalize_phy_inputs
    p = np.zeros((2, 4, 3), dtype=np.float32)
    a, b, c = normalize_phy_inputs(p, [], [], envs=2, radios=4)
    assert a.shape == (2, 4, 3) and b.shape == (0, 4) and c.shape == (0, 2)
    with pytest.raises(ValueError):
        normalize_phy_inputs(p[:, :3], [], [], envs=2, radios=4)
    with pytest.raises(ValueError):
        normalize_phy_inputs(p, [(1, 2, 3)], [], envs=2, radios=4)
    with pytest.raises(ValueError):
        normalize_phy_inputs(p, [], [(float("nan"), 1)], envs=2, radios=4)


def test_warp_status_not_just_method_existence():
    from scripts.maintenance.q1_gpu_preflight import q1_gpu_readiness
    report = q1_gpu_readiness()
    assert report["ready"] is False
    assert "full_rl_warp_rollout_update_eval_not_integrated" in report["blockers"]
    assert "gpu_cpu_packet_event_parity_unverified" in report["blockers"]
    assert report["v67_numpy_soa_ref_cpu_parity"] in ("IDEAL_HOP_SANITY_PASS", "IDEAL_HOP_SANITY_FAIL")


def test_sensing_inputs_reject_invalid_generation_and_false_detection_masks():
    from zhang_xue.gpu.runtime_v67 import normalize_sensing_inputs
    ids, detected, t_s = normalize_sensing_inputs(
        np.array([[0, -1]], dtype=np.int32), np.array([[1, 0]], dtype=np.int32),
        5, envs=1, uavs=2, cells=400)
    assert ids.shape == detected.shape == (1, 2) and t_s == 5
    with pytest.raises(ValueError):
        normalize_sensing_inputs([[400, -1]], [[1, 0]], 5, envs=1, uavs=2, cells=400)
    with pytest.raises(ValueError):
        normalize_sensing_inputs([[2, -1]], [[2, 0]], 5, envs=1, uavs=2, cells=400)
    with pytest.raises(ValueError):
        normalize_sensing_inputs([[2, -1]], [[1, 0]], 40000, envs=1, uavs=2, cells=400)


def test_compact_action_input_validation_prevents_invalid_sends():
    from zhang_xue.gpu.runtime_v67 import normalize_compact_actions
    act, xy = normalize_compact_actions([[2, 1]], [[[0, 0], [1, 2]]], envs=1, uavs=2)
    assert act.dtype == np.int32 and xy.dtype == np.float32
    with pytest.raises(ValueError):
        normalize_compact_actions([[3, 1]], [[[0, 0], [1, 2]]], envs=1, uavs=2)
    with pytest.raises(ValueError):
        normalize_compact_actions([[2, 1]], [[[0, float('nan')], [1, 2]]], envs=1, uavs=2)


def test_gpu_event_overflow_fails_closed_before_training_can_use_corrupted_time():
    source = _source('runtime_v67.py')
    assert 'self.data["overflow"].numpy()' in source
    assert 'raise RuntimeError("Warp packet-event budget exceeded' in source
    assert 'self.data["now"].numpy()' in source


def test_v67_30_uavs_at_2km_cannot_be_called_2048_env_t4_ready():
    from zhang_xue.gpu.layout_v67 import assert_network_allocation_fits
    from zhang_xue.gpu.layout_v67 import backing_bytes_per_environment_v67
    gib = 1024**3
    assert backing_bytes_per_environment_v67(30, 40) * 2048 > 190*gib
    with pytest.raises(MemoryError):
        assert_network_allocation_fits(2048, 30, 40, free_bytes=16*gib)
    result = assert_network_allocation_fits(2048, 6, 20, free_bytes=16*gib)
    assert result['estimated_network_bytes'] > 0
    assert result['safe_allocation_limit_bytes'] > result['estimated_network_bytes']


def test_warp_atomic_array_argument_uses_fixed_field_not_dynamic_conditional_array():
    mac = _source('kernels/mac_v67.py')
    events = _source('kernels/events_v67.py')
    assert 'wp.atomic_add(s.stats_gcs if' not in mac
    assert 'wp.atomic_add(s.stats_loop if' not in events
