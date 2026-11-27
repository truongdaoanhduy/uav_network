"""Adversarial tests that v5.7 did not cover; static Warp assertions are not GPU parity."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from zhang_xue.compat.belief_mobility import kinematic_safety_shield, trajectory_safety_violations
from zhang_xue.compat.env import ZhangXue7NodePacketEnv, make_v57_config
from zhang_xue.compat.warp_mac import SHAPES, backing_bytes_per_environment
from zhang_xue.gpu.kernels.phy import v57_gpu_memory_budget


def test_infeasible_shield_fallback_is_detected_not_claimed_safe():
    pos = np.array([[100., 100.], [160., 100.]])
    previous = np.array([[25., 0.], [-25., 0.]])
    safe, _ = kinematic_safety_shield(pos, previous, previous, np.ones(2, bool), [],
                                      map_size_m=1000., dt_s=1., v_max_mps=25., a_max_mps2=5.,
                                      uav_safe_distance_m=50.)
    np.testing.assert_allclose(safe, np.array([[20., 0.], [-20., 0.]]))
    assert np.linalg.norm((pos[0] + safe[0]) - (pos[1] + safe[1])) == 20.
    bad = trajectory_safety_violations(pos, safe, np.ones(2, bool), [],
                                       map_size_m=1000., dt_s=1., uav_safe_distance_m=50.)
    assert (0, 1) in bad["pairwise"]


def test_termination_penalty_on_infeasible_shield_motion():
    cfg = make_v57_config()
    env = ZhangXue7NodePacketEnv(cfg, seed=11)
    env.reset(seed=11)
    env.positions[:6,:2] = [
        [100., 100.], [160., 100.], [100., 550.], [200., 550.], [300., 550.], [400., 550.]
    ]
    env.velocities[:2] = [[25., 0.], [-25., 0.]]
    v = env.velocities.copy()
    _,_,_,r,terminated,truncated,info = env.step(v, np.zeros(6,dtype=np.int32))
    assert terminated and not truncated and info["safety_violation"]
    assert (0,1) in info["safety_violations"]["pairwise"]
    assert info["r_safety_failure"] == -cfg.safety_failure_penalty
    assert all(np.linalg.norm(env.velocities[u]-v[u]) <= cfg.a_max_mps2*cfg.dt_s+1e-7 for u in range(6))


def test_actual_warp_memory_budget_and_projection_are_not_confused():
    expected = sum(np.dtype(dt).itemsize*int(np.prod(shape)) for shape, dt in SHAPES.values())
    assert backing_bytes_per_environment() == expected == 635931
    budget = v57_gpu_memory_budget(2048,300_000,514)
    assert budget["actual_soa_bytes_per_env"] == expected
    assert budget["compressed_layout_implemented"] == 0
    assert budget["cuda_peak_verified"] == 0
    assert budget["network_soa_gib"] > 1.2
    assert budget["proposed_compressed_soa_bytes_per_env_unimplemented"] == 315710


def test_warp_queue_priority_fix_copies_payload_and_scans_64_slots_statically():
    """Static source contract only: full Warp compilation/running still required."""
    evt = (SRC / "zhang_xue/gpu/kernels/events.py").read_text()
    mac = (SRC / "zhang_xue/gpu/mac_functions.py").read_text()
    assert "for offset in range(64):" in evt
    assert "_promote_queued_slot(s, e, r, chosen_offset)" in evt
    assert "s.q_word0[e, r, head, k] = s.q_word0[e, r, chosen, k]" in mac
    assert "s.q_word1[e, r, head, k] = s.q_word1[e, r, chosen, k]" in mac
    for name in ('q_pid','q_src','q_dst','q_seq','q_expiry','q_visited','q_hops','q_entries'):
        assert f"s.{name}[e, r, chosen]" in mac


def test_effective_ewma_math_not_yet_wired_to_env_packet_ledger():
    env_code = (SRC / "zhang_xue/environment/observations.py").read_text()
    network_code = (SRC / "zhang_xue/communication/lifecycle.py").read_text()
    # This is a progress assertion, not a claim of equivalence with Xue.
    assert "fuse_provenance_counts(" in env_code
    assert "ewma_effective_counts_update(" not in network_code
    assert "ledger_rx_age_s" in network_code


def test_impossible_target_placement_raises_instead_of_infinite_loop():
    # Original v5.7 accepted num_targets=400, but buildings exclude some cells.
    env = ZhangXue7NodePacketEnv(make_v57_config(num_targets=400), seed=11)
    with pytest.raises(ValueError, match="Not enough legal target cells"):
        env.reset(seed=11)


def test_configurable_gcs_target_exclusion_applied():
    c = make_v57_config(num_targets=7, gcs_target_exclusion_m=200.)
    env = ZhangXue7NodePacketEnv(c,seed=12)
    env.reset(seed=12)
    for cid in np.flatnonzero(env.true_targets):
        xy = np.array([((int(cid)%20)+.5)*50., ((int(cid)//20)+.5)*50.])
        assert np.linalg.norm(xy-env.positions[6,:2]) >= 200.
