"""Integration smoke for the actual FP32 Warp PHY, DS, and slotted MAC kernels (CPU)."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from uav_network.communication.gpu.phy_sinr_fp32 import compute_phy_matrices_fp32
from uav_network.communication.gpu.mac_slotted_fp32 import mac_slotted_step_fp32
from uav_network.environment.gpu.ds_aoi_belief_fp32 import ds_sense_and_energy_step_fp32


def _a(data, dtype):
    return wp.array(np.asarray(data), dtype=dtype, device="cpu")


def test_phy_ds_kernel_executes_on_cpu():
    wp.init()
    E = 1
    pos = np.zeros((E, 7, 3), np.float32)
    pos[0, :6, :] = [60.0, 60.0, 50.0]
    pos[0, 6, :] = [40.0, 40.0, 10.0]
    ob = np.zeros((E, 22, 4), np.float32)
    dist = wp.zeros((E, 7, 7), wp.float32, device="cpu")
    los = wp.zeros((E, 7, 7), wp.int32, device="cpu")
    power = wp.zeros((E, 7, 7), wp.float32, device="cpu")
    cca = wp.zeros((E, 7, 7), wp.int32, device="cpu")
    wp.launch(compute_phy_matrices_fp32, dim=E, inputs=[_a(pos, wp.vec3f), _a(ob, wp.vec4f), 0, dist, los, power, cca], device="cpu")
    assert power.numpy()[0, 0, 6] > 0.0
    assert los.numpy()[0, 0, 6] == 1

    uav_pos = _a(pos[:, :6, :], wp.vec3f)
    zeros_i = lambda shape: wp.zeros(shape, wp.int32, device="cpu")
    zeros_f = lambda shape: wp.zeros(shape, wp.float32, device="cpu")
    qcount = zeros_i((E, 6))
    q_cell_id, q_ts, q_retry, q_hops, q_visit = (
        zeros_i((E, 6, 32)), zeros_f((E, 6, 32)), zeros_i((E, 6, 32)),
        zeros_i((E, 6, 32)), zeros_i((E, 6, 32)),
    )
    truth = _a(np.full((E, 400), -1, np.int32), wp.int32)
    dpos = _a(np.full((E, 400), 30, np.int32), wp.int32)
    dneg = zeros_i((E, 400))
    dmf = zeros_f((E, 400))
    dmu = _a(np.ones((E, 400), np.float32), wp.float32)
    reported = zeros_i((E, 400))
    battery = _a(np.full((E, 6), 150000.0, np.float32), wp.float32)
    alive = _a(np.ones((E, 6), np.int32), wp.int32)
    rand = _a(np.array([12345], np.uint32), wp.uint32)
    t = zeros_f(1)
    for _ in range(30):
        wp.launch(ds_sense_and_energy_step_fp32, dim=E, inputs=[
            uav_pos, wp.zeros((E, 6), wp.vec3f, device="cpu"), battery, alive,
            truth, dpos, dneg, dmf, dmu, reported, qcount, q_cell_id, q_ts,
            q_retry, q_hops, q_visit, t, rand, zeros_f(E), zeros_f(E), 60.0, 1.0,
        ], device="cpu")
    # Even though ALL cells are empty, false-positive sensor readings can create
    # hypothesis reports. Truth must never be an oracle gate for packet creation.
    assert int(qcount.numpy().sum()) > 0
    assert (q_cell_id.numpy() >= 0).all()


def test_mac_gcs_scores_truth_only_after_first_arrival():
    wp.init()
    E = 1
    ints = lambda shape: wp.zeros(shape, wp.int32, device="cpu")
    floats = lambda shape: wp.zeros(shape, wp.float32, device="cpu")
    qcount = _a(np.array([[2, 0, 0, 0, 0, 0]], np.int32), wp.int32)
    ids = np.zeros((E, 6, 32), np.int32)
    ids[0, 0, :2] = [0, 1]
    q_cell_id = _a(ids, wp.int32)
    q_ts = floats((E, 6, 32))
    q_retry = ints((E, 6, 32))
    q_hops = ints((E, 6, 32))
    q_visit = _a(np.full((E, 6, 32), 1, np.int32), wp.int32)
    rx = _a(np.full((E, 7, 7), 1.0e-6, np.float32), wp.float32)
    cca = ints((E, 7, 7))
    alive = _a(np.ones((E, 6), np.int32), wp.int32)
    act = _a(np.array([[6, 0, 0, 0, 0, 0]], np.int32), wp.int32)
    delivered = ints((E, 400))
    truth = np.full((E, 400), -1, np.int32)
    truth[0, 1] = 24
    truth = _a(truth, wp.int32)
    rand = _a(np.array([12345], np.uint32), wp.uint32)
    tm = floats(1)
    aoi = floats(E)
    new_true = ints(E)
    new_false = ints(E)
    multi = ints(E)
    collision = ints(E)
    deaf = ints(E)
    drop = ints(E)
    ttl = ints(E)

    def step():
        wp.launch(mac_slotted_step_fp32, dim=E, inputs=[
            act, rx, cca, alive, tm, qcount, q_cell_id, q_ts, q_retry, q_hops,
            q_visit, delivered, truth, rand, aoi, new_true, new_false, multi,
            collision, deaf, drop, ttl, 0.02,
        ], device="cpu")

    step()  # false cell 0 is delivered but cannot receive positive AoI reward
    assert int(new_true.numpy()[0]) == 0
    assert int(new_false.numpy()[0]) == 1
    assert float(aoi.numpy()[0]) == 0.0
    step()  # true cell 1 should be rewarded exactly once
    assert int(new_true.numpy()[0]) == 1
    assert int(new_false.numpy()[0]) == 0
    assert float(aoi.numpy()[0]) == pytest.approx(1.0)
    step()  # no pending packets => no reward farming
    assert int(new_true.numpy()[0]) == 0
    assert int(new_false.numpy()[0]) == 0
    assert int(delivered.numpy()[0, 0]) == 1
    assert int(delivered.numpy()[0, 1]) == 1


def test_slotted_cca_deferral_improves_two_sender_collision_case():
    """Compare identical seeded subslot draws with CCA enabled vs hidden nodes."""
    wp.init()
    E = 128
    base_ids = np.zeros((E, 6, 32), np.int32)
    base_ids[:, 1, 0] = 1
    q_init = np.zeros((E, 6), np.int32)
    q_init[:, :2] = 1
    src_mask = np.zeros((E, 6), np.int32)
    src_mask[:, :2] = 6
    truths = np.full((E, 400), -1, np.int32)
    truths[:, :2] = np.array([2, 7], np.int32)
    rx = np.full((E, 7, 7), 1.0e-6, np.float32)
    # A weak interferer makes both collide if they use the same subslot,
    # but each succeeds alone; all other UAVs are silent.

    def simulate(hear: bool):
        i = lambda shape: wp.zeros(shape, wp.int32, device="cpu")
        f = lambda shape: wp.zeros(shape, wp.float32, device="cpu")
        cca = np.zeros((E, 7, 7), np.int32)
        if hear:
            cca[:, 0, 1] = cca[:, 1, 0] = 1
        delivered = i((E, 400))
        new_true = i(E)
        wp.launch(mac_slotted_step_fp32, dim=E, inputs=[
            _a(src_mask, wp.int32), _a(rx, wp.float32), _a(cca, wp.int32),
            _a(np.ones((E, 6), np.int32), wp.int32), f(1),
            _a(q_init, wp.int32), _a(base_ids, wp.int32), f((E, 6, 32)),
            i((E, 6, 32)), i((E, 6, 32)), i((E, 6, 32)), delivered,
            _a(truths, wp.int32), _a(np.arange(1, E + 1, dtype=np.uint32), wp.uint32),
            f(E), new_true, i(E), i(E), i(E), i(E), i(E), i(E), 0.02,
        ], device="cpu")
        return int(new_true.numpy().sum())

    with_cca = simulate(True)
    no_cca = simulate(False)
    assert with_cca > no_cca, (with_cca, no_cca)
