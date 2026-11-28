"""A simultaneous slot is a collision opportunity, independent of radio index."""
import os
import numpy as np
import pytest
import warp as wp

from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
from zhang_xue.gpu.state_v67 import NEVER_US, ST_BACKOFF, ST_DIFS, ST_TX


def _put(net, name, values):
    arr = net.data[name]
    wp.copy(arr, wp.array(np.asarray(values), dtype=arr.dtype, device=net.device))


def contention_fixture(device, *, hidden=False, difs=False, equal=True):
    """Two broadcasters and one receiver; no routes/packet injections are mocked."""
    net = WarpCompressedRadioNetworkV67(1, num_uavs=2, grid_w=2, device=device, seed=44)
    gain = np.full((1, 3, 3), 1e-4, dtype=np.float32)
    gain[0, np.arange(3), np.arange(3)] = 0.0
    if hidden:
        gain[0, 0, 1] = gain[0, 1, 0] = 0.0
    net.set_channel_gain(gain)
    _put(net, "hello_due", [[NEVER_US] * 3])
    _put(net, "hello_pending", [[1, 1, 0]])
    _put(net, "state", [[ST_DIFS if difs else ST_BACKOFF] * 2 + [0]])
    _put(net, "deadline", [[100, 100 if equal else 120, NEVER_US]])
    if difs:
        # Frozen zero-slot counters expire immediately after the same DIFS.
        _put(net, "frozen", [[1, 1, 0]])
        _put(net, "backoff_rem", [[0, 0, 0]])
    return net


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("difs", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_equal_slot_both_radios_start_before_carrier_sense_changes(device, difs, masked):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    net = contention_fixture(device, difs=difs)
    if masked:
        from zhang_xue.gpu.kernels.mission_v79_reset_obs import step_v67_masked
        wp.launch(step_v67_masked, dim=1, inputs=[
            net.state, net.dims, wp.ones(1, dtype=wp.uint8, device=device),
            101, 4096, 0.1, 10 ** (-11.2), 4.38957709293155e-13, 10 ** 0.6,
        ], device=device)
    else:
        net.step(duration_us=101)
    assert net.data["state"].numpy()[0, :2].tolist() == [ST_TX, ST_TX]
    assert net.data["rx_interf"].numpy()[0, 2] > 0
    # Neither of the equal-power colliding HELLOs may establish a route.
    net.step(duration_us=400)
    assert net.data["route_hop"].numpy()[0, 2, :2].tolist() == [-1, -1]


def test_later_counter_freezes_on_an_earlier_transmission():
    net = contention_fixture("cpu", equal=False)
    net.step(duration_us=101)
    assert int(net.data["state"].numpy()[0, 0]) == ST_TX
    assert int(net.data["state"].numpy()[0, 1]) != ST_TX
    assert int(net.data["backoff_rem"].numpy()[0, 1]) == 20


def test_hidden_equal_slot_collision_is_not_a_successful_hello():
    net = contention_fixture("cpu", hidden=True)
    net.step(duration_us=501)
    assert net.data["route_hop"].numpy()[0, 2, :2].tolist() == [-1, -1]

