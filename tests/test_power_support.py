import numpy as np
import torch

from uav_network.configuration import Config
from uav_network.environment.env import Action
from uav_network.experiments.power_support import apply_power_support


def settings(fraction=0.0):
    c = Config()
    c.runtime.algorithm = "masac"
    c.runtime.power_support_fraction = fraction
    c.network.power_control = "learned"
    return c


def base_action(as_tensor=False):
    continuous = np.asarray([
        [[0.1, 0.2, 0.31], [0.3, 0.4, 0.32], [0.5, 0.6, 0.33]],
        [[0.7, 0.8, 0.34], [0.9, 1.0, 0.35], [0.2, 0.3, 0.36]],
    ], dtype=np.float32)
    destination = np.asarray([[1, 0, 2], [2, 1, 0]], dtype=np.int64)
    if as_tensor:
        continuous, destination = torch.from_numpy(continuous), torch.from_numpy(destination)
    return Action(continuous, destination)


def observations(as_tensor=False):
    obs = np.zeros((2, 3, 16), dtype=np.float32)
    obs[..., 9] = 1
    if as_tensor:
        obs = torch.from_numpy(obs)
    return obs


def test_zero_fraction_is_exact_noop_without_copying_rng_or_action_values():
    action = base_action()
    before_c = action.continuous.copy()
    before_d = action.destination.copy()
    result = apply_power_support(action, observations(), step=4096, config=settings(0.0))
    np.testing.assert_array_equal(result.continuous, before_c)
    np.testing.assert_array_equal(result.destination, before_d)


def test_full_support_changes_only_power_for_active_non_silent_rows_numpy():
    action = base_action()
    original = action.continuous.copy()
    result = apply_power_support(action, observations(), step=4096, config=settings(1.0))
    np.testing.assert_array_equal(result.destination, action.destination)
    np.testing.assert_array_equal(result.continuous[..., :2], original[..., :2])
    silent = action.destination == 0
    np.testing.assert_array_equal(result.continuous[..., -1][silent], original[..., -1][silent])
    grid = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    assert np.isin(result.continuous[..., -1][~silent], grid).all()


def test_full_support_preserves_tensor_device_dtype_and_dead_rows():
    action = base_action(as_tensor=True)
    obs = observations(as_tensor=True)
    obs[1, 1, 9] = 0
    original = action.continuous.clone()
    result = apply_power_support(action, obs, step=8192, config=settings(1.0))
    assert isinstance(result.continuous, torch.Tensor)
    assert result.continuous.dtype == original.dtype
    assert result.continuous.device == original.device
    torch.testing.assert_close(result.continuous[..., :2], original[..., :2], rtol=0, atol=0)
    assert result.continuous[1, 1, -1] == original[1, 1, -1]


def test_support_is_masac_learned_power_only():
    action = base_action()
    original = action.continuous.copy()
    c = settings(1.0)
    c.runtime.algorithm = "matd3"
    result = apply_power_support(action, observations(), step=0, config=c)
    np.testing.assert_array_equal(result.continuous, original)
    c.runtime.algorithm = "masac"
    c.network.power_control = "link_budget"
    result = apply_power_support(action, observations(), step=0, config=c)
    np.testing.assert_array_equal(result.continuous, original)


def test_runtime_rejects_power_support_fraction_outside_unit_interval():
    c = settings(1.01)
    try:
        c.validate()
    except ValueError as error:
        assert "power_support_fraction" in str(error)
    else:
        raise AssertionError("invalid power_support_fraction must be rejected")
