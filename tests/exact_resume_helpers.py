"""Shared small-world fixture and exact array oracle for CPU/CUDA resume tests."""
from test_continuation import small_config


def exact_world_config(folder="unused", algorithm="matd3"):
    config = small_config(folder, algorithm)
    config.runtime.environment_backend = "warp_packet"
    config.network.enabled = True
    config.world.max_steps = 12
    return config


def assert_snapshot_arrays(test, actual, expected):
    import torch
    test.assertEqual(actual.keys(), expected.keys())
    for name, value in expected.items():
        with test.subTest(array=name):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True)


def assert_current_world(test, vector, snapshot):
    """Compare device storage one array at a time, bounding host oracle memory."""
    import warp as wp
    arrays = {name: wp.to_torch(value) for name, value in vars(vector.s).items()
              if isinstance(value, wp.array)}
    test.assertEqual(arrays.keys(), snapshot["arrays"].keys())
    for name, expected in snapshot["arrays"].items():
        test.assertEqual(arrays[name].device, vector.device)
        assert_snapshot_arrays(test, {name: arrays[name].cpu()}, {name: expected})
