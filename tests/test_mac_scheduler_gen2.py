"""Exact ordering primitives for the semantics-preserving MAC Scheduler Gen2."""
import importlib
import importlib.util
import unittest

import numpy as np

from uav_network.configuration import Config

HAS_WARP = importlib.util.find_spec("warp") is not None

if HAS_WARP:
    import warp as wp

    from uav_network.environment.gpu.state import WorldState

    _events = importlib.import_module("uav_network.communication.gpu.events")
    _key_before = getattr(_events, "key_before", None)
    _candidate_before_root = getattr(_events, "candidate_before_root", None)

    if _key_before is not None:
        @wp.kernel
        def _key_before_probe(time_a: wp.float64, priority_a: int, order_a: wp.int64,
                              time_b: wp.float64, priority_b: int, order_b: wp.int64,
                              out: wp.array(dtype=wp.int32)):
            out[0] = int(_key_before(time_a, priority_a, order_a,
                                     time_b, priority_b, order_b))
    else:
        _key_before_probe = None

    if _candidate_before_root is not None:
        @wp.kernel
        def _candidate_root_probe(s: WorldState, time: wp.float64, priority: int,
                                  order: wp.int64, out: wp.array(dtype=wp.int32)):
            out[0] = int(_candidate_before_root(s, 0, time, priority, order))
    else:
        _candidate_root_probe = None
else:
    wp = None
    _events = None
    _key_before_probe = None
    _candidate_root_probe = None


@unittest.skipUnless(HAS_WARP, "optional warp_packet dependency not installed")
class ExactEventOrderingTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        self.assertIsNotNone(_key_before_probe, "key_before helper is missing")
        self.assertIsNotNone(_candidate_root_probe, "candidate_before_root helper is missing")

    def key_before(self, first, second):
        out = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(_key_before_probe, dim=1,
                  inputs=[wp.float64(first[0]), int(first[1]), wp.int64(first[2]),
                          wp.float64(second[0]), int(second[1]), wp.int64(second[2]), out],
                  device=self.device)
        return bool(out.numpy()[0])

    def test_key_before_orders_timestamp_priority_then_insertion_order(self):
        cases = [
            ((9.0, 1, 10), (10.0, 0, 0), True, "earlier timestamp"),
            ((11.0, 0, 0), (10.0, 1, 10), False, "later timestamp"),
            ((10.0, 0, 99), (10.0, 1, 1), True, "lower priority"),
            ((10.0, 2, 1), (10.0, 1, 99), False, "higher priority"),
            ((10.0, 1, 7), (10.0, 1, 8), True, "lower insertion order"),
            ((10.0, 1, 9), (10.0, 1, 8), False, "higher insertion order"),
            ((10.0, 1, 8), (10.0, 1, 8), False, "exact key equality"),
        ]
        for first, second, expected, label in cases:
            with self.subTest(label=label):
                self.assertEqual(self.key_before(first, second), expected)

    def state(self):
        c = Config()
        c.world.num_uavs = 2
        c.world.num_targets = 1
        c.world.num_obstacles = 0
        c.runtime.num_envs = 1
        state = importlib.import_module("uav_network.environment.gpu.state")
        return state.allocate_state(c, num_envs=1, seeds=[44], device=self.device)[0]

    def candidate_before_root(self, s, candidate):
        out = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(_candidate_root_probe, dim=1,
                  inputs=[s, wp.float64(candidate[0]), int(candidate[1]),
                          wp.int64(candidate[2]), out], device=self.device)
        return bool(out.numpy()[0])

    def test_candidate_before_root_is_strict_and_empty_heap_is_available(self):
        s = self.state()
        self.assertTrue(self.candidate_before_root(s, (10.0, 1, 8)),
                        "an empty heap has no competing root")

        s.event_count.assign(np.array([1], dtype=np.int32))
        s.event_time.assign(np.array([[10.0] + [0.0] * (s.event_time.shape[1] - 1)], dtype=np.float64))
        priority = np.zeros(s.event_priority.shape, dtype=np.int32)
        priority[0, 0] = 1
        s.event_priority.assign(priority)
        order = np.zeros(s.event_order.shape, dtype=np.int64)
        order[0, 0] = 8
        s.event_order.assign(order)

        before = {
            "event_count": s.event_count.numpy().copy(),
            "event_time": s.event_time.numpy().copy(),
            "event_priority": s.event_priority.numpy().copy(),
            "event_order": s.event_order.numpy().copy(),
            "event_sequence": s.event_sequence.numpy().copy(),
        }
        self.assertTrue(self.candidate_before_root(s, (9.0, 9, 99)))
        self.assertTrue(self.candidate_before_root(s, (10.0, 0, 99)))
        self.assertTrue(self.candidate_before_root(s, (10.0, 1, 7)))
        self.assertFalse(self.candidate_before_root(s, (10.0, 1, 8)),
                         "an exact-key tie is not strictly earlier")
        self.assertFalse(self.candidate_before_root(s, (10.0, 1, 9)))
        self.assertFalse(self.candidate_before_root(s, (11.0, 0, 0)))

        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(s, name).numpy(), expected, err_msg=name)


if __name__ == "__main__":
    unittest.main()
