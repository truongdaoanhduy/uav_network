"""Byte counts must follow actual replay tensors, not a fixed GiB slogan."""
import unittest

class MemoryBudgetTest(unittest.TestCase):
    def setUp(self):
        global replay_storage_bytes, network_storage_lower_bound
        try:
            from uav_network.experiments.zhang_xue_memory_budget import replay_storage_bytes, network_storage_lower_bound
        except ModuleNotFoundError:
            self.fail('Replay and network byte accounting is not implemented')

    def test_replay_counts_current_and_next_observation_state_and_masks(self):
        # One row: obs 48 + state 48 + actions 40 + reward/done 8 + masks 12.
        self.assertEqual(replay_storage_bytes(10, 2, 6, 12, 2, 3), 1560)

    def test_actual_pilot_one_million_rows_exceeds_five_gib(self):
        self.assertEqual(replay_storage_bytes(1_000_000, 6, 121, 726, 2, 6), 6_080_000_000)

    def test_network_lower_bound_includes_receiver_ledger_and_timestamps(self):
        # E1,R2,A1,C3,Q4,K5: payload320 + watermark24 + ledger24 + timestamp24.
        b = network_storage_lower_bound(1, 1, 2, 3, 4, 5)
        self.assertEqual(b['total_bytes'], 392)
        self.assertFalse(b['complete_runtime_budget'])

    def test_nonpositive_or_fractional_shape_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replay_storage_bytes(value, 6, 121, 726, 2, 6)


if __name__ == '__main__':
    unittest.main()
