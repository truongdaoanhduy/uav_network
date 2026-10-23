"""Exact derived frame bitmap primitives; Gen1 canonical arrays remain the oracle."""
import importlib
import importlib.util
import unittest

import numpy as np

from uav_network.configuration import Config


@unittest.skipUnless(importlib.util.find_spec("warp"), "optional warp_packet dependency not installed")
class FrameFreePrimitiveTests(unittest.TestCase):
    device = "cpu"

    def setup_state(self, num_uavs=6, max_queue_packets=64):
        import warp as wp
        state = importlib.import_module("uav_network.environment.gpu.state")
        frame_free = importlib.import_module("uav_network.communication.gpu.frame_free")
        c = Config()
        c.world.num_uavs = num_uavs
        c.world.num_targets = 1
        c.world.num_obstacles = 0
        c.network.max_queue_packets = max_queue_packets
        s, p = state.allocate_state(c, seeds=[44], device=self.device)
        words = frame_free.bitmap_words(p.frame_capacity)
        free_words = wp.zeros((1, words), dtype=wp.uint64, device=self.device)
        reclaim_words = wp.zeros((1, words), dtype=wp.uint64, device=self.device)
        return wp, frame_free, s, p, free_words, reclaim_words

    def rebuild(self, wp, frame_free, s, p, free_words, reclaim_words):
        wp.launch(frame_free.rebuild_frame_bitmaps_kernel, dim=1,
                  inputs=[s, p, free_words, reclaim_words], device=self.device)

    def take(self, wp, frame_free, s, p, free_words, reclaim_words):
        selected = wp.full(1, -99, dtype=wp.int32, device=self.device)
        wp.launch(frame_free.apply_reclaims_and_take_smallest_kernel, dim=1,
                  inputs=[s, p, free_words, reclaim_words, selected], device=self.device)
        return int(selected.numpy()[0])

    def test_word_count_and_tail_mask_for_1632_frames(self):
        wp, frame_free, s, p, free_words, reclaim_words = self.setup_state()
        self.assertEqual(p.frame_capacity, 1632)
        self.assertEqual(frame_free.bitmap_words(p.frame_capacity), 26)
        active = np.ones((1, p.frame_capacity), dtype=np.int32)
        active[0, 1631] = 0
        s.tx_active.assign(active)
        self.rebuild(wp, frame_free, s, p, free_words, reclaim_words)
        self.assertEqual(self.take(wp, frame_free, s, p, free_words, reclaim_words), 1631)
        self.assertEqual(self.take(wp, frame_free, s, p, free_words, reclaim_words), -1)

    def test_smallest_free_crosses_word_boundary(self):
        wp, frame_free, s, p, free_words, reclaim_words = self.setup_state(num_uavs=1)
        active = np.ones((1, p.frame_capacity), dtype=np.int32)
        active[0, 64] = active[0, 130] = 0
        s.tx_active.assign(active)
        self.rebuild(wp, frame_free, s, p, free_words, reclaim_words)
        self.assertEqual(self.take(wp, frame_free, s, p, free_words, reclaim_words), 64)
        self.assertEqual(self.take(wp, frame_free, s, p, free_words, reclaim_words), 130)

    def test_reclaim_mark_does_not_mutate_canonical_state(self):
        wp, frame_free, s, p, free_words, reclaim_words = self.setup_state(num_uavs=1)
        active = np.ones((1, p.frame_capacity), dtype=np.int32)
        receive = np.zeros((1, p.frame_capacity), dtype=np.int32)
        active[0, 3], receive[0, 3] = 2, 3
        s.tx_active.assign(active)
        s.tx_receive.assign(receive)
        self.rebuild(wp, frame_free, s, p, free_words, reclaim_words)
        wp.launch(frame_free.mark_reclaim_if_ready_kernel, dim=1,
                  inputs=[s, 3, reclaim_words], device=self.device)
        self.assertEqual(int(s.tx_active.numpy()[0, 3]), 2)
        self.assertNotEqual(int(reclaim_words.numpy()[0, 0]) & (1 << 3), 0)

    def test_apply_reclaims_canonicalizes_all_candidates_before_selection(self):
        wp, frame_free, s, p, free_words, reclaim_words = self.setup_state(num_uavs=1)
        active = np.ones((1, p.frame_capacity), dtype=np.int32)
        receive = np.zeros((1, p.frame_capacity), dtype=np.int32)
        active[0, 2], receive[0, 2] = 2, 3
        active[0, 50] = 0
        active[0, 120], receive[0, 120] = 2, 3
        s.tx_active.assign(active)
        s.tx_receive.assign(receive)
        self.rebuild(wp, frame_free, s, p, free_words, reclaim_words)
        self.assertEqual(self.take(wp, frame_free, s, p, free_words, reclaim_words), 2)
        actual = s.tx_active.numpy()[0]
        self.assertEqual(int(actual[2]), 0)
        self.assertEqual(int(actual[120]), 0)
        self.assertEqual(int(reclaim_words.numpy().sum()), 0)

    def test_no_free_frame_returns_minus_one_without_out_of_range_bit(self):
        wp, frame_free, s, p, free_words, reclaim_words = self.setup_state(num_uavs=1)
        s.tx_active.assign(np.ones((1, p.frame_capacity), dtype=np.int32))
        self.rebuild(wp, frame_free, s, p, free_words, reclaim_words)
        self.assertEqual(self.take(wp, frame_free, s, p, free_words, reclaim_words), -1)


if __name__ == "__main__":
    unittest.main()
