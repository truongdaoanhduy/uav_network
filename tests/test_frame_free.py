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

# Task 2 exactness tests intentionally import the not-yet-existing path first in TDD RED.
if importlib.util.find_spec("warp"):
    import warp as wp
    from uav_network.environment.gpu.state import WorldParams, WorldState
    from uav_network.communication.gpu.phy import emit, emit_framefree

    @wp.kernel
    def _emit_gen1_kernel(s: WorldState, p: WorldParams, selected: wp.array(dtype=wp.int32)):
        selected[0] = emit(s, p, 0, 0, p.n, 0, 1, wp.float64(1000.0))

    @wp.kernel
    def _emit_framefree_kernel(s: WorldState, p: WorldParams,
                               free_words: wp.array2d(dtype=wp.uint64),
                               reclaim_words: wp.array2d(dtype=wp.uint64),
                               selected: wp.array(dtype=wp.int32)):
        selected[0] = emit_framefree(s, p, 0, 0, p.n, 0, 1, wp.float64(1000.0),
                                     free_words, reclaim_words)


@unittest.skipUnless(importlib.util.find_spec("warp"), "optional warp_packet dependency not installed")
class FrameFreeEmitTests(unittest.TestCase):
    device = "cpu"

    def states(self):
        import warp as wp
        state = importlib.import_module("uav_network.environment.gpu.state")
        mac = importlib.import_module("uav_network.communication.gpu.mac")
        ff = importlib.import_module("uav_network.communication.gpu.frame_free")
        c = Config()
        c.world.num_uavs = 1
        c.world.num_targets = 1
        c.world.num_obstacles = 0
        c.energy.enabled = False
        a, p = state.allocate_state(c, seeds=[44], device=self.device)
        b, _ = state.allocate_state(c, seeds=[44], device=self.device)
        mask = wp.array([1], dtype=wp.int32, device=self.device)
        for s in (a, b):
            wp.launch(mac.reset_network, dim=1, inputs=[s, p, mask], device=self.device)
            s.pkt_fragment.assign(np.zeros((1, p.packet_capacity), dtype=np.int32))
            s.pkt_generation.assign(np.ones((1, p.packet_capacity), dtype=np.int32))
            s.pkt_power.assign(np.full((1, p.packet_capacity), .2, dtype=np.float64))
            s.pkt_id.assign(np.ones((1, p.packet_capacity), dtype=np.int64))
        words = ff.bitmap_words(p.frame_capacity)
        free_words = wp.zeros((1, words), dtype=wp.uint64, device=self.device)
        reclaim_words = wp.zeros((1, words), dtype=wp.uint64, device=self.device)
        return wp, ff, a, b, p, free_words, reclaim_words

    def run_pair(self, configure):
        wp, ff, gen1, candidate, p, free_words, reclaim_words = self.states()
        configure(gen1, p)
        configure(candidate, p)
        wp.launch(ff.rebuild_frame_bitmaps_kernel, dim=1,
                  inputs=[candidate, p, free_words, reclaim_words], device=self.device)
        left = wp.full(1, -99, dtype=wp.int32, device=self.device)
        right = wp.full(1, -99, dtype=wp.int32, device=self.device)
        wp.launch(_emit_gen1_kernel, dim=1, inputs=[gen1, p, left], device=self.device)
        wp.launch(_emit_framefree_kernel, dim=1,
                  inputs=[candidate, p, free_words, reclaim_words, right], device=self.device)
        return wp, gen1, candidate, p, int(left.numpy()[0]), int(right.numpy()[0])

    def assert_canonical_equal(self, wp, left, right):
        for name, value in vars(left).items():
            other = getattr(right, name, None)
            if isinstance(value, wp.array) and isinstance(other, wp.array):
                np.testing.assert_allclose(other.numpy(), value.numpy(), rtol=0, atol=0,
                                           equal_nan=True, err_msg=name)

    def test_emit_framefree_matches_gen1_retention_compaction_and_selected_index(self):
        def configure(s, p):
            active = np.zeros((1, p.frame_capacity), dtype=np.int32)
            active[0, 0] = active[0, 4] = 1
            s.tx_active.assign(active)
            order = np.zeros((1, p.frame_capacity), dtype=np.int32)
            order[0, :2] = [4, 0]
            s.frame_order.assign(order)
            s.frame_count.assign(np.array([2], dtype=np.int32))
            ends = np.zeros((1, p.frame_capacity), dtype=np.float64)
            ends[0, 4] = ends[0, 0] = 1000.0
            s.tx_end.assign(ends)
        wp, left, right, _, a, b = self.run_pair(configure)
        self.assertEqual(a, b)
        self.assert_canonical_equal(wp, left, right)

    def test_emit_framefree_reclaims_every_completed_receiver_frame(self):
        def configure(s, p):
            active = np.ones((1, p.frame_capacity), dtype=np.int32)
            receive = np.zeros((1, p.frame_capacity), dtype=np.int32)
            active[0, 2] = active[0, 120] = 2
            receive[0, 2] = receive[0, 120] = 3
            s.tx_active.assign(active)
            s.tx_receive.assign(receive)
        wp, left, right, _, a, b = self.run_pair(configure)
        self.assertEqual(a, 2)
        self.assertEqual(b, 2)
        self.assertEqual(int(right.tx_active.numpy()[0, 120]), 0)
        self.assert_canonical_equal(wp, left, right)

    def test_emit_framefree_preserves_frame_order_and_interference_order(self):
        def configure(s, p):
            active = np.zeros((1, p.frame_capacity), dtype=np.int32)
            active[0, 7] = active[0, 3] = 1
            s.tx_active.assign(active)
            order = np.zeros((1, p.frame_capacity), dtype=np.int32)
            order[0, :2] = [7, 3]
            s.frame_order.assign(order)
            s.frame_count.assign(np.array([2], dtype=np.int32))
            ends = np.zeros((1, p.frame_capacity), dtype=np.float64)
            ends[0, 7] = ends[0, 3] = 1000.0
            s.tx_end.assign(ends)
        wp, left, right, _, _, _ = self.run_pair(configure)
        self.assert_canonical_equal(wp, left, right)
        self.assertEqual(right.frame_order.numpy()[0, :3].tolist(), left.frame_order.numpy()[0, :3].tolist())

    def test_emit_framefree_matches_error_three_when_capacity_exhausted(self):
        def configure(s, p):
            s.tx_active.assign(np.ones((1, p.frame_capacity), dtype=np.int32))
        wp, left, right, _, a, b = self.run_pair(configure)
        self.assertEqual((a, b), (-1, -1))
        self.assertEqual(int(left.errors.numpy()[0]), 3)
        self.assertEqual(int(right.errors.numpy()[0]), 3)

if importlib.util.find_spec("warp"):
    from uav_network.communication.gpu.mac import event, event_framefree

    @wp.kernel
    def _event_gen1_kernel(s: WorldState, p: WorldParams, kind: int, arg: int, generation: int):
        event(s, p, 0, kind, arg, generation, wp.float64(1000.0))

    @wp.kernel
    def _event_framefree_kernel(s: WorldState, p: WorldParams, kind: int, arg: int, generation: int,
                                free_words: wp.array2d(dtype=wp.uint64),
                                reclaim_words: wp.array2d(dtype=wp.uint64)):
        event_framefree(s, p, 0, kind, arg, generation, wp.float64(1000.0),
                        free_words, reclaim_words)


@unittest.skipUnless(importlib.util.find_spec("warp"), "optional warp_packet dependency not installed")
class FrameFreeEventTests(FrameFreeEmitTests):
    def run_event_pair(self, kind, arg, generation, configure):
        wp, ff, left, right, p, free_words, reclaim_words = self.states()
        configure(left, p)
        configure(right, p)
        wp.launch(ff.rebuild_frame_bitmaps_kernel, dim=1,
                  inputs=[right, p, free_words, reclaim_words], device=self.device)
        wp.launch(_event_gen1_kernel, dim=1,
                  inputs=[left, p, kind, arg, generation], device=self.device)
        wp.launch(_event_framefree_kernel, dim=1,
                  inputs=[right, p, kind, arg, generation, free_words, reclaim_words], device=self.device)
        return wp, left, right, p, reclaim_words

    def packet(self, s, p):
        source = np.zeros((1, p.packet_capacity), dtype=np.int32)
        destination = np.full((1, p.packet_capacity), p.n, dtype=np.int32)
        generation = np.ones((1, p.packet_capacity), dtype=np.int32)
        power = np.full((1, p.packet_capacity), .2, dtype=np.float64)
        s.pkt_source.assign(source)
        s.pkt_destination.assign(destination)
        s.pkt_generation.assign(generation)
        s.pkt_power.assign(power)

    def test_event_framefree_tx_start_matches_gen1(self):
        def configure(s, p):
            self.packet(s, p)
        wp, left, right, _, _ = self.run_event_pair(9, 0, 1, configure)
        self.assert_canonical_equal(wp, left, right)

    def test_event_framefree_ack_start_matches_gen1(self):
        def configure(s, p):
            self.packet(s, p)
            active = np.zeros((1, p.frame_capacity), dtype=np.int32)
            active[0, 0] = 1
            s.tx_active.assign(active)
            s.frame_count.assign(np.array([1], dtype=np.int32))
            order = np.zeros((1, p.frame_capacity), dtype=np.int32)
            order[0, 0] = 0
            s.frame_order.assign(order)
            s.tx_packet.assign(np.zeros((1, p.frame_capacity), dtype=np.int32))
            src = np.zeros((1, p.frame_capacity), dtype=np.int32)
            dst = np.full((1, p.frame_capacity), p.n, dtype=np.int32)
            s.tx_source.assign(src)
            s.tx_destination.assign(dst)
            ends = np.zeros((1, p.frame_capacity), dtype=np.float64)
            ends[0, 0] = 1000.0
            s.tx_end.assign(ends)
        wp, left, right, _, _ = self.run_event_pair(15, 0, 0, configure)
        self.assert_canonical_equal(wp, left, right)

    def test_event_framefree_marks_receiver_cleanup_without_early_canonical_mutation(self):
        def configure(s, p):
            active = np.zeros((1, p.frame_capacity), dtype=np.int32)
            receive = np.zeros((1, p.frame_capacity), dtype=np.int32)
            active[0, 3], receive[0, 3] = 2, 2
            s.tx_active.assign(active)
            s.tx_receive.assign(receive)
            destination = np.zeros((1, p.frame_capacity), dtype=np.int32)
            s.tx_destination.assign(destination)
        wp, left, right, _, reclaim_words = self.run_event_pair(16, 3, 0, configure)
        self.assert_canonical_equal(wp, left, right)
        self.assertEqual(int(right.tx_active.numpy()[0, 3]), 2)
        self.assertEqual(int(right.tx_receive.numpy()[0, 3]), 3)
        self.assertNotEqual(int(reclaim_words.numpy()[0, 0]) & (1 << 3), 0)

@unittest.skipUnless(importlib.util.find_spec("warp"), "optional warp_packet dependency not installed")
class FrameFreeVectorTests(unittest.TestCase):
    def config(self, envs=2):
        c = Config()
        c.runtime.num_envs = envs
        c.runtime.device = "cpu"
        c.runtime.environment_backend = "warp_packet"
        c.world.num_uavs = 6
        c.world.num_targets = 2
        c.world.num_obstacles = 0
        c.world.max_steps = 20
        c.energy.enabled = False
        return c

    def test_framefree_scheduler_allocates_expected_bitmap_shape_and_bytes(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        from uav_network.communication.gpu.frame_free import bitmap_words
        env = WarpVectorEnv(self.config(), device="cpu", packet_scheduler="framefree")
        try:
            words = bitmap_words(env.p.frame_capacity)
            self.assertEqual(env.p.frame_capacity, 1632)
            self.assertEqual(words, 26)
            self.assertEqual(env._frame_free_words.shape, (2, 26))
            self.assertEqual(env._frame_reclaim_words.shape, (2, 26))
            derived = (env._frame_free_words.size + env._frame_reclaim_words.size) * 8
            self.assertEqual(derived, 2 * 26 * 2 * 8)
            self.assertEqual(2048 * 26 * 2 * 8, 851968)
        finally:
            env.close()

    def test_framefree_partial_reset_rebuilds_only_selected_rows(self):
        import warp as wp
        from uav_network.environment.gpu.vector import WarpVectorEnv
        env = WarpVectorEnv(self.config(), device="cpu", packet_scheduler="framefree")
        try:
            env.reset(seed=[44, 45])
            # Poison row 1 derived state; a row-0-only reset must leave it untouched.
            free = env._frame_free_words.numpy()
            reclaim = env._frame_reclaim_words.numpy()
            free[1, :] = np.uint64(0x1234)
            reclaim[1, :] = np.uint64(0x55)
            env._frame_free_words.assign(free)
            env._frame_reclaim_words.assign(reclaim)
            before_free = env._frame_free_words.numpy()[1].copy()
            before_reclaim = env._frame_reclaim_words.numpy()[1].copy()
            env.reset(seed=[46, None], options={"reset_mask": np.array([True, False])})
            np.testing.assert_array_equal(env._frame_free_words.numpy()[1], before_free)
            np.testing.assert_array_equal(env._frame_reclaim_words.numpy()[1], before_reclaim)
            self.assertTrue(np.any(env._frame_free_words.numpy()[0] != 0))
            self.assertTrue(np.all(env._frame_reclaim_words.numpy()[0] == 0))
        finally:
            env.close()

    def test_framefree_derived_arrays_are_absent_from_snapshot_arrays(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        env = WarpVectorEnv(self.config(), device="cpu", packet_scheduler="framefree")
        try:
            env.reset(seed=[44, 45])
            snap = env.snapshot_state()
            self.assertNotIn("_frame_free_words", snap["arrays"])
            self.assertNotIn("_frame_reclaim_words", snap["arrays"])
            self.assertEqual(snap["version"], 1)
        finally:
            env.close()

    def _step_zeros(self, env, count=3):
        actions = {"continuous": np.zeros((env.num_envs, env.p.n, 3), dtype=np.float32),
                   "destination": np.zeros((env.num_envs, env.p.n), dtype=np.int64)}
        for _ in range(count):
            env.step(actions)

    def _assert_snapshots_equal(self, left, right):
        self.assertEqual(left["version"], right["version"])
        self.assertEqual(left["config"], right["config"])
        self.assertEqual(left["device_type"], right["device_type"])
        self.assertEqual(left["arrays"].keys(), right["arrays"].keys())
        for key in left["arrays"]:
            np.testing.assert_allclose(left["arrays"][key].numpy(), right["arrays"][key].numpy(),
                                       rtol=0, atol=0, equal_nan=True, err_msg=key)

    def test_gen1_snapshot_restores_into_framefree_and_continues_exactly(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        c = self.config()
        gen1 = WarpVectorEnv(c, device="cpu", packet_scheduler="gen1")
        ff = WarpVectorEnv(c, device="cpu", packet_scheduler="framefree")
        try:
            gen1.reset(seed=[44, 45])
            self._step_zeros(gen1, 2)
            snap = gen1.snapshot_state()
            ff.restore_state(snap)
            self._step_zeros(gen1, 3)
            self._step_zeros(ff, 3)
            self._assert_snapshots_equal(gen1.snapshot_state(), ff.snapshot_state())
        finally:
            gen1.close(); ff.close()

    def test_framefree_snapshot_restores_into_gen1_and_continues_exactly(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        c = self.config()
        ff = WarpVectorEnv(c, device="cpu", packet_scheduler="framefree")
        gen1 = WarpVectorEnv(c, device="cpu", packet_scheduler="gen1")
        try:
            ff.reset(seed=[44, 45])
            self._step_zeros(ff, 2)
            snap = ff.snapshot_state()
            gen1.restore_state(snap)
            self._step_zeros(ff, 3)
            self._step_zeros(gen1, 3)
            self._assert_snapshots_equal(ff.snapshot_state(), gen1.snapshot_state())
        finally:
            ff.close(); gen1.close()
