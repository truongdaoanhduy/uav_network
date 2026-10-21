"""Exact ordering primitives for the semantics-preserving MAC Scheduler Gen2."""
import importlib
import importlib.util
import unittest

import numpy as np

from uav_network.configuration import Config

HAS_WARP = importlib.util.find_spec("warp") is not None

if HAS_WARP:
    import warp as wp

    from uav_network.environment.gpu.state import WorldState, WorldParams

    _events = importlib.import_module("uav_network.communication.gpu.events")
    _key_before = getattr(_events, "key_before", None)
    _candidate_before_root = getattr(_events, "candidate_before_root", None)
    _mac = importlib.import_module("uav_network.communication.gpu.mac")
    _try_macro_event = getattr(_mac, "try_macro_event", None)

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

    if _try_macro_event is not None:
        @wp.kernel
        def _gen1_prefix_probe(s: WorldState, p: WorldParams, now: wp.float64, packet: int, generation: int):
            _mac.event(s, p, 0, _mac.START, packet, generation, now)
            if (s.event_count[0] > 0 and s.event_kind[0,0] == _mac.MAC
                    and s.event_arg[0,0] == packet):
                kind = s.event_kind[0,0]
                arg = s.event_arg[0,0]
                gen = s.event_generation[0,0]
                time = s.event_time[0,0]
                _events.pop(s,0)
                _mac.event(s,p,0,kind,arg,gen,time)
            if (s.event_count[0] > 0 and s.event_kind[0,0] == _mac.IDLE
                    and s.event_arg[0,0] == packet):
                kind = s.event_kind[0,0]
                arg = s.event_arg[0,0]
                gen = s.event_generation[0,0]
                time = s.event_time[0,0]
                _events.pop(s,0)
                _mac.event(s,p,0,kind,arg,gen,time)

        @wp.kernel
        def _gen2_prefix_probe(s: WorldState, p: WorldParams, now: wp.float64, packet: int, generation: int,
                               counters: wp.array2d(dtype=wp.int64), handled: wp.array(dtype=wp.int32)):
            fused = _try_macro_event(s,p,0,_mac.START,packet,generation,now,counters)
            handled[0] = int(fused)
            if not fused:
                _mac.event(s,p,0,_mac.START,packet,generation,now)


        @wp.kernel
        def _gen1_tx_probe(s: WorldState, p: WorldParams, now: wp.float64, packet: int,
                           generation: int, consume_data_end: int):
            _mac.event(s,p,0,_mac.TX_START,packet,generation,now)
            if (consume_data_end != 0 and s.event_count[0] > 0
                    and s.event_kind[0,0] == _mac.DATA_END and s.event_arg[0,0] == packet):
                kind = s.event_kind[0,0]; arg = s.event_arg[0,0]
                gen = s.event_generation[0,0]; time = s.event_time[0,0]
                _events.pop(s,0); _mac.event(s,p,0,kind,arg,gen,time)

        @wp.kernel
        def _gen2_event_probe(s: WorldState, p: WorldParams, kind: int, now: wp.float64,
                              arg: int, generation: int, counters: wp.array2d(dtype=wp.int64),
                              handled: wp.array(dtype=wp.int32)):
            fused = _try_macro_event(s,p,0,kind,arg,generation,now,counters)
            handled[0] = int(fused)
            if not fused:
                _mac.event(s,p,0,kind,arg,generation,now)
    else:
        _gen1_prefix_probe = None
        _gen2_prefix_probe = None
        _gen1_tx_probe = None
        _gen2_event_probe = None
else:
    wp = None
    _events = None
    _mac = None
    _try_macro_event = None
    _key_before_probe = None
    _candidate_root_probe = None
    _gen1_prefix_probe = None
    _gen2_prefix_probe = None
    _gen1_tx_probe = None
    _gen2_event_probe = None


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


@unittest.skipUnless(HAS_WARP, "optional warp_packet dependency not installed")
class MacroPrefixTests(unittest.TestCase):
    device = "cpu"

    def config(self):
        c = Config()
        c.runtime.num_envs = 1
        c.runtime.device = c.runtime.replay_device = "cpu"
        c.runtime.environment_backend = "warp_packet"
        c.world.num_uavs = 1
        c.world.num_targets = 1
        c.world.num_obstacles = 0
        c.energy.enabled = False
        c.network.report_bytes = 512
        c.network.buffer_bytes = 1024
        c.network.payload_bytes = 512
        return c.validate()

    def prepared(self, competitor=None, full=False):
        state = importlib.import_module("uav_network.environment.gpu.state")
        s, p = state.allocate_state(self.config(), num_envs=1, seeds=[44], device=self.device)
        mask = wp.array([1], dtype=wp.int32, device=self.device)
        wp.launch(_mac.reset_network, dim=1, inputs=[s,p,mask], device=self.device)
        s.event_count.assign(np.array([0], dtype=np.int32))
        s.event_sequence.assign(np.array([20], dtype=np.int64))
        status = s.pkt_status.numpy(); status[0,0] = 2; s.pkt_status.assign(status)
        source = s.pkt_source.numpy(); source[0,0] = 0; s.pkt_source.assign(source)
        generation = s.pkt_generation.numpy(); generation[0,0] = 1; s.pkt_generation.assign(generation)
        attempts = s.pkt_attempt.numpy(); attempts[0,0] = 0; s.pkt_attempt.assign(attempts)
        versions = s.pkt_backoff_version.numpy(); versions[0,0] = 0; s.pkt_backoff_version.assign(versions)
        current = s.node_current.numpy(); current[0,0] = -1; s.node_current.assign(current)
        if full:
            s.event_count.assign(np.array([int(p.event_capacity)], dtype=np.int32))
        elif competitor is not None:
            time, priority, order = competitor
            s.event_count.assign(np.array([1], dtype=np.int32))
            times = s.event_time.numpy(); times[0,0] = time; s.event_time.assign(times)
            priorities = s.event_priority.numpy(); priorities[0,0] = priority; s.event_priority.assign(priorities)
            orders = s.event_order.numpy(); orders[0,0] = order; s.event_order.assign(orders)
            kinds = s.event_kind.numpy(); kinds[0,0] = _mac.POLL; s.event_kind.assign(kinds)
            args = s.event_arg.numpy(); args[0,0] = 0; s.event_arg.assign(args)
            gens = s.event_generation.numpy(); gens[0,0] = 0; s.event_generation.assign(gens)
        return s,p

    def arrays(self, s):
        import torch
        return {name: wp.to_torch(value).detach().clone() for name,value in vars(s).items() if isinstance(value,wp.array)}

    def assert_arrays_equal(self, left, right):
        import torch
        self.assertEqual(left.keys(), right.keys())
        for name in left:
            torch.testing.assert_close(left[name],right[name],rtol=0,atol=0,equal_nan=True,msg=name)

    def run_case(self, competitor=None, full=False):
        self.assertIsNotNone(_gen2_prefix_probe, "try_macro_event helper is missing")
        left,p_left = self.prepared(competitor,full)
        right,p_right = self.prepared(competitor,full)
        counters = wp.zeros((1,6), dtype=wp.int64, device=self.device)
        handled = wp.zeros(1,dtype=wp.int32,device=self.device)
        now = wp.float64(100.0)
        wp.launch(_gen1_prefix_probe,dim=1,inputs=[left,p_left,now,0,1],device=self.device)
        wp.launch(_gen2_prefix_probe,dim=1,inputs=[right,p_right,now,0,1,counters,handled],device=self.device)
        self.assert_arrays_equal(self.arrays(left),self.arrays(right))
        return bool(handled.numpy()[0]), counters.numpy()[0].copy()

    def test_safe_prefix_fuses_without_competitor_and_with_later_competitor(self):
        for competitor in (None,(200.0,0,20)):
            with self.subTest(competitor=competitor):
                handled,counters = self.run_case(competitor)
                self.assertTrue(handled)
                self.assertEqual(counters[0],1)  # attempted chains
                self.assertEqual(counters[1],1)  # completed chains
                self.assertGreaterEqual(counters[2],2)  # fused successors
                self.assertEqual(counters[3],0)
                self.assertEqual(counters[4],0)
                self.assertEqual(counters[5],3)  # START + MAC + IDLE

    def test_earlier_and_equal_key_competitors_force_exact_fallback(self):
        for competitor in ((99.0,0,20),(100.0,0,20)):
            with self.subTest(competitor=competitor):
                handled,counters = self.run_case(competitor)
                self.assertFalse(handled)
                self.assertEqual(counters[0],1)
                self.assertEqual(counters[1],0)
                self.assertEqual(counters[2],0)
                self.assertEqual(counters[3],1)

    def test_capacity_boundary_falls_back_without_preproof_mutation(self):
        handled,counters = self.run_case(full=True)
        self.assertFalse(handled)
        self.assertEqual(counters[0],1)
        self.assertEqual(counters[1],0)
        self.assertEqual(counters[4],1)

    def prepared_tx(self, equal_competitor=False):
        s,p = self.prepared()
        fields = {
            "pkt_destination": int(p.n), "pkt_target": 0, "pkt_fragment": 0,
            "pkt_attempt": 1, "pkt_generation": 1, "pkt_status": 2,
            "pkt_power": .4, "pkt_created": 0., "pkt_report_created": 0.,
        }
        for name,value in fields.items():
            a=getattr(s,name).numpy(); a[0,0]=value; getattr(s,name).assign(a)
        if equal_competitor:
            duration=(min(int(p.payload_bytes),int(p.report_bytes))*8+float(p.header_bits))/float(p.bit_rate)*1e6
            s.event_count.assign(np.array([1],dtype=np.int32))
            for name,value in (("event_time",100.+duration),("event_kind",_mac.POLL),
                               ("event_arg",0),("event_priority",1),("event_order",20)):
                a=getattr(s,name).numpy(); a[0,0]=value; getattr(s,name).assign(a)
        return s,p

    def test_tx_start_fuses_data_end_but_equal_key_competitor_wins(self):
        for competitor in (False,True):
            with self.subTest(equal_key_competitor=competitor):
                left,p1=self.prepared_tx(competitor); right,p2=self.prepared_tx(competitor)
                counters=wp.zeros((1,6),dtype=wp.int64,device=self.device)
                handled=wp.zeros(1,dtype=wp.int32,device=self.device)
                wp.launch(_gen1_tx_probe,dim=1,inputs=[left,p1,wp.float64(100),0,1,int(not competitor)],device=self.device)
                wp.launch(_gen2_event_probe,dim=1,inputs=[right,p2,_mac.TX_START,wp.float64(100),0,1,counters,handled],device=self.device)
                self.assert_arrays_equal(self.arrays(left),self.arrays(right))
                self.assertTrue(bool(handled.numpy()[0]))
                values=counters.numpy()[0]
                self.assertEqual(values[0],1)
                if competitor:
                    self.assertEqual(values[2],0); self.assertEqual(values[3],1)
                else:
                    self.assertGreaterEqual(values[2],1); self.assertEqual(values[1],1)


@unittest.skipUnless(HAS_WARP, "optional warp_packet dependency not installed")
class SchedulerSelectionParityTests(unittest.TestCase):
    def config(self):
        c = Config()
        c.runtime.num_envs = 2
        c.runtime.device = c.runtime.replay_device = "cpu"
        c.runtime.environment_backend = "warp_packet"
        c.world.num_uavs = 2
        c.world.num_targets = 2
        c.world.num_obstacles = 0
        c.world.max_steps = 8
        c.sensor.altitude_m = 20
        c.sensor.obstacle_occlusion = False
        c.network.report_bytes = 2048
        c.network.buffer_bytes = 4096
        c.network.payload_bytes = 512
        c.tracking.enabled = False
        return c.validate()

    def test_fallback_gen2_matches_gen1_every_array_for_identical_actions(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        left = WarpVectorEnv(self.config(), device="cpu", packet_scheduler="gen1")
        right = WarpVectorEnv(self.config(), device="cpu", packet_scheduler="gen2")
        try:
            obs_left, _ = left.reset(seed=[44, 45])
            obs_right, _ = right.reset(seed=[44, 45])
            for key in obs_left:
                import torch
                torch.testing.assert_close(obs_left[key], obs_right[key], rtol=0, atol=0, equal_nan=True)

            import torch
            continuous = torch.tensor([
                [[0.10, -0.05, 0.75], [-0.10, 0.05, 0.50]],
                [[0.05, 0.10, 0.25], [-0.05, -0.10, 1.00]],
            ], dtype=torch.float32)
            destinations = [
                torch.tensor([[2, 0], [2, 0]], dtype=torch.int64),
                torch.tensor([[1, 2], [0, 2]], dtype=torch.int64),
                torch.tensor([[2, 0], [2, 0]], dtype=torch.int64),
            ]
            for destination in destinations:
                actions = {"continuous": continuous, "destination": destination}
                obs_left, reward_left, done_left, truncated_left, _ = left.step(actions)
                obs_right, reward_right, done_right, truncated_right, _ = right.step(actions)
                torch.testing.assert_close(reward_left, reward_right, rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(done_left, done_right, rtol=0, atol=0)
                torch.testing.assert_close(truncated_left, truncated_right, rtol=0, atol=0)
                for key in obs_left:
                    torch.testing.assert_close(obs_left[key], obs_right[key], rtol=0, atol=0, equal_nan=True)
                snap_left, snap_right = left.snapshot_state(), right.snapshot_state()
                self.assertEqual(snap_left["arrays"].keys(), snap_right["arrays"].keys())
                for name in snap_left["arrays"]:
                    torch.testing.assert_close(snap_left["arrays"][name], snap_right["arrays"][name],
                                               rtol=0, atol=0, equal_nan=True, msg=name)
                torch.testing.assert_close(snap_left["seeds"], snap_right["seeds"], rtol=0, atol=0)
                torch.testing.assert_close(snap_left["action_generator"], snap_right["action_generator"],
                                           rtol=0, atol=0)
        finally:
            left.close()
            right.close()

    def test_active_direct_packet_exercises_tx_receive_ack_fusion(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        import torch
        c=self.config(); c.runtime.num_envs=1; c.world.num_targets=1
        c.sensor.altitude_m=10000; c.network.report_bytes=c.network.payload_bytes=512
        c.network.buffer_bytes=1024; c.network.gcs_range_m=c.network.peer_range_m=1_000_000
        c.network.sinr_threshold_db=-100; c.energy.enabled=False
        left=WarpVectorEnv(c,device="cpu",packet_scheduler="gen1")
        right=WarpVectorEnv(c,device="cpu",packet_scheduler="gen2")
        try:
            left.reset(seed=44); right.reset(seed=44)
            actions={"continuous":torch.zeros((1,2,3)),"destination":torch.tensor([[2,0]])}
            actions["continuous"][0,0,2]=1
            a=left.step(actions); b=right.step(actions)
            torch.testing.assert_close(a[1],b[1],rtol=0,atol=0); torch.testing.assert_close(a[2],b[2])
            for name,value in left.snapshot_state()["arrays"].items():
                torch.testing.assert_close(value,right.snapshot_state()["arrays"][name],rtol=0,atol=0,equal_nan=True,msg=name)
            metrics=right.scheduler_metrics()
            self.assertGreater(metrics["completed_chains"],1)
            self.assertGreater(metrics["fused_events"],2)
        finally:
            left.close(); right.close()

    def test_gen2_scheduler_metrics_are_external_to_snapshot_state(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        vector = WarpVectorEnv(self.config(), device="cpu", packet_scheduler="gen2")
        try:
            vector.reset(seed=[44,45])
            vector._scheduler_counters.assign(np.asarray([[1,1,2,0,3,3],[4,2,6,5,0,2]],dtype=np.int64))
            metrics = vector.scheduler_metrics()
            self.assertEqual(metrics, {"attempted_chains":5,"completed_chains":3,"fused_events":8,
                                       "competitor_fallbacks":5,"boundary_fallbacks":3,"max_chain_length":3})
            snapshot = vector.snapshot_state()
            self.assertNotIn("scheduler_metrics",snapshot)
            self.assertNotIn("scheduler_counters",snapshot["arrays"])
        finally:
            vector.close()


if __name__ == "__main__":
    unittest.main()
