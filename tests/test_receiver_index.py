"""Exact derived receiver-pending heap primitives for rxindex."""
import importlib
import importlib.util
import unittest

import numpy as np
import torch

from uav_network.configuration import Config

HAS_WARP = importlib.util.find_spec("warp") is not None
RX_SPEC = importlib.util.find_spec("uav_network.communication.gpu.receiver_index") if HAS_WARP else None

if HAS_WARP:
    import warp as wp
    from uav_network.environment.gpu.state import WorldState, WorldParams, allocate_state
    if RX_SPEC is not None:
        _rx = importlib.import_module("uav_network.communication.gpu.receiver_index")

        @wp.kernel
        def _heap_probe(s: WorldState, p: WorldParams,
                        frames: wp.array3d(dtype=wp.int32), sizes: wp.array2d(dtype=wp.int32),
                        receiver: int, first: int, second: int,
                        out: wp.array(dtype=wp.int32)):
            _rx.receiver_push(s,p,0,receiver,frames,sizes,first)
            _rx.receiver_push(s,p,0,receiver,frames,sizes,second)
            out[0] = _rx.receiver_pop_min(s,p,0,receiver,frames,sizes)
            out[1] = _rx.receiver_pop_min(s,p,0,receiver,frames,sizes)
    else:
        _rx = None
        _heap_probe = None
else:
    wp = None
    _rx = None
    _heap_probe = None


@unittest.skipUnless(HAS_WARP, "optional warp_packet dependency not installed")
class ReceiverHeapTests(unittest.TestCase):
    device = "cpu"

    def config(self):
        c = Config()
        c.runtime.num_envs = 1
        c.runtime.device = c.runtime.replay_device = "cpu"
        c.runtime.environment_backend = "warp_packet"
        c.world.num_uavs = 2
        c.world.num_targets = 1
        c.world.num_obstacles = 0
        c.energy.enabled = False
        c.network.report_bytes = 512
        c.network.buffer_bytes = 1024
        c.network.payload_bytes = 512
        return c.validate()

    def prepared(self):
        s,p = allocate_state(self.config(),num_envs=1,seeds=[44],device=self.device)
        frames = wp.zeros((1,p.n+1,p.frame_capacity),dtype=wp.int32,device=self.device)
        sizes = wp.zeros((1,p.n+1),dtype=wp.int32,device=self.device)
        return s,p,frames,sizes

    def run_pair(self, orders, first, second, receiver=0):
        self.assertIsNotNone(_heap_probe, "receiver heap primitives are missing")
        s,p,frames,sizes = self.prepared()
        values=np.zeros(s.tx_receive_order.shape,dtype=np.int64)
        for frame,order in orders.items(): values[0,frame]=order
        s.tx_receive_order.assign(values)
        out=wp.zeros(2,dtype=wp.int32,device=self.device)
        wp.launch(_heap_probe,dim=1,inputs=[s,p,frames,sizes,receiver,first,second,out],device=self.device)
        return out.numpy().tolist(), sizes.numpy(), frames.numpy()

    def test_heap_orders_by_receive_order(self):
        popped,sizes,_=self.run_pair({3:20,5:10},3,5)
        self.assertEqual(popped,[5,3])
        self.assertEqual(int(sizes[0,0]),0)

    def test_equal_order_tie_uses_smaller_frame_index(self):
        popped,sizes,_=self.run_pair({7:12,2:12},7,2)
        self.assertEqual(popped,[2,7])
        self.assertEqual(int(sizes[0,0]),0)

    def test_receivers_have_independent_sizes_and_heaps(self):
        self.assertIsNotNone(_rx, "receiver heap primitives are missing")
        s,p,frames,sizes=self.prepared()
        orders=np.zeros(s.tx_receive_order.shape,dtype=np.int64)
        orders[0,4]=1; orders[0,8]=2
        s.tx_receive_order.assign(orders)

        @wp.kernel
        def probe(s: WorldState,p: WorldParams,frames: wp.array3d(dtype=wp.int32),
                  sizes: wp.array2d(dtype=wp.int32),out: wp.array(dtype=wp.int32)):
            _rx.receiver_push(s,p,0,0,frames,sizes,4)
            _rx.receiver_push(s,p,0,1,frames,sizes,8)
            out[0]=_rx.receiver_pop_min(s,p,0,1,frames,sizes)

        out=wp.zeros(1,dtype=wp.int32,device=self.device)
        wp.launch(probe,dim=1,inputs=[s,p,frames,sizes,out],device=self.device)
        self.assertEqual(int(out.numpy()[0]),8)
        self.assertEqual(sizes.numpy()[0,:2].tolist(),[1,0])
        self.assertEqual(int(frames.numpy()[0,0,0]),4)


if __name__ == "__main__":
    unittest.main()


# Task-1 rebuild/reset/indexed-selection probes. Defined conditionally so the
# tests fail cleanly before the corresponding production functions exist.
if HAS_WARP and _rx is not None and getattr(_rx, "next_receive_indexed", None) is not None:
    @wp.kernel
    def _next_receive_probe(s: WorldState, p: WorldParams,
                            frames: wp.array3d(dtype=wp.int32), sizes: wp.array2d(dtype=wp.int32),
                            receiver: int, now: wp.float64, receive_kind: int):
        _rx.next_receive_indexed(s,p,0,receiver,now,receive_kind,frames,sizes)
else:
    _next_receive_probe = None

if HAS_WARP and _rx is not None:
    @wp.kernel
    def _pop_three_probe(s: WorldState, p: WorldParams,
                         frames: wp.array3d(dtype=wp.int32), sizes: wp.array2d(dtype=wp.int32),
                         receiver: int, out: wp.array(dtype=wp.int32)):
        out[0]=_rx.receiver_pop_min(s,p,0,receiver,frames,sizes)
        out[1]=_rx.receiver_pop_min(s,p,0,receiver,frames,sizes)
        out[2]=_rx.receiver_pop_min(s,p,0,receiver,frames,sizes)
else:
    _pop_three_probe = None


@unittest.skipUnless(HAS_WARP, "optional warp_packet dependency not installed")
class ReceiverIndexLifecycleTests(ReceiverHeapTests):
    def canonical_pending_state(self, num_envs=1):
        c=self.config(); c.runtime.num_envs=num_envs
        s,p=allocate_state(c,num_envs=num_envs,seeds=list(range(44,44+num_envs)),device=self.device)
        active=np.zeros(s.tx_active.shape,dtype=np.int32)
        dest=np.zeros(s.tx_destination.shape,dtype=np.int32)
        receive=np.zeros(s.tx_receive.shape,dtype=np.int32)
        order=np.zeros(s.tx_receive_order.shape,dtype=np.int64)
        # receiver 0: frame 1 and retained frame 2 are pending; frames 3/4/5 excluded.
        active[:,1]=1; dest[:,1]=0; receive[:,1]=1; order[:,1]=30
        active[:,2]=2; dest[:,2]=0; receive[:,2]=1; order[:,2]=10
        active[:,3]=1; dest[:,3]=0; receive[:,3]=0; order[:,3]=1
        active[:,4]=1; dest[:,4]=0; receive[:,4]=2; order[:,4]=2
        active[:,5]=1; dest[:,5]=0; receive[:,5]=3; order[:,5]=3
        # receiver 1: one pending frame.
        active[:,6]=1; dest[:,6]=1; receive[:,6]=1; order[:,6]=5
        s.tx_active.assign(active); s.tx_destination.assign(dest)
        s.tx_receive.assign(receive); s.tx_receive_order.assign(order)
        frames=wp.zeros((num_envs,p.n+1,p.frame_capacity),dtype=wp.int32,device=self.device)
        sizes=wp.zeros((num_envs,p.n+1),dtype=wp.int32,device=self.device)
        return s,p,frames,sizes

    def test_rebuild_uses_exact_pending_predicate_and_preserves_canonical_state(self):
        rebuild=getattr(_rx,"rebuild_receiver_index",None) if _rx is not None else None
        self.assertIsNotNone(rebuild,"rebuild_receiver_index is missing")
        s,p,frames,sizes=self.canonical_pending_state()
        before={name:value.numpy().copy() for name,value in vars(s).items() if isinstance(value,wp.array)}
        wp.launch(rebuild,dim=1,inputs=[s,p,frames,sizes],device=self.device)
        self.assertEqual(sizes.numpy()[0,:2].tolist(),[2,1])
        out=wp.zeros(3,dtype=wp.int32,device=self.device)
        wp.launch(_pop_three_probe,dim=1,inputs=[s,p,frames,sizes,0,out],device=self.device)
        self.assertEqual(out.numpy().tolist(),[2,1,-1],"tx_active==2 pending frame stays eligible")
        for name,expected in before.items():
            np.testing.assert_array_equal(getattr(s,name).numpy(),expected,err_msg=name)

    def test_masked_reset_clears_only_selected_world_sizes(self):
        rebuild=getattr(_rx,"rebuild_receiver_index",None) if _rx is not None else None
        reset=getattr(_rx,"reset_receiver_index",None) if _rx is not None else None
        self.assertIsNotNone(rebuild,"rebuild_receiver_index is missing")
        self.assertIsNotNone(reset,"reset_receiver_index is missing")
        s,p,frames,sizes=self.canonical_pending_state(num_envs=2)
        wp.launch(rebuild,dim=2,inputs=[s,p,frames,sizes],device=self.device)
        before_frames=frames.numpy().copy(); before_sizes=sizes.numpy().copy()
        mask=wp.array([1,0],dtype=wp.int32,device=self.device)
        wp.launch(reset,dim=2,inputs=[p,mask,frames,sizes],device=self.device)
        self.assertTrue(np.all(sizes.numpy()[0]==0))
        np.testing.assert_array_equal(sizes.numpy()[1],before_sizes[1])
        np.testing.assert_array_equal(frames.numpy()[1],before_frames[1])

    def test_indexed_selection_matches_gen1_canonical_updates_and_event_push(self):
        self.assertIsNotNone(_next_receive_probe,"next_receive_indexed is missing")
        s,p,frames,sizes=self.canonical_pending_state()
        rebuild=getattr(_rx,"rebuild_receiver_index",None)
        wp.launch(rebuild,dim=1,inputs=[s,p,frames,sizes],device=self.device)
        self.assertEqual(int(s.event_count.numpy()[0]),0)
        wp.launch(_next_receive_probe,dim=1,
                  inputs=[s,p,frames,sizes,0,wp.float64(100.0),14],device=self.device)
        self.assertEqual(int(s.node_receiving.numpy()[0,0]),1)
        self.assertEqual(int(s.tx_receive.numpy()[0,2]),2)
        self.assertEqual(int(sizes.numpy()[0,0]),1)
        self.assertEqual(int(s.event_count.numpy()[0]),1)
        self.assertEqual(int(s.event_kind.numpy()[0,0]),14)
        self.assertEqual(int(s.event_arg.numpy()[0,0]),2)
        self.assertEqual(int(s.event_priority.numpy()[0,0]),0)
        self.assertEqual(float(s.event_time.numpy()[0,0]),100.0)
        self.assertEqual(int(s.event_sequence.numpy()[0]),1)


if HAS_WARP:
    _mac = importlib.import_module("uav_network.communication.gpu.mac")
    if getattr(_mac,"event_rxindex",None) is not None:
        @wp.kernel
        def _event_rxindex_probe(s: WorldState,p: WorldParams,kind: int,arg: int,generation: int,
                                 now: wp.float64,frames: wp.array3d(dtype=wp.int32),
                                 sizes: wp.array2d(dtype=wp.int32)):
            _mac.event_rxindex(s,p,0,kind,arg,generation,now,frames,sizes)
    else:
        _event_rxindex_probe=None

    @wp.kernel
    def _event_gen1_probe(s: WorldState,p: WorldParams,kind: int,arg: int,generation: int,
                          now: wp.float64):
        _mac.event(s,p,0,kind,arg,generation,now)
else:
    _mac=None
    _event_rxindex_probe=None
    _event_gen1_probe=None


@unittest.skipUnless(HAS_WARP, "optional warp_packet dependency not installed")
class RxIndexEventParityTests(unittest.TestCase):
    device="cpu"

    def config(self):
        c=Config()
        c.runtime.num_envs=1
        c.runtime.device=c.runtime.replay_device="cpu"
        c.runtime.environment_backend="warp_packet"
        c.world.num_uavs=2; c.world.num_targets=1; c.world.num_obstacles=0
        c.energy.enabled=False
        c.network.report_bytes=512; c.network.buffer_bytes=1024; c.network.payload_bytes=512
        c.network.peer_range_m=c.network.gcs_range_m=10000
        return c.validate()

    def pair(self):
        c=self.config()
        left,p=allocate_state(c,num_envs=1,seeds=[44],device=self.device)
        right,_=allocate_state(c,num_envs=1,seeds=[44],device=self.device)
        frames=wp.zeros((1,p.n+1,p.frame_capacity),dtype=wp.int32,device=self.device)
        sizes=wp.zeros((1,p.n+1),dtype=wp.int32,device=self.device)
        return c,left,right,p,frames,sizes

    def copy_left_to_right(self,left,right):
        for name,value in vars(left).items():
            other=getattr(right,name,None)
            if isinstance(value,wp.array) and isinstance(other,wp.array):
                other.assign(value.numpy())

    def assert_canonical_equal(self,left,right):
        for name,value in vars(left).items():
            other=getattr(right,name,None)
            if isinstance(value,wp.array) and isinstance(other,wp.array):
                np.testing.assert_allclose(other.numpy(),value.numpy(),rtol=0,atol=0,equal_nan=True,err_msg=name)

    def seed_receive_queue(self,s,p,current=3,pending=(7,2),receiver=0,ttl_invalid=False):
        active=np.zeros(s.tx_active.shape,dtype=np.int32)
        dest=np.zeros(s.tx_destination.shape,dtype=np.int32)
        recv=np.full(s.tx_receive.shape,3,dtype=np.int32)
        order=np.zeros(s.tx_receive_order.shape,dtype=np.int64)
        active[0,current]=1; dest[0,current]=receiver; recv[0,current]=2
        s.node_receiving.assign(np.array([[1]*(p.n+1)],dtype=np.int32))
        # equal order deliberately exercises frame-index tie semantics
        for frame in pending:
            active[0,frame]=1; dest[0,frame]=receiver; recv[0,frame]=1; order[0,frame]=12
        s.tx_active.assign(active); s.tx_destination.assign(dest); s.tx_receive.assign(recv); s.tx_receive_order.assign(order)
        if ttl_invalid:
            ttl=np.zeros(s.tx_ttl.shape,dtype=np.int32); ttl[0,current]=p.n+2; s.tx_ttl.assign(ttl)
        return current

    def launch_pair(self,left,right,p,frames,sizes,kind,arg,now=100.0):
        self.assertIsNotNone(_event_rxindex_probe,"rxindex event path is missing")
        wp.launch(_rx.rebuild_receiver_index,dim=1,inputs=[right,p,frames,sizes],device=self.device)
        wp.launch(_event_gen1_probe,dim=1,inputs=[left,p,kind,arg,0,wp.float64(now)],device=self.device)
        wp.launch(_event_rxindex_probe,dim=1,inputs=[right,p,kind,arg,0,wp.float64(now),frames,sizes],device=self.device)
        self.assert_canonical_equal(left,right)

    def test_receive_done_busy_queue_preserves_equal_order_tie(self):
        _,left,right,p,frames,sizes=self.pair()
        current=self.seed_receive_queue(left,p)
        self.copy_left_to_right(left,right)
        self.launch_pair(left,right,p,frames,sizes,_mac.RECEIVE_DONE,current)
        self.assertEqual(int(left.event_arg.numpy()[0,0]),2)

    def test_invalid_ttl_receive_selects_same_next_pending_frame(self):
        _,left,right,p,frames,sizes=self.pair()
        current=self.seed_receive_queue(left,p,pending=(8,5),ttl_invalid=True)
        self.copy_left_to_right(left,right)
        self.launch_pair(left,right,p,frames,sizes,_mac.RECEIVE,current)
        self.assertEqual(int(left.event_arg.numpy()[0,0]),5)

    def test_direct_vector_delivery_matches_gen1_exactly(self):
        from uav_network.environment.gpu.vector import WarpVectorEnv
        from uav_network.environment.gpu.geometry import sense
        c=self.config(); c.world.num_uavs=1; c.runtime.num_envs=1; c=c.validate()
        left=WarpVectorEnv(c,device="cpu",packet_scheduler="gen1")
        right=WarpVectorEnv(c,device="cpu",packet_scheduler="rxindex")
        try:
            for vector in (left,right):
                vector.reset(seed=44)
                targets=vector.s.targets.numpy().copy(); targets[:,0]=vector.s.positions.numpy()[:,0]
                vector.s.targets.assign(targets)
                vector._launch(sense,(1,1,1),vector.s,vector.p)
                vector._launch(_mac.sense_reports,1,vector.s,vector.p)
                vector._publish(vector._buffers[vector._buffer])
            actions={"continuous":torch.tensor([[[0.,0.,1.]]],dtype=torch.float32),
                     "destination":torch.tensor([[1]],dtype=torch.int64)}
            left.step(actions); right.step(actions)
            self.assert_canonical_equal(left.s,right.s)
            self.assertGreater(int(left.s.packet_counts.numpy()[0,1]),0,"corpus must exercise receive path")
        finally:
            left.close(); right.close()
