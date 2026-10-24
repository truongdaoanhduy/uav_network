"""Finite-c arrival windows must affect carrier sense, collisions and ACK timing."""
import importlib.util
import math
import unittest

import numpy as np

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv

HAS_WARP = importlib.util.find_spec("warp") is not None
C = 299792458.0
if HAS_WARP:
    import warp as wp
    from uav_network.environment.gpu.state import WorldState, WorldParams
    from uav_network.communication.gpu.phy import emit, busy, evaluate

    @wp.kernel
    def emit_probe(s: WorldState, p: WorldParams, packet: int, source: int, dest: int,
                   now: wp.float64, result: wp.array(dtype=wp.int32)):
        result[0] = emit(s,p,0,packet,source,dest,0,now)

    @wp.kernel
    def busy_probe(s: WorldState, p: WorldParams, node: int, now: wp.float64,
                   result: wp.array(dtype=wp.int32)):
        result[0] = int(busy(s,p,0,node,now))

    @wp.kernel
    def evaluate_probe(s: WorldState, p: WorldParams, frame: int, now: wp.float64,
                       result: wp.array(dtype=wp.int32)):
        result[0] = int(evaluate(s,p,0,frame,now))


def config(enabled=True):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 1, 0
    c.world.max_steps = 20
    c.sensor.altitude_m = 20
    c.energy.enabled = False
    c.runtime.device = "cpu"
    c.network.propagation_delay_enabled = enabled
    return c


def idle(destination):
    return Action(np.zeros((len(destination),3),dtype=np.float32),np.asarray(destination))


class PropagationTests(unittest.TestCase):
    def native(self,c,positions=None):
        env = SwarmSearchEnv(c)
        positions = positions if positions is not None else [[500,100],[600,100]]
        env.reset(seed=44,options={"positions":positions,"targets":[positions[0]],"obstacles":[]})
        self.addCleanup(env.close)
        return env

    def test_data_and_ack_arrive_after_their_own_airtime_plus_distance_over_c(self):
        observations=[]
        for enabled in (False,True):
            env=self.native(config(enabled))
            env.step(idle([2,0]))
            events=env.network.simulator.event_bus.since(0)
            tx=[e for e in events if e["event_type"]=="packet_tx_started"]
            rx=[e for e in events if e["event_type"]=="packet_rx_succeeded"]
            self.assertEqual(len(tx),2)
            for sent,received in zip(tx,rx):
                distance=math.sqrt(100**2+20**2)
                expected=sent["sim_time_us"]+sent["data"]["duration_us"]
                if enabled:expected+=distance/C*1e6
                self.assertAlmostEqual(received["sim_time_us"],expected,places=9)
            self.assertAlmostEqual(tx[1]["sim_time_us"],rx[0]["sim_time_us"]+10,places=9)
            observations.append(env.info()["network"]["communication_energy_j"])
            env.close()
        self.assertEqual(observations[0],observations[1])

    def test_roundtrip_allowance_prevents_a_false_ack_timeout_on_a_long_link(self):
        c=config();c.world.map_size_m=20000;c.network.peer_range_m=c.network.gcs_range_m=15000
        c.network.sinr_threshold_db=-10
        env=self.native(c,[[10000,10000],[12000,10000]])
        env.step(idle([2,0]))
        tx=[e for e in env.network.simulator.event_bus.since(0) if e["event_type"]=="packet_tx_started"]
        self.assertTrue(env.reports.delivered[0])
        self.assertEqual(len(tx),2,"one DATA and one ACK, with no premature DATA retry")

    def test_arrival_crossing_mission_boundary_is_not_delivered_early(self):
        c=config();c.world.dt_s=.0054181
        env=self.native(c)
        env.step(idle([2,0]))
        self.assertFalse(env.reports.delivered[0],"last emitted bit has ended but has not arrived yet")
        env.step(idle([0,0]))
        self.assertTrue(env.reports.delivered[0])

    @unittest.skipUnless(HAS_WARP,"optional Warp dependency not installed")
    def test_native_warp_packet_energy_timing_and_cross_step_parity(self):
        import test_gpu_packets as packet_tests
        checker=packet_tests.GPUPacketTests(methodName="runTest")
        for enabled in (False,True):
            c=config(enabled)
            checker.compare(c,[[500,100],[600,100]],[([2,0],0)])
        c=config();c.world.dt_s=.0054181;c.world.num_targets=2
        checker.compare(c,[[500,100],[600,100]],[([2,0],0),([0,0],0)],
                        targets=[[500,100],[900,900]])
        c=config();c.world.num_targets=2;c.network.cca_threshold_dbm=-30
        checker.compare(c,[[500,100],[550,100]],[([2,2],0)],targets=[[500,100],[550,100]])

    @unittest.skipUnless(HAS_WARP,"optional Warp dependency not installed")
    def test_receiver_relative_cca_and_overlap_include_delayed_waves(self):
        from uav_network.environment.gpu.state import allocate_state
        c=config();c.world.num_uavs=3;c.network.bit_rate_bps=1e9
        c.network.report_bytes=c.network.payload_bytes=1
        positions=np.array([[100,100],[980,100],[1000,100]],dtype=float)
        env=self.native(c,positions)
        s,p=allocate_state(c,device="cpu")
        s.positions.assign(positions[None]);s.gcs.assign(np.array([[500.,0.]]));s.battery.fill_(c.energy.battery_capacity_j)
        powers=s.pkt_power.numpy();powers[0,:2]=.1;s.pkt_power.assign(powers)
        result=wp.zeros(1,dtype=wp.int32,device="cpu")
        channel=env.network.simulator.channel
        def native_emit(source,now,packet_id):
            env.network.clock._now=now
            packet=env.network.DataPacket(src_drone=env.network.simulator.drones[source],
                dst_drone=env.network.simulator.drones[2],creation_time=now,data_packet_id=packet_id,
                data_packet_length=472,simulator=env.network.simulator,channel_id=1)
            packet.mission_power_w=.1
            env.network.simulator.drones[source].mac_protocol.phy.unicast(packet,2)
            return channel.transmissions[-1]
        target=native_emit(0,0.,1001)
        wp.launch(emit_probe,dim=1,inputs=[s,p,0,0,2,wp.float64(0),result],device="cpu")
        target_frame=int(result.numpy()[0])
        # A wave has reached the closer GCS but not the remote intended receiver.
        env.network.clock._now=1.5
        for observer,expected in ((2,False),(3,True)):
            native_busy=channel.is_busy_for(env.network.simulator.drones[observer],1)
            wp.launch(busy_probe,dim=1,inputs=[s,p,observer,wp.float64(1.5),result],device="cpu")
            self.assertEqual(native_busy,expected)
            self.assertEqual(bool(result.numpy()[0]),expected)
        # Emission intervals do not overlap; receiver arrival intervals do.
        native_emit(1,2.8,1002)
        wp.launch(emit_probe,dim=1,inputs=[s,p,1,1,2,wp.float64(2.8),result],device="cpu")
        arrival_end=.472+900/C*1e6
        env.network.clock._now=arrival_end
        sinr,_,interferers=channel._evaluate(target,2)
        self.assertLess(sinr,c.network.sinr_threshold_db)
        self.assertTrue(interferers)
        wp.launch(evaluate_probe,dim=1,inputs=[s,p,target_frame,wp.float64(arrival_end),result],device="cpu")
        self.assertFalse(bool(result.numpy()[0]))

    @unittest.skipUnless(HAS_WARP,"optional Warp dependency not installed")
    def test_disabled_delay_table_has_no_payload_and_empty_medium_is_idle(self):
        from uav_network.environment.gpu.state import allocate_state
        for enabled in (False,True):
            c=config(enabled);c.network.cca_threshold_dbm=-200
            s,p=allocate_state(c,device="cpu")
            self.assertTrue(hasattr(s,"tx_propagation_delay"),"per-frame observer delay table missing")
            self.assertEqual(s.tx_propagation_delay.size>0,enabled)
            result=wp.zeros(1,dtype=wp.int32,device="cpu")
            wp.launch(busy_probe,dim=1,inputs=[s,p,0,wp.float64(0),result],device="cpu")
            self.assertFalse(bool(result.numpy()[0]),"empty medium cannot be busy at any threshold")


if __name__=="__main__":
    unittest.main()
