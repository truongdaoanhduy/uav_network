"""The fixed shared V2 physics must execute on real CUDA before cloud training."""
import unittest
import numpy as np
import torch

from uav_network.environment.motion import move_uavs
import test_gpu_packets as packets

@unittest.skipUnless(torch.cuda.is_available(), "requires real CUDA")
class SharedScenarioV2CUDATests(unittest.TestCase):
    def helper(self):
        helper=packets.GPUPacketTests()
        helper.device="cuda"
        return helper

    def test_selected_power_recovers_same_receiver_on_cuda(self):
        c=packets.config(peer_range_m=500,gcs_range_m=500,
            tx_power_min_w=.001,tx_power_mapping="log",power_limited_admission=True)
        helper=self.helper()
        wp,s,p,mac=helper.prepare(c,[[500,400],[900,900]])
        self.assertTrue(s.positions.device.is_cuda)
        failed=helper.step(c,s,p,mac,wp,[2,0],-1)
        self.assertEqual(failed["diagnostics"]["data_attempts"],5)
        recovered=helper.step(c,s,p,mac,wp,[2,0],1)
        self.assertEqual(recovered["generated_packets"],2)
        self.assertEqual(recovered["diagnostics"]["data_attempts"],6)
        self.assertTrue(s.delivered.numpy()[0,0])
        self.assertAlmostEqual(recovered["uav_communication_energy_j"],
            (5*.001+.4)*(1024*8+464)/2_000_000)

    def test_real_cuda_opportunities_apply_max_power_and_joint_silence(self):
        c=packets.config(peer_range_m=500,gcs_range_m=500,
            tx_power_min_w=.001,tx_power_mapping="log",power_limited_admission=True)
        helper=self.helper()
        c.world.num_obstacles=1
        wp,s,p,mac=helper.prepare(c,[[500,400],[950,950]],obstacles=[[500,200,30]])
        d=helper.step(c,s,p,mac,wp,[0,0])["diagnostics"]
        self.assertEqual(d["action_receiver_eligible"],0)
        self.assertEqual(d["action_no_opportunity"],2)
        self.assertEqual(d["action_eligible_silent"],0)
        c.world.num_obstacles=0
        wp,s,p,mac=helper.prepare(c,[[500,700],[500,350]])
        d=helper.step(c,s,p,mac,wp,[0,0])["diagnostics"]
        self.assertEqual(d["action_eligible_silent"],1)
        self.assertEqual(d["action_progress_peer_opportunity"],1)
        self.assertEqual(d["action_progress_peer_silent"],1)

    def test_real_cuda_wall_peer_conflict_matches_certified_native_motion(self):
        import warp as wp
        from uav_network.environment.gpu.state import allocate_state
        from uav_network.environment.gpu.motion import motion_step
        c=packets.config()
        positions=np.array([[0.,715.],[0.,735.]])
        velocities=np.zeros_like(positions)
        obstacles=np.empty((0,3))
        commands=np.array([[-1.,1.],[-1.,-1.]])
        s,p=allocate_state(c,num_envs=1,device="cuda")
        s.positions.assign(positions[None])
        s.velocities.assign(velocities[None])
        actions=np.zeros((1,2,3),np.float32)
        actions[0,:,:2]=commands
        values=wp.array(actions,dtype=wp.float32,device="cuda")
        trace=[]
        for _ in range(100):
            positions,velocities,blocked,clipped=move_uavs(
                positions,velocities,commands,obstacles,c)
            self.assertEqual((blocked,clipped),(0,0))
            wp.launch(motion_step,dim=1,inputs=[s,p,values],device="cuda")
            np.testing.assert_allclose(s.positions.numpy()[0],positions,rtol=0,atol=2e-8)
            trace.append(positions.copy())
        self.assertGreaterEqual(float(np.ptp(np.asarray(trace)[-20:],axis=0).max()),20)
        self.assertTrue(s.positions.device.is_cuda)

    def test_real_cuda_obstacle_peer_conflict_uses_the_certified_open_tangent(self):
        import warp as wp
        from uav_network.environment.gpu.state import allocate_state
        from uav_network.environment.gpu.motion import motion_step
        from uav_network.environment.geometry import segment_distance
        c=packets.config()
        c.world.num_uavs=6
        c.world.num_obstacles=1
        c.energy.enabled=True
        positions=np.array([[439.,500.],[439.,484.5],[100.,100.],
                            [250.,100.],[750.,100.],[900.,100.]])
        velocities=np.zeros_like(positions)
        obstacles=np.array([[500.,500.,50.]])
        commands=np.zeros((6,2),np.float32); commands[0,0]=1
        s,p=allocate_state(c,num_envs=1,device="cuda")
        s.positions.assign(positions[None])
        s.obstacles.assign(obstacles[None])
        actions=np.zeros((1,6,3),np.float32)
        actions[0,:,:2]=commands
        values=wp.array(actions,dtype=wp.float32,device="cuda")
        trace=[]
        for step in range(80):
            old,previous=positions.copy(),velocities.copy()
            positions,velocities,blocked,clipped=move_uavs(
                positions,velocities,commands,obstacles,c)
            self.assertEqual((blocked,clipped),(0,0))
            self.assertTrue(np.all(np.abs(velocities-previous)<=2+1e-8))
            self.assertGreaterEqual(segment_distance(old[0],positions[0],obstacles[0,:2]),60-1e-8)
            self.assertGreaterEqual(segment_distance(old[0]-old[1],positions[0]-positions[1],[0.,0.]),15-1e-8)
            wp.launch(motion_step,dim=1,inputs=[s,p,values],device="cuda")
            np.testing.assert_allclose(s.positions.numpy()[0],positions,rtol=0,atol=2e-8)
            np.testing.assert_allclose(s.velocities.numpy()[0],velocities,rtol=0,atol=2e-8)
            if step==0:
                self.assertGreater(positions[0,1],old[0,1])
            trace.append(positions.copy())
        self.assertGreater(positions[0,0],570)
        self.assertGreater(float(np.ptp(np.asarray(trace)[-20:,0],axis=0).max()),10)
        counts=s.motion_counts.numpy()[0]
        self.assertEqual((counts[2],counts[7]),(0,0))
        self.assertGreater(counts[4],0)
        self.assertTrue(np.all(s.battery.numpy()>0))
        self.assertTrue(s.positions.device.is_cuda)

if __name__=="__main__":
    unittest.main()
