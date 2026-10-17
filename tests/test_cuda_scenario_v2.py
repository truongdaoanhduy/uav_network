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

if __name__=="__main__":
    unittest.main()
