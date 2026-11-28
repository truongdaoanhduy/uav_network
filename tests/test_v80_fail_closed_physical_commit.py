"""A detected physical safety failure must NOT write unsafe positions to the world."""
from __future__ import annotations
import os,unittest,numpy as np
import warp as wp
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params, WarpParallelMissionV80

class TestV80FailClosed(unittest.TestCase):
    def check_device(self,device):
        p=GPUWorldV80Params.from_yaml(
            "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml",
            envs=4,device=device,seed=44,horizon=900)
        w=WarpParallelMissionV80(p)
        old=w.buffers["positions"].numpy().copy()
        vel=w.buffers["velocities"].numpy().copy()
        # World 0: irrecoverable boundary violation; active velocity would move.
        old[0,0,:]=[8.,500.,50.]
        vel[0,0,:]=[4.,0.]
        # World 1: pair already inside 50m collision distance.
        old[1,0,:]=[500.,500.,50.]
        old[1,1,:]=[525.,500.,50.]
        vel[1,0,:]=[3.,0.]
        vel[1,1,:]=[3.,0.]
        wp.copy(w.buffers["positions"],wp.array(old,dtype=wp.float32,device=device))
        wp.copy(w.buffers["velocities"],wp.array(vel,dtype=wp.float32,device=device))
        action=np.zeros((4,6,2),dtype=np.float32)
        action[0,0,:]=[1.,0.]
        action[1,0,:]=[1.,0.]
        action[1,1,:]=[1.,0.]
        action[2,0,:]=[1.,0.]  # live-world control: unaffected worlds still advance
        w.step(wp.array(action,dtype=wp.float32,device=device),
               wp.zeros((4,6),dtype=wp.int32,device=device))
        failed=w.buffers["failed"].numpy()
        self.assertEqual(failed[0],1)
        self.assertEqual(failed[1],1)
        current=w.buffers["positions"].numpy()
        current_velocity=w.buffers["velocities"].numpy()
        np.testing.assert_array_equal(current[:2],old[:2],
            "A world that fails terminal physical safety must not commit an unsafe move")
        np.testing.assert_array_equal(current_velocity[:2],vel[:2],
            "Fail-closed must not silently introduce impossible instantaneous braking")
        self.assertTrue(np.any(np.abs(current[2:]-old[2:])>0),
            "Nonfailed worlds must keep advancing normally")

    def test_cpu(self):self.check_device("cpu")
    @unittest.skipUnless(os.getenv("RUN_V80_CUDA_GATE")=="1","CUDA gate opt-in")
    def test_cuda(self):self.check_device(os.getenv("V80_CUDA_DEVICE","cuda:1"))
