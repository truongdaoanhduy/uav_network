"""Behavioral TDD for distinct terminal-reason diagnostics on real Warp CPU/CUDA."""
from __future__ import annotations
import os
import unittest
import numpy as np
import warp as wp
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params, WarpParallelMissionV79


class TestTerminalReason(unittest.TestCase):
    def _build(self,device,horizon=1):
        p=GPUWorldV79Params.from_yaml(
            "research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml",
            envs=4, device=device, horizon=horizon, seed=44)
        return WarpParallelMissionV79(p)

    def test_time_limit_vs_safety_on_cpu(self):
        self._verify("cpu")

    @unittest.skipUnless(os.environ.get("RUN_V79_CUDA_GATE") == "1", "CUDA smoke run is opt-in")
    def test_time_limit_vs_safety_on_cuda(self):
        self._verify("cuda:1")

    def test_safety_type_masks_cpu(self):
        self._verify_safety_types("cpu")

    @unittest.skipUnless(os.environ.get("RUN_V79_CUDA_GATE") == "1", "CUDA opt-in")
    def test_safety_type_masks_cuda(self):
        self._verify_safety_types("cuda:1")

    def _verify_safety_types(self,device):
        w=self._build(device, horizon=900)
        b=w.buffers
        positions=b["positions"].numpy().copy()
        # World 0: enforce dangerous UAV-to-UAV overlap.
        positions[0,0,:2]=[500.0, 500.0]
        positions[0,1,:2]=[515.0, 500.0]
        # World 1: enforce geometric/boundary violation.
        positions[1,0,:2]=[1.0, 500.0]
        wp.copy(b["positions"],wp.array(positions,dtype=wp.float32,device=device))
        acts=wp.zeros((4,6,2),dtype=wp.float32,device=device)
        dsts=wp.zeros((4,6),dtype=wp.int32,device=device)
        w.step(acts,dsts)
        why=b["safety_reason"].numpy()
        self.assertTrue(int(why[0]) & 2, f"pairwise safety flag missing: {why}")
        self.assertTrue(int(why[1]) & 1, f"geometry safety flag missing: {why}")
        self.assertEqual(int(b["terminated"].numpy()[0]),1)
        self.assertEqual(int(b["terminated"].numpy()[1]),1)

    def test_masked_reset_clears_only_selected_diagnostics_cpu(self):
        self._verify_masked_clear("cpu")

    @unittest.skipUnless(os.environ.get("RUN_V79_CUDA_GATE") == "1", "CUDA opt-in")
    def test_masked_reset_clears_only_selected_diagnostics_cuda(self):
        self._verify_masked_clear("cuda:1")

    def _verify_masked_clear(self,device):
        w=self._build(device,horizon=900)
        b=w.buffers
        tr=np.array([1,2,3,4],dtype=np.uint8)
        sr=np.array([4,3,2,1],dtype=np.uint8)
        wp.copy(b["termination_reason"],wp.array(tr,dtype=wp.uint8,device=device))
        wp.copy(b["safety_reason"],wp.array(sr,dtype=wp.uint8,device=device))
        wp.copy(b["done"],wp.array(np.array([0,1,0,0],dtype=np.uint8),dtype=wp.uint8,device=device))
        w.auto_reset_gpu()
        np.testing.assert_array_equal(b["termination_reason"].numpy(),[1,0,3,4])
        np.testing.assert_array_equal(b["safety_reason"].numpy(),[4,0,2,1])

    def _verify(self,device):
        w=self._build(device)
        b=w.buffers
        acts=wp.zeros((4,6,2),dtype=wp.float32,device=device)
        dsts=wp.zeros((4,6),dtype=wp.int32,device=device)
        w.step(acts,dsts)
        reason=b["termination_reason"].numpy()
        term=b["terminated"].numpy()
        trunc=b["truncated"].numpy()
        failed=b["failed"].numpy()
        self.assertTrue(np.any(trunc==1),"horizon=1 must produce at least one truncation")
        self.assertTrue(np.all((reason[trunc==1] & 8)!=0))
        self.assertTrue(np.all((reason[failed==1] & 1)!=0))
        self.assertTrue(np.all(((reason & 1)>0)==(failed>0)))
        self.assertTrue(np.all((reason[term==0] & 7)==0))
        self.assertTrue(np.all((reason[trunc==0] & 8)==0))


if __name__=="__main__":
    unittest.main()
