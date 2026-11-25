"""Conservative v56 gates: static SoA and actual Warp reset/inject when available.

These tests deliberately DO NOT describe the unverified event kernel as working.
"""
from __future__ import annotations
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from v56_warp_7radio_mac import (
    HAS_WARP,N_RADIO,N_UAV,GCS,Q,K,SHAPES,backing_bytes_per_environment,
    WarpSevenRadioNetwork,
)

class V56StaticContracts(unittest.TestCase):
    def test_roles_and_7radio_queue_payload(self):
        self.assertEqual((N_RADIO,N_UAV,GCS,Q,K),(7,6,6,64,62))
        self.assertEqual(SHAPES['q_word0'][0],(7,64,62))
        self.assertEqual(SHAPES['ledger_pos'][0],(7,6,400))
        self.assertEqual(SHAPES['ack_gen'][0],(6,7,400))
        self.assertEqual(SHAPES['route_hop'][0],(7,7))
    def test_memory_is_lower_bound_and_not_490001_claim(self):
        self.assertGreater(backing_bytes_per_environment(),560_000)
        self.assertLess(backing_bytes_per_environment(),1_000_000)
    def test_reset_indexing_matches_6uav_and_7radio_source_ids(self):
        text=(Path(__file__).resolve().parents[1]/'src/v56_warp_reset_kernels.py').read_text()
        self.assertIn('u=(tid//16)%6;p=tid%16',text,
                      'Reset pending had an out-of-bounds %7 bug at env index >=1')
        self.assertIn('s.hello_due[e,r]=2000*r',text,
                      'HELLO beacons need per-radio de-synchronization for hidden terminals')
        self.assertEqual(SHAPES['delivered_seq'][0],(7,7,16),
                         'E2E ACK packets originate at GCS radio 6, too')
    def test_runtime_cannot_claim_training_readiness_without_warp(self):
        if not HAS_WARP:
            with self.assertRaises(RuntimeError): WarpSevenRadioNetwork(2,device='cpu')

@unittest.skipUnless(HAS_WARP,'warp-lang not installed in artifact-building container')
class V56WarpExecutionContracts(unittest.TestCase):
    def test_separate_world_reset_and_device_evidence_injection(self):
        s=WarpSevenRadioNetwork(2,device='cpu',seed=44)
        self.assertEqual(s.snapshot(('now',))['now'].tolist(),[0,0])
        c=np.full((2,6),-1,np.int32);c[0,0]=12
        p=np.zeros((2,6),np.int32);p[0,0]=1
        s.sense(c,p,1)
        self.assertEqual(int(s.data['ledger_pos'].numpy()[0,0,0,12]),1)
        self.assertEqual(int(s.data['ledger_pos'].numpy()[1].sum()),0)
        a=np.zeros((2,6),np.int32);a[0,0]=6
        s.inject(a,1)
        self.assertEqual(int(s.data['q_count'].numpy()[0,0]),1)
        self.assertEqual(int(s.data['q_entries'].numpy()[0,0,0]),1)
        self.assertEqual(int(s.data['q_count'].numpy()[1].sum()),0)
        s.reset(np.array([1,0],np.uint8))
        self.assertEqual(int(s.data['ledger_pos'].numpy()[0].sum()),0)
        self.assertEqual(int(s.data['ledger_pos'].numpy()[1].sum()),0)
        with self.assertRaises(RuntimeError):s.run(duration_us=1000)

if __name__=='__main__':unittest.main()
