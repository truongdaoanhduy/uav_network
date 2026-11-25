from __future__ import annotations
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from zhang_xue_v54_7node_warp import v55_minimum_gpu_memory_budget
from v55_state_derived_replay import estimate_replay_bytes

class MemorySafetyContract(unittest.TestCase):
    def test_1m_replay_requires_more_than_one_t4(self):
        b=v55_minimum_gpu_memory_budget(2048,1_000_000)
        self.assertFalse(b['measured_cuda_peak'])
        self.assertFalse(b['cuda_full_mac_parity'])
        self.assertGreater(b['replay_gib'],20)
        self.assertGreater(b['v55_minimum_state_bytes_per_env'],490001)
        self.assertGreater(b['actor_observation_dim'],14)

    def test_state_derived_replay_halves_map_storage(self):
        a=estimate_replay_bytes(1_000_000,6,514,2,7,False)
        b=estimate_replay_bytes(1_000_000,6,514,2,7,True)
        self.assertLess(a,b)
        self.assertGreater((b-a)/1e9,12)

if __name__=='__main__':unittest.main()
