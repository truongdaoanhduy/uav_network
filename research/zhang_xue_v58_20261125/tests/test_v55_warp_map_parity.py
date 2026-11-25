"""Do not claim a GPU result unless the exact Warp kernel actually launched."""
from __future__ import annotations
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from uav_v54_7node_env import EnvConfigV54,ZhangXue7NodePacketEnv
from v55_warp_actor_maps import HAS_WARP,project_actor_maps_warp

@unittest.skipUnless(HAS_WARP,'warp-lang unavailable (no kernel launch claimed)')
class WarpActorProjectionTests(unittest.TestCase):
    def test_warp_cpu_map_exact_input_against_packet_cpu_reference(self):
        env=ZhangXue7NodePacketEnv(seed=7);env.reset(seed=7)
        env.net.record_local_observation(0,21,True,1)
        env.net.record_local_observation(0,21,False,2)
        env.t=3
        out=project_actor_maps_warp(env.net.ledger_pos[None],env.net.ledger_neg[None],
           env.net.ledger_gen_ts_s[None],env.net.ledger_rx_age_s[None],
           env.positions[None],np.array([env.t],dtype=np.int32),device='cpu')
        self.assertEqual(out.shape,(1,6,500))
        np.testing.assert_allclose(out[0,0],env._local_belief_features(0),rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(out[0,5],env._local_belief_features(5),rtol=2e-5,atol=2e-5)

if __name__=='__main__':unittest.main()
