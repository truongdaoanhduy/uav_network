"""RED-to-GREEN contracts for packet-local map, delayed DS, and reward credit."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from uav_v54_7node_env import EnvConfigV54,ZhangXue7NodePacketEnv
from uav_v54_ds_and_battery import (counts_to_canonical_ds_stable,
    discounted_mass,combine_ds_masses,fuse_provenance_counts)
from uav_v54_7radio_csmaca_sim import ACTOR_N,RADIO_N,NUM_CELLS,GCS_ID

class ObservationContract(unittest.TestCase):
    def test_actor_gets_local_spatial_belief_not_truth(self):
        env=ZhangXue7NodePacketEnv(seed=7)
        obs, state, mask=env.reset(seed=7)
        self.assertEqual(obs.shape,(ACTOR_N,514))
        self.assertEqual(state.shape,(ACTOR_N*514,))
        self.assertEqual(mask.shape,(ACTOR_N,7))
        before=obs.copy()
        env.true_targets[:]=~env.true_targets
        self.assertTrue(np.array_equal(before,env._build_obs_state_mask()[0]),'oracle targets leaked into actor')
        # Mutating only a distant receiver must not change the actor's local map.
        env.net.ledger_pos[GCS_ID,0,15]=2
        self.assertTrue(np.array_equal(before[0],env._build_obs_state_mask()[0][0]))
        env.net.record_local_observation(0,15,True,1)
        self.assertFalse(np.array_equal(before[0],env._build_obs_state_mask()[0][0]))
        self.assertTrue(np.array_equal(before[1],env._build_obs_state_mask()[0][1]))

    def test_actor_age_increases_without_fabricating_evidence(self):
        env=ZhangXue7NodePacketEnv(seed=7); env.reset(seed=7)
        env.net.record_local_observation(0,0,True,1)
        env.t=1
        t1=env._build_obs_state_mask()[0][0].copy()
        env.t=40
        t40=env._build_obs_state_mask()[0][0]
        self.assertFalse(np.array_equal(t1,t40))
        self.assertTrue(np.isfinite(t40).all())
        self.assertTrue(np.isfinite(t40).all() and (t40[14:] >= 0.0).all() and (t40[14:] <= 1.0 + 1e-6).all())

class DSEvidenceContract(unittest.TestCase):
    def test_discount_mass_vacuous_and_probability_conservation(self):
        m=(0.8,0.1,0.1)
        self.assertEqual(discounted_mass(m,0.0),(0,0,1))
        self.assertEqual(discounted_mass(m,1.0),m)
        new=discounted_mass(m,0.2)
        self.assertAlmostEqual(sum(new),1.0)
        self.assertAlmostEqual(new[0],0.16)

    def test_two_source_no_double_count_and_conflict(self):
        self.assertEqual(combine_ds_masses((0,0,1),(0.7,0,0.3)),(0.7,0,0.3))
        a=np.array([2,2],dtype=np.uint16); b=np.array([0,0],dtype=np.uint16)
        t1=fuse_provenance_counts(a,b,np.array([1,1]),np.array([0.,0.]),mf=0.8,me=0.8,eta=0.02,mode='sum')[0]
        t2=fuse_provenance_counts(np.array([4,0]),b,np.array([1,-1]),np.array([0.,0.]),mf=0.8,me=0.8,eta=0.02,mode='sum')[0]
        self.assertAlmostEqual(t1,t2,places=9)
        tdelayed=fuse_provenance_counts(a,b,np.array([1,1]),np.array([100.,100.]),mf=0.8,me=0.8,eta=0.02,mode='sum')[0]
        self.assertLess(tdelayed,t1)

    def test_zhang_max_is_distinct_from_provenance_sum(self):
        p=np.array([2,2],np.uint16); n=np.zeros(2,np.uint16)
        sum_m=fuse_provenance_counts(p,n,np.array([1,1]),np.array([0.,0.]),mode='sum')[0]
        max_m=fuse_provenance_counts(p,n,np.array([1,1]),np.array([0.,0.]),mode='zhang_max')[0]
        self.assertGreater(sum_m,max_m)
        self.assertAlmostEqual(max_m,counts_to_canonical_ds_stable(2,0)[0])

class RewardContract(unittest.TestCase):
    def test_peer_credit_requires_new_information_and_real_delivery(self):
        env=ZhangXue7NodePacketEnv(seed=7); env.reset(seed=7)
        self.assertEqual(env.net.step_new_peer_cells,0)
        env.net.record_local_observation(1,15,True,1)  # local evidence is not peer delivery
        self.assertEqual(env.net.step_new_peer_cells,0)
        self.assertGreater(env.cfg.w_p2p,0)
        self.assertIn('r_peer_information',env.step(np.zeros((6,2)),np.zeros(6,dtype=np.int64))[-1])

    def test_all_stopped_mission_does_not_teleport_or_credit(self):
        env=ZhangXue7NodePacketEnv(seed=7); env.reset(seed=7)
        env.battery_j[0]=0.0
        prev=env.positions[0].copy()
        env.step(np.ones((6,2))*20,np.zeros(6,dtype=np.int64))
        np.testing.assert_array_equal(prev,env.positions[0])

if __name__=='__main__': unittest.main()
