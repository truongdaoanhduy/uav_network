"""Map-only scientific contracts. Training and network deliberately not involved."""
import math
import unittest
import numpy as np

from uav_network.environment.zhang_xue_coop import ZhangXueConfig, ZhangXueCooperativeEnv
from uav_network.communication.zhang_xue_ref.ds_belief import ds_log_space_safe


class TestZhangXueMapFirst(unittest.TestCase):
    def make(self, seed=44):
        return ZhangXueCooperativeEnv(ZhangXueConfig(),seed=seed)

    def test_camera_footprint_90deg_at_altitude_100m_observes_nearby_cells(self):
        env=self.make()
        c=env.config
        self.assertEqual(c.camera_fov_degrees,90.)
        radius=c.fixed_altitude_meter*math.tan(math.radians(c.camera_fov_degrees/2))
        self.assertAlmostEqual(radius,100.,places=5)
        # Synthetic empty map, fixed position and no UAV movement.
        env.target_mask[:]=False
        env.positions[0]=[525,525]
        env.positions[1:]=[[125,125],[875,125],[125,875],[875,875],[825,525]]
        env._sync_physics()
        before=env.belief.pos[0,0].copy()+env.belief.neg[0,0].copy()
        env.step(np.zeros((6,2),np.float32),np.zeros(6,np.int32))
        seen=np.flatnonzero(env.belief.pos[0,0]+env.belief.neg[0,0]>before)
        self.assertGreater(len(seen),1)
        for cell in seen:
            col,row=int(cell%20),int(cell//20)
            center=np.array([(col+.5)*50,(row+.5)*50])
            self.assertLessEqual(np.linalg.norm(center-env.positions[0]),radius+1e-5)

    def test_map_scenario_geometry_and_no_fly_200_seeds(self):
        for seed in range(200):
            e=self.make(seed)
            self.assertEqual(int(e.target_mask.sum()),10)
            self.assertEqual(len(e.no_fly_zones),2)
            self.assertEqual(len(e.blackout_zones),2)
            for cy,cx in (divmod(int(cell),20) for cell in np.flatnonzero(e.target_mask)):
                x,y=(cx+.5)*50,(cy+.5)*50
                for fx,fy,fr in e.no_fly_zones:
                    self.assertGreater(math.hypot(x-fx,y-fy),fr)
            self.assertTrue(np.all(e.positions>=0))
            self.assertTrue(np.all(e.positions<1000))

    def test_sensor_probability_independent_seeded_and_zero_prior(self):
        e=self.make()
        self.assertEqual(e.belief.pos.sum()+e.belief.neg.sum(),0)
        f,empty,unknown=e.belief.get_belief(0)
        np.testing.assert_allclose(f,0,atol=0)
        np.testing.assert_allclose(empty,0,atol=0)
        np.testing.assert_allclose(unknown,1,atol=0)
        a=self.make(91)
        b=self.make(91)
        for _ in range(3):
            a.step(np.zeros((6,2),np.float32),np.zeros(6,np.int32))
            b.step(np.zeros((6,2),np.float32),np.zeros(6,np.int32))
            np.testing.assert_array_equal(a.belief.pos,b.belief.pos)
            np.testing.assert_array_equal(a.belief.neg,b.belief.neg)

    def test_false_targets_never_awarded_team_tp(self):
        e=self.make()
        empty=int(np.flatnonzero(~e.target_mask)[0])
        e.belief.record_own_observation(0,empty,True)
        e.belief.record_own_observation(0,empty,True)
        _,_,mu=e.belief.get_team_own_belief()
        mf=e.belief.get_team_own_belief()[0]
        r=e.reward_engine.compute_unified_step_reward(
            np.zeros(400,np.float32),np.zeros(400,np.float32),
            mf,0.,
        )
        self.assertFalse(e.reward_engine.team_tp_latch[empty])
        self.assertTrue(e.reward_engine.team_fp_latch[empty])
        self.assertLess(r,0)

    def test_ds_near_equal_uint16_counts_no_nan_and_mass_sum_one(self):
        vals=np.array([0,1,2,5,200,1000,65535],np.int32)
        p,n=np.meshgrid(vals,vals)
        for mf,me in ((.8,.8),(.85,.7)):
            f,e,u=ds_log_space_safe(p,n,mf,me)
            self.assertTrue(np.isfinite(f).all())
            self.assertTrue(np.isfinite(e).all())
            self.assertTrue(np.isfinite(u).all())
            np.testing.assert_allclose(f+e+u,1.,rtol=0,atol=3e-7)

if __name__=="__main__":
    unittest.main()
