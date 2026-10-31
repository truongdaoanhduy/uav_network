"""XYZ camera behavior through the real Native world and geometry."""
import numpy as np
import pytest

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv


def xyz_config(n=1, targets=1, obstacles=0):
    c = Config()
    c.world.motion_dimensions = 3
    c.sensor.model = "bayesian"
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = n, targets, obstacles
    c.network.enabled = False
    c.sensor.obstacle_occlusion = True
    return c


def world(c, z=25, seed=44, target=(512.5, 512.5), obstacles=None):
    env = SwarmSearchEnv(c)
    env.reset(seed=seed, options=dict(positions=[[512.5,512.5,z]],
              targets=[target], obstacles=[] if obstacles is None else obstacles))
    return env


def test_reference_camera_and_bayes_geometry():
    from uav_network.environment.bayesian import sensor_profile, bayes_update, cell_coverage
    c = xyz_config()
    pd,pf,r = sensor_profile(c.sensor,150)
    assert (pd,pf) == (.7,.3)
    assert r == pytest.approx(150)
    assert sensor_profile(c.sensor,50) == pytest.approx((.9,.1,50))
    assert bayes_update(.5,1,.9,.1) == pytest.approx(.9)
    assert bayes_update(.5,0,.9,.1) == pytest.approx(.1)
    assert cell_coverage(0,0,[0,0],25,25,1000) == pytest.approx(np.pi/4)
    assert cell_coverage(2,2,[0,0],25,25,1000) == 0


def test_high_altitude_is_evidence_not_confirmation_or_report():
    env = world(xyz_config(),z=100)
    env.belief[:] = .999999
    for _ in range(8):
        env.step(Action(np.zeros((1,4)),np.zeros(1,dtype=int)))
    assert env.positions.shape == (1,3)
    assert not env.reports.known.any()
    assert not env.reports.buffers[0]
    assert env.visited.any()
    assert env.belief[0,0,0] == .999999
    # Actor candidates may show noisy cell centers, never the exact hidden target.
    a = env.observation_dim
    assert env.state().shape == (env.state_dim,)
    assert a > 30


def test_low_fresh_positive_can_confirm_once_and_negative_cannot():
    from uav_network.environment.bayesian import sensor_uniform
    c = xyz_config()
    cell = 20*40+20
    seed = next(i for i in range(1,1000) if sensor_uniform(i,1,0,cell)<.945)
    env = world(c,z=25,seed=seed)
    env.belief[:] = .9999
    env.step(Action(np.zeros((1,4)),np.zeros(1,dtype=int)))
    assert env.reports.known[0,0]
    assert len(env.reports.buffers[0]) == 1
    before = env.reports.first_discovery.copy()
    env.step(Action(np.zeros((1,4)),np.zeros(1,dtype=int)))
    np.testing.assert_array_equal(before,env.reports.first_discovery)
    assert env.reports.bytes_used(0) == 1024


def test_unseen_and_occluded_cells_remain_prior_and_reads_do_not_resample():
    env = world(xyz_config(),z=25)
    assert env.belief[0,0,0] == .5
    before = env.belief.copy()
    from uav_network.environment.observations import local_observations
    local_observations(env); env.info(); env.state()
    np.testing.assert_array_equal(before,env.belief)
    c = xyz_config(obstacles=1)
    env = world(c,z=25,target=(560,512.5),obstacles=[[540,512.5,5,120]])
    assert env.belief[0,20,22] == .5


def test_xyz_vertical_motion_is_funded_and_constrained():
    env = world(xyz_config(),z=100,target=(950,950))
    before = env.battery_j.copy()
    env.step(Action(np.array([[0.,0.,1.,0.]]),np.array([0])))
    assert env.positions[0,2] == pytest.approx(102)
    assert env.velocities[0,2] == pytest.approx(2)
    assert env.battery_j[0] < before[0]-140
    for _ in range(30):
        env.step(Action(np.array([[0.,0.,1.,0.]]),np.array([0])))
    assert 0 <= env.positions[0,2] <= 150
    assert np.linalg.norm(env.velocities[0]) <= 20+1e-8


def test_xyz_swept_cylinder_checks_include_height():
    from uav_network.environment.xyz_geometry import cylinder_hit
    o = np.array([50.,0.,5.,40.])
    assert cylinder_hit([0,0,20],[100,0,20],o)
    assert not cylinder_hit([0,0,60],[100,0,60],o)
    assert cylinder_hit([50,0,60],[50,0,20],o)
    assert not cylinder_hit([60,0,60],[60,0,20],o)


def test_config_rejects_bad_xyz_sensor_settings():
    c = xyz_config()
    c.sensor.pd = [1,0]
    with pytest.raises(ValueError): c.validate()
    c = xyz_config()
    c.sensor.belief_prior = float("nan")
    with pytest.raises(ValueError): c.validate()

def test_bayesian_actor_does_not_see_unconfirmed_target_coordinates():
    c=xyz_config()
    first=world(c,z=100,target=(511,511))
    from uav_network.environment.observations import local_observations
    a=local_observations(first)
    second=world(c,z=100,target=(514,514))
    b=local_observations(second)
    assert not first.reports.known.any() and not second.reports.known.any()
    np.testing.assert_array_equal(a,b)


def test_false_confirmation_marks_verified_empty_cell():
    from uav_network.environment.bayesian import sensor_uniform
    c = xyz_config()
    c.sensor.verified_empty_belief = .01
    cell = 20 * 40 + 20
    seed = next(i for i in range(1, 10000) if sensor_uniform(i, 1, 0, cell) < .055)
    env = world(c, z=25, seed=seed, target=(950, 950))
    env.belief[:] = .5
    env.belief[0, 20, 20] = .9999
    env.step(Action(np.zeros((1, 4)), np.zeros(1, dtype=int)))
    assert env.sensor_counts["false_confirmation"] >= 1
    assert env.belief[0, 20, 20] == pytest.approx(.01)


def test_coarse_belief_uncertainty_is_global_and_compact():
    from uav_network.environment.bayesian import coarse_belief_uncertainty_map
    belief = np.full((40, 40), .5)
    coarse = coarse_belief_uncertainty_map(belief, 5)
    assert coarse.shape == (5, 5)
    np.testing.assert_allclose(coarse, 1.0)
    belief[:8, :8] = .01
    coarse = coarse_belief_uncertainty_map(belief, 5)
    assert coarse[0, 0] < .1
    np.testing.assert_allclose(coarse[1:, 1:], 1.0)


def test_xyz_observation_can_include_coarse_belief_without_hidden_targets():
    from uav_network.environment.observations import local_observations, observation_dim
    c = xyz_config()
    base_dim = observation_dim(c)
    c.sensor.belief_coarse_cells = 5
    first = world(c, z=100, target=(511, 511))
    second = world(c, z=100, target=(514, 514))
    a, b = local_observations(first), local_observations(second)
    assert a.shape == (1, observation_dim(c))
    assert observation_dim(c) == base_dim + 25
    np.testing.assert_array_equal(a, b)


def test_information_and_investigation_shaping_make_sensing_progress_dense():
    from uav_network.environment.bayesian import investigation_potential
    c = xyz_config()
    c.reward.discovery = c.reward.delivery = 0
    c.reward.blocked_motion = c.reward.expired_report = c.reward.dropped_report = 0
    c.reward.step_cost = c.reward.communication_energy = c.reward.propulsion_energy_per_kj = 0
    c.reward.shield_correction = c.reward.communication_efficiency = 0
    c.reward.information_gain = 10
    c.reward.investigation_shaping = 10
    c.reward.shaping_gamma = .99
    c.reward.investigation_belief_threshold = .8
    env = world(c, z=100, target=(950, 950))
    env.belief[:] = .5
    before = investigation_potential(env)
    _, reward, *_ = env.step(Action(np.zeros((1, 4)), np.zeros(1, dtype=int)))
    assert reward > 0
    assert env.reward_components["information_gain"] > 0
    env.belief[:] = .5
    env.belief[0, 20, 20] = .95
    env.positions[0, 2] = 100
    high = investigation_potential(env)
    env.positions[0, 2] = 25
    low = investigation_potential(env)
    assert before == 0
    assert 0 < high < low
