"""Calibrated 2kg rotary-wing Zeng2019 opt-in model, native/Warp agreement."""
import math

import numpy as np
import pytest
import torch
import warp as wp

from uav_network.configuration import load_config
from uav_network.environment.energy import propulsion_cost
from uav_network.environment.gpu.vector import WarpVectorEnv

OLD="configs/matd3-nooracle-map1000-25d-h35-130-vz4-20261123.yaml"
NEW="configs/matd3-nooracle-map1000-25d-aoi-zeng2019-h35-130-vz4-20261123.yaml"


def test_new_model_changes_only_energy_and_run_identity():
    x,y=load_config(OLD),load_config(NEW)
    assert x.energy.propulsion_model=="quadratic"
    assert y.energy.propulsion_model=="zeng2019"
    for part in ("world","sensor","network","reward","matd3"):
        assert getattr(x,part)==getattr(y,part)
    assert y.world.motion_safety=="reject"
    assert y.world.max_vertical_speed_mps==4
    assert y.runtime.episodes==20000
    assert y.runtime.warm_start_checkpoint=="none"


def test_calibrated_hover_cruise_and_positive_climb():
    c=load_config(NEW)
    velocities=np.array([[0.,0.,0.],[5.,0.,0.],[10.,0.,0.],
                         [12.,0.,0.],[15.,0.,0.],[20.,0.,0.]],dtype=float)
    power=propulsion_cost(velocities,velocities,c)/c.world.dt_s
    np.testing.assert_allclose(power,[167.3254546653501,140.533,128.873,131.493,141.63,175.614],
                               atol=.1,rtol=0)
    assert power[2]<power[0]  # slower-than-cruise hover is more costly
    assert c.energy.battery_capacity_j / power[0] > 1500
    uphill=velocities[2].copy();uphill[2]=2.
    plus=propulsion_cost(uphill[None],uphill[None],c)[0]/c.world.dt_s
    assert plus-power[2]==pytest.approx(c.energy.rotor_weight_n*2)


def test_native_equation_matches_same_zeng_parameterization():
    c=load_config(NEW);e=c.energy
    rho=e.rotor_air_density_kg_m3;area=e.rotor_disc_area_m2
    P0=e.rotor_profile_drag_coefficient/8*rho*e.rotor_solidity*area*e.rotor_angular_velocity_rad_s**3*e.rotor_radius_m**3
    Pi=(1+e.rotor_induced_correction_factor)*e.rotor_weight_n**1.5/math.sqrt(2*rho*area)
    Utip=e.rotor_angular_velocity_rad_s*e.rotor_radius_m
    v0=math.sqrt(e.rotor_weight_n/(2*rho*area))
    d0=e.rotor_fuselage_drag_ratio
    def official_formula(v):
        return (P0*(1+3*v*v/(Utip*Utip))
              + Pi*(math.sqrt(math.sqrt(1+v**4/(4*v0**4))-v*v/(2*v0*v0)))
              + .5*d0*rho*e.rotor_solidity*area*v**3)
    for v in (0,1,5,10,12,20):
        vxyz=np.array([[v,0.,0.]])
        actual=propulsion_cost(vxyz,vxyz,c)[0]/c.world.dt_s
        assert actual == pytest.approx(official_formula(v),rel=1e-11)


def test_invalid_rotor_parameters_rejected():
    c=load_config(NEW)
    for key,value in (("rotor_weight_n",0.),("rotor_radius_m",-1.),
                      ("rotor_fuselage_drag_ratio",-1.),
                      ("rotor_air_density_kg_m3",float("nan"))):
        original=getattr(c.energy,key)
        setattr(c.energy,key,value)
        with pytest.raises(ValueError):
            c.validate()
        setattr(c.energy,key,original)


def test_warp_cpu_one_step_energy_matches_native_zeng_energy():
    c=load_config(NEW)
    c.runtime.num_envs=1;c.runtime.device="cpu"
    c.world.num_targets=1;c.world.num_uavs=2;c.world.num_obstacles=0
    c.network.enabled=False
    env=WarpVectorEnv(c,device="cpu")
    try:
        env.reset(seed=44)
        wp.to_torch(env.s.positions)[0]=torch.tensor([[400.,400.],[600.,600.]],dtype=torch.float64)
        wp.to_torch(env.s.altitudes)[0]=100.
        wp.to_torch(env.s.velocities)[0]=torch.tensor([[10.,0.],[10.,0.]],dtype=torch.float64)
        wp.to_torch(env.s.vertical_velocity)[0]=0.
        before=wp.to_torch(env.s.battery)[0].clone()
        action={"continuous":np.zeros((1,2,4),np.float32),
                "destination":np.zeros((1,2),np.int64)}
        env.step(action)
        actual=wp.to_torch(env.s.flight_energy)[0].cpu().numpy()
        expect=propulsion_cost(np.array([[10.,0.,0.],[10.,0.,0.]]),
                               np.array([[10.,0.,0.],[10.,0.,0.]]),c)
        np.testing.assert_allclose(actual,expect,atol=1e-6,rtol=1e-7)
        np.testing.assert_allclose((before-wp.to_torch(env.s.battery)[0]).cpu().numpy(),
                                   expect,atol=1e-6,rtol=1e-7)
    finally:
        env.close()
