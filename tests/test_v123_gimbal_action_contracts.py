"""V123 research-only camera azimuth action RED/GREEN gates, no baseline edits."""
import hashlib
import math
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from zhang_xue.environment.configuration import EnvConfigV54
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.sensing.gimbal_cone import is_cell_in_gimbal_cone, slew_azimuth
from zhang_xue.gpu.mission_v79 import GPUWorldV79Params

ROOT=Path(__file__).resolve().parents[1]

def test_optical_cone_yaw_controls_ground_visibility():
    """Same UAV location, different non-motion action must alter sensor footprint."""
    assert is_cell_in_gimbal_cone(50, 0, 50, 0, 25, 25)
    assert not is_cell_in_gimbal_cone(50, 0, 50, math.pi, 25, 25)
    assert is_cell_in_gimbal_cone(0, 50, 50, math.pi/2, 25, 25)
    assert not is_cell_in_gimbal_cone(0, 50, 50, -math.pi/2, 25, 25)


def test_slew_rate_and_wrap():
    old=np.array([0.0, math.pi-0.1])
    tgt=np.array([math.pi, -math.pi+0.1])
    new=slew_azimuth(old,tgt,math.pi/4)
    assert np.isclose(new[0], -math.pi/4)  # shortest direction tie: -pi
    assert np.isclose(new[1], -math.pi+0.1, atol=1e-10)
    assert np.all(np.abs((new-old+math.pi)%(2*math.pi)-math.pi) <= math.pi/4+1e-10)


def test_invalid_gimbal_params_fail_closed():
    with pytest.raises(ValueError, match='Gimbal'):
        EnvConfigV54(enable_gimbal_azimuth_action=True,gimbal_tilt_deg=70,gimbal_half_fov_deg=40)
    with pytest.raises(ValueError, match='Gimbal'):
        EnvConfigV54(gimbal_max_rate_deg_s=0)
    with pytest.raises(ValueError):
        is_cell_in_gimbal_cone(1,2,50,float('nan'),25,25)


def test_action_contract_and_sensing_observation_cpu():
    cfg=EnvConfigV54(enable_gimbal_azimuth_action=True,obs_schema_v2=True,
                     comm_action_mode='compact_3',enable_peer_dissemination=False,
                     safety_horizon_s=900, map_size_m=1000, grid_w=20)
    env=ZhangXue7NodePacketEnv(cfg)
    obs,_,_=env.reset(seed=44)
    assert obs.shape==(6,551)
    assert np.allclose(obs[:,-2:],[0,1])
    vel=np.zeros((6,2)); comm=np.zeros(6,dtype=np.int32)
    with pytest.raises(ValueError, match='required'):
        env.step(vel,comm)
    for invalid in (np.full(6,1.1),np.full(6,np.nan),np.zeros(5)):
        with pytest.raises(ValueError): env.step(vel,comm,gimbal_azimuth_cmd=invalid)
    assert env.t==0  # rejects before advancing clock
    obs,_,_,_,_,_,info=env.step(vel,comm,gimbal_azimuth_cmd=np.full(6,0.5))
    assert np.allclose(env.gimbal_azimuth_rad,math.pi/4)
    assert np.allclose(obs[:,-2:],[math.sin(math.pi/4),math.cos(math.pi/4)],atol=1e-5)
    assert info['gimbal_pilot_cpu_only'] is True
    assert all(abs(a-b)<1e-10 for a,b in zip(info['gimbal_azimuth_rad'],env.gimbal_azimuth_rad))
    obs2,_,_,_,_,_,_=env.step(vel,comm,gimbal_azimuth_cmd=np.full(6,0.5))
    assert np.allclose(env.gimbal_azimuth_rad,math.pi/2)
    assert obs2.shape==(6,551)
    env.reset(seed=44)
    assert np.allclose(env.gimbal_azimuth_rad,0)


def test_gimbal_action_is_transactional_on_invalid_motion_or_comm():
    cfg=EnvConfigV54(enable_gimbal_azimuth_action=True,obs_schema_v2=True,
                     comm_action_mode='compact_3')
    env=ZhangXue7NodePacketEnv(cfg)
    env.reset(seed=44)
    unchanged=env.gimbal_azimuth_rad.copy()
    airtime=env.net.step_tx_airtime_us.copy()
    bad_actions=(
        (np.full((6,2),np.nan),np.ones(6,dtype=np.int32)),
        (np.zeros((6,2)),np.full(6,99,dtype=np.int32)),
    )
    for xy,comm in bad_actions:
        with pytest.raises(ValueError):
            env.step(xy,comm,gimbal_azimuth_cmd=np.full(6,.5))
        np.testing.assert_array_equal(env.gimbal_azimuth_rad,unchanged)
        np.testing.assert_array_equal(env.net.step_tx_airtime_us,airtime)
        assert env.t==0


def test_default_canonical_rejects_extra_sensor_action():
    cfg=EnvConfigV54(obs_schema_v2=True,comm_action_mode='compact_3')
    env=ZhangXue7NodePacketEnv(cfg)
    obs,_,_=env.reset(seed=44)
    assert obs.shape==(6,549)
    with pytest.raises(ValueError,match='required iff'):
        env.step(np.zeros((6,2)),np.zeros(6,dtype=np.int32),gimbal_azimuth_cmd=np.zeros(6))
    assert env.t==0


def test_warp_failclosed_on_cpu_only_research_action():
    cfg=EnvConfigV54(enable_gimbal_azimuth_action=True)
    with pytest.raises(ValueError,match='fail closed: gimbal'):
        GPUWorldV79Params.from_scenario(cfg,envs=1)


def test_original_yaml_algorithm_byte_identity():
    paths=(
        'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml',
        'research/zhang_xue/configs/scenario_v105_ds_rf_stress_1km.yaml',
    )
    expected=('3fcfe434b35c29310835c2aa99f982a96c4af58375dc5e916e1e4d017a832fc5',
              '3ee084e793716baa9ea83d842af0b51345690fdb8156ab0e5c527739ad033b76')
    for p,h in zip(paths,expected):
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    algos=sorted((ROOT/'uav_network/algorithms').rglob('*.py'))
    assert len(algos)==30
    digest=hashlib.sha256()
    for p in algos:
        digest.update(f'{p.relative_to(ROOT).as_posix()}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n'.encode())
    assert digest.hexdigest()=='96d57cd9c4b60679fe8f74eb0f8cb2d67ced12b9dbacec99858f8c9551ad1074'
