"""Source-level regression for scientific configuration fail-closed behavior."""
import math
import pytest
from zhang_xue.environment.configuration import EnvConfigV54, make_v57_config


@pytest.mark.parametrize("bad",[
    {"safety_mode":"typo"},
    {"obs_pool_mode":"unsupported"},
    {"peer_reward_mode":"unknown"},
    {"aoi_reward_mode":"unknown"},
    {"v_max_mps":0.0},
    {"a_max_mps2":-1.0},
    {"e_max_j":0.0},
    {"tx_power_w":-0.1},
    {"uav_alt_m":-1.0},
    {"fov_radius_m":0.0},
    {"uav_safe_distance_m":0.0},
    {"bba_reliability_cap":2.0},
    {"mac_difs_us":-1},
    {"w_loc_team":math.nan},
    {"e_max_j":math.inf},
    {"dt_s":math.nan},
    {"safety_horizon_s":0},
    {"enable_3d_sensing_occlusion": "true"},
])
def test_config_rejects_bad_physics_and_unknown_critical_modes(bad):
    with pytest.raises(ValueError):
        make_v57_config(**bad)


def test_current_benchmark_valid_and_scalar_bounds():
    cfg=make_v57_config()
    assert cfg.safety_mode=="kinematic_shield"
    assert cfg.e_max_j==270_000
    assert cfg.mac_queue_capacity==64


def test_retired_hard_veto_motion_no_longer_selectable():
    with pytest.raises(ValueError, match='safety_mode'):
        EnvConfigV54(safety_mode='hard_veto')


def test_canonical_cli_rejects_old_scenario_presets():
    from zhang_xue.configuration.loader import load_scenario
    with pytest.raises(ValueError, match='preset'):
        load_scenario(preset='v54')
