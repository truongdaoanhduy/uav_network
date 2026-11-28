from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from zhang_xue.environment.aoi_metrics import fixed_horizon_aoi_no_new_deliveries
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.configuration.loader import load_scenario

ROOT=Path(__file__).resolve().parents[1]
SCENARIO=ROOT/'research/zhang_xue/configs/scenario_v67_datamule_1km.yaml'


def test_fixed_horizon_terminal_padding_arithmetic():
    # Two cells have ages [0,1], then [0,2]; after termination they are [1,3],[2,4]
    got=fixed_horizon_aoi_no_new_deliveries(
        observed_sum_cell_seconds=3, current_mean_cell_age_s=1, t=2, horizon=4, num_cells=2)
    assert got==pytest.approx(13/8)
    at_h=fixed_horizon_aoi_no_new_deliveries(
        observed_sum_cell_seconds=3, current_mean_cell_age_s=1, t=2, horizon=2, num_cells=2)
    assert at_h==pytest.approx(3/4)


def test_fixed_horizon_requires_valid_nonnegative_values():
    with pytest.raises(ValueError):
        fixed_horizon_aoi_no_new_deliveries(observed_sum_cell_seconds=1,current_mean_cell_age_s=0,t=5,horizon=4,num_cells=2)
    with pytest.raises(ValueError):
        fixed_horizon_aoi_no_new_deliveries(observed_sum_cell_seconds=float('nan'),current_mean_cell_age_s=0,t=1,horizon=4,num_cells=2)


def test_observation_info_exposes_terminal_metric_only_when_done():
    c=replace(load_scenario(SCENARIO), safety_horizon_s=2)
    # eta>0 so short horizon is allowed by UQ8.8 guard
    e=ZhangXue7NodePacketEnv(c,seed=12)
    e.reset(seed=12)
    zero=np.zeros((c.num_uavs,2))
    silence=np.zeros(c.num_uavs,dtype=np.int64)
    _,_,_,_,a,b,info1=e.step(zero,silence)
    if not (a or b):
        assert info1['mean_gcs_aoi_fixed_horizon_no_future_delivery_s'] is None
        _,_,_,_,a,b,info2=e.step(zero,silence)
        assert a or b
        assert isinstance(info2['mean_gcs_aoi_fixed_horizon_no_future_delivery_s'],float)
