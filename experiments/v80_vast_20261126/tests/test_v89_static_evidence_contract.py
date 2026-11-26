"""Static-target evidence cannot fade solely due to radio blackout."""
from pathlib import Path
import pytest
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.mission_v80 import GPUWorldV80Params

SCENARIO = Path("research/zhang_xue/configs/scenario_v89_static_1km.yaml")

def test_static_evidence_scenario_loads_and_is_explicit():
    cfg=load_scenario(SCENARIO)
    assert cfg.static_evidence_mode == "raw_u16_count"
    assert cfg.ds_aoi_eta == 0.0
    p=GPUWorldV80Params.from_scenario(cfg,envs=1,device="cpu")
    assert p.static_raw_count is True

def test_static_900_step_mode_refuses_legacy_q88_semantics():
    cfg=load_scenario(SCENARIO)
    assert cfg.safety_horizon_s == 900
    assert cfg.num_targets == 15
    assert cfg.grid_w == 20
    assert cfg.map_size_m == 1000.0
