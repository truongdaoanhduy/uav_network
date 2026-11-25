"""Gate5H: source-faithful native counters and seed-matched Warp MAC."""
import pytest
import os
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1]
_UPSTREAM=Path(os.environ.get("UAV_CLASSIC_V1_ROOT",_ROOT/"outputs/reference_sources/UavNetSim-v1-classic"))
if not (_UPSTREAM/"entities/drone.py").is_file():
    pytest.skip("Pinned native UavNetSim-v1 source unavailable",allow_module_level=True)
pytest.importorskip("warp")
from scripts.verify_zhang_xue_network_v40 import run_parity

def test_native_backoff_draws_replayed_into_warp_match_six_seed_delivery():
    result=run_parity(start_seed=44,num_seeds=6)
    assert result["mac_backoff_matches_actual_native_rng"]
    assert result["native_metric_matches_physical_data_delivery"]
    assert result["native_pdr_by_seed"]==[0.,1.,1.,0.,0.,0.]
    assert result["warp_pdr_by_seed"]==result["native_pdr_by_seed"]
    assert result["pdr_seed_mismatch_count"]==0
    assert result["data_tx_attempt_total_native"]==result["data_tx_attempt_total_warp"]
    assert result["max_first_attempt_start_error_us"]<0.1
    assert result["max_any_attempt_start_error_us"]<=550.1
    assert result["full_gate5_status"]=="NOT_PASSED_ONE_HOP_ONLY"
    assert result["training_permitted"] is False


def test_native_rng_replay_scales_to_32_seeds_with_short_successful_tries():
    report=run_parity(start_seed=44,num_seeds=32)
    assert len(report["seeds"])==32
    assert len(report["native_draw_counts_by_seed"])==32
    assert report["native_metric_matches_physical_data_delivery"]
    assert report["censored_warp_retry_count"]>=0
    assert report["full_gate5_status"]=="NOT_PASSED_ONE_HOP_ONLY"
