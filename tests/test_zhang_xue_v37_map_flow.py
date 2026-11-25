"""Map-only end-to-end workflow gate: batch scene → motion → sensing → DS."""
import numpy as np
from scripts.verify_zhang_xue_map_v37 import run_map_flow


def test_full_warp_map_flow_64_envs_without_network_or_training():
    report=run_map_flow(envs=64,steps=4,seed=44,device="cpu")
    assert report["environment_count"]==64
    assert report["total_map_env_steps"]==256
    assert report["targets_per_world"]==10
    assert report["map_sensing_mass_finite"] is True
    assert report["all_map_positions_in_bounds"] is True
    assert report["no_network_or_training"] is True
    assert report["target_ground_truth_hidden_from_actor_state"] is True
    assert 0.<report["coverage_fraction_mean"]<=1.
    assert report["map_device"]=="cpu"
