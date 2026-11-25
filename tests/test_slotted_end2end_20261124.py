"""Comprehensive Verification Suite for Streamfix v3 Production & Speed Pipeline (2026-11-24)."""

import ast
import gzip
import json
import math
import pathlib
import sys
import tempfile

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_network.environment.gpu.slotted_2d_env import (
    DISC_ACTION_DIM,
    OBS_DIM,
    STATE_DIM,
    Slotted2DUavEnv,
    generate_scenario_snapshot,
)
from uav_network.experiments.aggregate_five_seed_rliable import aggregate_runs, probability_of_improvement
from uav_network.experiments.dual_fidelity_eval import EpisodeKPI, aggregate_kpis
from uav_network.experiments.telemetry_wandb_hf import (
    SlottedTelemetryLogger,
    iqm_stratified_bootstrap_ci,
)
from uav_network.experiments.uavnetsim_csma_evaluator import (
    evaluate_on_reference_csma,
    evaluate_on_slotted_warp,
)


def test_zero_delivery_q1_metric_hygiene():
    """Verify Streamfix v3: 0 deliveries MUST return None for energy_kj_per_delivered_target, mean_aoi_s, multihop_ratio."""
    ep0 = EpisodeKPI(
        episode_return=4.427, targets_discovered=0, targets_delivered=0, total_targets=25,
        mean_aoi_s=None, p95_aoi_s=None, multihop_deliveries=0, mac_collisions=0,
        hd_drops=0, retry_drops=0, ttl_expired=0, propulsion_kj=3.022, safety_violations=0,
        censored_mean_aoi_s=120.0, aoi_samples_s=[],
    )
    agg = aggregate_kpis([ep0])
    assert agg["energy_kj_per_delivered_target"] is None, "Must NOT fabricate 3.02 kJ/target on 0 deliveries!"
    assert agg["mean_aoi_s"] is None and agg["p95_aoi_s"] is None, "Must NOT fabricate 120s AoI on 0 deliveries!"
    assert agg["multihop_delivery_ratio_pct"] is None
    assert math.isclose(agg["mean_propulsion_kj"], 3.022, abs_tol=1e-6)
    assert math.isclose(agg["censored_mean_aoi_s"], 120.0, abs_tol=1e-6)

    # n=1 CI must return (point_iqm, None, None) rather than 0-width CI
    iqm1, lo1, hi1 = iqm_stratified_bootstrap_ci(np.array([4.427]))
    assert math.isclose(iqm1, 4.427, abs_tol=1e-6) and lo1 is None and hi1 is None

    # sim_to_net_gap must never explode on 0 baseline
    with tempfile.TemporaryDirectory() as tmp:
        lg = SlottedTelemetryLogger(tmp, {"world": {"num_uavs": 6, "num_obstacles": 22, "num_targets": 25}}, "matd3", 44)
        rep = lg.log_final_dual_fidelity_test(8, [ep0], [ep0])
        lg.close()
        assert rep["sim_to_net_gap"]["delta_abs_energy_kj_per_delivered_target"] is None
        assert rep["sim_to_net_gap"]["delta_rel_pct_mean_mac_collisions"] is None


def test_real_discovery_and_multihop_delivery_pilot():
    """Verify Streamfix v3 scenario: 250-step patrol+relay policy achieves real discovery, multi-hop delivery & non-null Q1 metrics."""
    def patrol_relay_policy(obs: np.ndarray, mask: np.ndarray):
        e, n, _ = obs.shape
        ac = np.zeros((e, n, 2), dtype=np.float32)
        ad = np.zeros((e, n), dtype=np.int32)
        for ei in range(e):
            for ui in range(n):
                # Fan out across map sectors to explore & trigger DS confirmations
                ang = (ui / float(n)) * 1.45 + 0.1
                ac[ei, ui, 0] = math.cos(ang) * 0.65
                ac[ei, ui, 1] = math.sin(ang) * 0.65
                # Prefer GCS (6) if physically decodable, else first valid peer (1..5), else Silent (0)
                if mask[ei, ui, 6] == 1:
                    ad[ei, ui] = 6
                else:
                    peers = np.where(mask[ei, ui, 1:6] == 1)[0]
                    ad[ei, ui] = int(peers[0] + 1) if len(peers) > 0 else 0
        return ac, ad

    seeds = [44, 45]
    eps = evaluate_on_slotted_warp(patrol_relay_policy, seeds, max_steps=200, backend="numpy")
    agg = aggregate_kpis(eps)
    assert agg["total_targets_delivered"] >= 1, f"Expected real deliveries in 200-step pilot, got {agg}"
    assert agg["energy_kj_per_delivered_target"] is not None and agg["energy_kj_per_delivered_target"] > 0.0
    assert agg["mean_aoi_s"] is not None and 0.0 <= agg["mean_aoi_s"] < 120.0
    assert agg["confirmation_precision_pct"] is not None and agg["confirmation_precision_pct"] >= 80.0
    print(f"  [200-step CPU Pilot] deliv={agg['total_targets_delivered']:.0f}, AoI={agg['mean_aoi_s']:.2f}s, kJ/deliv={agg['energy_kj_per_delivered_target']:.2f}, prec={agg['confirmation_precision_pct']:.1f}%")


def test_warp_fast_jit_and_production_contracts():
    # 1) All 4 Warp kernel files must disable backward codegen & avoid wp.static(...) unrolling + zero fp64
    for rel in [
        "uav_network/communication/gpu/phy_sinr_fp32.py",
        "uav_network/communication/gpu/mac_slotted_fp32.py",
        "uav_network/environment/gpu/ds_aoi_belief_fp32.py",
        "uav_network/environment/gpu/slotted_2d_warp_kernels.py",
    ]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert '"enable_backward": False' in src, f"enable_backward=False missing in {rel}"
        assert "range(wp.static(" not in src, f"Unrolled range(wp.static(...)) found in {rel}"
        assert "wp.float" + "64" not in src, f"Forbidden 64-bit Warp float found in {rel}"
        ast.parse(src)

    # 2) Verify YAMLs pass strict validation with rebalanced max_accel=4.0, w4=3.0, w2=15.0
    from uav_network.experiments.train_slotted_vanilla import validate_scenario_config
    for cfg_name in [
        "configs/vanilla_matd3_shared_2d_ds_aoi.yaml",
        "configs/vanilla_masac_shared_2d_ds_aoi.yaml",
    ]:
        cfg = yaml.safe_load((ROOT / cfg_name).read_text(encoding="utf-8"))
        assert cfg["world"]["max_accel_mps2"] == 4.0
        assert cfg["reward"]["w4_safety_violation"] == 3.0
        assert cfg["reward"]["w2_gcs_aoi_delivery"] == 15.0
        validate_scenario_config(cfg)

    # 3) Historical Kaggle notebook was explicitly retired from v5.9 HEAD.
    # Its old cell AST and dependencies belong to v5.8, not the active CPU/GPU
    # scenario under test. Preserve traceability in the immutable Git manifest.
    retired_name = "RUN_SLOTTED_VANILLA_T4X2_END2END_20261124.ipynb"
    assert not (ROOT / "notebooks" / retired_name).exists()
    archive = (ROOT / "docs/benchmarks/NOTEBOOK_ARCHIVE_V59_20261125.md").read_text(encoding="utf-8")
    assert retired_name in archive
    assert "7d7ab1ccea3d31a121f9708e428bbf96c3c529a6" in archive

    meta = json.loads((ROOT / "notebooks/kaggle_account1_public_kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["is_private"] is False and meta["enable_gpu"] is True


if __name__ == "__main__":
    test_zero_delivery_q1_metric_hygiene()
    test_real_discovery_and_multihop_delivery_pilot()
    test_warp_fast_jit_and_production_contracts()
    print("ALL STREAMFIX V3 END-TO-END & Q1 PILOT VERIFICATION TESTS PASSED!")
