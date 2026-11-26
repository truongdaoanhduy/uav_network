#!/usr/bin/env python3
"""Evidence-led static *research scenario* audit; NOT a dynamic parity verifier."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.layout_v67 import Q_V67, Q_V67_MAX, backing_bytes_per_environment_v67

UPSTREAM_UAVNETSIM_NOISE_W = 4.0e-11  # Zhou et al. (2025), Table I (scenario-specific, not universal)
UPSTREAM_URL = "https://arxiv.org/abs/2507.09852"
ZHANG_URL = "https://doi.org/10.1109/TIV.2024.3352581"
XUE_URL = "https://doi.org/10.3390/electronics15091875"


def analyze(cfg, *, warp_q_cap=Q_V67, num_envs=4096):
    findings = []
    def record(fid, kind, status, message, fix, evidence):
        findings.append({"id": fid, "class": kind, "status": status,
                         "finding": message, "recommended_action": fix, "evidence": evidence})
    record("MAP-01", "scope", "methodological_mismatch",
           f"Velocity command is XY in 2D with fixed altitude {cfg.uav_alt_m} m, "
           f"although 3D sensing occlusion is {cfg.enable_3d_sensing_occlusion}.",
           "Call it 2D motion with height-aware LOS; port altitude control only with 3D energy/obstacle parity.",
           ["research/zhang_xue/src/zhang_xue/environment/transition.py", "research/zhang_xue/src/zhang_xue/communication/channel.py"])
    record("MAP-02", "assumption", "ablation_needed",
           f"grid {cfg.grid_w}×{cfg.grid_w} @50m, Pd={cfg.pd}, Pf={cfg.pf}, "
           f"DS masses=({cfg.ds_mf},{cfg.ds_me}), calibration={cfg.evidence_calibration}.",
           "Ablate Pd/Pf, fixed DS mass vs sensor likelihood calibration; calibrate to real sensor, not by paper popularity.",
           [ZHANG_URL, "research/zhang_xue/src/zhang_xue/belief/dempster_shafer.py"])
    if cfg.ds_fusion_mode != "zhang_max" or cfg.ds_aoi_eta > 0:
        record("DS-01", "paper_fidelity", "open",
               f"Actual fusion {cfg.ds_fusion_mode}, AoI eta={cfg.ds_aoi_eta}; not a literal Zhang MAX or Xue direct BPA implementation.",
               "Declare new count-power/EWMA variant and benchmark Zhang MAX and Xue direct BPA separately.",
               [ZHANG_URL, XUE_URL, "research/zhang_xue/src/zhang_xue/belief/dempster_shafer.py"])
    ratio = UPSTREAM_UAVNETSIM_NOISE_W / cfg.mac_noise_power_w
    db = 10.0 * math.log10(ratio)
    if abs(db) > 1e-6:
        record("PHY-01", "model_mismatch", "ablation_needed",
               f"Noise power is {cfg.mac_noise_power_w:.6g} W vs {UPSTREAM_UAVNETSIM_NOISE_W:.6g} W in UavNetSim Table I; delta={db:+.2f} dB (reference/current).",
               "Document interference/noise model and compare matched PHY assumptions; avoid attributing PDR gains solely to policy.",
               [UPSTREAM_URL, "research/zhang_xue/configs/scenario_v67_datamule_1km.yaml"])
    record("NFZ-01", "research_assumption", "ablation_needed",
           f"RF obstacle model {cfg.rf_obstacle_model!r}; per-building loss={cfg.rf_per_building_loss_db}dB. "
           "Only physical building cylinders obstruct radio; legal no-fly zones alone have no RF absorption.",
           "Keep binary as provenance baseline; compare multi_obstacle and rf_opaque with reference parity and real packet workload.",
           ["research/zhang_xue/src/zhang_xue/communication/common.py", "https://www.itu.int/rec/R-REC-P.2040"])
    record("PHY-02", "model_mismatch", "open",
           "Custom PHY uses NLoS inverse-cubic path loss plus 12 dB excess loss; not a line-for-line copy of UavNetSim channels.",
           "Compare received SNR/PDR across LOS, NLOS, blackout, contention; use provided PHY reference gate first.",
           ["research/zhang_xue/src/zhang_xue/gpu/kernels/phy_v67.py", UPSTREAM_URL])
    if int(cfg.mac_queue_capacity) != int(warp_q_cap):
        record("NET-01", "config_conflict", "fixed_by_explicit_v76_opt_in",
               f"CPU MAC queue={cfg.mac_queue_capacity}; Warp requested queue={warp_q_cap}.",
               f"Choose --warp-q-cap {cfg.mac_queue_capacity} --require-queue-parity only when <= {Q_V67_MAX}.",
               ["uav_network/experiments/radio_config.py", "research/zhang_xue/src/zhang_xue/gpu/layout_v67.py"])
    record("NET-02", "paper_fidelity", "open",
           "Local custom SevenRadioCsmaCaNetwork/Warp packet simulator is inspired by UavNetSim; upstream SimPy routing/MAC code is not executed here.",
           "Implement a statistical cross-simulator benchmark rather than claiming upstream implementation equivalence.",
           ["research/zhang_xue/src/zhang_xue/communication/network.py", "https://github.com/Zihao-Felix-Zhou/UavNetSim"])
    if cfg.w_p2p == 0 or cfg.lambda_b == 0:
        record("REW-01", "objective_gap", "ablation_needed",
               f"Network reward weights: w_p2p={cfg.w_p2p}, lambda_b={cfg.lambda_b}; direct rewards/airtime costs may be absent.",
               "Keep zero-weight task baseline; add separately registered information gain, delivery, airtime, AoI ablations.",
               ["research/zhang_xue/src/zhang_xue/environment/transition.py"])
    record("REW-02", "theorem_scope", "open",
           "Monotone minimum-uncertainty reward watermark bounds revisit farming but is NOT guaranteed potential-based reward shaping.",
           "Compare learned returns and behavior to a task reward baseline; do not claim Ng 1999 policy invariance.",
           ["https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf",
            "research/zhang_xue/src/zhang_xue/environment/transition.py"])
    if not cfg.enable_accel_energy:
        record("ENE-01", "energy_scope", "ablation_needed",
               "Kinematic acceleration limit is on, but accelerated propulsion energy option is disabled.",
               "State as steady-level propulsion simplification or enable and validate acceleration-aware power ablation.",
               ["research/zhang_xue/src/zhang_xue/mobility/flight_safety_energy.py"])
    record("MET-01", "metric_bias", "open",
           "Mapwide censored AoI is accumulated over actual episode length; comparisons with unequal termination time are confounded.",
           "Report completed-episode length and delivered-cell coverage, plus a fixed-horizon censored AoI.",
           ["research/zhang_xue/src/zhang_xue/environment/transition.py", "https://arxiv.org/abs/2007.08564"])
    record("GPU-01", "incomplete_pipeline", "open",
           "Capacity probe only executes Warp radio and repeats four CPU-generated transitions into replay; no vectorized GPU flight/sensing/reward/reset.",
           "Gate motion→sensor→belief→network→reward→reset per transition on GPU before full MARL claims.",
           ["uav_network/experiments/capacity_probe.py"])
    record("GPU-02", "memory_inefficiency", "open",
           "TensorReplayBuffer stores obs+state and next_obs+next_state as separate FP32 allocations even when state is flattened obs.",
           "Before switching to FP16 state-derived replay, enforce exact state semantics and measure quantization effects on a validation set.",
           ["uav_network/algorithms/core/replay.py",
            "research/zhang_xue/src/zhang_xue/learners/state_derived_replay.py"])
    per_env = backing_bytes_per_environment_v67(cfg.num_uavs, cfg.grid_w, q_cap=warp_q_cap)
    return {"scenario": {"map_m": cfg.map_size_m, "grid_w": cfg.grid_w,
                          "num_uavs": cfg.num_uavs, "num_targets": cfg.num_targets,
                          "horizon_steps": cfg.safety_horizon_s, "battery_j": cfg.e_max_j,
                          "dt_s": cfg.dt_s, "comm_action_mode": cfg.comm_action_mode,
                          "data_mule": cfg.enable_data_mule_forwarding, "peer": cfg.enable_peer_dissemination},
            "radio_soa_estimate": {"envs": num_envs, "warp_q_cap": warp_q_cap,
                                   "bytes_per_env": per_env, "total_gib": per_env*num_envs / 2**30,
                                   "not_a_training_vram_guarantee": True},
            "classification": "static_source_audit_not_simulator_parity",
            "findings": findings}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario",type=Path,default=ROOT/"research/zhang_xue/configs/scenario_v67_datamule_1km.yaml")
    parser.add_argument("--warp-q-cap",type=int,default=Q_V67)
    parser.add_argument("--envs",type=int,default=4096)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    cfg=load_scenario(args.scenario)
    report=analyze(cfg,warp_q_cap=args.warp_q_cap,num_envs=args.envs)
    rendered=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__=='__main__':
    main()
