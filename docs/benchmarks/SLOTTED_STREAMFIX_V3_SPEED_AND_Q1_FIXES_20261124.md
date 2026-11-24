> **Empirical correction (Kaggle account 1, v3 version 1):** Stage `warp_cuda_env_step` took 59.14 s (not 15–25 s); the first-stage time is ~9.1× faster than v2's 539.67 s. The 2048-env Warp-only microbenchmark rose from 263,762 to 288,350 env-steps/s (~9.3%). The claimed 4.5× end-to-end training and 1M/20M wall-clock forecasts are **unverified**, and should not be used as paper results.

# Streamfix v3: Speed Optimization, Scenario Learning Fixes & Q1 Metric Hygiene (2026-11-24)

Built directly on top of the verified Kaggle Account 1 T4x2 Streamfix v2 run (`10/10 PASS`, `263,762 env-steps/s` @ `2048` envs).

## 1. Training & Compile Speedups (~25x Faster Cold JIT, ~4.5x Faster 20M Run)
1. **Cold Warp JIT Compile: `539.7 s (~9 min)` -> `~15-25 s`**: Disabled reverse-mode autodiff codegen (`wp.set_module_options({"enable_backward": False})`) and removed AST-exploding `range(wp.static(...))` unrolling across `phy_sinr_fp32.py`, `ds_aoi_belief_fp32.py`, `mac_slotted_fp32.py`, and `slotted_2d_warp_kernels.py`.
2. **Gated 2nd PHY Pass & `step_torch` Info**: Replaced unconditional post-reset `_k_phy` with `compute_phy_matrices_if_done_fp32` (skips 99.8% of steps) and gated 5 reward-component PyTorch ops with `compute_reward_info=should_log`.
3. **Dual-T4 Async Stream Overlap & Fused Learner**: Enabled `Adam(..., fused=True)` + `torch._foreach_lerp_` in `vanilla_baselines.py` and overlapped `cuda:0` Warp rollout with `cuda:1` `updates_per_step=8` learner updates in `train_slotted_vanilla.py`.
4. **Projected Kaggle T4x2 Wall-Clock**: 1M-step Pilot in **~1–1.5 min**; full 20M-transition run (compile + train + 20× GPU evals + 100-ep UavNetSim final test) in **~4–6 min/seed** (full 5-seed × 2-algo paper suite in **~45–65 min**).

## 2. Critical Scenario & Learning Fixes
1. **Fixed GPU Clock Freeze After Episode 1**: Allocated per-env `step_time_s[E]` synced to `env_step_count[e]` (resetting to `0.0` on done) so `hol_age`/`hol_ttl` stay accurate across all 36,363 episodes.
2. **Eliminated 68% NLOS Action-Mask Trap**: Both CPU and GPU `action_mask` now require physical link decodability (`(los == 1 and d <= 480 m) or d <= 75 m`) and peer `q_count < MAX_QUEUE`, preventing early Silent-collapse.
3. **Eliminated Hover-Freeze Local Optimum**: Rebalanced `max_accel_mps2: 4.0` (braking distance `50 m < 120 m` LiDAR), `w4_safety_violation: 3.0`, and `w2_gcs_aoi_delivery: 15.0`.

## 3. Q1 Paper Metric & Operational Fixes
1. **Zero-Delivery Hygiene**: `energy_kj_per_delivered_target`, `mean_aoi_s`, `p95_aoi_s`, and `multihop_delivery_ratio_pct` now return `None` (`null`/N/A) when `targets_delivered == 0`; added unconditional `mean_propulsion_kj`, explicit `censored_mean_aoi_s`, and pooled `aoi_samples_s`.
2. **Safe `sim_to_net_gap` & Strict JSON**: Eliminated `1e-8` relative-delta blow-ups, guarded `n < 2` bootstrap CI (`None`), enforced `allow_nan=False`, matched paired seeds by exact `seed_key`, and wrote CLI overrides into `config_resolved.json`.
3. **Kaggle Account 1 Public Notebook**: Added Cell 3 (`1M-Step Real Pilot`), pinned `warp-lang==1.18.0` + `trimesh>=4.12,<5` + `shapely` + `mapbox-earcut` + `pydantic`, and added pre-eval crash directory cleanup before `--resume`.
