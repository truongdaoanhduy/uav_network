# Streamfix v3 integration forensic audit — 2026-11-24

## Source
- User patch SHA-256 `fa23603b98c44d90d4d5420121b8e38c24138189bece52e3d9f98dda2d6b1b5e`, 18 files; all 18 entries validated against bundled `STREAMFIX_V3_SHA256.json` and ZIP CRC.
- Base v2 commit `120e65ba9abed25728708ac0214656293db4cb74`.
- Worktree branch `experiment/uav-slotted-streamfix-v3-fast-20261124`.
- Vendored `third_party/UavNetSim` restored with its upstream LICENSE. Previous Git commit did not track this ignored directory, so a fresh clone could not complete full UavNetSim evaluation.

## Defects discovered beyond supplied ZIP
1. `compute_phy_matrices_if_done_fp32` was declared inside `analytical_link_budget_summary()` after a `return` (not registered as a module symbol). Moved to `if wp is not None` module scope.
2. Gated PHY kernel passed a `wp.vec3f` pair into `check_link_los_fp32`, whose actual signature takes 7 scalar coordinate arguments plus obstacle buffers. Corrected the call.
3. `assemble_reward_and_done_fp32` zeroed the per-env episode clock on `done`, then immediately overwrote it with `sc` unconditionally. Removed the duplicate write.
4. Previously ignored vendored UavNetSim files were brought into Git tracking to make the repository portable and reproducible.

## Actual verification, not performance projections
- Ubuntu pytest selected suite: **28 passed / 1 GPU-only skipped**, 0 failed, after integration and vendor recovery.
- Warp 1.18.0 CPU JIT 2 envs × 7 steps through two 3-step episodes: reset clock sequence 1,2,0,1,2,0,1, PASS; initial CPU kernel module compilation ~10.45s (this does NOT validate CUDA JIT).
- 200-step heuristic patrol+relay CPU test on seeds 44,45: **19 target reports delivered, mean AoI 1.58s, confirmation precision 97.0%**. This policy is handcrafted and does not prove MATD3/MASAC convergence.
- Kaggle account 1 GPU smoke link: https://www.kaggle.com/code/longlonthiu/uav-slotted-v3-fast-smoke-a1-20261124 ; submitted as Public; source dataset intentionally Private.

## Research limits and gaps
- Claims that cold CUDA JIT takes 15–25s, 1M transitions takes 1–1.5min, and full 20M+100 UavNetSim episodes takes 4–6min remain **unverified until logs from the new Kaggle GPU run and real production pilot**. Do not cite them as actual timings.
- Kaggle mini-runs with 8 transitions do not cross `learning_starts=20000`, so standalone learner kernel tests do not establish long-run convergence.
- Avoid reusing CPU pilot metrics as trained-policy evaluation, and do not claim that action masking alone corrects a learning failure.
- Full 550-step reliability, GPU/CPU trajectory parity after autoreset, replay resume fidelity, 5 seeds and real W&B/HF upload are still pending acceptance.

## Kaggle T4x2 v3 smoke — notebook version 1 (downloaded actual logs)
- 11 of 12 smoke stages PASS, one FAIL solely in packaging: test contract reads `notebooks/RUN_SLOTTED_VANILLA_T4X2_END2END_20261124.ipynb` but source tar omitted `notebooks/`; not a CUDA/algorithm failure. Updated dataset version adds only the two required notebook assets.
- Cold Warp CUDA env stage (including initial JIT): **59.14 seconds** vs v2 **539.67 seconds**, approximately **9.1x faster**, not proven 25x.
- Pure environment-only 2048-env throughput: **288,349.9 vs 263,761.6 env-steps/s**, approximately **9.3% faster**. Does not include actor, replay, optimizer, evaluation or logging.
- GPU CUDA Graph PASS; per-world episode clocks `[1,2,0,1,2,0,1]` PASS; MATD3 and MASAC standalone critic+actor updates PASS; real UavNetSim injected packet PASS.
- Mini runs with 8 transitions each still perform **zero** optimizer updates; do not mistake these for learned policy performance.
- Further validation: user-requested 1M-transition GPU pilot pending; wall-clock forecasts remain unverified.
