# UAV Slotted Streamfix v2 — Integration Verification (2026-11-24)

## Source provenance
- User patch: `uav_slotted_end2end_streamfix_v2_20261124.zip` SHA256 `d4d4ffca9732194005b15523bd5e1da82a5b391596e64e2c9db67db50de3c6ae` (17 source/config/test/docs files; all 17 hash checks passed).
- Integrated on top of previously tested `uav_network_slotted_prod_streamfix_20261124` source (not the active 3D project).
- Git worktree branch: `experiment/uav-slotted-streamfix-v2-integrated-20261124`.

## Fixes found and applied after ZIP merge
1. `WarpSlotted2DBackend` used undefined `td` when allocating `terminated_t`.
2. Wrong position of `node_pos`/`last_act_norm` kernel arguments: removed spurious `assemble_reward_and_done_fp32` signature params; added missing `selective_auto_reset_fp32` params to match actual `wp.launch` inputs.
3. Reset unconditionally invoked `torch.cuda.current_stream(...)`, breaking Warp CPU JIT: conditional CUDA scoped stream / CPU nullcontext.
4. Dempster–Shafer closed-form fusion underflowed and divided by zero at large observation counts. Reimplemented normalized log-space formula in CPU and Warp FP32.
5. NumPy action shape validation, plus physical UAV-UAV minimum separation projection (CPU & Warp) instead of penalty only.
6. Slotted MAC CPU defers at final busy CCA subslot rather than transmitting forcibly.
7. Radio transmission draws shared battery: CPU battery stored in float64 for sub-0.01 J TX precision.
8. Real UavNetSim adapter imports vendor GainEstimate from `phy.a2a`, creates compatible urban scene, invokes `Simulator` supported constructor, keeps the vendored drone processes and carries proper `AckPacket`/unicast fields.
9. Updated old regression tests to the real Warp/UavNetSim interfaces instead of expecting `NotImplementedError`.

## Tested
- Integrated pytest suite: **30 passed / 1 skipped** on Ubuntu (Warp GPU preflight skipped because current Ubuntu host has no CUDA driver).
- Independent Warp CPU kernel pipeline: init, reset, and three steps passed again after the final UAV-separation GPU kernel adjustment (verified JIT module db88045).
- Vendored UavNetSim SimPy evaluation: one seed two steps succeeded.

## Still not verified or not completed
- New v2 changes and latest GPU safety projection not yet executed on Kaggle T4x2 in this release.
- Full 550-step acceptance, multi-seed throughput and policy performance, 20M env transitions, W&B/HF online continuity and full replay/RNG/environment state restoration after resume.
- GPU autoreset reuses the staged same scenario rather than generating fresh independent scenes; confirm scientific protocol before long-run learning.
- The previous Kaggle 10/10 GPU smoke is for **Streamfix v1**, not evidence of v2 acceptance.

## Security and isolation
Active `/home/aduy/Documents/NCKH/uav_network_study_1200` and user checkpoints were not modified. No local `key.txt` or API credentials were copied into the Git branch/source release.
