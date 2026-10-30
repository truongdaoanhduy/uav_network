# XYZ Bayesian Sensing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Add controllable XYZ search with altitude-dependent Bayesian sensing and 2D visualization.

**Architecture:** Keep legacy XY behavior/configurations loadable. New XYZ profiles use dimension-aware policies/replay, a shared mathematical sensing contract and matching Native/Warp paths. Store full beliefs in worlds/resume and compact local patches in replay.

**Tech Stack:** Existing Python, NumPy, PyTorch, NVIDIA Warp and Matplotlib; no new dependencies.

**Spec:** ../specs/2026-10-30-xyz-bayesian-sensing-design.md

## Global Constraints

- Approved by user on 2026-10-30: implement inline, then one independent whole-change review.
- 6 UAV,1000m,20 targets,15 obstacles,1200 steps; legacy XY YAMLs unchanged.
- XYZ action [ax,ay,az,power], learned RF; legacy XY [ax,ay,power].
- Altitude0..150m,initial100m; finite-height obstacles30..120m; ground targets/GCS.
- 25m cells,prior0.5,FOV90°,Pd/Pf anchors and confirmation<=50m/posterior>=0.99/fresh-positive per spec.
- Keep full beliefs out of replay; local5x5 patch; preserve actor locality and active index9.
- Keep FP32,E4096,B8192,replay1M,20k episodes in requested profiles; no cloud launch.
- Separate validation/final banks; keep best actors and final full-resume semantics.
- <=500 lines per owned text source. Preserve preexisting CURRENT_TASK edits.
- Existing isolated linked worktree and working-branch push authorized by AGENTS.md.

## Review Focus

- Masked resets must not resample/scrub untouched worlds' beliefs/RNG.
- Unconfirmed target truth must not leak via actor target slots or false-positive handling.
- Hidden hardcoded three-slot assumptions must not corrupt az versus RF in inference/replay/critics.
- Native/Warp radio admission/LOS must use real altitude, not the old fixed100m.
- Restoring incompatible/invalid state must fail before mutation; legacy XY state stays readable.

### Task 1: Native XYZ world and Bayesian camera

**Files:** configuration.py; environment/world.py,env.py,sensing.py,observations.py,motion.py; create environment/xyz_geometry.py,xyz_motion.py,bayesian.py; tests/test_xyz_sensing.py.
**Interfaces:** Config.motion_dim/continuous_dim; sensor model and grid settings; bayes_update(),cell_coverage(),sensor_uniform(); sense_bayesian(env); move_xyz(...); local XYZ feature helpers.
- [x] Write tests for prior/geometry/Pd/Pf,positive/negative updates,altitude confirmation gates,false positives,XYZ swept safety and real environment report creation/locality.
- [x] Run python -m pytest -q tests/test_xyz_sensing.py; observe missing-feature failures.
- [x] Implement validated opt-in XYZ config and Native math/world/sensor/observations without altering XY defaults.
- [x] Run Native tests plus existing environment/sensing/motion contracts; require green.
- [x] Commit working Native slice and record test evidence.

### Task 2: Dimension-aware algorithms, replay and checkpoint identity

**Files:** algorithms/models.py,action_utils.py,masac.py,matd3.py,replay.py,hybrid_replay.py,checkpoints.py,critic_chunks.py; experiment/environment/vector/replay/continuation callers; tests/test_xyz_algorithms.py.
**Interfaces:** models/replay optional continuous_dim=3; XYZ callers pass4; final power slot, all preceding slots motion; configured observation schema.
- [x] Write tests for both algorithms/sharing topologies:4-slot stochastic/deterministic/target actions,az gradients,power masking,real learner update,replay roundtrip and incompatible warm-start rejection.
- [x] Run python -m pytest -q tests/test_xyz_algorithms.py; observe shape/identity failures.
- [x] Generalize shape plumbing and entropy masks; leave learning equations unchanged; extend legacy defaults and checkpoint schema matching.
- [x] Run new algorithm tests and existing sharing/replay/checkpoint tests; require finite updates and unchanged XY behavior.
- [x] Audit scientific-source diff, update pinned hashes only for reviewed shape plumbing,commit and ledger.

### Task 3: Warp XYZ state, motion, sensing and observations

**Files:** environment/gpu/state.py,vector.py,world.py,geometry.py,validation.py,observations.py; create gpu/xyz_motion.py,gpu/bayesian.py,gpu/xyz_observations.py; tests/test_xyz_warp.py.
**Interfaces:** Warp stores XY and separate altitude/vertical velocity arrays as actual XYZ state; mission wrapper exposes Nx3 positions. Counter-based sensor draws match Native. Full belief/visited state checkpointed.
- [x] Write real Warp CPU tests for altitude control,sensing/confirmation,Native agreement,masked reset isolation,next-measurement exact resume,invalid restore atomicity.
- [x] Run python -m pytest -q tests/test_xyz_warp.py; observe missing state/kernel/shape failures.
- [x] Implement opt-in Warp kernels with device-local arrays; avoid host transport in the production step.
- [x] Run XYZ Warp plus existing Warp parity/snapshot tests; require green with honest CUDA skips.
- [x] Commit and ledger exact commands/results.

### Task 4: XYZ radio physics and end-to-end training

**Files:** communication/link_budget.py,backend.py,upstream.py,telemetry.py and gpu/phy.py,power.py,mac.py; experiment resource/identity/planner seams; tests/test_xyz_radio.py,tests/test_xyz_training.py.
**Interfaces:** endpoint XYZ and separate obstacle heights; circular camera LOS,polygon-prism RF LOS; actual3D range,admission/delay/SINR; Native/Warp same conventions.
- [x] Write real radio/mask tests above/below obstacles,3D out-of-range and learned RF last-slot; tiny BASE/PQL train/eval tests for both learners.
- [x] Run python -m pytest -q tests/test_xyz_radio.py tests/test_xyz_training.py; observe wrong-physics/shape failures.
- [x] Implement actual-height radio integration and all remaining pipeline dimension/metadata/resource handling.
- [x] Run end-to-end tests and related communication/continuation/best-checkpoint regressions; require finite updates.
- [x] Commit and ledger.

### Task 5: Profiles,2D rendering,evidence and final verification

**Files:** create three configs/*-xyz-sensing.yaml,docs/xyz-sensing.md; modify experiments/rendering.py,docs/CURRENT_TASK.md; tests/test_xyz_rendering.py,test_project_structure.py.
**Interfaces:** frame-specific altitude/FOV/local belief; XY-projected paths/hops; selected-UAV heatmap; science/config names identify XYZ lineage.
- [x] Write rendering/profile tests for correct frame data,2D axes,FOV radii,local unknown0.5 and unchanged baseline values.
- [x] Run python -m pytest -q tests/test_xyz_rendering.py; observe missing presentation/profile failures.
- [x] Implement requested profiles/rendering/docs; compute precise memory estimates; generate and inspect a real PNG/short animation.
- [x] Run all XYZ tests,then full pytest and required checks; inspect failures,fix root causes and rerun affected checks.
- [x] Commit,request one independent review against spec/plan/rulings,fix Important/Critical in one RED→GREEN pass and rerun whole suite.
- [x] Record honest CPU/CUDA/training limits,push working branch,verify remote SHA,preserve original CURRENT_TASK dirty suffix,remove only this plan's scratch workspace.
