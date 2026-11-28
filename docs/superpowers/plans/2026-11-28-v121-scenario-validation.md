# V121 scenario validation implementation plan

> For agentic workers: use the execution and debugging workflow task by task. Every scientific change requires observed RED, GREEN and regression evidence. Independent DS and network reviews may run concurrently; GPU runs remain sequential.

**Goal:** audit and repair the DS-only search/network scenario before building or tuning MATD3/MASAC baselines, with a reproducible full-source ZIP.

**Architecture:** preserve the V120 handoff and its evidence; add a separately named V121 audit layer and explicit model-scope contract. Repair only reproduced scenario or validation defects. Keep learner implementations byte-identical.

**Tech stack:** Python, NumPy, YAML, unittest/pytest, NVIDIA Warp 1.18.0, Vast RTX 3060.

**Spec:** the user's current request, the attached V119 context and V120 source; when these disagree, inspect source and raw run evidence and record the decision below.

## Global constraints

- Canonical: 1000m square, 6 UAVs, 15 static targets, 20x20 grid, altitude50m, horizon900, dt1s, TX0.1W fixed, pure-GCS application communication.
- DS-only: raw provenance counters, no Bayesian runtime. Keep historical evidence clearly labelled.
- No learner, reward-weight or canonical YAML tuning to improve scores.
- Do not train or push a Git branch during the scenario audit.
- Keep code files under500 lines; preserve full failed and successful run logs and checksums.
- Freeze is an evidence decision. Unsupported claims are corrected, not converted into PASS by renaming.

## Review focus

- A reset-based stress test must not certify a single uninterrupted900-step mission.
- CPU commands are velocity in m/s; normalized Warp/actor commands need explicit conversion.
- A straight-line route ignoring obstacles/acceleration/sensing/network is not a mission-feasibility proof.
- Correlated evidence can increase false confirmations despite correct DS arithmetic and packet deduplication.
- Pure-GCS still includes HELLO, routing and multi-hop forwarding; zero app send is not radio-off.

### Task1: establish provenance and audit V120 claims

- [ ] Preserve original source tree/hash, algorithm and two canonical YAML hashes.
- [ ] Read all72 conflict rows; match claims to code, actual steps, units and logs.
- [ ] Retrieve primary paper, upstream and NVIDIA documentation; retain access failures and claim limits.
- [ ] Record root causes and the distinction between bugs, design choices and research limitations.

### Task2: close residual CPU Bayesian runtime access

Files: environment configuration/DS dispatch, tests/test_v121_ds_only_runtime.py, affected legacy integration tests.

- [ ] RED: direct configuration and CLI override must reject Bayesian inference.
- [ ] Remove runtime Bayes dispatch and stale descriptions while preserving unrelated source and historical standalone math evidence.
- [ ] GREEN: direct and loader rejection plus ordinary DS mass update/observation contracts.
- [ ] Regression: full test suite on the verified Vast runtime.

### Task3: correct calibration action units and scientific bounds

Files: scripts/maintenance/v120_*calibration.py, tests/test_v121_calibration_contracts.py.

- [ ] RED: patrol velocity command norm must equal configured cruise15m/s away from goal.
- [ ] Convert normalized patrol XY to CPU m/s before env.step; log units, actual steps and early termination.
- [ ] Relabel the idealized straight-line route calculation; assert it does not certify physical or network feasibility.
- [ ] GREEN and regression; rerun paired seeded DS and RF calibration for900 requested steps, without resetting failures or suppressing early terminations.

### Task4: validate true single-episode CUDA endurance

Files: scripts/gpu/v121_uninterrupted_cuda_gate.py and tests/test_v121_horizon_gate_contracts.py.

- [ ] RED: gate must reject reset/early-terminal/unfinished-world evidence and accept complete pinned non-reset evidence only.
- [ ] Add on-device all-step counters, first-terminal step, physical/state/reward/energy monitors and machine-readable result.
- [ ] Run4096x900 on base and isolated pure-GCS RF noise; do not call auto_reset_gpu.
- [ ] Retain per-world counter summaries, finite/overflow/safety outcomes, source/YAML hashes and actual VRAM/time.

### Task5: audit multi-step shield and network boundaries

- [ ] Run CPU/Warp physical shield differential with identical pre-step states and shared normalized action conversion, including actual intervention.
- [ ] Retain counterexamples, numerical limits and distinction from whole-world stochastic/event parity.
- [ ] Reproduce any network custody/TTL/ACK defect before changes; apply only source-justified minimal repairs with RED/GREEN evidence.
- [ ] Keep upstream UavNetSim event-equivalence unclaimed unless differential trace proves it.

### Task6: evaluate conflicts and package verified deliverable

- [ ] Update allC01-C72 rows and add every new issue, with severity, current evidence, precise closure condition and paper/code provenance.
- [ ] Inventory map/sensing/DS/reward/actions/PHY/MAC/routing/energy/Warp; identify inactive, unsupported or unjustified components.
- [ ] Write a fail-closed freeze decision; keep Phase2 deferred if any in-scope blocker remains.
- [ ] Fresh review, full regression, compile/policy/checksum validation; record all failures and remaining limits.
- [ ] ZIP full source, diff, historical and current raw logs, primary-source evidence, report, conflict matrix and runbook; save and provide the file.
