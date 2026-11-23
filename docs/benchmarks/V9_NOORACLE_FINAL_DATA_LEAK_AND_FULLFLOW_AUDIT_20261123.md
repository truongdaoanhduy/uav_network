# NoOracle Adaptive Altitude — final data-leak and full-flow audit

**Date:** 2026-11-23  | **Review basis:** `b37e5361adf1649dbb9210c93fe50d1ef4fab9b0`  | **Fix branch:** `audit/nooracle-adaptive-altitude-final-20261123`

## Decision

**CPU scientific/privacy gate passed for the audited scenarios, subject to listed limitations. CUDA T4x2 / E4096 / long-horizon learned policy and production W&B/Hugging Face remain unverified.** Never use this audit as evidence of complete 20,000-episode success or GPU performance. This is a NEW, science-changing NoOracle **observation version 10**. Do not resume v8 historical or v9 pre-audit Actors into v10.

## Verified scenario pipeline

1. **Generation and motion:** 6 UAVs, 35 hidden targets, 25 finite-height obstacles, 1000m square, 0–150m altitude, deterministic seed and 1500-step episode horizon. The policy chooses horizontal and vertical motion, plus radio destination/power. Collision and energy logic stays in Warp.
2. **Camera and belief:** FoV radius follows altitude with 90-degree full angle, Pd decreasing and Pf increasing with altitude. Only positive/negative camera samples update local Bayesian belief. Candidate confirmation requires positive sensor sample AND posterior≥0.99 AND marginal positive likelihood ratio≥4; no fixed ≤50m confirmation restriction in adaptive mode. The Actor sees altitude, vertical speed, local positive evidence and belief; it receives no true target IDs or coordinates.
3. **Obstacle visibility:** a new *conservative line-of-sight approximation* checks other cylinders before revealing obstacle position/radius/height. Only 3 nearest visible obstacles are included; when camera is off at z=0, no obstacle geometry is exposed. This is a modeled local sensing rule, NOT a real camera calibration or a proof of perfect perception.
4. **NoOracle REPORT:** a confirmed cell ID is reported, not the simulator's hidden true target index; report slots are an internal allocation detail. Report forwarding through UavNetSim-style Warp PHY/MAC and GCS delivery is asynchronous with battery and packet constraints. True target coordinates are used only for sensor outcome sampling and post-transition reward/scoring; using truth in those simulator internals does NOT mean the Actor can observe it.
5. **Learning:** decentralized Actor observation; centralized training state in NoOracle mode uses observation-derived cell sketches and zeroes true target positions/3D obstacle oracle fields. Replay/PQL and MATD3 training operate on the issued observations and masks. The old native XYZ reference could send true IDs and is now explicitly rejected for NoOracle.
6. **Evaluation:** CPU WarpMissionEnv aligns internal 140 report hypotheses to 35 true targets in host-side metrics and rendering, not during policy inference. Validation seed 1,000,000,044 and final evaluation seed 2,000,000,044 are disjoint from training seed44. Checkpoint source schema verifies v10; metadata now describes the posterior PLUS likelihood-ratio gate, and report animation labels use `Cell 820` rather than incorrectly `T820` for observed cell IDs.

## Confirmed bugs reproduced RED, fixed GREEN

| Bug | Proof before fix | Correction and scope |
|---|---|---|
| **Hidden-obstacle ground-truth geometry in Actor** | Modifying a fully occluded obstacle changed Actor observation columns100/102/159 even though visibility was blocked | `camera_sees_obstacle` LOS; stable nearest-3 visible obstacle ranking and paired height; adaptive NoOracle only; historical policies preserved |
| **Camera-off-at-zero-altitude leaked obstacle** | At z=0 Actor still received nearby obstacle radius, XY and valid flag | Strict adaptive observation requires z>0 for camera geometry |
| **Native NoOracle silently used true target ID** | Direct `SwarmSearchEnv(c)` did not reject `report_cell_hypotheses=true`; Native `ReportStore.discover(i,t)` used true target index | `SwarmSearchEnv` rejects direct NoOracle; `validate_xyz` forbids native backend for NoOracle |
| **Incorrect confirmation provenance** | `world_info()` said `posterior_only_no_truth_filter` for likelihood-ratio mode | New mode publishes `posterior_plus_likelihood_ratio_no_truth_filter`; historical mode metadata unchanged |
| **Misleading radio event/render labels** | NoOracle trace containing cell ID820 was rendered as `T820` | Host trace includes `report_identity=observed_cell_id`; renderer labels `Cell 820`; historical target IDs retain T label |

**No direct hidden target coordinate leak in v10**: a test modified true target XY without running sensing and verified Actor and centralized state were bitwise unchanged. Existing tests also verify GCS hidden obstruction does not leak via exact nominal RF power, another UAV's private state cannot alter Actor’s mask, and global report-slot permutations do not encode private sensing order. No code in Actor/radio admission imports the CPU evaluation truth projection helper.

## Runtime evidence, not learning performance

**16 seeded full-horizon real Warp CPU worlds** (`seeds400..415` ×1500 steps, 24,000 world steps) completed without internal errors. Observed max altitude150m, max allocated hypotheses50/140, total true found486/560, true GCS delivered293/560, false-confirmed cell hypotheses169. Complete per-seed JSON was saved in local Downloads: `V9_ADAPTIVE_NOORACLE_16WORLD_FULL_HORIZON_STRESS_20261123.json`.

**Important:** These are hand-crafted/random stress actions with altitude targeting. They are NOT metrics for a trained MATD3 Actor or scientific performance comparison. A smaller two-world long-horizon stress at seeds44 and45 also completed.

**Real training/evaluation path tested on CPU:** short MATD3 training with model updates → checkpoint save/load (v10) → held-out CPU evaluation/metrics → figure generation, plus extraction of the pinned Kaggle notebook source archive followed by an actual bounded CPU train from the extracted files. No complete E4096 CUDA training/validation is claimed.

**Code/artefact integrity:** The notebook builder embeds 547 source files including197 Python modules; checked all547 matched local source byte-for-byte, and extracted one actual notebook archive and ran a CPU training worker from it. A heuristic scan found zero credential-like filenames and zero strings matching selected standard HuggingFace/OpenAI/W&B/AWS token formats. This is not a security guarantee against every possible secret format.

**Test suite evidence:** Post-fix group A:95 passing/3 skipped (NoOracle, privacy, source-bundling, checkpoint schemas). Group B:71 passing/7 warnings (XYZ training/resume, CPU evaluation, W&B/HF handoff, async evaluation). Subsequent focused tests covered provenance and trace label fixes. Combined **166 passing, 3 skipped**, 7 warnings. These are *selected regression tests*, not the full 1,791 collected project cases.

## Outstanding scientific/engineering risks (not silently patched)

1. **CUDA/E4096:** This Ubuntu machine cannot execute the T4x2-only pilot; source packaging, VRAM pressure, Warp JIT on CUDA, GPU throughput and exact 4096-env behavior remain unproven. The exported Kaggle notebook has a hard64-vector-step limit and cannot validate full-episode behavior or convergence.
2. **REPORT capacity:** 140 slot limit can fail closed in a policy that confirms too many distinct cells; highest observed50/140 across16 stress worlds, insufficient to prove worst case. Test adversarial corner/false-positive-heavy runs before full training.
3. **Sensor/Belief calibration:** The configured cell prior is0.5 whereas target occupancy density is35/1600≈2.19% in the uniform 40×40 cell layout. A prior of0 would be irreversible under exact Bayes without a floor. Whether to recalibrate is a **scientific experiment decision**, not a pure software bug fix. `Pd/Pf`, partial-cell area `q`, LR4 and false-positive rates should undergo calibration/ablation with outcome uncertainty.
4. **No perfect camera claim:** The LOS blocker is a conservative center/near-face ray approximation and visible obstacle geometry is still observed perfectly when selected. It has no learned real-world camera uncertainty, perspective occlusion mask or perception-noise model. Such realism needs separate scientific specification; NoOracle specifically excludes hidden target identities, not every sensing idealization.
5. **Research validation:** No full20,000 episodes, GPU checkpoint validation, independent training-seed runs, held-out Final Test1000, W&B/HF archival upload or production convergence was launched during this audit. Historical v8 actors are not scientifically interchangeable with this v10 MDP.

## Reproducibility

Use `PYTHONPATH=. /home/aduy/Documents/NCKH/uav_network/.venv/bin/python -m pytest -q` on the test modules shown in repository history for reproducible focused checks. New regression modules:

- `tests/test_nooracle_adaptive_privacy_final_audit_20261123.py`
- `tests/test_nooracle_trace_identity_final_audit_20261123.py`
- `tests/test_nooracle_adaptive_altitude_20261123.py`
- `tests/test_nooracle_adaptive_gpu_notebook_20261123.py`
- `scripts/audit_adaptive_nooracle_16world_stress_20261123.py`

**Model/scenario config:** `configs/matd3-nooracle-adaptive-altitude-lr4-h256x512-20261123.yaml`.
**Kaggle CPU-proven source bundle / bounded CUDA pilot:** `notebooks/MATD3-NOORACLE-ADAPTIVE-ALTITUDE-LR4-H256x512-E4096-U4-B8192-T4X2-BOUNDED-PILOT-20261123.ipynb`. Its GPU path remains to be tested by Kaggle, not on Ubuntu.
