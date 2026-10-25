# Local report sensing implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep report creation and recovery independent of unannounced remote GCS delivery.

**Architecture:** Camera sightings update local knowledge and finite local custody. Global delivery is used by the GCS to deduplicate delivery rewards and by centralized training/metrics, never to suppress a distant UAV's report. Native and Warp implementations retain identical observation layouts and radio/TTL behavior.

**Tech Stack:** Python, NumPy, PyTorch, Warp 1.18, pytest.

**Spec:** Latest user audit/fix request, operating constraints in `docs/superpowers/CURRENT_TASK.md`.

## Global constraints

- Work sequentially in the existing linked worktree, branch `feat/cpu-replay-20261022`.
- Primary report/payload1024 B, buffer3072 B; env4096, batch8192, replay1M, CPU75%, horizon1800, episodes20000, seed44 remain pinned.
- Preserve existing completed results and source identities; any changed observation semantics require fresh episode-zero training.
- Preserve policy-selected next-hop, PA/TX/RX energy, PHY/MAC/retry and terminal mission semantics.
- Keep global critic inputs and unique GCS delivery rewards.

## Review focus

- Fresh sightings after remote delivery must create the same local report.
- Recovery after local expiry must be independent of remote delivery.
- Full buffers must retain local recovery status regardless of remote delivery.
- Multiple sources delivering one target must yield one unique delivery.
- Warp CPU and CUDA must follow the same local sensing behavior.

## Task 1: local sensing invariance

**Files:** Modify `uav_network/communication/reports.py`, `uav_network/communication/gpu/reports.py`; create `tests/test_decentralized_observations.py`.

**Interfaces:** Keep `ReportStore.discover(uav_id, target_id, time_s)` and `sense_reports_at(s, p, e)` signatures unchanged. Actors consume the existing `local_observations()` / Warp `observe` layout.

- [x] Run clean baseline suite and record exit/counts.
- [x] Add real native/Warp tests comparing observations and custody when only hidden `delivered[target]` changes, for fresh/recovery/full-buffer sightings; assert report presence/TTL with hand-derived expected values. Add duplicate GCS delivery assertion.
- [x] Run new tests and record expected failures on the existing global suppression branches.
- [x] Remove global delivery checks from local sighting branches in native and Warp code; retain local capacity checks and unique GCS delivery logic.
- [x] Run focused sensing/report/observation/native-Warp parity tests, then full pytest suite. Run the CUDA tests on Kaggle T4 if available; report CPU-only skips accurately.
- [x] Document evidence and source-lineage consequence, update CURRENT_TASK, commit and push after verification.

## Follow-up audit

After Task1 verification, review actors/critics/replay/continuation/evaluation/cloud publication sequentially. Reproduce each confirmed defect before changing its owning function. Preserve a file/module inventory and list audit limits instead of claiming exhaustive proof of correctness.

## Task 2: checkpoint observation compatibility

**Files:** `environment/observations.py` owns observation version3; `algorithms/checkpoints.py` saves and validates it; `experiments/evaluation_snapshots.py`, `fixed_checkpoints.py`, `pql_checkpoints.py` reuse the validator; `comparison_metrics.py` includes it in scenario hashes; `tracking.py` publishes it in future run configs. Paths are under `uav_network/`.

**Interfaces:** Add `validate_observation_version(saved)` and constant `OBSERVATION_VERSION=3`. Missing or older metadata is rejected before accepting a policy, evaluation snapshot, or resume checkpoint. Historical policies remain usable with their pinned historical source.

- [x] Add real policy/snapshot/full BASE and PQL resume tests with missing/old version but identical dimensions; watch all fail for lack of rejection.
- [x] Save version3 in both policy formats, validate through all loading/preflight paths, and include it in scenario provenance.
- [x] Run focused checkpoint/resume/evaluation tests, then full pytest. Existing exact-resume tests must still pass.
- [x] Record evidence, update current-task status, commit and push.

## Task 3: final review corrections to CPU reporting

**Files:** `experiments/tracking.py`, `experiments/dashboard.py`, `cloud/visualize.py`; tests `test_diagnostics_axes.py`, `test_visualization_tracking.py`, `test_visualization_provenance.py`.

**Interfaces:** Reuse `start_tracking(config, output, initial_progress)` to mark source observation version unknown until a checkpoint/protocol is loaded. Keep aggregate validation checkpoint-axis behavior. Define the five CPU episode metric names once, use the episode-count axis in SDK definitions and authored panels, and publish actual checkpoint step plus saved observation version as evaluation provenance.

- [x] Review source and reproduce CPU axis defect:3failed4passed; historical publication provenance:3failed.
- [x] Add exact CPU metric overrides and matching panel mapping; preserve periodic aggregate axes.
- [x] Keep CPU initial source version unknown; derive result version from saved protocol, not the recovery runtime; log true checkpoint step.
- [x] Focused real CPU evaluation/publication resume tests and full pytest must pass.
- [x] Record independent review boundaries/rulings, evidence, and final commit; update historical W&B source labels without changing measured outcomes.

Ruling: Treat historical publication provenance as an Important correction, because labeling older observations as current can contaminate scientific comparisons; retaining current labeling would risk mixing incompatible results. Preserve historical media publication without policy re-evaluation.
