# Independent whole-change review

Review date: 2026-10-26. Base `8f8e3fe`; reviewed head `9faf67add3b711ed549c6d4ffcf0c3994ea7fb6f`.

Inputs: `review-package.diff`, the robust-training design and implementation plan, `progress.md`, changed production/test files, and their checkpoint, benchmark, continuation, telemetry, and resource-estimation consumers. Review was read-only apart from this report. No additional suite, cloud job, source edit, Git mutation, or credential inspection was performed.

## Verdicts

**Spec verdict: partial compliance; fix the benchmark identity path before declaring Task 3 complete.** The certificate-backed generator, explicit-scene bypass to strict validation, generation-3 rejection guards, device-resident bank, and checkpointed destination enumeration implement the intended architecture. Production numerical budgets and scientific configuration were preserved in the reviewed diff. Full CPU/seed-schedule completion and real CUDA proof remain outstanding evidence rather than established failures.

**Quality verdict: request changes for the Important integration issues below.** The implementation is small and localized, and its optimizer/loss boundaries remain intact. There is also one Minor accounting regression. Counts: Critical 0, Important 2, Minor 1.

## Critical

None found.

## Important

### I1 — Actual PQL benchmarks bypass the shared identity and inherit the training stage

Locations: `scripts/benchmark_pql.py:45`; `uav_network/cloud/remote.py:172`; integration with changed `uav_network/experiments/tracking.py:140` and `uav_network/cloud/cli.py:84`.

The new names are placed in the launch manifest, but the actual mandatory PQL preflight still sets `case.tracking.run_name` to `MASAC-PQL-benchmark-E4096-B8192-seed44`. `OnlineTracker` invokes the shared helper only when the configured name is `auto`, so this explicit value wins. FULLGPU and hybrid benchmarks therefore still have identical display names and omit replay capacity, generator version, and a run-unique suffix. Standalone PQL benchmark launches also discard the meaningful manifest name because the benchmark subprocess receives only its run ID.

For mandatory preflight under a training launch, `pql_benchmarks` copies the parent's environment without changing `UAV_STAGE`. The parent kernel sets it to `train`, and `OnlineTracker` records that value at `tracking.py:156`. Thus the benchmark is still classified as training in W&B metadata even though its old display name says benchmark. This directly misses the review focus on separating train/controller/benchmark and recognizing the actual replay layout.

Concrete fix: construct each benchmark case's display name with the shared identity helper after applying its algorithm/environment/batch overrides, give preflight its own unique suffix/ID, and explicitly set `UAV_STAGE=benchmark` in the benchmark subprocess. Preserve the reserved training identity. For standalone benchmark launches, either pass the manifest benchmark name through or generate the same complete identity at the final tracker boundary. Add a focused test that traverses the benchmark subprocess/entry-point path and captures the actual tracker name and stage for FULLGPU and hybrid cases; checking only `cli.launch`'s manifest is insufficient.

### I2 — Low-level state allocation now imposes automatic-generation feasibility on manual-world callers

Location: `uav_network/environment/gpu/state.py:341`; production caller `uav_network/environment/gpu/vector.py:47`.

The certificate call is unconditional in the raw `allocate_state` utility. That utility is also used to allocate physics worlds whose geometry is injected by the caller, including boundary fixtures and deliberate impossible-geometry/error-code probes. These callers can now fail before they install their explicit geometry or launch the intended error-producing kernel. The implementing agent reported that the already-running full CPU suite exposed three boundary subtest failures and an impossible-geometry kernel-test failure on this seam. This report does not claim an independent rerun of those failures; the unconditional call and low-level usages establish the integration cause.

Concrete fix: distinguish raw/manual state allocation from automatic environment startup. Require certification explicitly in `WarpVectorEnv` construction, while allowing the low-level allocator to return a raw state for manual geometry and kernel error-code tests. A raw state without a bank must expose `fallback_count=0`, valid empty/sentinel array fields as needed, and preserve explicit placement errors; it must never be the path used by production automatic collection. Keep the regression asserting that an uncertifiable production `WarpVectorEnv` fails during construction. This matches the implementing agent's proposed seam fix and does not justify weakening production certification.

## Minor

### M1 — The shared certificate bank is multiplied once per environment by the memory estimator

Locations: introduced shared allocation at `uav_network/environment/gpu/state.py:353`; affected consumer `uav_network/experiments/state_resources.py:9` and `:13`.

`allocate_state` now creates `fallback_positions`, `fallback_targets`, and `fallback_obstacles` once for the whole vector. The resource estimator still allocates one world, calls all of its bytes `bytes_per_env`, and multiplies the entire allocation by the requested environment count. It therefore counts the fixed certificate bank 4,096 times for the production case. With four primary scenes the nominal bank is 5,024 bytes, so the estimate overstates storage by roughly 19.6 MiB at 4,096 environments. Actual allocation and `WarpVectorEnv.allocated_bytes` are unaffected; this is an estimate/reporting error rather than an OOM cause.

Concrete fix: separate fixed shared bytes from per-world bytes and return `shared_bytes + per_env_bytes * count`. Update the resource-estimator regression to check the affine relationship and compare with two small actual allocations, rather than asserting that all state is strictly proportional to environment count.

## Reviewed invariants and evidence

- Recovery first certifies complete scenes using a reset-independent bounded search. Automatic native generation retries whole scenes and copies a certified scene on exhaustion; Warp uses the same bank and an equivalent single uniform draw for selection. Failure to certify rejects startup rather than weakening geometry.
- Explicit native scenarios go directly to `_sample_scenario` and existing validation; their errors are not caught as automatic recovery. Random placement exceptions are a distinct subtype. The primary profile retains its geometry, 20,000 episodes, 4,096 environments, 8,192 batch, 1M replay, 1 KiB reports, 1,800 steps, and seed 44.
- The GPU reset mask returns before modifying an unselected row. Recovery initializes transient geometry state through `_place_geometry`, clears successful selected-row errors, and retains per-row fallback/attempt indicators. No host scene copy occurs in the reset path.
- Generic Warp snapshots include all new arrays, including certificates and generation indicators. Restore verifies the configuration, field set, shape, dtype, and action-generator state before copies. Existing policy, evaluation, and training checkpoint checks derive the accepted generation from the default, and added tests exercise generation-2 rejection.
- The chunk helper encloses only deterministic destination-enumerating critic work. Sampling remains outside; losses are still normalized over the complete logical batch, and optimizer stepping/clipping stays at the original update boundary. Non-reentrant checkpointing avoids retaining each chunk's internal activation graph. The model paths inspected contain no stochastic critic layer requiring RNG preservation inside that helper.
- Continuation preserves its root run name. Controller and CPU-evaluation names reuse the parent identity with distinct stage prefixes. The fresh ordinary launch path removes stale CPU75/GPU25 replay tags and supplies its actual layout; I1 is the missed production benchmark path.
- Changed text files checked are at most 500 lines; the largest is `cloud/controller.py` at 495.
- Reported focused results supplied by the implementing agent/ledger: scenario 35 passed with 4 CUDA skips; critic 39 passed with 376 subtests; cloud 90 passed with 31 subtests. I inspected relevant tests but did not rerun these suites or treat supplied counts as independent execution evidence.

## Evidence still needed before launch/acceptance

- Resolve the CPU-suite allocation-seam failures noted in I2, then finish and inspect the full CPU suite and frozen-config 74,126-seed scan. Do not count the interrupted earlier scan as completed coverage.
- Run real CUDA recovery/masked-reset cases, including forced fallback. Compare native and Warp final PCG state explicitly after random success, retry, and fallback; geometry equality alone is weaker evidence when several RNG outcomes can select the same certificate.
- Complete production-shaped 2-T4 BASE/PQL memory, updates, and resume checks. CPU retained-storage tests and source inspection do not establish actual T4 headroom or CUDA numerical equivalence.
- Existing exact-world tests inspect full snapshot tensors; ensure the hardware evidence includes the added fallback arrays/counters and a forced-recovery scenario. The new production resume smoke check establishes continued progress, not by itself equality with an uninterrupted oracle.

## Declined to judge

- Real 2-T4 fit, CUDA kernel behavior, peak VRAM, or completed CUDA exact-resume equivalence: proof was pending, and this reviewer was instructed not to launch jobs.
- Final full-suite and 74,126-seed scan outcomes: both were still running when review was requested. The implementing agent subsequently reported the failures documented in I2, but no final post-fix suite or completed scan was supplied for independent inspection.
- Current account availability, lineage reconciliation, remote W&B progress, and correctness of future submissions: no live cloud reconciliation was performed in this read-only review.
