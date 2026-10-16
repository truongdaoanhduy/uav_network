# Boundary escape and communication visualization

> Implementation plan for the user's requested video diagnosis and fixes.

Goal: prevent the verified stationary boundary fallback and make completed report hops to peers/GCS visible without changing packet or learning semantics.

Architecture: retain the target-blind dynamic-window safety certificate and change only its boundary fallback direction. Add opt-in whole-report hop telemetry to native and Warp single-world evaluation. Render those timestamped events from their actual positions and keep an exportable JSON ledger.

Constraints: preserve original checkpoints, original 30-seed metrics, and original videos; no training restart; no new dependencies; no host synchronization added to vectorized training. Report power as last received fragment TX power, never as the power/energy of an entire multi-hop route. Re-evaluation under the repaired environment must be labelled separately from the original benchmark.

Review focus: constant outward corner/normal-edge commands; acceleration and obstacle/peer certificates; unsuccessful or incomplete packets must not be labelled delivered; events between sampled video frames; trace enabled/disabled numerical equivalence.

## Task 1 — boundary fallback

- [x] Reproduce sustained 1500-step stationary edge/corner failure in native and Warp CPU.
- [x] Add regression assertions for escape, per-axis acceleration limits, and backend parity.
- [x] Implement the smallest target-blind fallback that maintains progress while remaining certified.
- [x] Run safety regressions; retain original safe interior command identity.

## Task 2 — successful hop telemetry

- [x] Add red/green tests for whole-report custody transfer, GCS identity, opt-in/reset/drain semantics, actual event coordinates, power, and trace parity.
- [x] Expose enable_communication_trace() and drain_communication_events() on both single-world environments.
- [x] Record only successful whole-report transfers after fragment reassembly; retain each event until drained, including between sampled frames.

## Task 3 — renderer and evaluation

- [x] Add red/green tests that frame capture drains all events and arrows distinguish relay/GCS with power labels.
- [x] Add communication events to animation_frame, with static trajectories retaining their existing meaning.
- [x] Enable traces for animated evaluation, export communication_events.json, and draw the latest sampled time window with explicit simulation timestamps.
- [x] Include pending reports, dead UAVs and flight/RF energy if exposed without costly instrumentation.

## Task 4 — validation and delivery

- [x] Run full CPU test suite with isolated pinned Warp and report unavailable CUDA checks.
- [x] Re-evaluate the same final checkpoint/last original evaluation seed for four new visualizations; compare against preserved originals separately.
- [x] Review the complete diff, then commit/sync the authorized fixes.
- [x] Provide accessible videos, event ledgers and a Vietnamese explanation of the original metrics and model limits.

## Verification record

Final whole suite: 394 tests, 347 passed, 47 skipped because CUDA is unavailable; Warp1.18.0 CPU supplied as isolated optional dependency. Native/Warp motion matched exactly in8 scenarios×1500steps. Reviewer found native NumPy target IDs were not JSON serializable and a zero-size trace array had a null pointer in memory accounting; both were reproduced RED, fixed, and covered in the green whole suite. A scale-aware label regression covers50/100/1000m maps. Diagnostic warnings remain tested in a real closed pocket. No training job restarted. Replayed media and publication evidence live under outputs/audit-boundary-20261016 and remain separate from original30-seed results.

Limitation: boundary escape can still travel boundary strips; this is not learned interior search or proof of energy optimality. Completed GCS report receipts can include duplicate targets. Power labels represent the last fragment only. CUDA execution has not been validated on this host.
