# General failure handling for supervised UAV training

The user requests a general response to MASAC failed/crashed jobs, beyond the historical 11-second incident. The exact platform-side cause remains unproven; the four old runs have no recoverable replay/optimizer/RNG. New code must handle failure boundaries safely without claiming every external failure can be prevented.

## Approach

Reuse GitHub reconciliation, immutable full-state snapshots and existing diagnostics. Do not add another controller or silently change scientific settings. Each account/job is isolated; cloud observations distinguish active, waiting, blocked, cancelled and confirmed submission. Recovery requires monotonic exact state and is bounded to three consecutive crash recoveries; a successful normal segment resets the counter.

## Confirmed gaps to fix

1. An exception listing one account escapes main and prevents later accounts from being checked.
2. Blocked/missing recovery and unknown kernel states can leave the workflow green. Preserve a sanitized JSON report even on failure.
3. Verified recovery result omits initial_global_step/environment_state_saved required by next_stage, breaking continuation-prefix and zero-episode recovery. Save/publish the actual environment-state flag for BASE/PQL; legacy absent evidence must remain blocked for zero-episode recovery.
4. Recovery submission reports submitted even when the submit helper returns waiting_for_quota.
5. A cancelled GPU/CPU job can be restarted automatically; respect explicit cancellation.
6. Consecutive crashes have no explicit retry budget. Store/reset a three-recovery counter in child specs and enforce it at the shared builder and supervisor.

## Boundaries

No production launch/restart/cancel, no history changes, no cross-account lineage moves, no raw error messages/secrets in reports. Keep pinned source, 20k/1800/E4096/B8192/replay1M/precision policy/eval protocol intact. Missing credentials in an explicitly enabled deployment remain a configuration error. Invalid account APIs do not block healthy accounts. Read-only live credential/quota checks do not certify GPU availability.

Automatic recovery does not repair deterministic learner bugs, revoked access or missing/corrupt full state; these must be visible and bounded. Network/quota failures defer transitions without creating duplicate children or consuming a submitted recovery attempt. Recovery never invents lost replay.

## Verification

Observed RED then GREEN for each gap, actual small BASE/PQL saved-world metadata, focused lifecycle regressions, full offline CPU suite, independent review, static/secret checks and hosted manual dry-run. No production training is used as a deployment test.
