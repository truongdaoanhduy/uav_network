# Resolution added by implementer after the independent review

The sole Important finding below is FIXED in 0fe0e7c. Both periodic/final failed-wait alias cases were reproduced RED, then passed GREEN after deferred persisted alias promotion. Focused combined39tests and final full CPU834passed; actual2T4 diagnostic passed. No other finding is open. This is a parent resolution record, not a claim that the reviewer reran tests. See crash-flow-recovery.md for every limitation and measured GPU/network boundary.

---

# Independent whole-change review — crash visibility and recovery

Reviewed once against BASE b01aa1c, HEAD 8cfc5da plus staged implementation, in /home/aduy/Documents/NCKH/uav_network-public-hf-20261022. Read AGENTS.md, CURRENT_TASK.md, the design, implementation plan, progress ledger, the complete changed source/test surface, and relevant checkpoint serialization, replay, PQL drain/restore and dashboard context. No source, index, HEAD, cloud state or credentials changed. This report is the sole written artifact. No tests or live-service calls were run by this reviewer.

## Findings

### Important — resumable alias can advance before publication confirmation

Location: uav_network/experiments/tracking.py:297-300 (especially alias append at 298 and log_artifact at 299).

Trigger: a previously confirmed resumable artifact exists; the next backup uploads and verifies HF bytes, then calls log_artifact with aliases including resumable. W&B accepts the asynchronous publication but logged.wait times out or otherwise fails before the caller receives confirmation.

Consequence: the asynchronous publication already carries permission to move the resumable alias. It can complete server-side and move that alias even though this attempt is recorded as failed and recovery/checkpoint_artifact remains at the preceding confirmed artifact. Thus the independent recovery selectors disagree and the last-confirmed-alias guarantee in the design is not enforced. This does not imply the verified HF bytes themselves are corrupt; the defect is the publication/confirmation boundary and its promised failure semantics.

Suggested fix: initially log the artifact without resumable, wait successfully, then promote the resumable alias through the SDK's persisted alias update and publish the summary pointer in a deliberate order. Preserve the previous alias when the initial wait fails. Extend the failure test with an alias-aware fake whose log_artifact applies the supplied aliases before wait raises, and assert the alias as well as the summary pointer remains at the previous artifact. Include final full-state publication, which takes the same path.

The existing test test_resumable_publication_bypasses_light_throttle_and_keeps_separate_pointer checks only summary preservation after the failed wait; its fake does not track remote alias state, so it cannot catch this gap.

Critical: none found.
Other Important: none found.
Minor: none separately actionable.

## Review-focus assessment

- Failed HF/W&B publication: summary recovery pointer updates only after wait and remains unchanged on that failure; resumable alias has the finding above. Existing checkpoint_storage revision is upload metadata, not the separate confirmed recovery pointer.
- PQL consistency: the full callback drains workers before environment/model serialization; engine.save also drains. Periodic invocation passes publish=False and suppresses its internal light publication. Calls occur after collection/reset boundaries and before the next submission.
- Disk staging: one unique owned TemporaryDirectory is used synchronously per attempt and removed before the next attempt or final save. Policy and resume .tmp files remain under that owned directory. The implementation does not accumulate periodic full states or overwrite the final resume_latest checkpoint. Existing cloud-owned restored-input release occurs before the coordinator starts.
- Timing/caps: attempts increment before save/upload, including failures; clock is checked on every non-final collection, independent of completed episodes. Generic opt-in and HF/legacy backend guards are present. Effective interval is max(requested, soft budget / cap); next due is measured after attempt completion, so this is an upper bound on attempts rather than a promise to produce four checkpoints.
- Resume semantics: BASE retains pending_learning and RNG in the existing full-state serializer. PQL uses existing drained serializer and separate collector RNG. New integration tests compare continued small synchronous BASE/PQL runs after a before-first-episode snapshot for both algorithms.
- Workspace behavior: shared reconciliation changes runset sort/visibility/columns and does not alter panel configuration or run states. Existing personal reconciliation preserves chart objects. Explicit saved-view live updates and chart hashes are recorded as verified by the parent; those remote facts were not independently re-fetched here.

## Declined to judge

1. Exact external cause of the historical near-simultaneous G3 heartbeat losses: no kernel/platform evidence in this review establishes it. Authentication failure now is not proof of the original termination mechanism.
2. Fresh full-suite outcome: the supplied initial suite predates a config fix and is not final proof. Parent is responsible for the fresh post-fix suite.
3. Production-shape 15.8 GB snapshot peak RAM, disk headroom and serialization/upload duration on two T4s: source establishes one owned stage at a time, not actual resource sufficiency under production load. A bounded diagnostic is pending; no GPU/resource-proof claim made.
4. Live W&B alias-server behavior, current run counts, saved-view chart hashes, and owner/account/quota state: no live-service calls performed. The alias finding follows the application ordering and asynchronous publication contract, without claiming a reproduced real-cloud incident.
5. Bitwise equivalence of asynchronous PQL resumed schedules: asynchronous scheduling is explicitly nondeterministic; the new exact continuation tests use synchronous PQL.
6. Prevention of future external SIGKILL/account restrictions and reconstruction of replay already absent from historical policy-only artifacts: outside the capabilities of these changes.
7. Existing create_dashboard reconstructs predefined chart layouts when deliberately creating/rebuilding a dashboard; that pre-existing behavior is not a regression introduced by this diff. This review assesses the new reconciliation path and records the supplied live explicit-view verification separately.

## Verdict

Changes requested: resolve the single Important resumable-alias confirmation gap, run affected tests and the fresh full suite, and keep GPU/resource and external-cause limitations explicit. No second independent review is requested by this report.
