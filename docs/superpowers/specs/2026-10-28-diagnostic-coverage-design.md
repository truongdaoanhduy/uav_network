# Diagnostic coverage for failed, stalled and weak-policy runs

User explicitly requests an audit and implementation of justified missing logs,
Runs columns and Personal Workspace evidence, including cases beyond crashed /
failed / poor metrics. Baseline a13430b. Routine implementation choices below are
our proposals within that authorized scope; no production training or scientific
configuration change is authorized/needed here.

## Findings before changes

- Reproduced: TrainingDiagnostics.event raises ValueError on closed stdout and
  OSError on disk-full. Those optional diagnostics can terminate healthy training.
- Reproduced: failure_report fails to persist a report when selected context
  contains NaN, before/while preserving the original error. Shared redaction does
  not yet cover nested context and all environment credential families.
- Journal has selected losses/means but omits warning detail, important replay /
  PQL / disk evidence, metric-source steps and evaluation protocol/seed/count.
- No no-progress thread dump; a deadlock or native wait can leave only old scalar
  progress. Parent runner retains numeric exit codes but no named signal semantics.
- Canonical periodic evaluation is disabled. Historical final CPU30 has real
  aggregate delivery/success and evaluation/episodes_completed; training quality
  curves alone cannot establish held-out performance or a root cause.
- Source/actual Personal UI currently17 columns and26 plots+guide. Mode duplicates
  BASE/PQL in names; eval sample count is missing. Count/partial and disk headroom
  matter more in the main view than actor loss.

## Approaches and selected tradeoff

1. Add every metric / a custom grade / a monitoring service: rejected; clutter,
   speculative thresholds, overhead and another dependency do not prove causes.
2. Reuse existing scalar, cohort, evaluation, console, native system and supervisor
   evidence; harden shared writers and fill concrete gaps: selected. Stdlib only.
3. Rebuild the training/evaluation protocol: not this task. Multi-training-seed
   experiments and periodic evaluation are scientific decisions, not UI fixes.

## Deliverables

- Shared bounded JSON-safe sanitization/redaction reused by journals, diagnostic
  events and failure reports. Preserve nested evidence, nonfinite markers, primary
  traceback/cause chain, explicit truncation markers; never repr tensor/object data.
  Optional file/console channels fail independently; scientific history/checkpoint
  I/O keeps its existing correctness semantics.
- Failure report includes real run/source/stage identity, available process/disk/
  host RAM/Torch allocator snapshots and a conservative observed exception category.
  A SIGKILL or heartbeat loss is never labelled OOM/ban without independent proof.
- Reuse existing replay/PQL/network/reward/disk metrics in bounded journal evidence;
  include warning flags and source steps. Add evaluation count, seed/protocol /
  checkpoint evidence from actual results. No fake historical logs or quality score.
- Stdlib faulthandler no-progress dump, default300s, repeat without terminating the
  process; reset only on actual progress. Own/cancel timer and keep FD open safely.
  Dump is evidence of a long call/wait, not a verdict of deadlock. Native fatal
  traceback enabled for learner subprocess startup. No new poller/service.
- Parent nonzero exits record signal name when negative, unknown cause explicitly;
  preserve existing failure/continuation contracts and states.
- Personal Runs remains17 columns: remove redundant Mode, add actual completed
  evaluation episodes (normalize periodic summary when available); keep Algorithm
  hidden. Retain4 pins and all authentic states. Move actor loss to advanced view;
  add cohort sample count and free disk to core:27plots+guide. Guide distinguishes
  insufficient evidence, poor outcomes, regression, numerical fault, slow progress,
  evaluation failure, logging failure and pipeline blockage.
- Coverage matrix maps each symptom to evidence, next check and what remains
  unknowable. Hard kill, hidden platform/account failures and causal/generalization
  claims still need external logs or controlled experiments; no universal promise.

## Constraints and verification

No training/reward/scenario/precision/evaluation schedule/replay/checkpoint-format
changes, no production launch/cancel, no credential writes or historical state /
history edits. New observations apply to future source-pinned jobs. New resources
bounded at existing progress cadence (about60s), not each optimizer update.

TDD on reproduced serialization/I/O failures; real subprocess watchdog capture and
cancellation; exception chain/resource fallbacks; selected journal evidence /
evaluation count+protocol; negative process exit semantics; SDK table/plot axes.
Run focused tests then one frozen-source full offline CPU suite, independent code
review, logging-only cloud probe if publication behavior changes, Personal live
API readback and signed-in UI verification. Source/evidence pushed to both branches.
