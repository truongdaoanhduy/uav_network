# Live W&B/Kaggle diagnosis: two training runs remain

Observed W&B 2026-10-26 18:55:13 +07:00; Kaggle 18:50:06 +07:00.
This supersedes the 16:51 handoff snapshot. Requested scope: current running count, failure causes and proposed fixes. No production source/config changes or cloud submissions/cancellations.

## Current counts and progress

The complete W&B active-state query returned exactly four Running rows:
two production learners and their two cloud controllers.
Kaggle independently reports both learners and their controllers running.
Both learners progressed across independent W&B reads.

| Replay | Case | Run | State | Completed episodes / 20,000 |
| --- | --- | --- | --- | --- |
| CPU75/GPU25 | MASAC BASE | d17a279d | running | 4,096 |
| CPU75/GPU25 | MASAC PQL | 60fc6245 | running | 4,096 |
| CPU75/GPU25 | MATD3 BASE | da65a1ba | failed | 5,519 |
| CPU75/GPU25 | MATD3 PQL | 6f538a33 | failed | 5,512 |
| FULL GPU | MASAC BASE | b18d6a0d | failed: CUDA OOM | 0 |
| FULL GPU | MASAC PQL | benchmark 8b988c64 | preflight OOM; production not started | n/a |
| FULL GPU | MATD3 BASE | 8c599ca4 | failed | 5,551 |
| FULL GPU | MATD3 PQL | cfad2583 | failed | 5,562 |

At 18:55:13, MASAC BASE had 8,769,536 transitions / 16,872 updates;
MASAC PQL had 10,878,976 transitions / 20,976 updates.
Global transitions are not episodes.
Reserved MASAC PQL production ID 003e327e must not be reported as an existing production run.

## New confirmed cause: environment reset fails

All four MATD3 failure.json tracebacks terminate at:
training.py:302 or pql_training.py:200 -> WarpVectorEnv.reset ->
reset_geometry -> _check_errors: Warp world capacity/configuration error 102.

world.py:96 assigns error 102 only when obstacle placement exhausts
the configured 5,000 attempts. This is not a CUDA OOM traceback.
The last BASE diagnostics report zero nonfinite learner parameter names.

Both BASE failure contexts independently identify the failing episode seed:
- hybrid da65a1ba: world index 3256, seed 9642;
- full-GPU 8c599ca4: world index 50, seed 9642.

PQL failure contexts do not persist their episode seed arrays. Their matching
error 102 is confirmed; seed 9642 was not directly recovered from those runs.

Fresh unchanged-source reproduction with the primary configuration:
- native seeds 44, 302, 9641, 9643 succeed;
- native seed 9642 fails with Cannot place obstacles;
- 24 of 25 obstacles were placed; the next radius is 67.92984974735757 m;
- Warp CPU reset(seed=9642) independently fails with [(0, 102)].

The generator commits each obstacle greedily, fixes its sampled radius, and only
retries its center. It has no whole-layout retry/backtracking on exhaustion.
This establishes a reproducible scenario-generation failure. It does not prove
no geometrically valid final placement exists after unlimited search.

Both remaining MASAC learners use the same generator and monotonically assigned
episode seeds. They have not reached the failing seed in the observed progress.
If they continue far enough under the unchanged protocol, the same failure is
expected; this is an inference from the shared reset and seed-assignment code.

## Independent confirmed cause: MASAC FULL GPU exceeds VRAM

The earlier full-GPU failure remains independently supported by the downloaded
BASE traceback and PQL benchmark summary:
- 1M replay consumes about 12.2 GiB on learner GPU1, including bookkeeping;
- actual T4 capacity is 14.56 GiB per device;
- BASE target critic/vmap requests 336 MiB with only 244.81 MiB free;
- PQL preflight requests 336 MiB with only 176.81 MiB free.

The second GPU does not automatically pool memory with GPU1. The OOM precedes
successful production updates in BASE; PQL stops before production initialization.
Do not attribute the four MATD3 reset errors to this separate memory problem.

## Why automatic continuation/visualization did not proceed

The six failed lineages have W&B controllers failed/phase=blocked/error_type=ValueError.
Their Kaggle controller kernels are complete; complete here is the execution
container state, not proof the pipeline succeeded. W&B state and phase reveal failure.

next_stage requires a successful training summary, verified progress, immutable
checkpoint and episode_budget/time_budget stop metadata. Fatal training does not
satisfy these gates, so the controller blocks instead of declaring success or
submitting final evaluation. These are consequence/guard behavior, not six
additional independent training root causes. No final CPU completion was established.

The two surviving controllers have matching root/source/current GPU and fresh
11:51 UTC heartbeats, phase training. Their readiness does not guarantee later success.

## Proposed fixes, not implemented

1. Fix the shared scenario generator first. Add a bounded deterministic whole-layout
   restart/backtracking path after placement exhaustion, preserving 25 obstacles,
   radius range, clearance, map and all validation invariants. Mirror RNG/behavior
   in native and Warp; retain explicit failure for unsupported configurations.
   Do not silently drop obstacles, skip a bad seed, enlarge the map or shrink radii.
2. Regress exact seed9642 on native/Warp CPU/real CUDA, test repeat determinism,
   masked resets and native/GPU parity; scan the actual training/evaluation seed
   schedule before launching another 20k cohort. If generation semantics change,
   record a new protocol version and keep old/new scientific results distinct.
3. For MASAC FULL GPU, validate lower-peak critic execution: sequential/chunked
   critic evaluation and, if needed, sample microbatches with correctly weighted
   accumulated gradients and one optimizer step per unchanged logical batch8192.
   Keep replay1M/env4096/scientific settings. Verify loss/gradient equivalence and
   peak allocation on actual2T4 before a new attempt. Existing vectorize_critics=False
   is a candidate to benchmark, not a proven sufficient fix; actor graphs also matter.
   torch.vmap supports chunk_size per official PyTorch documentation.
4. Keep fatal-error continuation fail-closed. Improve controller diagnostics to
   preserve the actionable parent exception; only launch continuation/evaluation
   after valid successful checkpoints/progress. Do not blind-retry deterministic
   seed failures or restart the active learners.

## Evidence and validation

Raw downloads and query/reproduction receipts:
outputs/status-audit-20261026-1849/.
Tracked compact receipt: 2026-10-26-live-running-diagnosis.json.
Source remained f39f456 for production; local branch HEAD before this record773c484.

No production learner/controller was cancelled, restarted or launched.
No HF/W&B data was deleted or modified by this diagnosis.
Only local operational evidence/current-task records are changed.
Verification of these documentation changes is recorded in CURRENT_TASK.md.

Documentation verification: tests/test_project_structure.py -> 2 passed, 370 subtests;
git diff --check clean; compact receipt consistency and credential leak checks pass.
