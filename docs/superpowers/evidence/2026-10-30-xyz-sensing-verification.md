# XYZ Bayesian sensing verification — 2026-10-30

Approved request: use altitude-dependent sensing from the local
`uav_search_target` reference, with real XYZ flight and 2D visualization.
Implementation/runtime commit: `7dd41d01c7b0c0df8af0ff35d0e2a7314f0a71af`; branch `feat/cpu-replay-20261022`.
Comparison base: `3382feee81047c82d000e5c598ef685aec435297`.
Design/review base: `c18c65c89630607792c8e6da0b6068b1dbcf6588`.

## Implemented behavior

- Three opt-in XYZ profiles preserve the original scenario/training settings:
  6 UAV,1000m,20 targets,15 obstacles,1200 steps,20k episodes,E4096,B8192,
  replay1M,FP32 and learned RF. Existing three XY YAMLs are byte-identical
  to the runtime base.
- Actual XYZ motion/velocity, swept cylinder and peer safety, finite-height
  obstacles,3D radio range/LOS/pathloss/admission and existing energy accounting.
- Continuous action `[ax,ay,az,power]`; both MASAC/MATD3, sharing variants,
  replay, warmup,inference,BASE/PQL and critic dimensions use the configured width.
- Per-UAV25m belief grid starts0.5. FullFOV90deg; radius increases with altitude.
  Reference Pd/Pf anchors0/50/100/150m are .99/.90/.80/.70 and .01/.10/.20/.30.
  Partial-cell coverage, LOS and positive/negative Bayesian evidence are applied.
  Confirmation needs active UAV,z<=50m,posterior>=.99,fresh positive and a
  real target in the camera footprint/LOS. False positives do not reset beliefs
  using hidden truth and do not create fake real-target reports.
- Actor inputs contain local noisy cell candidates and5x5 belief patches;
  unconfirmed target coordinates are not exposed. Compact critic features are
  documented as partial belief representations, not a complete Markov state.
- Counter-based sensing does not resample on observe/info/render. Masked resets
  preserve untouched worlds. Full resume stores beliefs,coverage,altitude,
  vertical velocity and sampling state. XYZ schema4 is distinct from legacyXY3.
- Rendered frames remain2D and use captured z/battery/FOV/belief, with ground
  truth explicitly diagnostic. Validation-best actors and full last-resume
  checkpoints keep their separate roles.

## Follow-up: Warp reset compilation

Core XYZ final fixes were committed/pushed inc63c8f7 by an intervening
continuation. Its full suite passed1145 tests,97 skips,1554 subtests577.22s;
another immutable-source run passed the same counts655.14s.
This follow-up continues the existing uncommitted reset extraction.

Archived CUDA phase logs reached vector allocation but stalled at reset:
a BASE MASAC probe ended after a600s timeout. A later reset-kernel probe first
failed because Warp was missing; v2 progress reached reset_network,with no
completed success report. These are historical partial attempts,not a passing
CUDA preflight or evidence that E4096/B8192/replay1M fits the intended hardware.

Warp compiles kernels by Python module. Reset previously shared the scheduler
module with packet_step. The existing reset is moved into network_reset.py,
retaining mac.reset_network for callers. The function body/decorators are
AST-identical to c63c8f7; its compile registry contains only reset_network and
is distinct from packet_step. No scheduler equations,state writes or RNG
ordering were changed. Targeted packet/physics/vector/XYZ tests:
93 passed,1 skipped,12 subtests,30.79s. The boundary guard was also observed
RED on an isolated c63 source archive and GREEN on the live extracted code;
no live source mutation was used to reproduce the baseline.

Official references consulted through Context7:
[NVIDIA code generation](https://github.com/nvidia/warp/blob/main/docs/user_guide/programming_model/code_generation.rst)
and [module boundaries](https://github.com/nvidia/warp/blob/main/docs/user_guide/execution_and_performance/reducing_compilation_and_startup_time.rst).
Local checks used installed Warp1.18.0. Actual CUDA improvement is unmeasured.
The previous automatic approval-review rejection of an external Kaggle launch
remains a blocker; its saved record has no detailed rejection reason.
This continuation submitted no job,changed no account and used no bypass.
Task3 CUDA preflight/tiny GPU smoke remains pending; the local code fix is verified.

## Verification

Final full suite: **1146 passed, 98 skipped, 8 warnings, 1555 subtests passed in 648.58s (0:10:48)**. Exit0; JUnit reports no failures/errors.
Command: `python -m pytest -q --junitxml=outputs/xyz-verification-20261030/closure-suite.xml`.

Initial full suite:5 failures,1123 passed,97 skipped,1549 subtests passed,
613.60s. Four failure instances were legacy lightweight-render adapter
compatibility; one was the XY map rejected by an irrelevant Bayesian grid cap.
Both causes were fixed. The initial red result is retained alongside the final
run; it is not presented as a green suite.

Post-review full run:4 failed,1140 passed,97 skipped,1554 subtests,624.61s.
The stricter height guard also checked Warp's legal zero padding slot when
num_obstacles=0. All4 failures were at restore,not model equality. A new
zero-obstacle roundtrip test reproduced this RED; height validation now covers
only actual obstacle entries. The invalid-height rejection remains strict.

Targeted final fixes:32 passed,3 subtests,7.12s; then40 passed,23.24s
including the valid zero-obstacle case and4 full learner resume cases. The first attempted command
referenced nonexistent `tests/test_rendering.py` and collected nothing;
the corrected command used `tests/test_communication_rendering.py`.
Compileall and diff whitespace checks passed. After the final documentation update,project structure checks passed:
2 tests,506 subtests,0.22s; all owned text sources remain<=500 lines. No CUDA results are inferred
from Warp CPU execution.

New coverage includes Native/Warp sensing/geometry and radio parity,high/low
confirmation,false-positive locality,masked reset isolation,atomic restore,
all sharing combinations,az gradients,power masks,replay and schema guards,
small real BASE/synchronous/async PQL learner updates,CPU evaluate,
validation-best selection,legacy identities,study naming and2D frame data.
Full learner uncut/resume equality compares model hashes,mission records
and every world array for serial BASE and synchronous PQL. Raw world snapshots
independently preserve the next measurement under the same action.

## Independent whole-change review and fixes

One fresh reviewer reviewed `c18c65c..5ab785e` against spec,plan and rulings;
no Critical,4 Important and1 Minor. No second review was dispatched.
Important findings were accepted by effect and fixed in one pass:

| Finding | Regression evidence |
| --- | --- |
| Invalid nonpositive obstacle heights accepted by restore; new discrete sensor domains unchecked |8 invalid-domain cases and1 valid zero-obstacle padding roundtrip RED→GREEN; all arrays,seeds,action RNG unchanged after rejection |
| Bayesian128-cell cap rejected valid large XY maps | Existing long-link propagation case RED→GREEN; cap restricted toXYZ |
| Relay audit validated3 slots and fixed power wrote az |3 communication-control cases RED→GREEN; motion slots and input action preserved |
| XY oracle accepted unsupported XYZ and crashed in shape math |3 public-entry cases RED→GREEN; explicit XY-only error before executing |

A Minor remains deferred: per-category XYZ obstacle/peer/boundary intervention
counters are unavailable and currently0. Total interventions remain recorded.
The user-facing document discloses this; do not use zero categories as evidence
of no obstacle/peer avoidance. This does not change physical safety behavior.

## Rulings in execution order

1. Warp retains XY arrays plus dedicated altitude/vertical-velocity arrays,
   exposing XYZ at its boundary, to retain legacy kernels/checkpoints.
   Cost if wrong: inconsistent coupling could break3D safety/physics; parity
   tests cover motion,sensing,radio and full world continuation.
2. Hosted skill helper scripts are unavailable remotely; equivalent briefs,
   ledger and test logs were used. Cost if wrong: losing the ledger can cause
   repeated work after compaction; final records are now committed.
3. Bring rendering forward because evaluation always renders; do not bypass
   that integration seam. Cost if wrong: bad frame projection invalidates
   evaluation visualization; frame-specific renderer tests and inspected
   PNG/GIF cover this.
4. Scope uninterrupted full-learner equality to serial BASE/synchronous PQL.
   Existing final forced drains change overlapping/async scheduling; do not
   replace the scheduler to manufacture parity. Cost if wrong: falsely
   promising bitwise overlap/async equivalence. No such guarantee is made;
   raw world next-measurement resume is verified separately.
5. A new XYZ feasibility oracle is a separate study; old public oracle
   entrypoints explicitly rejectXYZ. Cost if wrong: no XYZ20/20 oracle
   feasibility claim is available; this is not proof of mission impossibility.
6. Production CUDA performance/convergence remain unmeasured; preserve requested
   E/B/replay/FP32 and launch no jobs here. Cost if wrong: profile fit and
   convergence remain unknown until measured on the intended hardware.

7. Isolate the reset kernel into its own compile module while retaining its
   public alias,as a minimal response to archived reset startup stalls.
   Cost if wrong: dependency compilation/extra load overhead may leave startup
   slow; functional parity is tested,but CUDA performance is not yet measured.

## Demo, memory and remaining limits

A real Native demo uses a scripted altitude controller,2UAV/2target/2obstacle,
seed44,60steps:1 target confirmed and1 report delivered toGCS.
This is pipeline evidence,not a trained MASAC/MATD3 or20/20 feasibility result.
The PNG and22-frame GIF were visually inspected.

- [Usage and configurations](../../xyz-sensing.md)
- [Demo JSON](xyz-camera-demo-20261030.json)
- [PNG](xyz-camera-low-20261030.png)
- [2D animation](xyz-camera-demo-20261030.gif)
- [Static resource estimates](2026-10-30-xyz-resources.json)
- [Structured verification](2026-10-30-xyz-sensing-verification.json)

Observation173,critic565,continuous4. Replay1M uses12,976,000,000B
(~12.085GiB) plus measurement arrays; Warp E4096~2.629GiB plus second observation
buffer,models/optimizers,activations and CUDA overhead. Device placement matters.
Belief/geometry arefloat64; learners remainFP32,AMPoff. These are static
estimates,not a measured CUDA fit or throughput benchmark.
No20k study or policy-convergence claim. Historical lineages remain distinct.
Host-managed worktree and initial uncommitted CURRENT_TASK suffix are preserved.
Raw check logs remain in `outputs/xyz-verification-20261030/`; only this plan's
ignored execution workspace is removed after its records/fixes are committed.
