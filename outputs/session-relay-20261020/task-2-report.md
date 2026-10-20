# Task 2 report — paired relay/power audit and backend feasibility

Status: DONE. Implementation and final targeted verification complete.
Commit: this report, evidence, task files, and the user steering spec/plan are committed together (resolve with git log for this report).

## Implemented interfaces

- `communication_control(action, observation, config, mode="learned", fixed_power_w=None)`: returns a copy. Direct-only preserves silence/GCS selections and XY/power, redirecting peers only through the original GCS admission mask. Fixed-power preserves XY/destination and uses existing inverse power mapping, with finite physical-bound validation.
- `run_relay_audit(checkpoint, seeds=None, modes=MODES, fixed_power_w=None, output_dir=None, trace=False, on_progress=None)`: one immutable checkpoint, exact saved config/horizon, 30 default seeds from saved eval_seed or a 1–30 seed subset. Fingerprints SHA256, full config/scenario/protocol, modes/power, seed bank, trace flag and control version. It records actual saved episodes/global steps and target budget; old zero-completed-episode policies remain partial.
- `scripts/audit_relay.py`: CLI for the paired audit, emitting flushed JSON progress every 50 steps and at episode boundaries. Example: `PYTHONPATH=/tmp/uav-launch-http:.:tests .venv/bin/python scripts/audit_relay.py CHECKPOINT --output-dir OUTPUT --seeds 2000000044 --modes learned direct_only fixed_power --trace`.
- Existing `evaluate(...)` defaults/signature and metrics remain compatible; its episode loop is now shared through `evaluation.run_episode(...)`.
- `run_feasibility(config, seeds, margin_db=3.0, on_progress=None)`: selects the supplied backend. Warp uses exactly one CPU world even when supplied runtime.num_envs is 4096. Its controller reuses the existing assignment, movement, nominal power and destination logic through a controller-specific read-only report view and real observation admission masks. Native explicit feasibility stays native.

## Progress and persisted evidence

Audit paths under output_dir:

- `protocol.json`: written before episodes, immutable compatibility check.
- `cases/MODE-SEED.json`: atomically published only after a complete episode. Matching completed cases are reused; incomplete/malformed cases are rerun. Foreign checkpoint/protocol outputs are rejected.
- `summary.json`: initially partial, updated atomically after every completed case. Fields `complete`, `status`, `completed_cases`, `expected_cases`, and `by_mode[mode].complete` distinguish partial matrices.
- `progress.json`: audit callback's latest episode boundary or every tenth completed step. This file is progress evidence, not proof of persisted matrix completion.

The shared `run_episode`/feasibility callback receives dictionaries:

- Start: `status="episode_started", seed, step=0`.
- Every completed step: `status="running", seed, step, max_steps, targets_found, targets_delivered`.
- Terminal step: same step fields with `status="episode_complete"`.
- Audit adds `mode` and protocol `fingerprint` before forwarding to its caller. The callback is synchronous; exceptions propagate, and the environment is closed. An interrupted feasibility call returns no complete result.

Records contain backend, seed, initial layout hash, observed MissionMetrics and raw episode/network/report/energy evidence. Feasibility also includes scenario and explicit `not_a_policy_baseline` labeling. Equal numeric seeds across native/Warp do not establish equal geometry.

Summaries use OutcomePool and MissionMetrics for search/delivery/success/battery/PDR/underpower/required-power diagnostics, and expose successful direct/relayed GCS report counts. Explicit totals separate all-mission UAV propulsion+TX/RX energy from UAV radio-only energy; both per-delivery ratios are null at zero deliveries and exclude GCS. Optional traces record actual successful whole-report hops and last-fragment TX power independently of animation, without claiming packet-level logs.

## New production profile

`configs/comparison-20k-steps1800-scenario-v3.yaml` copies the old 2000-step comparison profile, changing only world.max_steps and truthful output/group/tags/comments. Tests assert 4096 environments, 20000 episodes, 1800 steps, seed44, 35 targets, 25 obstacles, 1000000-byte reports, and equality of all other config parameters apart from run labels. Original profile/checkpoint payloads remain unchanged. Old checkpoints still evaluate at their saved 2000-step horizon; full-state continuation across this horizon change is incompatible.

## RED/GREEN evidence

RED command:

```bash
PYTHONPATH=/tmp/uav-launch-http:.:tests .venv/bin/python -m unittest test_relay_audit -v
```

`task-2-red.txt`: five tests failed before implementation for missing relay_audit module, missing 1800 profile, and unsupported feasibility progress/backend interface. An initial incorrect unittest module name was corrected before recording this RED evidence.

Final GREEN command:

```bash
PYTHONPATH=/tmp/uav-launch-http:.:tests .venv/bin/python -m pytest \
  tests/test_relay_audit.py tests/test_feasibility.py \
  tests/test_comparison_evaluation.py tests/test_cpu_visualization.py \
  tests/test_communication_trace.py tests/test_async_evaluation.py -q
```

Final result: **37 passed, 9 subtests passed in 103.87s (0:01:43)**. Clean evidence: task-2-green-final.txt.

Seven new tests cover controls/copies/admission, existing inverse power mapping and invalid powers, real immutable paired checkpoint/config/seeds/layouts, partial budget, null energy denominators, interruption/restart/incomplete recovery/protocol rejection, CLI execution, exact profile constraints, real Warp feasibility, and explicit direct/far-peer/no-route read-only oracle cases. Existing evaluation, CLI comparison, animation, successful-hop traces and async evaluator tests are included.

Intermediate verification: first GREEN had one invalid test fixture (.02W below default .1W); corrected test to the requested V3 .001–.4W range. A 60-second shell call timed out while the existing native deterministic mission test continued in its child; that run subsequently reported 20 tests OK in 86.65s. A concurrent targeted run reported 35 passed/1 deselected/9 subtests, but shared its old log file with the surviving child. Neither overlapping log is the final evidence; the final run uses a separate unique log and includes the slow native case.

`git diff --check` passed. `git diff --exit-code 2889fcac -- uav_network/algorithms uav_network/communication uav_network/environment` passed with no changes.

## Files changed

- configs/comparison-20k-steps1800-scenario-v3.yaml
- docs/relay-audit.md
- docs/superpowers/plans/2026-10-20-v3-session-relay-plan.md (existing user steering)
- docs/superpowers/specs/2026-10-20-v3-session-relay-design.md (existing user steering)
- scripts/audit_relay.py
- tests/test_relay_audit.py
- uav_network/experiments/evaluation.py
- uav_network/experiments/feasibility.py
- uav_network/experiments/relay_audit.py
- outputs/session-relay-20261020/task-2-report.md
- outputs/session-relay-20261020/task-2-red.txt
- outputs/session-relay-20261020/task-2-green-final.txt

## Limits and root follow-up

No expensive checkpoint matrix or full35 mission was launched by this worker. Root will run actual saved2000 checkpoint controls, supplied full35/25-obstacle/1MB/3MB feasibility at 1800, full CPU discovery, and available CUDA verification. No CUDA driver is available in this worker's CPU runtime. The audit expects one writer per output directory. A failed privileged oracle episode is evidence about that controller, not mission impossibility. Controls are fixed-policy ablations; later trajectories may diverge after feedback. No dependencies, credentials, cloud launches, pushes, scientific kernels, channel equations, reward equations, or algorithm blobs changed.


## Review fix round 1 — initial staging and malformed case recovery

Fix base: f9124b22fe4e65ba8732f0721f562f34f274d436. Addressed independent review I1 and M1 without changing controls, scenario, evaluator or scientific behavior.

Initial protocol recovery now accepts only a directory consisting entirely of regular, non-symlink `protocol.json.<eight tempfile characters>.tmp` staging files. Complete staged JSON must match the requested protocol; incomplete/empty JSON may be rewritten. All entries are validated before publication or cleanup. After atomic protocol publication, old staging files are removed. Unknown entries, directories, symlinks, incompatible complete staging, and incompatible finalized protocols remain rejected under the existing single-writer contract. Case roots decoded as null/list/scalar are treated as incomplete and rerun; completed cases claiming foreign protocol/mode/seed still reject without overwrite.

RED command:
```bash
PYTHONPATH=/tmp/uav-launch-http:.:tests .venv/bin/python -m pytest tests/test_relay_audit.py -q -k 'initial_protocol_sigkill or valid_json_malformed'
```
Evidence `task-2-fix1-red.txt`: **6 failed, 1 passed, 8 deselected in 12.24s**. The real child was SIGKILLed at initial fsync (exit -9), leaving staged protocol output; restart rejected it. The five malformed roots (null, list, number, string, boolean) raised AttributeError before the fix. The test also verifies conservative rejection of unrelated startup content and foreign completed cases.

GREEN command (covering file only, as requested):
```bash
PYTHONPATH=/tmp/uav-launch-http:.:tests .venv/bin/python -m pytest tests/test_relay_audit.py -q
```
Evidence `task-2-fix1-green.txt`: **10 passed, 8 subtests passed in 21.22s**. `git diff --check` passed; scientific directories have no diff from FIX_BASE. No larger suite repeated, expensive scenario runs launched, or running root feasibility state changed.

Fix files: uav_network/experiments/relay_audit.py; tests/test_relay_audit.py; this report; task-2-fix1-red.txt; task-2-fix1-green.txt.
