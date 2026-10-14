# Mission comparison and GPU resource validation

> For agentic workers: use superpowers:executing-plans for the shared implementation and dispatching-parallel-agents for independent audit/benchmark work.

Goal: explain the existing fixed/PQL and native/Warp flows using source evidence, measure matched GPU workloads, and add failure-aware checkpoint comparison metrics.
Architecture: keep simulator, reward, observations, and learner equations unchanged. Observe CPU evaluation trajectories at step boundaries; pair the same test scenes, and sample physical GPU boards separately from Torch allocator counters.
Tech stack: existing NumPy, Torch, Warp, matplotlib, Kaggle and W&B; stdlib CSV, threads and subprocesses.
Spec: the user's 2026-10-14 request and docs/algorithm-comparison-research-20261014.md.

Constraints:
- Use the existing pql-async-20261014 worktree; preserve the previous configuration and native backend.
- The matched resource profile has E4096/B8192/replay500000, warmup32/rate8, six UAVs,25 targets,15 obstacles,H1000.
- A failed requested allocation remains an explicit OOM; no batch reduction.
- GPU benchmarks compare equal completed update budgets and exclude startup from the pipeline window.
- Simulation UAV energy and physical GPU energy have different definitions and units/scopes.
- Scenario uncertainty for one checkpoint pair is not uncertainty across independent training runs.

Review focus:
- Zero deliveries must not become zero delay or cheap J/delivery.
- Per-hop packet PDR must not become report/GCS delivery.
- Early battery exhaustion must not look like a fast completed mission.
- Final test seeds can overlap the sequential training seed range.
- Later cloud-wave failures must retain completed checkpoints and CPU handoff.
- CPU child submission precedes parent exit; strict session nonoverlap cannot be claimed.

Tasks:
- [x] Read source/upstream/papers and persist cited research.
- [x] Add failing mission metrics and paired comparison tests; observe missing-feature failures.
- [x] Implement MissionMetrics(config,positions,obstacles,found,delivered), observe(...,step), finish(info,return).
- [x] Add compare_evaluations(left,right,bootstrap_samples,seed) and JSON/CSV export.
- [x] Add real checkpoint/CLI failures, integrate evaluation and verify unchanged checkpoint bytes.
- [x] Reproduce CPU credential and later-wave result-loss defects; implement shared guards.
- [x] Add physical-board sample parsing/integration tests and matched legacy/PQL tests.
- [x] Complete isolated GPU-case runner and cloud dispatch.
- [x] Integrate paired metrics into the separate CPU child.
- [x] Run the full CPU suite, clean diff and independent review.
- [ ] Pin/publish source; run matched T4 comparisons and an autonomous PQL→CPU smoke.
- [ ] Persist actual results and explain their limits.

Validation: .venv/bin/python -m unittest discover -s tests -v; git diff --check.
Cloud benchmarks are short implementation checks, not convergence trials or production training.

Independent review found and regression-tested fixes for incomplete comparison
reports, final-reset/episode-logging timing asymmetry, and validation-bank reuse.
Telemetry also rejects integration across long missed polling intervals.

CPU full suite:306 tests,46 CUDA skips,151.605s. Final cloud exit-code edge regression:4/4 passed. A fresh pinned-source full run is also recorded before the cloud result report.
