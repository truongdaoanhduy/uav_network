# Scenario V2 shared fixes and diagnostic training - 2026-10-18

User authorization: fix shared logical/environment bugs, research the channel, and use existing Kaggle accounts to run validation/diagnostic and baseline training. The prior design remains docs/superpowers/specs/2026-10-17-scenario-v2-common-baseline-design.md.

## Accepted bounded design
- Keep standard MASAC and MATD3 source unchanged; same physical scenario, reward, observations and seed banks for comparisons.
- Retire terminally failed packet pending state so a new selected power to the same receiver can be used next mission step. Do not escalate power automatically. Preserve arrived fragments, custody on DATA failure, report TTL and actual RF energy.
- If a nonzero wall guide fails the existing swept peer/static certificate, try the existing certified alternative wall tangents. Preserve acceleration/speed/clearance and pass-through for safe policy commands.
- Add passive joint counters for silence with a receiver available and for silence when GCS is unavailable but a valid closer-to-GCS peer exists. Such a peer is not proof of an end-to-end route. Do not affect random draws, dynamics or reward.
- Reuse V2 log power 0.001-0.4 W, max-power admission and link-state/cost formulation. Do not alter urban/suburban constants without measurement or primary-source justification. Current configured channel is an analytical SINR-threshold model, not calibrated stochastic fading.
- Report receiver-knowledge, reliable free beacons, reception-time custody-before-ACK and missing RX/circuit energy as explicit modeling assumptions; do not silently invent physical values.
- Existing V1 jobs and results stay intact. New jobs use an immutable fixed V2 source/config in the main W&B project.

## Acceptance
1. Failing tests reproduce next-step power recovery and feasible two-peer wall stall; minimal Native and Warp fixes pass.
2. Opportunity joint counts and pooled fractions agree on Native/Warp and remain passive.
3. Full active suite via python3 -m pytest -q tests; both learner hashes and V1 config hash remain unchanged.
4. Publish reachable feature branch, run real Kaggle CUDA validation plus BASE/PQL smokes, inspect online logs/metrics before long jobs.
5. Launch matched fixed-V2 jobs only with live quota, private credentials and no conflicting active job. Collect diagnostic evidence; policy quality is not a correctness gate.

## Verification before GPU release
- Baseline:444 passed,47 skipped,385 subtests; fresh full active suite.
- Retry regression:66 passed,20 subtests; RED6 failed/2 passed before fix.
- Boundary regression:6 passed,43 subtests;25 failing progress scenes before fix, related51 passed/1 skipped/4 subtests.
- Opportunity regression:40 passed,15 subtests; actual RED7 failed/1 passed before fix.
- Independent source review: no critical or important findings.
- First integrated run:458 passed,50 skipped,422 subtests; one structural line-limit failure in GPUmac (503lines). Reduced five blank lines to498; structure2 passed/240subtests. Full rerun:458 passed,50 skipped,423 subtests,8 pre-existing warnings in295.18s,exit0.
- New actual-CUDA tests are mandatory cloud startup checks; locally3 skipped because no CUDA driver.
- Learner and diagnostics-V1 SHA256 invariants match the prior baseline.

Release profile contract:all four long jobs share physical scenario SHA256 1f4c6bef03c37fe0a182d9722f01173afa699b2842b899f26d978431e20d7183 (world/sensor/network/energy/reward JSON), env4096,batch8192,updates8,episodes25000,horizon1800,seed44,final30seed bank2000000044. BASE/PQL and learner identity are the explicit experiment axes.
