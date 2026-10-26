# Scenario correctness fixes — 26/10/2026

The user requested actual Kaggle CUDA testing, scenario logic before algorithms, and safe joint W&B/HF cleanup. This source checkpoint resolves confirmed scenario defects; the corrected CUDA rerun remains required.

## Findings and changes

1. **Boundary predicate mismatch:** native validation and CPU/Warp generation admitted exact obstacle/clearance boundary points that motion and sensing reject within1e-9. An explicit UAV could remain stuck while commanding motion away; an admitted boundary target could be permanently occluded. Align the shared placement predicates with the existing geometry tolerance. Focused regressions cover native UAV/target/GCS and Warp CPU/CUDA predicates, including safe points beyond tolerance. Prior RED8fail/3pass/2skip was observed before implementation; original temporary log was lost on Sandbox reset, while the test and source changes persisted.
2. **Unobservable generated mission:** native primary seed302 target6 is isolated behind overlapping clearance-expanded obstacles. Physical gaps14.509269/19.818775m are below the required2x10m. The conservative proof even relaxes reachability and visibility; nearest admissible camera point remains131.250575m from the target vs100m footprint. The frozen geometry and reproducible proof are in `2026-10-26-unobservable-layout.json` and `2026-10-26-unobservable-proof.py` (Shapely diagnostic only, no new production dependency).
3. **Construction repair:** inset expanded obstacle disks within the map, keep them pairwise disjoint, and exclude the unobservable annulus when camera footprint is smaller than clearance. CPU and Warp implement the same constraints. This guarantees connected single-UAV free center space and an unoccluded geometric vantage for generated targets, not universal dynamic-window, multi-UAV, battery or finite-horizon mission success.
4. **Generation provenance:** `world.generation_version=2` is persisted in config and included in the scenario hash. Policy/evaluation-snapshot/BASE/PQL loading rejects missing, older and invalid generation versions before filling defaults. Observation_version remains3. Historical policies need their pinned historical source; a corrected cohort starts episode0. Report/payload1024B, buffer3072B, all reward/battery/learner settings remain unchanged. Layout distribution changes intentionally as the correctness repair.
5. **Historical-publication regression:** independent review found that fresh config defaults could relabel historical media as generation2. RED9publication cases reproduced this. Tracking now serializes an explicit unknown generation for visualization initialization, and final evaluation summary takes the saved source protocol's generation or unknown. GREEN28focused tests passed. No historical re-evaluation is performed.
6. **CUDA-host fixture crash:** actual Kaggle v1 on2TeslaT4/Torch2.11.0+cu128/CUDA12.8 segfaulted with exit-11 at a CPU Warp reset kernel. Five CPU-fixture masks in4testfiles lacked device selection and inherited CUDA default allocation. Bind them explicitly to CPU. This is a test-fixture correction; actual corrected-T4 verification is still pending. All executable owned production Warp allocations already select/infer their proper device.

## Verification

- Generator/protocol RED:13failed,3passed,1skipped. The subsequent test matrix was expanded to include both small/large camera footprints on Warp.
- Focused generator/locality/checkpoint integration:52passed,3skipped,8subtests before that final parameter expansion.
- Historical publication RED9failed; focused GREEN28passed.
- Fresh final complete CPU suite: **774 passed, 81 skipped, 8 warnings, 1282 subtests passed in 422.28s (0:07:02)**, exit0.
- 21/21 YAMLprofiles validate; compileall/AST and gitdiff whitespace checks pass.
- Primary profile SHA256:503cc6fb34133b7ffb0a43e3c9238b07a5e587958a49d68c45d3f491f711e952.
- Scan1030native layouts (train seeds44..1043 and all30final-eval seeds2000000044..2000000073): zero placement/invariant failures; minimum pair clearance surplus0.003338m and map-boundary surplus0.003998m. This scan checks construction, not learning quality or deadline feasibility.
- LegacyV2 physical parameter hash remains unchanged after removing the observation/generator protocol markers. The actual protocol hash changes intentionally; the original frozen-physics expected hash was not replaced.

## Review decisions and remaining work

Independent review found no Critical/Minor issue and one Important historical-publication issue, now fixed with RED/GREEN. The final exact tree has the full CPU verification above. No repeated review was requested after the fix; covering tests are the gate.

Rulings: battery calibration, memoryless actors and training-seed robustness remain methodological limits because no arithmetic/implementation defect was proved; the cost is no guarantee of good learning or physical calibration. Arbitrary explicit hand-authored layouts may intentionally be disconnected, so connectivity is a synthetic-generator contract rather than a new universal custom-scene feasibility solver; callers can still author infeasible fixtures. Storage deletion/global simulator equivalence were not approved by the source review and require separate fresh operational evidence.

Next: corrected source must pass the scenario-first CUDA diagnostic (including new tests), production-shape four-learner preflights/runtime checks and full CUDA suite. Existing v1 is terminal failed and preserved. No20kproduction training cohort is launched. Existing W&B/HF cleanup is tracked separately; do not confuse a reviewed proposal with completed deletion.

## Test isolation follow-up

Two real-render handoff tests did not stub external HF publication, so a local suite with valid fallback credentials wrote synthetic fixture media. A guard before credential lookup reproduced2failures; the minimal test-only fix stubs the external publisher, keeps the guard and all real train/load/render/evaluation assertions. Independent review accepted. Focused31passed/9subtests; final offline whole suite774passed/81skipped/8warnings/1284subtests,exit0,211.15s. Source/test commit823bf16 follows production fixes7223f97 and fixture9f60ce9. The CUDAv2run pins9f60 and predates only this test-isolation fix.

Additional profile generation smoke: all21YAMLprofiles,105native scenes plus42WarpCPUworlds, allpass. This checks the strengthened construction across configured scenarios without changing any profile.
