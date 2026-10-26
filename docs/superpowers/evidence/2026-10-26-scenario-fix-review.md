# Independent review: scenario generation and CUDA test-device fixes

Reviewed 2026-10-26T00:46:58Z in `/home/aduy/Documents/NCKH/uav_network-public-hf-20261022`, branch `feat/cpu-replay-20261022`. HEAD was `69446816c0c8a4672979bf6210d548f5b75cee1d` (documentation only above source `dc63f27c12e298c3f495fff8e1c4e84e7f191369`).

Scope: the nine-file tracked working diff, untracked `tests/test_scenario_clearance.py` and `tests/test_scenario_generation.py`, their native/Warp geometry and sensing consumers, policy/evaluation/BASE/PQL loading paths, evaluation protocol and historical media publication provenance. Read `AGENTS.md`, `docs/superpowers/CURRENT_TASK.md`, and `docs/superpowers/plans/2026-10-25-cuda-scenario-storage-audit.md`. No source, test, index, HEAD, job or cloud state was modified by this reviewer. This report is the sole intentional file write.

Review snapshot: tracked candidate diff SHA256 `738871ef419ced38eda6ab2c7ee63ddc2e7a490d10dc849dee6ff6c6d8ce0ea4`; clearance test SHA256 `aebe6c54ca16109ed336b96ce3512fe3c6c26b88d720bd6b0c1457267803ec1a`; generation test SHA256 `8a8b9831b9ab7cce557cced6fd8024ebe9892d71cf1a0e9c7e9a0569efeaf085`. These anchors exclude any subsequent root-agent fixes.

## Findings

### Critical: none found in scope

### Important: historical publication recovery advertises generation 2 in W&B configuration

**Introduced field:** `uav_network/configuration.py:12`.

**Affected path:** `uav_network/cloud/visualize.py:261`, `:277`, and `:316`; `uav_network/experiments/tracking.py:174`.

A publication-only recovery loads a fresh current configuration before reading the old media protocol. The newly added default supplies `world.generation_version=2`. It calls `start_tracking(..., {"observation_version": None})`, and the tracker serializes the entire current `config.to_dict()` into the W&B run configuration. After recovering the assets, visualization corrects only `evaluation/observation_version`; it never sources a generation version from `protocol.scenario.world`.

Consequently, old or unknown-generation media retain their original metrics protocol but the new CPU-publication run advertises `config.world.generation_version=2`. Filtering or exporting scientific runs by that field can falsely identify historical geometry as the corrected generator. This is a concrete provenance regression caused by the new default, even though no historical policy is loaded or re-evaluated.

A read-only, mocked execution of the actual `OnlineTracker` initialization used the same initial-progress value as this path, intercepted `wandb.init`, replaced all file writes, and made no service calls. Observed:

```text
initialized_observation_version: None
initialized_world_generation_version: 2
evaluation_generation_summary: absent
```

Please source historical generation metadata from the saved protocol, preserve a missing version as unknown, and avoid writing current generation 2 into scientific metadata during publication-only recovery. Preserve the valid simulator configuration separately as needed. Extend `tests/test_visualization_provenance.py:15` to cover absent, old and current generation versions in both serialized run configuration and evaluation provenance. Existing tests cover observation version only and therefore miss this regression.

### Minor: none found that warrants a change

## Strengths

- The generator fix has a sound geometric invariant. Native `environment/world.py:37` and Warp `environment/gpu/world.py:86` inset each clearance-expanded disk from the map boundary. Their pair acceptance requires center distance greater than the two physical radii plus twice the clearance and an additional margin. The finite set of disjoint expanded disks remains strictly inside the rectangular map, so it cannot divide the single-UAV free flight region into disconnected pockets. The margins exceed the existing motion collision tolerance.
- Target rejection at `world.py:58` and GPU `world.py:118` handles footprints smaller than clearance. With inflated disks disjoint, a target outside every inflated disk is itself a valid vantage point; a target in one disk's annulus has a radial, unoccluded outward vantage within the footprint when the new radial condition is met. Other inflated disks cannot intersect that segment. This supports geometric observability, without claiming finite-horizon or battery feasibility.
- The `1e-9` placement guard now matches `environment/geometry.py:14` and the Warp motion obstacle threshold. Explicit UAV/target/GCS boundary acceptance and generated layout topology have focused regressions. The explicit target validator also rejects the forbidden annulus for small footprints.
- The counterexample evidence is stronger than an oracle failure. I inspected `reachability_proof.py` and its JSON: clearance polygons are inscribed, and physical-obstacle shadows are built from inscribed polygons. Both relax the true restrictions, so the resulting reachable/visible set is optimistic. The reported seed-302 target-6 lower bound of 131.250575 m at resolution 128, above the 100 m footprint, supports impossibility for that legacy layout. Shapely remains confined to the diagnostic proof.
- Generation provenance is checked before `_apply` can fill in defaults. The shared validator rejects absent, old, non-integer and unsupported versions, and all four required paths call it: policy, evaluation snapshot, BASE continuation and PQL continuation. Adding the field to the world configuration also puts it into the existing evaluation scenario hash.
- The five temporary Warp-mask allocations now explicitly select the CPU used by their launch. An AST inventory found no remaining executable `wp.array/zeros/ones/empty/full` allocation in the test files without an explicit device; the remaining device-less uses are kernel type annotations. The fixes directly address the supplied CUDA-host segfault mechanism without changing production algorithms.
- The regressions exercise actual checkpoint saves/loads and real BASE/PQL resume payloads, with positive current-version controls. Existing native/Warp reset tests compare exact generated layouts and PCG state, and existing CUDA tests compare CPU/CUDA reset and physics behavior.

## Verification evidence and limits

- Fresh `git diff --check` passed during this review.
- Supplied RED log records 13 failed, 3 passed and 1 skipped; supplied focused GREEN log records 52 passed, 3 skipped and 8 subtests passed. I read those logs rather than rerunning a competing suite.
- The current generation test includes the altitude 1/100 parameterization, whereas the saved RED log collected 17 generation cases. Therefore those existing log counts should not be treated as final verification of every later test edit.
- The full CPU suite was still running when checked; its latest output had passed the 42% progress marker. No full-suite success is asserted here.
- The downloaded CUDA v1 receipt genuinely records Torch 2.11.0+cu128, CUDA 12.8 and two Tesla T4 devices. Its first scenario-test stage failed with exit -11, so it establishes neither corrected-generator CUDA success nor algorithm/full-suite completion. The relevant CUDA stage and final suite still require a fresh run against the final source.
- The mathematical argument supports the stated construction invariant. It does not establish universal trajectory realizability by the dynamic-window controller or policy, or guarantee mission completion.

## Declined to judge as bugs

- Battery tightness, propulsion calibration, memoryless actor limits, single-training-seed inference, or general learning quality: these are outside this diff and are not evidence against the geometric repair.
- Arbitrary hand-authored explicit layouts can still be disconnected; the new generator's connectivity contract is not a universal feasibility checker for all custom fixtures. No requirement to reject all intentionally constructed disconnected test scenes was inferred.
- Storage-retirement safety, current production job state, and global simulator equivalence: this source review made no live operational decisions and provides no approval for deletions or job changes.

## Verdict

**Request changes for the historical-publication generation provenance gap.** The native/Warp topology repair, tolerance alignment, loading guards and explicit test-device fixes are otherwise technically sound within the reviewed scope. After the provenance fix, use fresh focused/full CPU evidence and the required actual-CUDA rerun before claiming the audit complete. Root owns further TDD changes, verification and commits.
