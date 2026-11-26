# V72 verifiable scenario and packet audit implementation plan

> **For agentic workers:** Execute independent environment, CPU-radio and Warp fixes with RED → GREEN; integrate and review all changes before the final regression.

**Goal:** Verify the supplied V71 claims, repair reproducible defects, and deliver one complete `file.zip` with scientific rationale and raw evidence.

**Architecture:** Keep the CPU environment, CPU packet simulator, Warp packet runtime and MATD3/MASAC separate. Preserve the original task and learner losses. Use executable Warp CPU tests to test shared kernel code while reporting CUDA and full GPU RL independently.

**Tech stack:** Python, NumPy, PyTorch CPU, NVIDIA Warp 1.18.0, pytest, YAML.

**Spec:** User request, uploaded V70 handover and V71 source; `AGENTS.md` and `docs/PROJECT_ENGINEERING_CONTRACT_20261125.md`.

## Constraints

- Only the current planar seven-radio scenario is active; no old 3D scenario imports.
- Decentralized execution cannot use unobserved peer truth or target truth.
- Every owned source/text file stays at or below 500 lines.
- No credentials, checkpoint blobs, caches or `.git` in the release.
- A CPU backend kernel run is not CUDA execution or full RL training.
- Keep prior input ZIP unchanged; retain a local Git baseline and export the patch.
- Do not change reward objective, RF action or DS calibration without an explicit variant.

## Review focus

1. Hidden peer state must not alter actor masks or compact3 resolution.
2. Old UQ8.8 evidence must decay on the same clock in observations and reward.
3. Expired or unaccepted packets cannot count as application delivery or custody.
4. Partial reset, one-cell grids and exact event budgets must be valid.
5. Evidence, smoke and memory estimates cannot imply CUDA/training/parity success.

## Tasks and evidence

- [x] Preserve source ZIP; extract safely; read current contracts, V71 report and issue register.
- [x] Run unchanged full baseline. Result: 242 tests and 64 subtests pass; real Warp CPU compilation also works.
- [x] Environment: add behavioral failing tests for hidden-state access, DS aging, safe reset and action decoding; fix the root causes; run targeted regression.
- [x] CPU network: add failing tests for TTL, queue custody, coalescing lifecycle and fair origin service; fix and run targeted regression.
- [x] Warp network: add real kernel tests for rank preservation, event completion, ACK cleanup, expired routes and overflow handling; repair and execute.
- [x] Gate implementation: run real one-/six-UAV packet exchange through the same CPU/CUDA workload; verify delivery and ACK assertions; preserve fail-closed CUDA status.
- [x] Scientific comparison: read original UavNetSim source and primary DS/AoI/MARL/NVIDIA material; map each recommendation to an assumption and code path.
- [x] Check authorized Kaggle account 2 availability without exposing secrets; only attempt a bounded CUDA gate after CPU packet checks.
- [x] Audit configs, memory, maps and learner adapters; generate a Vietnamese component inventory, V71 verdict matrix, issue register and staged GPU roadmap.
- [x] Run full suite, CPU MATD3/MASAC optimizer smoke, policy, matched scenarios and manifests; capture exit codes/JUnit.
- [ ] Final packaging gate: package source/config/tests/docs/raw QA as `file.zip`, scan, extract afresh and run regression; results will be stored in `outputs/qa_evidence_v72/ROUNDTRIP_VERIFICATION.md`.

## Interfaces

Environment fixes own `environment/*`, `mobility/*`, `learners/actor_adapter.py` and `tests/test_v72_environment*.py`.
CPU network fixes own `communication/*` and `tests/test_v72_network*.py`.
Warp fixes own `gpu/*` and `tests/test_v72_warp*.py`.
The coordinating worker owns scripts, release docs, manifests and final evidence. Changes to cross-boundary APIs are agreed before editing.

## Completion criteria

Each repaired finding has an executable regression and an observed RED/GREEN result. The final ZIP is independently reproducible and names every remaining blocker, including full GPU environment, CPU–Warp event fidelity, upstream statistical validation and multi-seed learning.
