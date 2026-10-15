# Main W&B workspace and checkpoint storage implementation plan

> **For agentic workers:** Use `superpowers:executing-plans`; implement natively in this existing isolated worktree.

**Goal:** Every run is visible in one main Personal Workspace, while large checkpoint files live in a private external store and autonomous GPU/CPU handoff remains intact.

**Architecture:** Reuse W&B artifacts as immutable transport envelopes, replacing bulk files with HF commit/file/checksum manifests. Keep the legacy reader. Reuse existing private Kaggle credential datasets for the additional single HF credential. Preserve running pinned jobs.

**Tech Stack:** Python, unittest, W&B GraphQL/SDK, huggingface_hub, Kaggle CLI.

**Spec:** `docs/superpowers/specs/2026-10-15-main-wandb-storage.md`

## Global constraints

- Main W&B destination: `truongdaoanhduy-uit/uav_search_target`.
- No Saved Views, no default group/ID/search restrictions, no secrets in output.
- GPU continuation retains full version-3 training state; visualization stays CPU-only.
- Existing production jobs are not restarted or hot-patched.
- Removal requires verified private immutable backup; active artifacts stay protected.

## Review focus

- Legacy artifacts without a manifest still resume correctly.
- Corrupt/path-traversal/mutable-revision manifests fail before Torch load.
- HF authentication or privacy failures never fall back to multi-GB W&B uploads.
- CPU evaluation failures close the online run with failure status.
- Workspace reconciliation preserves chart content and shows future groups.

### Task 1: Main Personal Workspace

Files: `experiments/dashboard.py`, `experiments/tracking.py`, `tests/test_dashboard.py`, `tests/test_tracking.py`.

- [x] Add failing tests for group-independent visibility and preserved chart content.
- [x] Implement `ensure_project_workspace(config)` and make dashboard defaults unfiltered; invoke when online tracking starts and reject destination mismatch.
- [x] Run dashboard/tracking tests; reconcile the live Personal Workspace and verify all project runs match.

### Task 2: External checkpoint envelope

Files: new `experiments/checkpoint_store.py`, `configuration.py`, tracking, cloud remote/kernel/client/cli, pyproject, configs, new `tests/test_checkpoint_store.py` and existing cloud tests.

Interfaces: `upload_checkpoint_files(config, files, prefix) -> dict`; `download_checkpoint_file(artifact, entry, destination) -> str`. Manifest schema pins backend, repo, full commit SHA, relative file paths, bytes and SHA256.

- [x] Test checksum round-trip and refusal of corrupt/mutable/unsafe manifests, using real files and fake external transport.
- [x] Implement private HF upload/download and manifest validation; add config/backend validation and declared optional dependency.
- [x] Upload only checkpoint manifest to W&B in HF mode; read both old binary and new manifest formats.
- [x] Add single-file private HF credential dataset, propagate its reference through GPU continuation and CPU spec/bootstrap, perform preflight before cloud submission.
- [x] Run checkpoint/tracking/cloud/continuation checks, including unchanged full-state validation.

### Task 3: CPU visibility and live verification

Files: cloud visualize, evaluation callback, tests, verification documentation.

- [x] Test real CPU evaluation with logging callback and failed-run cleanup.
- [x] Log CPU evaluation online in the same project, retain output links, and keep videos on Kaggle.
- [x] Run complete unittest discovery and a small real online CPU/external-storage smoke; verify both run links, immutable restore and artifact size.
- [x] Commit/push the reviewed branch; launch one CPU-only cloud smoke for credential/bootstrap and visualization proof.

### Task 4: Verified migration and cleanup

Files: a small archive script and safe archive-index output, verification report.

- [x] Inventory completed project-owned legacy artifacts and active handoff references.
- [ ] Upload scoped candidates to the private store, verify remote metadata plus fresh-download checksums, publish tiny replacement manifests and archive mappings.
- [ ] Remove only verified obsolete original versions; keep running dependencies and all run histories.
- [ ] Read back W&B storage and workspace, record exact changes and any pinned-job limitation.
