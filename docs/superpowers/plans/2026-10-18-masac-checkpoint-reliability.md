# MASAC Checkpoint Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent multi-gigabyte fixed-schedule resume checkpoints from exhausting Kaggle local disk while preserving exact MASAC/MATD3 continuation at every graceful segment boundary.

**Architecture:** Periodic checkpoints remain lightweight policy snapshots. The complete optimizer/replay/RNG resume snapshot is written exactly once at the final batch or graceful segment end, where the run has enough free disk because no older local full-state snapshot has accumulated. Existing final artifact upload continues to publish the full resume checkpoint to Hugging Face/W&B manifest for the next continuation segment.

**Tech Stack:** Python, PyTorch, pytest/unittest, W&B, Hugging Face checkpoint store, Kaggle continuation pipeline.

**Spec:** `docs/scenario-v2-fixes-20261018.md`

## Global Constraints

- Keep standard MASAC and MATD3 source unchanged.
- Keep Scenario V2 physical scenario, reward, observations and seed banks unchanged.
- Preserve exact optimizer, replay, RNG and schedule state in the final version-3 resume checkpoint.
- Preserve atomic checkpoint writes in `uav_network/algorithms/checkpoints.py`.
- Keep W&B online and final checkpoint storage on the configured Hugging Face backend.

## Review Focus

- Periodic checkpoint callbacks must not create/require a 7 GB `resume_latest.pt` before the segment ends.
- A final batch that coincides with a checkpoint boundary must write the full resume exactly once.
- A final batch between checkpoint boundaries must still write the full resume exactly once after the loop.
- Resumed MASAC/MATD3 must preserve replay/optimizer/RNG schedule exactly.
- Final artifact publication must still include the full resume checkpoint.

---

### Task 1: Bound local full-state checkpoint storage

**Files:**
- Modify: `uav_network/experiments/training.py`
- Modify: `tests/test_baseline_resume.py`

**Interfaces:**
- Consumes: existing `save(path, final=False, full_state=True)` helper and `save_fixed_checkpoint(...)`.
- Produces: periodic policy snapshots with no full resume; one final `resume_latest.pt` per segment.

- [ ] Adjust regression tests so a two-checkpoint segment observes resume existence `[False, True]`, and full-state writes occur only once at segment completion.
- [ ] Run focused tests and verify they fail on the current implementation.
- [ ] Change the periodic `latest.pt` save to request full state only when `final_batch` is true.
- [ ] Run focused tests and verify they pass.
- [ ] Run baseline continuation tests proving replay/optimizer/RNG restoration is unchanged.
- [ ] Commit the change.

### Task 2: Full regression and remote verification

**Files:**
- No new production files expected.

**Interfaces:**
- Consumes: Task 1 commit.
- Produces: verified branch ready for Kaggle continuation.

- [ ] Run `git diff --check` and the full `tests` suite.
- [ ] Verify MASAC/MATD3 source hashes remain unchanged.
- [ ] Push the branch and verify the commit on GitHub.

### Task 3: Recover or relaunch MASAC V2 continuation

**Files:**
- Generated launch/evidence files under `outputs/` only.

**Interfaces:**
- Consumes: verified code commit plus latest valid MASAC BASE/PQL full-state checkpoints.
- Produces: new Kaggle continuation run(s) targeting 25,000 episodes and W&B online evidence.

- [ ] Inspect failed Kaggle outputs for a valid preserved `resume_latest.pt` and validate it as a version-3 fixed-schedule checkpoint.
- [ ] If a newer valid failed-run resume is recoverable, use it; otherwise use the last verified remote parent artifact.
- [ ] Select usable Kaggle account(s), launch continuation with the new code commit and unchanged scenario/baseline config.
- [ ] Verify Kaggle status and W&B run creation/config/state live.
