# Crash visibility and bounded recovery implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline; one independent review at the end.

**Goal:** Make W&B failures visible and preserve verified resumable progress before external preemption.
**Architecture:** Reuse workspace GraphQL and current exact BASE/PQL checkpoint + HF manifest code; one bounded clock/staging helper. Preserve old run states and immutable manifests.
**Tech Stack:** Python,pytest,PyTorch,wandb-workspaces,Hugging Face Hub.
**Spec:** docs/superpowers/specs/2026-10-27-crash-visibility-recovery-design.md

## Global Constraints
No running job mutation/relaunch/cancellation; no HF/W&B deletion; keep scientific/acceleration protocols; no new account selection or circumvention; credential-safe output; existing linked worktree and branch retained.

## Review Focus
- Failed backup after remote bytes upload but before W&B confirmation must not replace last confirmed pointer.
- PQL full snapshots must drain workers and not publish a light duplicate.
- Repeated15.8GB snapshots must not coexist on the20GB disk.
- Interval/cap must work before the first completed episode and not perturb resume RNG/update clocks.
- Workspace updates must keep charts and authentic failed/crashed states, including explicitly selected saved views.

### Task1: Failure visibility
- [x] Add a failing real-serialization test for crash-first sort and completion/status columns; run RED.
- [x] Share the run-sort payload between workspace creation and reconciliation; preserve plots/selections/all-state visibility.
- [x] Run focused GREEN; update existing primary live views without changing run data; read back and verify query order.

### Task2: Bounded resumable publication
- [ ] Add RED tests for publisher resumable inclusion/aliases/throttle/failed-confirmation pointer.
- [ ] Extend tracker checkpoint publication and configuration validation.
- [ ] Add RED tests for wall-clock timing/caps/failure/staging cleanup and BASE/PQL integration with real small full checkpoints.
- [ ] Implement recovery_checkpoints.py; wire snapshot callbacks after full reset boundaries, using optional resume_path and suppressing duplicate PQL publication.
- [ ] Verify real checkpoint validation and future resumed updates for BASE/PQL, including overflow state and pending-learning clocks.

### Task3: Verification and evidence
- [ ] Run full offline suite and targeted real2T4 checks when a legitimate available account passes gates.
- [ ] Obtain one fresh independent review; resolve confirmed findings and rerun affected checks.
- [ ] Record the exact states, alternative-cause evidence/uncertainty, workspace readback, backup bounds/cost/limits; update CURRENT_TASK and commit/push.
