# Parameter-Sharing Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in A/B/C/D parameter-sharing variants for MASAC and MATD3 while preserving the existing default behavior.

**Architecture:** Add two default-false flags to the shared off-policy config. Shared actors are one-hot agent-conditioned and stored once; shared critics are one centralized team TwinCritic. Algorithm helpers route action and Q evaluation through either shared or independent topology without changing environment/replay schemas.

**Tech Stack:** Python 3, PyTorch, unittest/pytest, existing MASAC/MATD3/PQL infrastructure.

**Spec:** `docs/superpowers/specs/2026-10-28-parameter-sharing-variants-design.md`

## Global Constraints
- Default/current topology and canonical YAML behavior must remain unchanged.
- No changes to reward, observations, replay schema, environment, or jobs already running.
- Shared actor receives only local observation plus static one-hot agent identity.
- Shared checkpoint topology is reconstructed from saved config; no implicit conversion from independent checkpoints.
- FP32/AMP/compile/PQL paths must retain existing behavior.

## Review Focus
- Shared actor action shape/masking for all six UAVs.
- Shared actor gradient uses one optimizer and does not scale sixfold.
- Shared critic target/broadcast shapes for MASAC and MATD3.
- PQL policy snapshot/inference-copy with a one-element actor ModuleList.
- Checkpoint round-trip and old default-false config compatibility.

---

### Task 1: Configuration and topology contracts

**Files:**
- Modify: `uav_network/configuration.py`
- Create: `tests/test_parameter_sharing.py`

**Interfaces:**
- Produces: `OffPolicyConfig.share_actor_parameters: bool`, `OffPolicyConfig.share_critic_parameters: bool`.

- [ ] Write failing tests for default A topology and B/C/D actor/critic module counts for both algorithms.
- [ ] Run those tests and verify they fail because the flags/topologies do not exist.
- [ ] Add the two default-false config fields and minimal constructor topology selection.
- [ ] Run the task tests and verify topology assertions pass.

### Task 2: Shared actor execution and optimization

**Files:**
- Modify: `uav_network/algorithms/masac.py`
- Modify: `uav_network/algorithms/matd3.py`
- Modify: `uav_network/experiments/pql_workers.py` if inference-copy metadata requires it
- Test: `tests/test_parameter_sharing.py`

**Interfaces:**
- Produces: agent-conditioned shared actor action generation and one shared actor optimizer.

- [ ] Write failing tests for B/D `act_batch`, deterministic checkpoint output, one-hot input dimension, and optimizer count.
- [ ] Run RED tests.
- [ ] Implement actor selection/input helpers and average shared actor losses before one optimizer step.
- [ ] Run GREEN tests plus existing MASAC/MATD3 tests.

### Task 3: MATD3 shared team critic

**Files:**
- Modify: `uav_network/algorithms/matd3.py`
- Test: `tests/test_parameter_sharing.py`

**Interfaces:**
- Consumes: Task 1 critic flag.
- Produces: one team TwinCritic target/update and per-agent actor gradients through that critic.

- [ ] Write failing tests that C/D perform finite critic and delayed actor updates with one online/target TwinCritic.
- [ ] Run RED tests.
- [ ] Implement shared critic TD target and actor joint-action evaluation.
- [ ] Run GREEN tests plus `tests/test_matd3.py`.

### Task 4: MASAC shared team critic

**Files:**
- Modify: `uav_network/algorithms/masac.py`
- Test: `tests/test_parameter_sharing.py`

**Interfaces:**
- Consumes: Task 1 critic flag and Task 2 shared actor helpers.
- Produces: shared team critic with coordinate expected-Q and mean soft target; six alpha pairs remain.

- [ ] Write failing tests that C/D perform finite critic/actor/alpha updates with one online/target TwinCritic.
- [ ] Run RED tests.
- [ ] Implement shared expected-Q and shared critic target/update paths.
- [ ] Run GREEN tests plus `tests/test_masac.py`.

### Task 5: PQL/checkpoint/compile regression and variant observability

**Files:**
- Modify: relevant naming/telemetry helper only if needed to expose sharing flags.
- Test: `tests/test_parameter_sharing.py` and existing PQL/checkpoint/compile tests.

**Interfaces:**
- Produces: exact shared checkpoint round-trip, PQL snapshot/install/inference support, and visible sharing metadata.

- [ ] Write failing integration tests for checkpoint and PQL snapshot/inference-copy in shared variants.
- [ ] Run RED tests.
- [ ] Make minimal compatibility changes.
- [ ] Run focused suite, then full `python -m pytest -q`.
