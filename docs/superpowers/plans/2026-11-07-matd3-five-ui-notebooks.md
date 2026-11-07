# Five MATD3 UI Notebooks Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for native execution.

**Goal:** Deliver exactly five distinct MATD3 Kaggle UI notebooks.
**Architecture:** One five-case notebook builder uses the existing shared
kernel bootstrap and scaled YAML with explicit MATD3 overrides. The old
independent/scaled builders retain their behavior and delivered files.
**Tech Stack:** Python,PyYAML,nbformat,PyTorch,Warp CPU.
**Spec:** ../specs/2026-11-07-matd3-five-ui-notebooks.md

## Global Constraints

No MASAC export, job submission or scientific/algorithm changes.
Actor-only LayerNorm; critic remains unnormalized.
All files <=500lines; source committed before archive export.

## Review Focus

- Shared/independent tags match actual actor/critic module counts.
- Updates8/16 reach active MATD3 config; delay2 yields4/8 actor calls.
- Case05 changes width only relative to01 on the same35t25o scene.
- Edited seeds affect config/run identity without stale labels.
- Default existing notebook builders and credential routes remain compatible.

## Task1: Five-case builder and executable boundary

Create uav_network/cloud/matd3_ui_notebooks.py,
scripts/build_matd3_ablation_ui_notebooks.py,
tests/test_matd3_ui_ablation.py. Reuse generic PREFLIGHT from ui_notebooks.py;
its common status line becomes architecture-neutral.

- [x] Write parameterized tests executing notebook identity and resolved config.
- [x] Observe missing-builder failures before implementation.
- [x] Implement only the five requested cases and a clean-source export CLI.
- [x] Test module sharing/LayerNorm, real CPU replay updates and round counts.
- [ ] Run related tests and the repository test command.
- [ ] Request read-only review and resolve material findings.

## Task2: Export and handoff

- [x] Document the matrix and manual UI steps, including actor-only LayerNorm.
- [ ] Commit/push source, then build all five from one common archive.
- [ ] Validate actual schemas/code/configs, all archived source bytes and old-file hashes.
- [ ] Save five separately named notebooks, update/push actual evidence and task record.
