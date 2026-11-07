# Scaled Independent Baseline Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for this native implementation.

**Goal:** Export a separate35T25O/H512/U8 matched baseline pair.
**Architecture:** Reuse the existing notebook builder with one explicit
scaled option. Preserve its existing default profile and filename.
**Tech Stack:** Python, PyYAML, nbformat, PyTorch, Warp CPU.
**Spec:** ../specs/2026-11-07-scaled-independent-baseline.md

## Global Constraints

Keep updates8; user only asked advice about16/32. No GPU job submission.
Do not change algorithm equations, old profiles, reward/CBF or battery.
All owned text files remain below500lines; commit/push current-task notes.

## Review Focus

- Default invocation continues to select the old profile and identity.
- Scaled notebook resolves both algorithms to512/512 and35/25.
- Notebook labels and filenames distinguish the new scenario.
- Dense layouts initialize and produce correctly shaped finite transitions.
- CPU backward checks do not imply full-batch GPU fit or convergence.

## Task1: Profile and notebook boundary

Files: new scaled YAML; ui_notebooks.py; existing build script;
new tests/test_scaled_ui_notebooks.py.
Interface: build_independent_notebook(..., scaled=False); CLI --scaled.

- [x] Write boundary tests executing generated notebook identity and config.
- [x] Run and observe missing scaled API failures.
- [x] Copy only requested scene/width values and experiment metadata.
- [x] Add explicit scaled option, preserving default labels and filenames.
- [x] Run new and existing notebook tests; actual scene/learner CPU checks.
- [x] Run repository suite and report all observed failures.

## Task2: Export and handoff

- [x] Document dimensions, costs and controlled update experiment.
- [x] Review diff; commit source before building frozen archives.
- [x] Export only new distinct notebook files; validate both archives.
- [x] Save user-facing files and push exact evidence/current-task record.
