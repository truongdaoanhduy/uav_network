# Independent repeated final audit — no-oracle MATD3 / Warp UavNetSim (2026-11-22)

## Scope and reproducibility

- Source branch: `experiment/18d756ef-nooracle-fullflow-20261122`
- **Executed functional code commit**: `bdf3190` (the audit report was added afterward, without code edits).
- Development machine: Ubuntu; PyTorch `2.7.1+cu126`; Warp `1.18.0`; **no working CUDA driver**. Native Warp behavior was executed on the CPU backend. This is **not** a T4x2 CUDA certification.
- M2 profile: `configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml`.
- Production profile: 6 UAV, 35 stationary ground targets, 25 obstacles; 1500 maximum steps per episode; E4096/U4/B8192; replay 1,000,000 transitions with 75% CPU and 25% GPU; MATD3 Actor BatchNorm H256x512, plain Critic. Full production GPU training **not executed in this local audit**.
- Git working tree was clean before and throughout both functional pytest suites.

## Repeated full functional suite: two independent Python processes

Both repetitions ran exactly:

```bash
python -m pytest -q tests --ignore=tests/test_project_structure.py --disable-warnings
```

| Run | Outcome | Skipped | Warnings | Subtests passed | Duration | Exit |
|---|---|---|---|---|---|---|
| 1 | **1,602 passed; 0 failed** | 105 | 8 | 1,111 | 346.58 s | 0 |
| 2 | **1,602 passed; 0 failed** | 105 | 8 | 1,111 | 342.01 s | 0 |

First run: `/tmp/matd3_v7_user_reaudit_full_round1.log`.
Second run: `/tmp/matd3_v7_user_reaudit_full_round2.log`.

## Independent additional tests (distinct from pytest)

1. `python scripts/audit_nooracle_horizon_20261122.py --envs 3 --repeats 15`:
   **15/15 batches, 45/45 native Warp CPU mission worlds passed**, with up
   to 1500 steps/world. Assertions checked finite sensor/energy/reward state,
   completed worlds, search/delivery consistency, total reporting, coverage
   percent and packet accounting. Random action policies yielded 12–27
   true targets found, 8–23 delivered, and 0–9 false confirmations per world.
   **These are simulation consistency results, not MATD3 policy scores.**
2. Independent full-horizon bitwise metamorphic comparison of two native Warp
   worlds with identical seeds `[44, 45]` and identical random action streams.
   A **third environment was restored from a real mid-episode WorldState
   snapshot at step 350**. All three subsequently produced identical actions,
   Actor observations, Critic state, reward, Bayesian belief, battery,
   reports, delivered flags, packet counters and sensor counters up to step
   1500. The 167-array state snapshot was restored without recomputation.
   Result: `FULL_1500_STEP_DETERMINISM_SNAPSHOT_RESTORE_PASSED 1500`.
3. `python scripts/audit_nooracle_train_validation_20261122.py`: **3/3
   small CPU MATD3 training/validation tests passed**, each with 132 real
   optimizer updates, 10 Actor snapshots, and 100 fixed-seed evaluations
   for each of two selected snapshots. Repeat training with seed44 produced
   byte-identical Actor hash:
   `b817bfd0111d2e6a4eaf3d60635772c11cc27da091e1e2347820a4ef2066e369`.
   Seed45 produced a distinct hash. This does not establish learning quality,
   CUDA numerical equivalence, or successful final Test4000.
4. `python -m compileall -q uav_network scripts notebooks` and
   `git diff --check` succeeded. The Kaggle UI notebook's embedded source
   archive SHA256 validated, and the `matd3.py`, `masac.py`, `mac.py`,
   `bayesian.py`, `training.py`, and `best_checkpoints.py` bytes inside it
   matched the corresponding files in the audited repository.
5. Pinned legacy models remain unchanged:
   - MATD3 SHA256 `e106277d2695936d983e2b558ad5cefac7a1015b24cca3b8032c727563bfeefa`
   - MASAC SHA256 `7e5fd9cb00c664bbea88321dcfdd3d0b75527c992b041a1a42434162a92796ce`

## Remaining known limitations — do not call the whole repository error-free

The repository's **separate strict 500-lines-per-file structure gate still fails
in seven pre-existing files**, although its checksum and structural checks have
other passing subtests. Run:

```bash
python -m pytest -q tests/test_project_structure.py
# 7 failing subtests (line-count contract)
```

The oversize files are `uav_network/configuration.py`,
`uav_network/experiments/training.py`,
`uav_network/experiments/deferred_gpu_evaluation.py`,
`uav_network/experiments/tracking.py`,
`uav_network/experiments/cause_evidence_report.py`,
`tests/test_tracking.py`, and `docs/superpowers/CURRENT_TASK.md`.
This is a maintenance/structure contract violation rather than a reproduced
math or packet-simulation runtime failure. An intrusive refactor merely to
make these files shorter was deliberately not mixed into the frozen-algorithm
final verification.

`motion_safety` is **`reject` in the audited production profile**, not DWA.
3D `dynamic_window` explicitly raises `NotImplementedError` and remains outside
this branch's supported configuration. 3D obstacle reconstruction and central
GCS DS-belief-map synchronization are also out of scope.

Due to missing local CUDA, no claim of T4x2 performance, 4096-CUDA-environment
memory safety, true 20,000-episode convergence, or 4,000-episode held-out final
test success can be made. No one can guarantee the absence of all unknown bugs;
within the executed CPU-backed functional and stress coverage, **no new
reproducible functional defect was discovered**.
