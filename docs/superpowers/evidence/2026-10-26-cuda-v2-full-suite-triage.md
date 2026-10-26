# CUDA v2 full-suite failure triage

Updated: 2026-10-26T01:46:28.315460+00:00

## Scope

Triage the 14 failures from Kaggle CUDA diagnostic v2 without changing scientific production behavior. v2 was pinned to `9f60ce9ac4bcb313ed26822343231e5e1db3e6e5`; current branch HEAD before this test-only fix was `5cc07ca`.

## Fresh v2 receipt

`outputs/cuda-scenario-audit-20261026/kaggle-v2-latest/cuda-audit-receipt.json` records:

- real hardware: 2 x Tesla T4, Torch 2.11.0+cu128, CUDA 12.8, Warp 1.18.0;
- scenario tests: 280 passed, exit 0;
- production-shaped scenario probes: exit 0;
- four bounded CUDA learner paths: `FOUR_CUDA_LEARNERS_PASS`;
- repository full suite: 834 passed, 7 skipped, 14 failed.

All 14 failures occurred before the intended assertions because CPU-only maintenance/visualization entry points correctly rejected a CUDA-visible host:

- 2 checkpoint archive runner tests: `Checkpoint maintenance must use a CPU-only session`;
- 9 visualization provenance parameter cases;
- 2 visualization resume cases;
- 1 visualization tracking case: `Visualization must run in a separate CPU-only session`.

This is a test-environment classification defect, not evidence that the production CPU-only guards are wrong.

## RED reproduction on current HEAD

A temporary `sitecustomize.py` forced `torch.cuda.is_available()` to return true while running only the 14 affected cases. Result before the fix:

```text
14 failed
```

The failure messages matched the real T4 v2 failure class. The temporary shim was not committed.

## Minimal fix

Only the affected tests declare the same CPU-only constraint already used by `tests/test_cpu_replay.py`:

```python
pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only host")
unittest.skipIf(torch.cuda.is_available(), reason="requires a CPU-only host")
```

Production `uav_network/cloud/archive.py` and `uav_network/cloud/visualize.py` are unchanged.

## GREEN verification

Simulated CUDA host after the test-only fix:

```text
14 skipped in 0.37s
```

Real local CPU host, all four affected files:

```text
23 passed in 4.87s
```

Fresh full CPU suite with `HF_HUB_OFFLINE=1`, `WANDB_MODE=disabled`:

```text
774 passed
81 skipped
8 warnings
1285 subtests passed
exit 0
213.42 s
```

## Ruling

Do not weaken CPU-only production guards. Run a fresh T4 diagnostic on the latest committed HEAD. On CUDA, these 14 cases must be reported as explicit CPU-only skips; the CUDA-relevant suite and scenario/learner probes must still execute normally.
