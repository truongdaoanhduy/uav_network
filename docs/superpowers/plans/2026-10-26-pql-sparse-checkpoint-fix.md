# PQL sparse checkpoint storage fix

User-authorized minimal fix. Retain no-deletion restriction for W&B/HF, all scientific settings and checkpoint schema. Existing linked feature worktree is used.

1. Reproduce actual serialized capacity growth with real CPU/CUDA tests; preserve RED logs.
2. Compact only populated transition/born/seen storage on CPU, without extra full replay allocation on CUDA. Preserve device/RNG/layout and full-ring synchronous borrowing.
3. Verify serialized-size independence and next-sample/measurement exactness; run fullCPU and fullCUDA suites, plus4096PQLresume cases. Compare to prior8.94GBcheckpoint.
4. Record source/tests/evidence and unresolved limits, commit/push and verify remote. No data deletion, hyperparameter changes or production cohort.
