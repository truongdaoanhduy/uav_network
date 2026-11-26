# Current task — V82 validation of V81 (26 November 2026)

Active branch: experiment/v82-evidence-audit-20261126.
Implementation commit: e81bab955cde37c87addeca606906040fa4e93fd.
Active source: experiments/v80_vast_20261126. Its runner filename remains run_v81_real_rollout.py, reporting version V82.
The V81 ZIP supplied by the user is preserved immutably outside Git; input SHA256 024a9db5af8b04c945bd6e1b5a9b4542cb6af630bcee4b0bdaa994ef5b6dd2b5.

Read docs/methods/v82_20261126/01_AUDIT_VI.md and its 24-issue CSV before changing science.
CPU:424 passed,5 skipped,100 subtests. CUDA isolated physical GPU1:425 passed,4 skipped,100 subtests.
Active source manifests296+122; policy298 files/0violations; all owned files<=500 lines.
Verified dead/live pair detector with V80 RED and V82 GREEN; strict bool/norm/warmup runner contracts; real numerical controls and legacy checkpoint migration.
DS400-cell same-ledger reference errors<=3.1e-6. This does not prove paper max-count/aging or upstream SimPy mission parity.
V81 original E64/H64/B128/s300: MATD3 safety66, MASAC5 (V80 history38/3; budgets/configs differ).
V82 policy E64/H64/B128/s900: MATD3 safety210/horizon45, MASAC safety5/horizon59;837 updates each; zero all15-target success.
Final E4096/B8192/H64/horizon900 smoke:3steps/two updates each algorithm, not a900-step training run.
All run seeds44; no claim of multi-seed confidence, convergence, full upstream parity, T4 V82,2GPU or formal viability.
Do not re-label candidate safety flags as committed physical crashes; fail-closed prevents the bad move.
No CBF was reintroduced, no physics threshold weakened, no Actor truth input added, no Kaggle job submitted.

External evidence directory on Vast: /workspace/uav_v82_verified_audit_20261126.
Final deliverable: UAV_V82_V81_REVIEW_CODE_LOG_KAGGLE_EVIDENCE_20261126.zip.
Next: physical viability/peer observation gates, pinned upstream event comparison, one-factor learner/reward ablations and held-out evaluation before full-budget training.
Maintain prior experiment history in Git. Never push main or force-push. Never include key.txt or credentials.
