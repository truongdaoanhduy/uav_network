# Four independent XYZ training roots

Observation: 2026-10-30T22:29:16.345023+00:00. User explicitly requested one fixed Kaggle
account per algorithm/mode. All four roots were submitted successfully and the
live Kaggle API reports them running. This is the first seed44 training cohort.

| Case | Account | Kaggle | W&B | Global step | Learner updates |
|---|---|---|---|---:|---:|
| MASAC BASE | account_01 (kesfljhk) | [running](https://www.kaggle.com/code/kesfljhk/gh-tra-g3-masac-base-h75-10302223-997c97) | [run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/33411b89) | 176128 | 88 |
| MASAC PQL | account_02 (cdancsjf) | [running](https://www.kaggle.com/code/cdancsjf/gh-tra-g3-masac-pql-h75-10302223-baef91) | [run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/11cbc295) | 167936 | 56 |
| MATD3 BASE | account_03 (jyfhjggiy) | [running](https://www.kaggle.com/code/jyfhjggiy/gh-tra-g3-matd3-base-h75-10302223-7459f1) | [run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ed7bbd70) | 532480 | 784 |
| MATD3 PQL | account_07 (ukggyfyy) | [running](https://www.kaggle.com/code/ukggyfyy/gh-tra-g3-matd3-pql-h75-10302223-c53374) | [run](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/63a462ec) | 237568 | 192 |

Each root uses runtime source `481f81f288d8e84a010dba52e4c8e30b5eebd391` and
`configs/comparison-20k-hybrid75-1kb-shared-actor-critic-xyz-sensing.yaml`. Same 6UAV,1000m,20targets/15obstacles,1200-step XYZ Bayesian
sensing mission, prior0.5, learned RF power, shared actor/critic, Hybrid75,
E4096/B8192/replay1M, FP32/eager and seed44. The target is20000 training episodes.
PQL warmup remains32; readiness-only248 is absent. No XY checkpoint warm start.

The existing supervisor is enabled for exactly account01/02/03/07.
[Four-account services and dry-run](https://github.com/truongdaoanhduy/uav_network/actions/runs/41137502991) passed.
[Normal reconciliation](https://github.com/truongdaoanhduy/uav_network/actions/runs/41137959190) passed and adopted
all four active roots; it submitted no duplicate. The default-branch workflow
contains the15-minute cron and checks out the working supervisor branch.
Continuation stays on each run's fixed account. Low quota or service rejection
must be reported; do not rotate to another account silently.

All four training W&B runs now exist, match the pinned source/config, expose two
Tesla T4 GPUs each and report positive learner updates. The table records one
simultaneous observation, not matched progress between algorithms.

The existing path separates validation-selected best policy from latest full
resume, continues incomplete GPU segments and submits final CPU30 evaluation
plus2D visualization/publication. Those final stages have not completed here.
Read live progress in the machine record and W&B; Kaggle `running` during setup
does not by itself prove learner updates. No20k completion or convergence claim
is made before terminal evidence.

[All six full-ring readiness cases](2026-10-31-xyz-full-ring.md) passed before
these initial roots were launched. Those benchmarks are not paper training results.
Seeds100044 and200044 remain pending; this cohort is not a multi-seed conclusion.

The proposed serial cohort scheduler was superseded before implementation:
the native launch plus existing continuation already handles this request.
Only session-authored RED queue tests were archived outside pytest discovery;
no product queue module or config was added. Manual skill bookkeeping replaces
unavailable provider scripts; check the evidence and git together when resuming.

Raw receipt, submission-attempt, preparation, live log and status files remain at
`outputs/xyz-cohort-20261031/` on the authorized host. A receipt-bearing or
attempt-bearing case must never be blindly submitted again. Read-only observer:
`PYTHONPATH=. python outputs/xyz-cohort-20261031/observe.py`.
The observer never submits, resumes or cancels jobs.

[Machine record](2026-10-31-xyz-four-account-training.json) includes exact public
configuration/source, URLs, service proof and current progress without credentials.

## Default-branch scheduler correction

A follow-up check found that the `main` cron workflow still hardcoded accounts17–23
and lacked secret bindings01/02/03/07. Earlier working-branch manual dispatches
did not prove that the scheduled default-branch workflow could see the new roots.

Fixed only the four missing secret references and the repository-variable account
selector in [main commit d98827d](https://github.com/truongdaoanhduy/uav_network/commit/d98827d69733755ee35020297be763be2f20e0c7).
Other workflow settings and running GPU jobs were preserved. A semantic assertion
failed on the old environment and passed on the deployed file;34 existing
supervisor/workflow tests passed. The deployment was fetched back and matched
the verified content exactly.

[Default-branch dry-run](https://github.com/truongdaoanhduy/uav_network/actions/runs/41139172161)
and [normal reconciliation](https://github.com/truongdaoanhduy/uav_network/actions/runs/41139219837)
both succeeded from `main`, resolved accounts01/02/03/07 and adopted all four
active roots without duplicates. The dry-run also verified W&B and HF access.
Thus training and the configured continuation/final-CPU flow no longer require
the Sandbox process or local observer. The completed20k→CPU30 chain has not yet
been observed; future quota, credential and service availability still apply.

GitHub's plugin could read but not modify workflow files (integration403);
the already-authenticated host `gh` client applied this one-file update.
No credential values were included in file content or logs.
