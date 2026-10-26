# Accelerated 20k cohort readiness preflight — NOT READY

Date: 2026-10-27 (Asia/Ho_Chi_Minh)

## Scope

Read-only operational preflight before any new accelerated 20k submission. No historical run/controller was cancelled, restarted, resumed, or mutated. No new 20k kernel was submitted.

Candidate source: current branch after measured acceleration policy. Canonical profile remains `configs/comparison-20k-hybrid75-1kb.yaml`.

## Existing G3 lineage state — fresh W&B + Kaggle agreement

Historical source for the eight lineages is `77a1d0f52ed4e78bc2823c3ada03d661568cd923`.

Fresh W&B query and exact Kaggle kernel-status query agree:

| Case | W&B training | Kaggle training | W&B controller | Kaggle controller | Progress evidence |
| --- | --- | --- | --- | --- | --- |
| fullgpu MASAC BASE `25eecfd1` | running | running | running / training | running | global_step ~14.43M |
| hybrid MASAC BASE `63202ad9` | running | running | running / training | running | global_step ~16.16M |
| fullgpu MASAC PQL `3b39b0c8` | running | running | running / training | running | global_step ~15.97M, updates ~30.9k |
| hybrid MASAC PQL `6f0bcde3` | running | running | running / training | running | global_step ~13.71M, updates ~26.5k |
| fullgpu MATD3 BASE `93fdf127` | finished | complete | finished / complete | complete | 20,000 episodes, global_step 35.26M |
| hybrid MATD3 BASE `ce8f3be6` | finished | complete | finished / complete | complete | 20,000 episodes, global_step 35.67M |
| fullgpu MATD3 PQL `76b73021` | finished | complete | finished / complete | complete | 20,000 episodes, 69,424 updates |
| hybrid MATD3 PQL `3fcf25c7` | finished | complete | finished / complete | complete | 20,000 episodes, 68,784 updates |

The W&B project currently has eight `running` rows corresponding exactly to the four MASAC learners plus their four controllers. Two reads during the audit showed MASAC global steps increasing, so these are not merely stale W&B rows.

Therefore accounts16–19 and those four MASAC lineages must remain untouched. The completed MATD3 lineages are historical results and must not be exact-resumed with new acceleration flags.

## Canonical production configuration

Fresh resolution of the canonical profile:

- episodes=20,000
- max_steps=1,800
- envs=4,096
- seed=44
- held-out eval seed=2,000,000,044
- final eval episodes=30
- 35 targets / 25 obstacles / 6 UAVs
- report/payload=1 KiB
- batch=8,192
- replay=1,000,000
- replay layout CPU75/GPU25 hybrid
- learning_starts=131,072
- updates_per_step=8
- checkpoint store=Hugging Face
- `tracking.log_checkpoint=true`
- `pql.checkpoint_replay=true`
- tracking group=`uav-v3-full20k-e4096-b8192-r1kb-cpu75-accel-policy-20261027`

Measured acceleration policy resolves to:

- MASAC BASE: FP16 + compile ON + fused OFF
- MATD3 BASE: FP16 + compile ON + fused OFF
- MASAC PQL: FP16 + compile ON + fused OFF
- MATD3 PQL: FP16 + compile OFF + fused OFF

This policy has already passed local full regression and real 2xT4 bounded smoke; see `2026-10-27-measured-acceleration-policy.md` and `2026-10-27-acceleration-policy-kaggle-smoke.md`.

## Account readiness

Quota metadata alone is not accepted. Each production account must also authenticate now, list kernels, be idle, and access its owned canonical private W&B credential dataset.

### Fully verified idle accounts

- `account_01` (`duytrngoanh`): ~23.42 GPU-hours at the quota snapshot; no active kernel; canonical W&B credential dataset files accessible.
- `account_02` (`vuliu123456`): ~25.03 GPU-hours; no active kernel; canonical W&B credential dataset files accessible.

### Not production-safe now

- accounts16–19: actively own the four still-running MASAC G3 lineages/controllers; must not be reused.
- account_24: previously showed high quota but current token no longer authenticates; username resolution fails and dataset-files access returns 403.
- account_07: ran the bounded smoke successfully earlier, but a fresh credential-path check now fails authentication; do not use until credentials are replaced/revalidated.
- account_13: similarly produced a successful smoke earlier but later fresh auth checks failed; do not use now.
- accounts03,04,05,06,08,09,10,11,12,14,15: fresh authentication checks failed during this audit.
- accounts20–23: their old MATD3 kernels/controllers are complete, but fresh current-token checks fail authentication, so they cannot be reused now.

Thus there are only **two** fully verified idle production accounts, not the required four independent accounts.

## Decision

**NOT READY for a fresh four-mode simultaneous accelerated 20k cohort.**

The blocker is account credentials/capacity, not model code, CUDA execution, scenario generation, replay layout, or the measured acceleration policy. Launching only two modes, reusing accounts16–19, or silently relying on quota metadata would violate the requested four-account isolation / no-duplicate rule.

No long-running job should be submitted until at least two additional Kaggle accounts pass all four gates: current authentication, sufficient quota, zero active kernels, and canonical credential-dataset access.

## Safe next action

Replace/refresh at least two invalid Kaggle API tokens in the local credential file (without exposing them in chat), then rerun this same account preflight. Once four accounts pass, launch four fresh episode-0 lineages separately (one algorithm/mode per account) with auto-continuation and CPU visualization enabled; never warm-start the old off/off cohort.
