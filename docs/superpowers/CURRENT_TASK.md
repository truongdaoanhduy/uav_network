# Current task — V84 independent validation of V83 (26 November 2026)

Active branch: experiment/v84-v83-independent-validation-20261126.
Active source: experiments/v80_vast_20261126.
Runner filename remains run_v81_real_rollout.py; version V84.
Final implementation:13b6cc499883853f1178b5833c8492c23db534e9.
Earlier replay/geometry implementation:093d304434381c62df1e693152179fa737decbfe.
Base V82:22fa5c2115fd57185f395a27bdf9f62bdab60d3a.
User V83 ZIP SHA256:d14f42fcff542bf0a5ea3594212375383d56e0c77d938ce44c8c05ad1a2dfc33.

Read docs/methods/v84_20261126/01_V83_AUDIT_V84_FIXES_VI.md and the8-claim/24-issue CSVs.
V83 L2 wall braking is correct within a sufficient collinear wall-backup set;2/300 old failures→0/300.
CPU/WarpCUDA each10000 state checks passed; not a proof of joint viability.
Raw replay is valid when shield is part of environment; dual is an experimental formulation.
Six V83 replay/warmup/logging bugs were reproduced RED and fixed.
A seventh inherited diagnostic bug used ReLU for SiLU Actor probes; corrected and4 activation/shared cases pass.
Actual Actor output, learner state and RNG were checked; old SiLU pre_tanh probe values must not be treated as actual Actor measurements.

CPU439 passed/5skipped; gatedCUDA440 passed/4skipped;100subtests each.
Active manifests301+122, policy303files/0violations; source<=500 lines.
30 current cohorts:6 immutable V83,12 V84 pre-diagnostic fix,12 final V84.
Final source frozen and clean for all12 final runs; counters/window sums and real-update budgets verified.
E64/300 final MATD3 raw21/dual2/raw_penalized11 safety failures; MASAC3/3/3; all boundary0.
E64/900 dual:MATD3 safety5/full15success4/horizon56; MASAC safety3/full15success0/horizon62.
All safety failures above are pair flags. Detector prevents unsafe physical commit.
E4096/B8192/3steps raw/dual both algorithms:2 real learner updates each, no safety flags.
No held-out evaluation, multi-seed confidence, full SimPy event parity,2GPU,T4V84 or fullE4096×900 proof.
Historical64 Kaggle archive/member records contain32 distinct payload hashes, not64 independent runs.
No new Kaggle job or credential access.

External evidence:/workspace/uav_v84_v83_verified_20261126.
ZIP:UAV_V84_V83_VERIFIED_CODE_LOG_KAGGLE_EVIDENCE_20261126.zip.
Next gates:joint safety state bank, HELLO observation schema, chosen replay MDP,
controlled reward/belief/RF ablations, upstream packet traces, held-out multi-seed evaluation.
Preserve prior history. Dedicated branch only, never main or force push. Never include credentials.
