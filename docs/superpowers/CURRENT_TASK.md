# 2026-11-05 — New-chat handoff prompt created; source unchanged

User requested a complete pasteable prompt for continuing work in a new chat.
Created UAV_NEW_CHAT_PROMPT_2026-11-05.md (347 lines) with verified branch/source
pointers, module map, current resolved XYZ config/dimensions, four-run W&B audit,
LayerNorm changes for both algorithms, checkpoint compatibility, tests and
remaining convergence limits. Temporary sandbox copy:
/tmp/UAV_NEW_CHAT_PROMPT_2026-11-05.md; user received a persistent copy.
No credentials were read or included. No training/live-service status query
or algorithm/config change was performed for this handoff.

Verified source commit remains a5841eab1f328bac0bcd86bc75f18a89adcada5a;
previous continuity HEAD ba495b6863fcd76ab910af3438b81b134900fa81.
Prior full suite for that source:1287 passed,103 skipped,8 warnings,
1624 subtests; CPU integration only, not convergence/GPU validation.
Handoff-only validation: reread source/config/current-task, resolved dimensions
motion3/continuous4/destinations7/observation198/state565, checked text line cap
and git diff --check. No test rerun needed for this documentation-only record.
Next action: follow the user's new request after rereading current source;
do not launch production jobs merely because the handoff mentions old plans.

---

# 2026-11-05 — MASAC/MATD3 hidden LayerNorm completed and pushed

User requested the same actor architecture as MATD3. Current reference XYZ
profile enables masac.actor_layer_norm=true and matd3.actor_layer_norm=true:
two256-unit Linear -> nn.LayerNorm -> ReLU hidden blocks. Both use the same
optional encoder builder. MASAC Gaussian/categorical heads, SAC objectives,
critics, rewards and safety unchanged; MATD3 output_normalization=false.
Default false and missing-field loading preserve original checkpoint layouts.

Final full suite:1287 passed,103 skipped,8 warnings,1624 subtests in332.67s;
JUnit0 failures/0 errors. Log/JUnit:/tmp/masac-layernorm-verified-20261105.log
and .xml. Focused101 checks/556 subtests passed. New tests observed22 RED
failures before implementation and42 GREEN afterward. Actual pre-change
MASAC policy loads with exactly identical deterministic actions. CPU BASE/PQL
smoke each4 episodes/48 transitions/21 updates, finite nonzero actor gradients;
full6 UAV/20 targets/15 obstacles/256x256 retained but reduced2env/horizon12.
This proves integration, not GPU behavior, convergence or long-run saturation.
Fresh read-only whole-change review found0 Critical/Important/Minor issues.

First full run:1286 passed,1 failed in the pre-XYZ compatibility fixture.
It retained the newly added MASAC flag while claiming a pre-XYZ identity.
Reproduced alone, proved config hash accepted and identity matches with
postdated flags removed. Fixed only that test fixture,44 actor/legacy tests
then passed; final suite above is green. Production validation not weakened.

Source commit:a5841eab1f328bac0bcd86bc75f18a89adcada5a.
Pushed to experiment/hfbest-embedded-source-steps1500-20261105 at 2026-11-05T20:26:46.335840+07:00.
Verification above applies to this source commit; this follow-up only records delivery.
Next research action: fresh seed44 training with unchanged other settings;
monitor raw XYZ, saturation, gradients, search and delivery. LayerNorm is not
min-max normalization. ReLU produces nonnegative hidden features, while signed
Linear weights still allow signed actions. Final heads can still saturate tanh.
No production Kaggle training launched in this implementation task.
Plan: docs/superpowers/plans/2026-11-05-masac-actor-layernorm.md.
Evidence: docs/superpowers/evidence/2026-11-05-masac-actor-layernorm.json.

---

# 2026-11-05 — MATD3 hidden LayerNorm completed and pushed

User approved standard nn.LayerNorm in both MATD3 actor hidden layers.
The current reference XYZ profile enables matd3.actor_layer_norm=true;
output_normalization remains false. Each hidden block is
Linear(256) -> LayerNorm(256) -> ReLU. BASE/PQL, actor targets and collector
copies use the same flag. Heads, critics, MASAC, rewards and safety are unchanged.
Default false and missing-field loading preserve old policy architectures.

Final verification: 1263 passed, 103 skipped, 8 warnings, 1623 subtests passed
in316.39s. Full log/JUnit: /tmp/matd3-layernorm-complete-20261105.log and .xml.
64 focused checks passed. CPU BASE/PQL smoke each completed4 episodes,
48 transitions and21 critic updates with finite nonzero sampled actor gradients.
This is integration evidence, not convergence evidence or GPU verification.
Two actual Nov2 W&B policies db61bd0c/d51b0350 retain identical actions after
legacy loading at atol1e-7. Review's historical best-state hash finding was
fixed with8 RED-to-GREEN historical cases. No material review item remains.

Source-size/SHA guards pass. The full suite also exposed a pre-existing gzip
timestamp equality test flaw: the test now validates the actual embedded bundle
and every member's source bytes. Production packaging code is unchanged.
Source commit: 4a5aa227920eea08410d212ff92a8a935d42c0f3.
Pushed to experiment/hfbest-embedded-source-steps1500-20261105 at 2026-11-05T20:02:49.904298+07:00.
Verification above applies to this source commit; this follow-up only records delivery.
Next research action: fresh seed44 training, monitor raw XYZ/tanh derivatives,
actor gradients, search and delivery. Hidden LayerNorm cannot bound the final
Linear head or establish that the old tanh saturation is permanently fixed.
Do not launch Kaggle training as part of this implementation task.

Plan: docs/superpowers/plans/2026-11-05-matd3-actor-layernorm.md.
Evidence: docs/superpowers/evidence/2026-11-05-matd3-actor-layernorm.json.

---

# 2026-11-05 19:16 ICT — Built-in normalization follow-up (research only)

User asks whether standard torch.nn norm blocks can replace bespoke output
normalization and simplify paper reporting. No architecture change authorized
or implemented in this follow-up. Read PyTorch2.11 LayerNorm/RMSNorm/BatchNorm1d
docs via Context7 + official docs; rechecked original sfujim/TD3 actor source.
Hidden nn.LayerNorm(256) is a candidate architecture choice, not a proven cure:
the final Linear can still produce saturated tanh logits. Direct toy autograd
probe [20,-15,-19] gives tanh Jacobian0; pre-tanh LayerNorm/RMSNorm with
affine=false gives nonzero Jacobian (~0.0226/0.0229), but changes action mapping.
LayerNorm(3,affine=false) maps [1,1,1] to zero; RMSNorm boosts small vectors.
Pre-tanh nn.RMSNorm(3,elementwise_affine=False) bounds XYZ pre-tanh magnitude
by sqrt(3), but remains output normalization and is not the old conditional
mean-absolute scaling. Norm after tanh cannot restore its dead derivative.
Standard library modules still require architecture disclosure; no new-method
claim is necessary. Original TD3 actor has no LayerNorm/RMSNorm. TorchRL MLP
permits optional norm_class; this does not establish it as a default TD3 choice.
Next: explain options/limitations; any hidden-LayerNorm change needs a separately
identified architecture and seed44 learning check before calling it a fix.

---

# 2026-11-05 19:04 ICT — MATD3 02-Nov versus 05-Nov regression diagnosed

User identified old runs db61bd0c / d51b0350 and new runs 140e6169 / ead641c4.
All finished 20,000 episodes. Old final 20-episode delivery 59% / 67.75%;
new 0% / 0%, final rolling return -4221.481. Source inspected at da3033d;
old runtime commit 05beef6c; new BASE 574f36c6, PQL ab2e953d. Environment,
communication source and world/sensor/network/energy/reward configs match.

Downloaded actual checkpoints and reproduced dead XYZ gradients on real reset
observations. New raw XYZ ~[+20,-15,-19] -> [+1,-1,-1], FP32 tanh derivative0.
Three reset seeds checked. Same-weight old-normalization counterfactual gives
nonzero derivative; this is not a retraining/convergence proof. New CPU probes
reach y=0,z=0 for all6 UAVs by step30; remain there with no delivery at step300.
Old final models move and deliver in the same bounded seed1000000044 probe.
Main causal attribution still needs one-factor training: output norm, clipping,
Gumbel/peer actions and CUDA build changed together. Do not silently restore
output normalization or call the shared/hybrid implementation exact upstream.

Audit: docs/methods/MATD3_REGRESSION_AUDIT_2026-11-05.md. Durable numeric
evidence: docs/superpowers/evidence/2026-11-05-matd3-saturation-audit.json.
Focused source/math checks:24 passed (reference contract, reference gradients,
MATD3). No algorithm edits or production jobs launched in this audit.
Next if requested: design/run a single-factor seed44 diagnostic with pinned
software; retain declared baseline output_normalization=false and separately
name any normalized-output variant. Follow the user's one-task-at-a-time rule.

---

# 2026-11-05 16:30 ICT — Three PQL job retries submitted successfully

Commit `ab2e953d37520eb5896700f1264d1e4ba80fdfad` contains the corrected
preflight environment isolation. `kaggle kernels push` updated the SAME
notebook refs for cases 02, 04 and 05, each successful, with NO duplicate
new refs. Per-case receipts: `outputs/hfbest-1500-five-20261105/
case-02-preflight-fixed.json`, `case-04-preflight-fixed.json`,
`case-05-preflight-fixed.json`. Source push log:
`/tmp/restart_failed_pql_3_20261105.log`, `RESTART_FAILURES []`.

At 2026-11-05 16:30:14 Asia/Ho_Chi_Minh, Kaggle API reported all five
cases 01-05 `running`. Note this confirms only kernel execution state,
not yet successful CUDA/PQL preflight or W&B metrics. Original Kaggle
PQL smoke failure traces stored at `/tmp/uav-pql-test-logs/pql-smoke-tests.log`
(case 02), and parsed job-level stdout in
`outputs/hfbest-1500-five-20261105/case-{02,04,05}-latest-kaggle-logs.json`.
Follow-up: check 3 newest kernel versions; confirm PQL preflight passes and
then W&B run IDs show rising training episodes. Do not relaunch without a new
observed error; leave cases 01/03 unchanged.

---

# 2026-11-05 16:24 ICT — Three PQL Kaggle failures diagnosed and fixed

Observed via each original Kaggle API account (all five keys valid, quota 29.30, 29.88,
29.36, 29.93, 29.92 hours remaining, account_01..05 respectively): cases 01/03
running, cases 02/04/05 error. For PQL, `kaggle kernels logs` shows the
`test_pql_training.py` preflight exits 1. Downloaded Kaggle output file
`pql-smoke-tests.log` for case 02; it shows 12 tests, 3 failures and 1 error,
all associated with missing `resume_latest.pt` and no calls to PQLSystem.save.
Root cause: parent kernel sets `UAV_TRAINING_ONLY=true`; preflight subprocess
inherits that flag and consequently disables resume/checkpoint writes that its
unit tests explicitly require. This is a test-environment mismatch, not
scientific algorithm code failure.

Fixed only `uav_network/cloud/remote.py`: pass `preflight_env` with
`UAV_TRAINING_ONLY=false` to `unittest discover` subprocesses, leaving the
parent/production learner training-only flag untouched. Added
`tests/test_cloud_preflight_environment.py` which asserts isolation in all
four preflight commands. Focused 11 tests passed; actual PQL unittest
preflight local: 12 tests, 0 failed, 2 skipped due no local GPU.

NEXT: commit/push this patch, repush existing Kaggle notebook versions for
cases 02/04/05 only (do not duplicate notebook slugs and do not modify
cases 01/03). Recheck Kaggle execution logs and verify production W&B run IDs
appear, with episodes/steps increasing. Original W&B planned IDs:
02 `550c3c6b`, 04 `ead641c4`, 05 `78d36ef3`.

---

# CURRENT OPERATION — Five 1500-step Kaggle GPU kernels (2026-11-05 15:46 ICT)

User explicitly authorized submitting all five 1500-step runs. Source code is
embedded in the Kaggle kernels push payload, with no git clone/checkouts. Every
job uses shared actor+critic, seed44, deterministic=true, 20,000 episodes,
CUDA T4x2, periodic validation to select best.pt on Hugging Face; no CPU
visualization, no GitHub Actions continuation.

Job group: `hfbest-shared-seed44-steps1500-20261105`; W&B project:
`truongdaoanhduy-uit/uav_search_target`; Hugging Face model repo:
`duy95/uav-search-checkpoints` (public per existing config).

| Case | Account | Algorithm | Kaggle notebook ref | W&B run ID |
|---|---|---|---|---|
| 01 | account_01 | MASAC BASE shared | thienlong123456/tra-g3-masac-base-h75-11050-6eac30 | 0006c2a7 |
| 02 | account_02 | MASAC PQL shared | bienlong/tra-g3-masac-pql-h75-11050839-44fb15 | 550c3c6b |
| 03 | account_03 | MATD3 BASE shared | fwlejfewf/tra-g3-matd3-base-h75-11050840-5796b3 | 140e6169 |
| 04 | account_04 | MATD3 PQL shared | fewlfj/tra-g3-matd3-pql-h75-11050841-2ed224 | ead641c4 |
| 05 | account_05 | MASAC PQL replica shared | wlfjwfw/tra-g3-masac-pql-h75-11050842-e85c98 | 78d36ef3 |

Submission manifests (not committed) are in
`outputs/hfbest-1500-five-20261105/case-01-submitted.json` through 05, and
retry receipts `case-01-retry.json` through 05.

Important incident:
- Initial submissions used source commit `20f0680` but ALL FIVE Kaggle kernels
  failed pre-training because `uav_network.cloud.remote.run_job()` executes
  several `unittest discover -s tests -p test_*.py` preflight suites, while
  the compressed bundle lacked `tests/`; unittest exit status 5 (no tests).
- Fixed bundle to include all 144 test Python files; commit `574f36c6d36d6182d3204a53f9e497498d3470f8` pushed.
  After fix, repushed all FIVE existing notebook refs (new versions, SAME URL
  and SAME W&B run IDs). All five API pushes succeeded; at the last check,
  all five Kaggle statuses were `running`. The initial versions were `error`;
  do not confuse previous failure with replacement version.
- At last check, W&B run IDs had no visible run objects yet (`CommError` from
  lookup), consistent with CUDA preflight not yet completing. Do not claim
  actual policy training has begun until confirmed from new Kaggle logs/W&B.
- Config hashes for cases 02 and 05 match once `runtime.output_dir` is omitted:
  `ff04b4ebb8895b23deb218a1e36a558d75dad2596433d2f1da19a77814a85ce2`.
- Full local repo suite before the fix: 1243 passed, 103 skipped,
  1619 subtests passed (8 warnings). Focused bundle tests after fix:
  33 passed, 2 subtests passed. Kaggle CUDA execution not yet verified.
- All three GitHub Actions workflows remain disabled.

NEXT IMMEDIATE TASK: poll the FIVE *newest* Kaggle kernel versions and read
`kaggle kernels logs` or Kaggle output files. Determine if CUDA preflight passes,
then verify actual training starts (W&B run present and episodes/global_step
increasing). If errors occur, diagnose exact new log, fix source, rerun focused
tests, commit/push code, replace the existing notebook versions without
creating duplicates; do not silently declare training started. Don't alter
scientific algorithm definitions or the 1500-step matrix without approval.

---

# CURRENT TASK — Kaggle local-source payload, training-only, 1500 steps (2026-11-05)

User explicitly paused all full training and required a complete preflight. Do not
launch any production Kaggle GPU job until the user gives a new explicit command.
All future planned runs use **world.max_steps=1500**, 20,000 episodes, seed 44,
shared actor + shared critic. Planned five cases (not launched):
01 MASAC BASE, 02 MASAC PQL, 03 MATD3 BASE, 04 MATD3 PQL,
05 MASAC PQL replica identical to 02 except its output/run identity.
Training-only is for validated **best.pt -> Hugging Face**, with no CPU visualize
or GitHub Actions continuation. No training was scheduled in this task.

Safety status:
- Deleted the two accidental 1200-step Kaggle kernels on account_01 and
  account_02. Account_05 launch process was terminated before any confirmed
  notebook submission. Old temporary 1200-step launcher was disabled.
- All three GitHub Actions workflows in truongdaoanhduy/uav_network remain
  disabled; no workflow runs queued/running on last check.
- Cleaned key.txt remains account_01..account_05, secret file at original
  external location, never copied into code payload.
- A local private Kaggle --no-run code-payload test was **actually pushed** with
  account_03, pulled back byte-for-byte (467,791-byte kernel.py matching the
  original SHA-256), then deleted. No GPU session started by this smoke test.
- The Kaggle CLI's kernels_push() submits only metadata.code_file, *not sibling
  files* in the local payload folder. To satisfy 'push all source via Kaggle
  API, never git clone in kernel', uav_network/cloud/source_bundle.py selects
  reviewed Python/config/runtime sources plus the existing pinned local
  third_party/UavNetSim (HEAD 04daafb...), makes a gzip archive and embeds it
  base85 inside the generated kernel.py. The kernel verifies SHA-256, safely
  extracts into /tmp/uav_network, installs dependencies via pip and executes
  the source. It contains **no git clone/checkout or repo URL**. No scripts
  containing secrets, private keys, outputs or git metadata are uploaded.
- Kaggle payload source was independently unpacked and imported under a separate
  working directory: Bayesian 3D config resolved max_steps 1500 and native
  UavNetSim simulator loaded successfully.
- Focused tests: 68 passed + 555 subtests, then 9 formerly failing cloud tests
  were traced to subprocess.check_output mocks colliding with local bundling.
  Removed subprocesses from bundle builder; isolated recheck: 32 passed,
  9 subtests. Added future five-case step-1500/replica-equality regression test.
- Full repository test suite PASS: 1243 passed, 103 skipped, 8 warnings,
  1619 subtests passed (265.24 seconds). Real Kaggle T4 execution, installation and full
  20k convergence have **not** been verified by a --no-run smoke.

Source is now being developed on branch
experiment/hfbest-embedded-source-steps1500-20261105.
The old branch experiment/matched-architecture-steps1200-20261105 is preserved.
2026-11-05 latest user explicitly authorized launching all five 1500-step
training runs after local validation. Next: commit/push code, preflight account
quotas, and launch five separate GPU kernels using local-source payload.
Do not report successful launches without Kaggle confirmation.

---

# Latest task — Kaggle training-only, validated best to Hugging Face (2026-11-05)

User requested: first prune/renumber `key.txt` by valid Kaggle GPU quota, then train and publish only the best policy to Hugging Face; remove CPU visualization and GitHub Actions.

Verified 2026-11-05 Vietnam local time:
- Checked ten Kaggle API credentials via Kaggle CLI quota. Old five returned no valid GPU quota; five newly appended keys each returned 30h GPU quota. Rewrote `/home/aduy/Documents/NCKH/uav_network/key.txt` as a single valid, mode-0600 JSON file with account_01..account_05. A private mode-0600 copy of original is kept under `~/.local/share/uav-network/credential-backups/`. Do not commit or print credentials.
- Disabled all three GitHub Actions workflows in `truongdaoanhduy/uav_network` remotely. No GitHub Actions required for future training-only jobs.
- In this experiment worktree, added opt-in `--training-only` to Kaggle launch. It forbids CPU visualization / automatic continuation and requires HF storage plus periodic validation; the notebook downloads pinned GitHub code and uses Kaggle GPU, W&B metrics, HF private credential dataset.
- Training-only suppresses periodic full-state recovery uploads and 17GB resume checkpoint, skips post-training one-episode evaluation and CPU visualization, promotes only validated `best.pt`. Each improved validation-best actor is uploaded to stable HF path `uav-network/<W&B run id>/best/checkpoint.pt`; final W&B artifact points to the same HF checkpoint and does not contain replay/optimizer.
- Target HF repository `duy95/uav-search-checkpoints` is reachable and currently PUBLIC, matching existing config. This task did not change visibility.
- Focused tests: 75 passed, 2 subtests passed. The unrelated cloud-cleanup real-process test initially failed once but passed when rerun alone. Git diff check clean.
- Previously launched MATD3 BASE and PQL runs completed 20k episodes. Original MASAC BASE, MASAC PQL and replica runs were marked crashed at 4096 episodes; do not misreport them as finished.

Next: commit/push new training-only implementation to this experiment branch; launch MASAC BASE, MASAC PQL and PQL replica as fresh jobs on verified newly numbered Kaggle accounts, no GitHub/visualization. Monitor only those new runs. Do not rerun completed MATD3 models.

---

# Latest task — PQL seed-44 reproducibility CPU smoke (2026-11-04)

Without changing algorithm/source/YAML settings, ran the existing deterministic PQL regression test (1 passed) and 8 bounded real-XYZ Warp CPU trainings: MASAC and MATD3, twice with deterministic=true and twice with deterministic=false, seed44. All have 6 UAVs/20 targets/15 obstacles but only 2 environments, 4 episodes, horizon4 and batch4 for quick verification. Deterministic=true produced identical model/episode/settings SHA-256 hashes, critic/actor updates and no concurrent collector for both algorithms. Deterministic=false produced **different model and episode hashes** for both, despite equal settings hashes and update counts, with collector_overlap=true. Not a CUDA T4 or full20k guarantee. Evidence JSON: `docs/superpowers/evidence/2026-11-04-pql-seed44-cpu-repro.json`. The exact test harness and full raw console output remain under ignored `outputs/repro-seed44-20261104/`; no training jobs launched.

---

# Latest task completed — 2026-11-04

On branch `baseline/reference-masac-matd3-20261103`, commit `81902df57fad8bc3ce9cadac8f0f10b099b24a89` returned `compile_models` and `fused_optimizers` to `false` in the canonical reference YAML. AMP remains off, chunk size 0, critic vectorization off and MATD3 output normalization off. Focused tests: 28 passed, 534 subtests passed. No training relaunched. W&B last queried: zero running or pending jobs; GitHub supervisor remains disabled. Compare future source and cloud state afresh. Older notes below refer to earlier milestones.

---

# Source-backed hybrid UAV MASAC/MATD3 baseline — VERIFIED

Updated 2026-11-03, Vietnam time.
Branch: baseline/reference-masac-matd3-20261103; runtime/source commit `c0a935ef51159406cb4d5e17ce1d4bf02453c4a4`; parent config commit0baea78.

- User requested critic_chunk_size=0, output_normalization=false, and source-backed baseline equations without bespoke stabilizers. No20k training or cloud job was launched.
- Reference profile: configs/comparison-reference-masac-matd3-xyz-sensing.yaml. Select this explicitly; historical XYZ study profiles still contain their recorded architecture/entropy variants.
- Both algorithms / BASE-PQL config resolution verified: chunk0, MATD3 output normfalse, reference entropy anneal0. MASAC has no custom vector output normalization.
- Reference actors/critics are independent; alpha init1, discrete entropy weight0.2 from BenchMARL, SAC logstd[-20,2], no hardalpha cap, noSACgradientclip. MATD3 uses original noise/delay, replay peers, softGumbel and per-variableclip0.5.
- Author-source review caught an extra agent average before per-variable critic clipping. Fixed in source per-parameter mode; independent2- and6-agent/twin Adam oracles observed RED-to-GREEN. Legacy global mode retains its existing scaling.
- Disabled/per-variable clipping telemetry corrected with observed RED-to-GREEN tests; per-variable grad_norm reports max individual tensor norm.
- All world/sensor/network/energy/reward namespaces match the existing common XYZ scenario. An omitted unsafe_action/shield_intervention pair was restored to1.0/0.5, with an observed config equality RED-to-GREEN.
- This is a source-backed hybrid UAV implementation, not byte-identical vanilla SAC/TD3 or exact Ackermann MPE reproduction. Hard replay/soft categorical training/argmax RF gating and effective conditional entropy targets are explicit task adapters.
- Provenance: docs/methods/REFERENCE_BASELINE_PROVENANCE.md. Primary refs include Fujimoto TD3, Ackermann MATD3 pinnedfd6c7d0, BenchMARL MASAC config, Delalleau HybridSAC1912.11077, FuCoRL2024/PMLR2025 Hybrid-MASAC.
- Fresh full CPU suite:1217passed,103skipped,8warnings,1605subtests passed in631.68s; JUnit0errors/0failures. Supplemental final focused check:17passed,534subtests passed, including the extra scenario guard added after full-suite collection. No new GPU memory benchmark or learning-performance claim.
- CURRENT_TASK historical508lines archived without byte loss into the adjacent two history files; source-size test now passes.
- No merge to main or historical supervisor migration. Preserve worktree and select the reference profile for future authorized runs.

Historical record, concatenated exactly in order:
[Part1](CURRENT_TASK_HISTORY_2026-11-03_1.md) | [Part2](CURRENT_TASK_HISTORY_2026-11-03_2.md).

[Final verification evidence](evidence/2026-11-03-reference-baseline-verification.json).
