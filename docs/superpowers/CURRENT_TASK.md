# 2026-11-08 — PAPER-CORE CTDE HYBRID BASELINE CONTRACT + 2 UI BUILDERS

**User asked for an actual source-faithful baseline, not another
posthoc W&B recommendation.** Created separate strict 35T25O config
`configs/comparison-paper-core-35t25o-xyz-sensing.yaml`:
6 independent local Actors, 6 independent centralized twin Critics,
No Actor LayerNorm, no PQL, mask-aware/alpha MASAC, MATD3 original
delay2/twin min/Gaussian smoothing/replay peer actions/default init;
H512×512, U8, B8192, replay1M, seed44. World/CBF/reward/network
identical to preexisting independent H512 scenario. Not source-paper
byte-exact because the domain has hybrid XYZ/RF + masked destination
and uses full global CTDE state for Critics.

**Prepared verification** `scripts/verify_paper_core_baseline.py`,
offline contract JSON `docs/methods/evidence/paper_core_35t25o_contract_2026-11-08.json`,
unit/math/negative tests `tests/test_paper_core_35t25o_baseline.py`,
and UI spec tests `tests/test_paper_core_ui_notebooks.py`.
Added opt-in `paper_core=True` to existing
`uav_network/cloud/ui_notebooks.py`; default legacy notebook
generation remains unchanged. `scripts/build_paper_core_35t25o_ui_notebooks.py`
writes two new source-pinned UI notebooks **after commit**. Do not
submit any training automatically. Historical ten W&B experiments
remain archived, unmodified, and are NOT the same strict-U8-NoLN run.

**Actual algorithm equations:** audit
`docs/methods/PAPER_CORE_VS_EMPIRICAL_BASELINE_35T25O_2026-11-08.md`
with source references for MASAC and MATD3. Since original code's
math update functions passed source-oriented numerical tests
and source commit comparison revealed only comment/docstring edits,
we have NOT modified the algorithm loss/smoothing/discrete logic.
Do not claim an exact copy of Ackermann's original MPE code or
Fu's attention critic, or that shared/LN belongs to the
paper-core architecture. Existing Shared H512 U8 ActorLN MATD3
`ccaadcd6` remains the empirically best tested 20k run and
MASAC Shared H256 U8 ActorLN `7f5fd73f` a completed measured
baseline; neither run relabeled.

**Next step if requested after this:** full-strict source-aligned
baseline experiments are new, distinct 2-run cohort, potentially
computationally expensive, only when explicitly approved. Do not
run the same earlier ten jobs again. Retain W&B/HF evidence and
do not rewrite loss based only on Q(soft)-Q(hard) speculation.

---

# 2026-11-08 — COMPLETED: checkpoint-based MATD3 Gumbel/hard action support diagnostic

**Task closed without algorithm change or new training.** After the completed
10-run W&B analysis, inspected live original W&B model artifacts and pinned
Hugging Face best snapshots `ccaadcd6` (MATD3 Shared H512 U8 ActorLN)
and `635d7134` (Shared H256 U8 ActorLN). Both checkpoint files
have verified SHA256 and type `purpose=evaluation_only` with only
`actors` tensors (12), zero trained Critic tensors. The W&B model artifacts
have only `checkpoint.json` pointing back to these same HF best policies.
**A real trained Q(soft)-Q(hard) result is therefore not available from
the saved best-checkpoint evidence.** No random Critic Q numbers were used.

Executed read-only rollout of the exact Warp CPU backend on the 5 fixed
validation seeds 1000000044..1000000048, deterministic trained Actors,
one Gumbel(tau1,hard=False) per observation and one-hot of same argmax:
- `ccaadcd6`: 32,484 UAV observations; 338 with >1 valid destination
  (1.04%), soft-hard eligible L1 mean 0.03724, max 0.86058.
- `635d7134`: 32,544 observations; 348 with >1 valid destination
  (1.07%), soft-hard eligible L1 mean 0.01076, max 0.78315.
Both show mostly forced-silent masks. Action mismatch is **not** Q error.

Read the detailed
`docs/methods/MATD3_TRAINED_ACTOR_GUMBEL_SUPPORT_AUDIT_2026-11-08.md`
and numeric evidence
`docs/methods/evidence/matd3_trained_actor_gumbel_support_2026-11-08.json`.
Reproduction script `scripts/diagnose_matd3_gumbel_support.py` and test
`tests/test_matd3_gumbel_support_diagnostic.py`. Original downloaded
Actor checkpoints and raw outputs stay local under
`outputs/wandb-ten-20261108/diagnostics/`; do not commit weights or secrets.

**Current decisions:** Shared Actor + Shared Critic, Actor LayerNorm on
but no isolated Shared LN benefit evidence, Critic LayerNorm off, U8 over
U16 for the studied 35T25O/seed44 setup. Do NOT change MATD3 Gumbel or
other algorithm mathematics. Do NOT repeat the original ten training runs
unless scientific code/config/scenario changes; no run started here.

**Next research task if user explicitly requests:** look for a *real
historical full-learner critic checkpoint* if any (do not confuse
W&B best actors with trained critics); otherwise study the existing
delivery and admissible-link/mask diagnostic metrics without training.
Any future optional critic-state snapshot must be a newly named
instrumentation capability, not a silent change to baseline objectives.

---

# 2026-11-08 — COMPLETED live comparison of 10 latest 35T25O BASE W&B runs

**Current task finished; do not retrain duplicates.** Read
`docs/methods/WANDB_10_LATEST_35T25O_COMPARISON_2026-11-08.md`,
the machine-readable `docs/methods/evidence/wandb_latest_ten_2026-11-08.json`,
and `docs/methods/evidence/wandb_10_case_reuse_registry_2026-11-08.json`
before discussing MATD3/MASAC 35T25O runs.

Read-only direct W&B API snapshot on 2026-11-08. Ten most recent BASE runs:
MATD3: 01 `ccaadcd6` (20k), 02 `f55774e2` (20k),
03 `55158605` (16384), 04 `baf27732` (20k),
05 `635d7134` (20k). MASAC: 01 `d9c1c5e9` (11675),
02 `c70aa6f4` (4096), 03 `70eaad3a` (4096),
04 `c6cf7916` (4096), 05 `7f5fd73f` (20k).

All ten W&B source commits = `a6e07a8f4b199028298b6ffd9a47f2cecd654fc9`;
world/sensing/network/energy/reward scientific SHA-256 and runtime
protocol hashes match across all ten. Only five reached target 20k,
despite W&B state `finished`. At all-run common comparison point:
last 1024 training episodes through episode4096; independently,
5-episode fixed-seed validation at checkpoint transition 7,372,800.
Pairwise comparisons instead use each pair's minimum completed
episode and latest exact shared validation checkpoint. No future-point
lookahead; best checkpoints across unequal training horizons are
not a fair same-budget ranking.

**Observed decision** within seed44, 5-episode validation:
MATD3 Shared H512 U8 ActorLN `ccaadcd6` is primary
(best validation delivery58.86%, final58.29%, W&B lifetime2.63h);
MATD3 Shared H256 U8 ActorLN `635d7134` is faster
(20k in1.74h, best51.43%); MASAC Shared H256 U8 ActorLN
`7f5fd73f` is the only completed 20k MASAC
(best52.0%, lifetime8.33h). MATD3 Shared H512 U8 beats
Shared H512 U16 by13.71 delivery percentage points at same
23,961,600-transition checkpoint, and the U8 run completed faster.
Independent U16 MATD3 substantially underperformed Shared U16
under pairwise matched comparisons. MASAC H512 U8 improved later but
stopped at 11,675, so do not claim a 20k ranking for it.

**Best checkpoint metadata:** all ten W&B-reported Hugging Face pinned
revisions and `uav-network/<run_id>/best/checkpoint.pt` file entries
verified read-only on 2026-11-08, see
`docs/methods/evidence/wandb_ten_hf_checkpoint_verification_2026-11-08.json`.
Full checkpoint bytes were not downloaded/hashed.

**IMPLEMENTED:** Read-only capture/analysis script
`scripts/audit_latest_ten_wandb_baselines.py`; offline
report/figures/registry script
`scripts/render_wandb_ten_case_registry.py`;
regression test `tests/test_wandb_ten_run_audit.py`.
No Kaggle training, W&B changes, HF mutation or scientific code changes.
Raw history cached locally under `outputs/wandb-ten-20261108/`
and dated copy in `~/Downloads`. Next chat must reuse these records
instead of re-running identical experiments. If sources or scientific
settings materially change, treat as a new cohort with its own fingerprint.

---

# 2026-11-08 — Spinning Up SAC/TD3 → CTDE Hybrid-MASAC/MATD3 source audit

User authorized proceeding from a research-only review to an implementation/audit phase,
with a one-task-at-a-time rule and no unverified baseline algorithm or reward changes.
Work verified in sandbox at initial HEAD `cc1885f` on
`experiment/hfbest-embedded-source-steps1500-20261105`. Initial worktree was clean.

**IMPLEMENTED in commit `ec2e069`:**

- Added `tests/test_paper_faithfulness_audit_20261108.py` (5 executable regression
  tests) for MASAC shared-critic exact conditional expectation against a nonlinear
  joint-Q reference, MATD3 target noise on all UAVs in both parameter-sharing
  topologies, soft-vs-hard categorical critic input, and replay command under CBF.
- Added source-level provenance comments to `masac.py` and `matd3.py`, without
  modifying their mathematical expressions or training behavior.
- Added `docs/methods/SPINNINGUP_SAC_TD3_MARL_AUDIT_2026-11-08.md`
  with paper/source findings and the remaining research caveats.

**VERIFIED:**

- Reference: OpenAI Spinning Up SAC/TD3; Ackermann MATD3 2019 Eq. (5).
  MATD3 paper smooths **all** agents' target actions, as current implementation
  does, even though one author TF2 implementation smooths only the learner's own.
- Shared-MASAC categorical estimator matches the exact joint categorical
  expectation on a non-additive synthetic Q under a factorized joint policy.
- Soft Gumbel Q inputs and hard replay one-hot are a documented representational
  difference, **not** a proven correctness bug or reason to change baseline.
- The MDP action stored in replay is the issued pre-CBF command; safety filtering
  lives in the environment transition. A CPU boundary test confirms this.
- Regression command:
  `python -m pytest -q tests/test_paper_faithfulness_audit_20261108.py tests/test_algorithm_math_regressions.py tests/test_reference_baseline_contract.py tests/test_reference_critic_gradients.py tests/test_parameter_sharing.py tests/test_matd3.py tests/test_masac.py tests/test_split_updates.py tests/test_xyz_cbf.py tests/test_xyz_cbf_warp.py tests/test_masac_ui_ablation.py tests/test_matd3_ui_ablation.py`
  → **118 passed, 43 subtests passed**. `python -m py_compile` and
  `git diff --check` also passed.

**NOT CHANGED:** No policy/critic loss modification, no Gumbel estimator switch,
no reward, world, CBF, observation, replay schema, seed 44, hyperparameter,
notebook, Kaggle execution, cloud storage or checkpoint alteration.
No full GPU-training benchmark was run.

**NEXT ACTION IF REQUESTED:** For a fixed trained checkpoint/replay minibatch,
measure MATD3 Q(soft Gumbel) versus Q(hard one-hot) and assess any downstream
bias. Consider any straight-through estimator only as a separately named
ablation; do not silently modify BASE.

---

# 2026-11-08 — Double validation cadence BEFORE starting 10 Kaggle UI runs

User authorized changing validation frequency immediately instead of waiting for training results. Started ~08/11/2026 02:42 UTC+7; verified 08/11/2026 02:46:28 UTC+7. **Active 20k baseline configurations now use `tracking.evaluation_interval_steps=1843200`**, exactly half of old 3686400. Changed only three current profiles: `comparison-independent-35t25o-h512-xyz-sensing.yaml`, `comparison-independent-baseline-xyz-sensing.yaml`, `comparison-reference-masac-matd3-xyz-sensing.yaml`. Did not change old/historical 20k/25k profiles. Kept 5 periodic validation episodes, fixed seeds 1000000044..48, evaluator timeout 900 s, best criterion, CPU async evaluator, checkpoint cadence256 and max20k episodes. No reward/algorithm/CBF/world/observation/scientific changes.

Main source code commit **`a6e07a8`** embeds the new cadence into both MATD3 and MASAC Kaggle UI intros + active YAMLs; test expectations updated. **105 passed**, including baseline MATD3/MASAC notebook tests, async evaluator and controlled study; historical `test_controlled_study` remained on original historical profile. Git diff --check clean.

**Re-exported all 10 Kaggle UI `.ipynb` files** on user's laptop `~/Downloads` and in repo ignored `outputs/kaggle` using the updated source commit `a6e07a8`. Replaced same-name earlier `.ipynb` files plus old-name ZIP to avoid accidental stale imports. Created clearly labeled ZIP: `~/Downloads/UAV_MATD3_MASAC_10_KAGGLE_UI_35T25O_S44_VAL1843200.zip` (same copy in `outputs/kaggle`), with manifest `~/Downloads/UAV_MATD3_MASAC_10_KAGGLE_UI_MANIFEST.json`. Verified actual embedded archive YAML values=1843200/5/256, independent case recipe overrides, PQL disabled, compiled notebook cells, schema, 10 ZIP entries, source integrity and copied notebook byte-for-byte. Notebook source commit `a6e07a8` is intentional snapshot even if subsequent continuity note commit advances HEAD. The downloaded files are **self-contained and sensitive source**, never publish to public git. User must import/reimport updated `.ipynb` in Kaggle; previously imported Kaggle notebooks will **not** auto-update. No training submitted and no W&B/HF mutation.

For old actual MATD3 BASE ~21.19M transitions the new interval gives 11 theoretical validation threshold crossings vs 5 old; at max30M transitions new16 vs8 old. These are thresholds, not guaranteed successful validations: single CPU evaluator may skip_busy, deadline, fail/timeout, and terminal threshold may be skipped. Future work only if user requests changes to busy-skip or evaluation method; avoid modifying algorithm/reward or validation protocol otherwise.

---

# 2026-11-08 — Export 10 Kaggle UI notebooks for user (MATD3 BASE x5, MASAC BASE x5)

User explicitly requested stopping reward plotting/changes and preparing (without running) **exactly 10 manually importable Kaggle UI `.ipynb` files**: MATD3 5 architecture ablations and MASAC 5 identical architecture/update cases. Start ~08/11/2026 02:25:48 UTC+7, verified 08/11/2026 02:33:00 UTC+7.

**Implementation commit `89e6274`**, pinned inside every notebook. Only added MASAC Kaggle notebook builder, 10-case export script, tests and guide. No changes to `reward`, algorithm update equations, CBF, world, sensor, episode aggregation, training code, MATD3 builder, validation cadence or checkpoint logic.

Matrix for both MASAC & MATD3: case01 SH H512 U8 ActorLN; case02 SH H512 U16 ActorLN; case03 IND H512 U16 ActorLN; case04 IND H512 U16 ActorNoLN; case05 SH H256 U8 ActorLN. Both Actor and Critic share/independent **across UAVs**, not with each other. Critic never has LN (existing code option actor only). MASAC uses mask-aware target entropy and alpha loss, MATD3 default initialization + policy_delay2. BASE only; PQL disabled.

Actual 10 exported files and one 6.9MB ZIP are ON THE USER'S LAPTOP under `/home/aduy/Downloads/`, duplicated in `/home/aduy/Documents/NCKH/uav_network_study_1200/outputs/kaggle/`; zip exact name `UAV_MATD3_MASAC_10_KAGGLE_UI_35T25O_S44.zip`. Local SHA/file verification manifest in `/home/aduy/Downloads/UAV_MATD3_MASAC_10_KAGGLE_UI_MANIFEST.json`. **Never commit/embed these private notebooks into public repository** — they include full bundled source for manual Kaggle UI, and anyone with notebook access can inspect them.

Source bundle SHA256 `f28cd52ef906e5623a592bffab0822a4b8406bceef5e78de11d89c8ceef96b28`; extracted and compared **all 429 bundled source files byte-for-byte** against repo; validated 10 nbformat schemas, compiled all code cells, full runtime `SPEC` from actual launch cell and reward equality; duplicated Downloads vs outputs/kaggle content hash verified; ZIP contains exactly 10 files and testzip passed. Current full config: 6 UAV, 35 targets, 25 obstacles, 1km, 1500 max steps, 20k episodes, seed44, 4096 env, batch8192, warmup131072, replay1M 75%CPU/25%GPU, no chunk, no compile/AMP/fused, deterministic true, checkpoint every256 episodes, fixed validation 5 episodes per3,686,400 transitions, best HF + W&B online. Manual Kaggle UI private, Internet on, T4x2, Secrets WANDB_API_KEY/HF_TOKEN and edit NOTEBOOK_REF. Kaggle remote runner max nominal 39600s minus cleanup and startup, possibly stops early.

Tests: **70 passed** (`tests/test_masac_ui_ablation.py`, `tests/test_matd3_ui_ablation.py`, reference/scaled UI checks, checkpoint retention). MASAC genuine U16 independent learner updates on CPU passed. No GPU training launched and no Kaggle/W&B/HF resources mutated. Next user action is private Kaggle notebook import and Save & Run All. Notebook source snapshot is pinned to `89e6274`; subsequent note-only commits do not change its program source. Source guide: `docs/methods/MATD3_MASAC_10_KAGGLE_UI_ABLATIONS.md`.

---

# 2026-11-08 — User-selected MATD3/MASAC BASE Reward plots at windows 8/16/32/64

User corrected earlier 'latest' run preference: **MATD3-BASE-SH-NOCHUNK-DEFAULT-S44-971f7cc2** paired with **MASAC-BASE-SH-NOCHUNK-MASKAWARE-S44-f6a196a7**, explicitly **no PQL** and no Kaiming. Completed and verified 08/11/2026 02:16:55 UTC+7. Code + 9 PNGs + README commit: `2dd9229`.

- Read exact 20,000 episode History records per run from live W&B, same world/reward/sensor/network/energy, 6UAV, 20 targets, 15 obstacles, 1km, 1500 horizon, seed44, 4096 envs. Not the 35t25o new config. Differing MATD3 default init vs MASAC mask-aware means this is a selected-run exploratory graph, not single-factor controlled comparison.
- Added `scripts/plot_latest_base_small_reward_windows.py`, a four-window **moving average** (8/16/32/64, no downsampling) and **non-overlapping block means** (8/16/32/64), zoom at 4096 boundary. Read-only W&B, no PQL or validation used. Images in `docs/research/reward-small-window-selected-base/`, raw records + analysis.json in local ignored `outputs/reward-small-window-default-matd3-maskaware-masac/`.
- Tests: 32 passed, no failure (7s); Git diff --check clean. Δ reward around 4096 (mean of 64 raw completed episodes each side): MATD3 +1121.64, MASAC -425.32. This persists at even 8-window, so the 200/256 smoothing choice did **not cause** the episode cohort discontinuity; using smaller windows resolves sampling/detail but cannot remove real logged jumps. It remains uncertain what fraction is episode completion-order vs actual policy learning.
- **Kept previous checkpoint save interval 256 episodes** untouched; the '256' checkpoint interval is completely separate from the graphical aggregation windows. Also no W&B Workspace mutations, checkpoint selection/validation adjustments, Kaggle run, algorithm training modifications, or PQL modifications.
- Next: user to inspect these figures and decide display window or request an alternate validation-Return convergence chart. Do not switch Validation schedule without permission. Github branch `experiment/hfbest-embedded-source-steps1500-20261105`.

---

# 2026-11-08 — Plot training Reward from actual W&B History using source-backed methods

Task: user explicitly requested making Reward learning-curve figures from Training History (MADDPG-style block means / EPyMARL-style rolling mean) first; user will decide later if switching the paper plot to periodic fixed-policy validation reward. Started ~08/11/2026 01:54 UTC+7; verified 08/11/2026 02:02:20 UTC+7. Main implementation + PNG commit: `db2ff5b`.

- New script `scripts/plot_training_reward_from_wandb.py` and four pure regression tests. Read-only W&B History, no metric update, no job submission, no training/validation/reward/CBF/observation/optimizer changes. Tests: **28 passed in 6.79s**, 0 errors; git diff --check clean.
- Generated and committed three original analysis PNGs plus README under `docs/research/reward-convergence-preview/` and cached full source records/JSON locally under `outputs/reward-method-study/` (ignored by git).
- Four verified W&B runs: MATD3 BASE default `971f7cc2` (20k), MATD3 PQL `30bdc182` (20k), MASAC BASE `62cfc303` (20k), MASAC PQL `43494a2c` (**16,384 incomplete**). All their W&B configs **match world/sensor/reward/energy/network**, actual scenario 6 UAV, 20 targets, 15 obstacles, 1km, horizon 1500, seed44; they are not new 35t25o. Never mistake plots for final benchmark.
- Figure 1: mean Reward over each complete 256-episode block (MADDPG-inspired, final <256 partial window omitted). Figure 2: sliding 200-completed-episode mean (exact formula from official EPyMARL `plot_results.py::smooth_data`, display only downsampled). Figure 3: per-0.25M-global-step completed-episode-bin means with sparse bins blank (diagnostic, not code taken from a paper).
- All 3 reveal reward/episode-cohort artifacts: abrupt steps at multiples of **4096 episodes**, unchanged materially by smoothing; global-step view has disconnected traces from sparse completion intervals. MATD3 BASE start/end complete block mean -622/+118; MATD3 PQL -3360/-125; MASAC BASE -478/-3467; MASAC PQL partial -401/-3290. Do NOT claim this proves all reward differences artificial; real policy updates occur between cohorts, and some algorithms have poor training reward.
- Current main W&B Workspace is **not modified**; original 25 plots unaffected. Validation still 3,686,400 steps, 5 seeds, best chosen from held-out validation; checkpoint 256 per prior commit. No Kaggle old notebooks regenerated. Next task: user decides whether paper uses these logged training-return curves or requests more-frequent fixed-seed validation reward; do not modify validation until user instructs. Re-render figures using latest run IDs with script options `--run LABEL=RUN_ID`.

---

# 2026-11-08 — Active checkpoint save cadence 128 → 256

Task start: 08/11/2026 01:30:30 UTC+7 | Verified: 08/11/2026 01:31:30 UTC+7. Main code commit: `27bc84d`.

User explicitly requested changing checkpoint saves to 256 completed episodes, then continuing discussion of validation frequency (do not change validation yet). Updated 3 **current 20k/41400s** YAML profiles: comparison-reference-masac-matd3-xyz-sensing, comparison-independent-baseline-xyz-sensing, comparison-independent-35t25o-h512-xyz-sensing. Added targeted config regression test `tests/test_active_checkpoint_cadence.py`. Targeted suite: 94 passed, 2 skipped, 608 subtests passed in 19.02s; git diff --check clean. No algorithms, reward, learning, validation, checkpoint retention (keep=2), or final saving touched. Old/specialized YAMLs and existing embedded Kaggle UI notebooks deliberately unchanged. Changes affect only future runs using latest source/config.

Validation remains every 3,686,400 global transitions with 5 independent fixed-seed CPU episodes and single active evaluator; busy/timeout can skip scheduled points. Source step snapshots are separate from 256-episode regular checkpoints; `best.pt` is validation-selected, `final.pt` is saved on normal exit. User is considering a variable cadence tied to 20k episodes (e.g. 10 milestones); no validation schedule or changes were authorized. Future requirement: study an episode-derived validation schedule robust to bursts from 4096 parallel environments and CPU evaluator busy before coding.

---

# 2026-11-08 — Training and held-out evaluation trade-off metrics

User requested that 4 supplemental plots appear in BOTH Training Convergence and Periodic Evaluation, and asked whether training metrics alone should select best checkpoint before a harder-seed final test.

Start: 08/11/2026 00:21:52 UTC+7 | End (implementation verified): 08/11/2026 00:26:50 UTC+7 | Code commit: `609cc2c`.

- Added passive per-completed-training-episode scalars under `training/*`: `delivery_given_discovery_fraction`, `first_discovery_to_gcs_delay_delivered_mean_s`, `false_confirmation_per_1000_measurements`, `map_coverage_percent`. No learning/reward/CBF/checkpoint-selection changes; missing denominators remain null (not fabricated as zero).
- Moved all 4 existing `evaluation/*` plots into fixed-seed Periodic Evaluation and placed 4 training counterparts in Training Convergence. Removed only redundant Supplemental SECTION, not history. Current dashboard: 9 training + 9 periodic evaluation + 7 collapsed learning diagnostics = 25 charts. Remote W&B Workspace read-back verified the exact 8 metrics and episode X axes scaled by 10.
- Tests: 71 passed, 594 subtests passed, 7 third-party warnings (plus CPU/Warp parity smoke). No Kaggle notebooks regenerated, no jobs submitted, no historical metrics backfilled. Old training runs may have empty `training/*` lines until future runs log them.
- Method recommendation: training episode metrics monitor learning only; select best on disjoint held-out validation seeds (same scenario parameters, new layouts), then one in-distribution final test and optional harder OOD stress tests. Final-test data never selects the checkpoint. Existing code still uses 5 periodic validation seeds at 3,686,400-transition spacing and lexicographic success/delivery/search/reward selection; no protocol change was requested or applied.
- Five existing MATD3 UI notebooks are still older embedded source/timeout than repo. Do not claim they include this metrics update until explicitly rebuilt and verified; do not automatically rebuild or submit them.

---

# 2026-11-07 — Timed handoff after later metrics and budget commits

User requests a paste-ready new-chat prompt including work times. Read
current source and five Library notebook headers before writing the handoff.
Runtime snapshot isfa5ee6507e6c6a78c9258b7ac46e5d3cf9faac4c,23:40:32+07.
The earlier sections below remain historical records, not latest runtime.

Verified post-export history (all07Nov,+07):9243c40 21:26:59 adds five
core evaluation metrics;8efcdb2 21:33:15 changes episode display to tens;
2853c9d 22:13:25 revises train/eval views and compresses journals;
e19c14c 22:40:45 restores compact diagnostics;37eb302 23:19:26 adds coverage
and search/relay trade-offs;fa5ee65 23:40:32 changes three YAML train limits
36000->41400. No learner equations/reward/CBF changes in these commits.

Version mismatch is CONFIRMED and must not be silently described as done:
- Library five-case notebook headers still cite0ca4f76 and10h.
- Downloads five-case files now embed9243c40696188e9faf6d6f4d9aed29517fb23257;
  archived YAML still36000s; file mtimes21:27:14+07.
- Current three active YAMLs have41400s. Builder specs still39600s and
  cloud/remote.py clamps available time against39600 before overhead reserve.
Do not claim full11h30 end-to-end notebook execution is already implemented.

Current evaluation still uses5 fixed validation layouts every3686400 steps
for best selection. Final-only evaluation was discussed, not applied;
training_only notebooks skip auto final30. Best criteria remain success,
delivery,search,return (lexicographic). New core metrics are evaluation-only.
Current chart definitions separate train,held-out checkpoint curves,
search/relay trade-offs and diagnostics. Online view state not reread here.

Fresh targeted checks onfa5ee65:58 passed,7 third-party warnings in4.02s
(core mission/comparison metrics,dashboard,axes,journal,reference contract).
The earlier1374-pass full suite in309.13s belongs to0ca4f76 and is not a
whole-suite verification of the later runtime. No GPU job was submitted or
live training status claimed in this handoff. Historical quota is not fresh.

NEXT: deliver PROMPT_UAV_NETWORK_2026-11-07_CO_MOC_THOI_GIAN.md with exact
commit chronology,configuration,artifact identities and pending work.
Future actions should record actual start/end/duration and source/run IDs.
This task changes continuity notes only; no notebook/runtime regeneration.

---

# 2026-11-07 — Five MATD3 UI ablation notebooks

User requests only manual Kaggle UI files, no MASAC or GPU submission.
Matrix on common35t25o/6UAV/1km/1500step/seed44 scenario:
01 SH,H512,U8,actorLN=true;02 SH,H512,U16,actorLN=true;
03 IND,H512,U16,actorLN=true;04 IND,H512,U16,actorLN=false;
05 SH,H256,U8,actorLN=true. Both actor AND critic sharing/width change;
LayerNorm uses the existing actor-only option; critics stay plain.

Started clean at6c5cb7b. New five-case builder composes the existing kernel
bootstrap and scaled profile with explicit active MATD3 overrides. Original
ui_notebooks.py PREFLIGHT has only a generic status/comment text change.
No algorithm,scenario,observation,CBF,reward,energy,init or noise changes.
Old notebooks/profiles are controls;8 existing local copies were hashed.
All5 new files share one cohort group, explicit SH/IND/ALN/H/U/seed labels,
unique per-execution run IDs, no warm-start, no auto-continue, training-only.

Observed8 RED missing-builder failures before implementation. Fresh related
suite89 passed,590 subtests passed, including actual requested architecture,
CPU Warp transitions/replay,finite U8/U16 updates and delayed4/8 actor calls
for all5 cases. Generated SPEC/config/name propagation checked atseeds44/45.
Fresh full repository suite PASSED:1374 passed,103 skipped,8 third-party
warnings,1662 subtests passed in309.13s; actual exit0.
Read-only review approved: no Critical/Important/Minor findings.
No GPU job was submitted.

Spec/plan:superpowers specs/plans2026-11-07-matd3-five-ui-notebooks.md.
Guide:../../methods/MATD3_FIVE_KAGGLE_UI_ABLATIONS.md.
Implementation commit0ca4f76837a20195b3c06f0279561aaec5ff0758 was pushed.
Exported exactly5 new MATD3 files from that clean source into Downloads
and outputs/kaggle. Actual notebook schemas/code,seeds44/45 resolved
configs,names,tags and all421 source archive files verified. All5 use
archive SHA256cd9ac15600e344b7949d6849bb0a9aca7dfbfb6af40be4d404a7ad95cf02b376.
Eight old notebook copies retain exact checksums. Five user-facing notebooks
were saved separately and byte-verified. No MASAC file was exported.
Evidence:2026-11-07-matd3-five-ui-notebook-verification.json and
2026-11-07-matd3-five-ui-test-results.json in evidence/.
NEXT: user imports the five Private notebooks in Kaggle UI,uses T4x2,Internet,
Secrets and real NOTEBOOK_REF,then fresh Save & Run All. Do not submit jobs
automatically. Compare01vs02,02vs03,03vs04,01vs05 at equal transitions.
T4 batch8192 memory,learning performance and convergence remain unverified.

---

# 2026-11-07 — Actor privacy audit and requested credential maintenance

User asked whether actor198->228 violates Dec-POMDP, and requested checking,
pruning and formatting newly appended Kaggle keys. Started clean atf7c0133.
No runtime algorithm, observation, CBF, reward or notebook source was changed.

Verified observation_dim=158+2*num_targets: only own known/report-status
vectors add30 fields for20->35 targets. Obstacles15->25 alone add ZERO actor
fields. Actor uses3 sensor-footprint obstacle slots; privileged full geometry
is reserved for critic state and the separate CBF layer, not actor tensors.
Local belief/coarse memory and ideal in-range radio/admission metadata are
part of the declared observation model. These assumptions must be stated.

12 counterfactual/positive-control probes passed over Native and Warp CPU:
hidden target/obstacle geometry, other-agent private known data and an
unannounced GCS receipt leave observer inputs and deterministic MASAC/MATD3
raw actions unchanged; critic state changes. Own acquired knowledge or a
visible obstacle changes actor observation as expected. Related regression
suite:59 passed,3 CUDA-hardware tests skipped. No GPU job submitted/needed.
Finite probes do not establish a formal all-rollout privacy theorem.

Important deployment qualification: current CBF reads all-agent geometry
and all obstacles. This does not feed global state into the actor or by
itself invalidate a Dec-POMDP transition model. Do not describe the ENTIRE
safety controller as local-only decentralized execution. Current truthful
claim: CTDE policies with a safety filter using privileged global geometry.
See ../../methods/DEC_POMDP_OBSERVATION_AUDIT_2026-11-07.md and the two
2026-11-07-decpomdp-observation-* evidence files, including repro probe code.

Credential maintenance was local/private, outside this repository. Parsed
6 existing accounts+11 appended tokens. The default CLI was too old to use
KAGGLE_API_TOKEN; its failed probes were NOT used for removal. Rechecked
with the installed automation environment's Kaggle2.2.4:13 accounts returned
GPU quota30h each (2 old+all11 new);4 old entries still returned no quota
after2 attempts. Removed those4 from the active JSON and renamed retained
entries account_1..account_13. Verified original private backup, source-hash
concurrency guard, candidate token set, JSON/load_credentials,0600 mode and
unchanged W&B/HF/non-account fields before/after atomic replacement.
No tokens or credential files were copied to Git, Library or notebook.

Audit/evidence commit d2ee64348831edb2f452c2926290bda8818c4c0f is the verified report.
Final structure check:2 passed,585 subtests passed.
NEXT: answer actor-dimension concern with the exact local-memory breakdown,
mention CBF deployment qualification and completed credential cleanup.
Do not change safety/observation assumptions or launch training based only
on this question. Existing20t15o/256 and35t25o/512 notebooks remain unchanged.

---

# 2026-11-07 — Separate35t25o/512x512 independent experiment

User asks whether16/32 updates help and explicitly requests wider networks
and35 targets/25 obstacles. Updates remain8: the16/32 question is advisory.
Starting clean source ee6a097. Created a separate scaled YAML and --scaled
notebook export option; default profile/files remain the20t15o/256 controls.
No algorithm/reward/CBF/battery, learning rate, seed or sensor-slot changes.

Actual dimensions: independent actor198->228, state565->830,
critic including joint actions631->896. Historical SH actor input was204
(198 observation+6 agent-ID). Online parameters grow about3.18x for both
algorithms; this combines scene and width changes, not a width-only effect.
At batch8192/env4096, U8/U16/U32 consume16/32/64 replay samples per new
joint transition. These are NOT literal per-transition optimizer counts.

RED:2 missing-scaled-API failures observed. GREEN:34 relevant tests and
582 subtests passed, including real Warp CPU transitions,512-network
critic+actor updates and finite metrics for both algorithms. The initial
new probe used Action instead of vector dict; corrected the probe adapter,
not production.36 scenarios generated successfully on first attempt with
no fallback. Baseline notebook hashes remain exactly unchanged.
Fresh full repository suite PASSED:1361 passed,103 skipped,8 third-party
warnings,1654 subtests passed in293.69s; actual exit0.
Read-only review approved with no Critical/Important/Minor findings.
No GPU training was submitted;
T4 batch8192 fit and convergence remain unverified.

Evidence: evidence/2026-11-07-scaled-independent-preflight.json.
Methods: ../../methods/SCALED_INDEPENDENT_BASELINE_35T25O_H512.md.
Implementation source89f3301a976d0f85478d6ce4fce8ce076e06b2ca was pushed.
Exported only labelled35T25O_H512_U8 MASAC/MATD3 notebooks from that clean
source. Both copies, actual schemas/cell compilation, all419 archive files,
seed44/45 launch configs and control hashes verified. Archive SHA256:
6c7516078ef75c76cefc36e80a352e49d48df445332beda0e8d5a80c6439efa8.
New matched notebooks were saved separately; old controls were not replaced.
Export manifest:evidence/2026-11-07-scaled-ui-notebook-verification.json.
Full suite:evidence/2026-11-07-scaled-independent-test-results.json.
NEXT: user may run the new U8 pilot in fresh Kaggle sessions. Keep equal
transition budgets; test16 separately on the same scene/width before32.
Do not claim512/scaling improves learning before GPU evaluation, and do
not launch a job or alter update count without the user selecting that action.

---

# 2026-11-07 — Independent baseline Kaggle UI notebooks

User requested continuing the source-baseline work. Prepared two separate
manual Kaggle notebooks for the existing independent profile: MASAC and MATD3,
BASE, NOCHUNK, plain MLPs, seed44. Scientific YAML/algorithm/reward/CBF unchanged.
The common bootstrap can consume an explicit frozen archive and Kaggle
Secrets; legacy private-dataset credentials remain default. The repository
builder needs no old notebook template from Downloads. Run name/tags track
edited seed and algorithm; each launch is fresh.

Read AGENTS.md and previous task record; initial clean source was
3c46ce91dcac1c358c49bc81479c14582bbdbc9d on the existing experiment branch.
Observed11 RED failures, then79 focused tests and6 subtests passed, including
the actual source-profile BASE/PQL CPU collector/reload checks.

The sandbox tunnel temporarily disconnected before the first full-suite
command was observed to launch. During that outage, fetched415 main/upstream
runtime files from exact GitHub revisions and verified every Git blob SHA.
Ran the11 notebook-boundary tests independently in the artifact runtime.
Exported provisional files from published3c46ce9 runtime and saved them with
a recovery patch. GitHub API write returned403, so no branch/commit was
created through that connector. Sandbox later recovered. SHA256 checks
confirmed all4 edited Python files were BYTE-IDENTICAL across both machines.

The initial full run exposed historical MASAC best-state identity failures:
metadata could retain explicit temperature_loss=log_alpha while canonical
identity omits that equivalent default. Restore accepts both forms while
still rejecting changed alpha objectives, architecture, config or provenance.
The historical actor-normalization matrix includes the changed-objective guard.

Final broad verification on recovered source: 1356 passed, 103 skipped,
8 third-party warnings,1644 subtests passed in 634.26s. The 4 reported failures
were static guards in tests/test_project_structure.py: three >500-line source
subtests and one stale algorithm-hash test. Refreshed only three reviewed pins
for earlier user-authorized 9cd3a15/458d7fb/2e50114 changes; optimization/action
utils pins remain unchanged. Removed blank lines only from configuration.py
and tracking.py, with exact AST-equivalence hashes recorded. Older notes were
archived byte-for-byte below 500 lines/file. Guard+notebook rerun: 13 passed,
578 subtests passed. The broad run is retained with its original exit1; do not
describe it as a fresh whole-suite exit0 after this format-only guard refresh.
No known failures remain after the targeted guard rerun.

Independent read-only review: no Critical/Important findings. Its minor
fresh-session guidance was added to notebook and guide. Same-session rerun
support, T4 memory and convergence are not claimed. No GPU job was submitted.

Implementation source commit 789bc1a0f0976922e351f59fe2b72a32d0e9a368 was pushed
to experiment/hfbest-embedded-source-steps1500-20261105. Built only the TWO
new independent BASE notebooks from that clean source; old SH/init notebooks
were not modified. Downloads and outputs/kaggle copies match. Actual notebook
schema/code compilation, 417 archived source files against current bytes,
same archive in both notebooks, scenario equality, default MATD3 init,
MASAC alpha-dual/MASKAWARE, independent plain MLPs, and seed 44/45 config/name
propagation were verified. Archive SHA256:
47ce6d1bc875b932cdfd03976f3a4f9d9d629a886623211e91528470d5764ee7.
Manifest: evidence/2026-11-07-independent-ui-notebook-verification.json.

Final downloadable notebooks, patch and verification manifest have replaced
all provisional versions successfully; local checksums match the Downloads
copies. Verification notes commit 1df65401971d18cb335389fa13b3884903badc67 was
pushed. No GPU job was submitted. The remaining handoff is manual Kaggle UI:
separate Private notebooks, T4x2 and Internet, WANDB_API_KEY/HF_TOKEN Secrets,
real NOTEBOOK_REF, seed 44, fresh Kaggle execution via Save & Run All.

NEXT: user runs the new matched pair; inspect the resulting real GPU logs and
checkpoint metrics before claiming memory fit, convergence or improved MASAC
ranking. Preserve old SH runs as controls; compare equal transition budgets.

---

# 2026-11-07 — Main MASAC/MATD3 pair audited before initialization ablations

User explicitly prioritized MATD3 `971f7cc2` and MASAC `f6a196a7`; both use
the default initialization scheme. Starting clean source was
`e29e729d5c12270df3417e3d338237c54c82b7ac` on
`experiment/hfbest-embedded-source-steps1500-20261105`.

Verified online W&B finished status, full episode/learner/alpha histories,
raw logs, saved config and exact validation ledgers. At matched18.432M steps
on the same5 validation layouts: delivery63% vs47%, return181.345 vs-331.266.
SHA-verified the exact18.432M historical actors from HF revision history:
MATD3 `a1ffcfcc377d0276ff7fe4292e87b76a5735ab17`,
MASAC `7f6f98935f7a6b947e8b433996aa5a0a3632e1ff`.
Ran30 NEW paired CPU layouts, seeds2110000044..2110000073, plus30
seeded sampled-policy MASAC diagnostic episodes. Holdout MATD3 vs MASAC:
delivery65% vs54.5%, return20.769 vs-286.855. Paired scenario-bootstrap
MASAC-minus-MATD3 delivery=-10.5pp,95%CI[-16,-5.333]; return=-307.624,
CI[-502.912,-101.134]. These are conditional-on-policy scenario intervals,
not training-seed uncertainty. Both0/30 full20-target missions; observed
post-filter safety violations0. MASAC sampling did not resolve the gap.
All-UAV-battery termination already works:29/30 MATD3 and30/30 MASAC mean
episodes ended on all-UAV depletion before the1500-step cap.

The main run's temperatures are adaptive and bounded in observed history,
not the old2734dbf3 explosion. Observed extra safety cost plus lower mission
score explain the return deficit arithmetically. No unique causal training
defect is proven from one training seed; do not force MASAC's ranking or
silently change reward/CBF to make curves agree.

Implemented opt-in `masac.temperature_loss=alpha`, matching the SAC author's
alpha-weighted dual objective. Default `log_alpha` and old checkpoint
selection identities remain compatible. Added a separate
`configs/comparison-independent-baseline-xyz-sensing.yaml` for independent
plain-MLP actors/critics in BOTH algorithms, same scene/reward/CBF/battery,
default MATD3 init, MASKAWARE hybrid interface, original alpha objective.
This is a declared adapted recipe, not literal whole-author-code reproduction
or a verified performance fix. Existing SH profile and historical runs stay
reproducible. No new GPU training, job submission or convergence test.

Fresh focused suite:139 passed,8 skipped,3 subtests passed in82.72s.
Additional old-vs-new temperature checkpoint-identity guard:1 passed.
Code provenance and complete findings:
`docs/methods/MASAC_MATD3_MAIN_PAIR_AUDIT_2026-11-07.md`.
Per-scenario numeric evidence and unsmoothed comparison chart are in
`docs/superpowers/evidence/2026-11-07-*`. Primary author sources are pinned
in the methods report; OpenAI itself has TD3>SAC benchmark cases.

Main-pair implementation/evidence commit
`9cd3a1553639a5928df68a4c91104d86e026a08c` was pushed to the working branch.
Secondary work, after the main audit: corrected cd07de3c's W&B display name
from DEFAULT to KAIMING; readback config/summary hashes were unchanged.
Fixed the notebook generator's DEFAULT-for-KAIMING label mapping and tested
all three generated run names. Labeling implementation commit:
`a2adb5d605005f42ca1c52ed908605eae9ba0667`. Fresh initialization/header suite:
11 passed. Regenerated ONLY the manual Kaiming notebook in Downloads and
outputs/kaggle, from that clean committed source; no Kaggle job submission.
Embedded archive SHA256
`632643cb1e0c6f91381121c79bdf91ea27a41e8feda6ed80d3a228137e31e72a`,
415 files. Notebook cells compile; nbformat validates; actual generated
header resolves KAIMING; archived runtime sources match; both copies match.
Verification manifest: evidence/2026-11-07-kaiming-notebook-verification.json.

NEXT: explain the measured main-pair gap and causal limits to the user.
Use source recipe only as a new labelled baseline; do not claim improved
learning before full GPU training or launch training without a new request.
Preserve old SH runs as controls. A paper comparison needs several training
seeds and matched transition checkpoints.
Do not launch production training automatically. If the new source profile
is selected for a paper, BOTH algorithms need fresh training lineages and
multiple training seeds with matched transition checkpoints.

---

Historical notes preserved byte-for-byte in order: [Part1](CURRENT_TASK_HISTORY_2026-11-07_1.md) | [Part2](CURRENT_TASK_HISTORY_2026-11-07_2.md).
