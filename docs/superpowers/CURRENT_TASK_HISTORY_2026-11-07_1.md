# 2026-11-06 — MATD3 BASE Kaiming/He Kaggle UI notebook prepared (no launch)

User requested a **single manual Kaggle UI MATD3 BASE Kaiming initialization notebook**,
following existing self-contained DEFAULT and ORTHOGONAL notebooks; user will
submit from the Kaggle UI. Read AGENTS.md and CURRENT_TASK.md with a clean
worktree on branch experiment/hfbest-embedded-source-steps1500-20261105,
at 200db5cf12fc6f844c58a1eeaad4e5b9f8365d69 before editing.

**Implementation commit** `2e5011447badbc669d84c1ea115c92e230e11fa7`: \n- Added opt-in `matd3.initialization=kaiming` (other choices `default`,
  `orthogonal` still supported; default unchanged).
- New `initialize_matd3_kaiming()` applies `torch.nn.init.kaiming_normal_`
  (`mode=fan_in`, `nonlinearity=relu` for all hidden Linear layers,
  `nonlinearity=linear` for actor motion/routing and both Q output heads);
  all Linear biases zero. Actor LayerNorm affine params remain 1/0.
- Initialization applies to online shared actors and critics **before**
  target deepcopy and Adam optimizer creation. The added init runs under
  `torch.random.fork_rng(devices=[])` so subsequent CPU RNG state matches
  the default network-construction path. Existing action, losses, optimizer
  settings, CBF, reward and canonical YAML unchanged.
- `scripts/build_matd3_init_ui_notebooks.py --mode kaiming` builds only the
  Kaiming notebook and does not overwrite earlier DEFAULT/ORTHOGONAL files.

**User deliverable:**
`/home/aduy/Downloads/MATD3_BASE_NOCHUNK_KAIMING_Kaggle_UI.ipynb`
(copy `outputs/kaggle/MATD3_BASE_NOCHUNK_KAIMING_Kaggle_UI.ipynb`).
It embeds the committed source with 414 bundled files, SHA-256
`e02e3514eb817b0af38e1e543a1f7a8d8f24e51fee3082efb6d582822aeccacc`.
The user's Kaggle UI notebook gets a fresh run ID/name at launch.

**Comparison:** resolved config is byte-equivalent to the saved DEFAULT
notebook's effective config after excluding only `matd3.initialization`
and informational tracking tags. This includes BASE (PQL false), NOCHUNK
(chunk=0), seed44 deterministic, 6 UAV, 20 targets, 15 obstacles,
1500-step horizon, 4096 env, batch8192, replay1M CPU75%/GPU25%, 20k
episodes, shared actor/critic LayerNorm, FP32, 2xT4 request, W&B and HF best.

**Verification:** focused 54 tests passed, including new Kaiming construction,
Linear application/intercept, LayerNorm unchanged, target copy, RNG state,
finite forward and actor/critic update. Notebook nbformat validation, both
code-cell Python compiles, archive hash/content, downloaded copy equality
and resolved-config comparison with DEFAULT all passed. No Kaggle GPU
training, GPU VRAM peak or mission-score performance was tested; no cloud job
was submitted and the two older notebooks were not modified.

**Next:** User imports the Kaiming notebook in Kaggle UI, enables private
T4x2 + Internet, grants `WANDB_API_KEY` and `HF_TOKEN` Secrets, replaces
`NOTEBOOK_REF` with real username/slug, runs preflight and Save & Run All.
Compare search/delivery, mission_score, return, Q, saturation and training
stability at matched environment steps and fixed evaluation layouts.

---

# 2026-11-06 — Paired MATD3 BASE Kaggle UI: default vs Orthogonal (prepared only)

User requested two self-contained manual Kaggle UI notebooks in `~/Downloads`,
one preserving MATD3 BASE defaults, another changing **initialization only**.
Read the 2026-11-06 handoff and AGENTS.md first; starting worktree was clean
on experiment/hfbest-embedded-source-steps1500-20261105 at e45a6473d3b0.

**New source commit** `458d7fbb7a304995b2316d681cdb8c9ac89042ec`.
`matd3.initialization` is an explicit validated `default|orthogonal` field,
with `default` preserving the PyTorch Linear initialization exactly.
`orthogonal` initializes every online actor encoder/continuous/destination
Linear and both online twin-critic MLPs before target cloning/optimizers.
Gain sqrt(2) for ReLU hidden, 0.01 for actor heads and 1.0 for Q heads;
biases zero; LayerNorm affine remains default. `torch.random.fork_rng` restores
CPU RNG after the extra orthogonal QR draws so following CPU RNG state
matches the default model construction. No changes to losses, action semantics,
reward, CBF, replay, optimizers, target updates or original canonical YAML.

**Notebook outputs:**
- `/home/aduy/Downloads/MATD3_BASE_NOCHUNK_DEFAULT_Kaggle_UI.ipynb`
- `/home/aduy/Downloads/MATD3_BASE_NOCHUNK_ORTHOGONAL_Kaggle_UI.ipynb`
- Matching copies in `outputs/kaggle/`, source generator
  `scripts/build_matd3_init_ui_notebooks.py`.
- Both embed the exact 413-file source bundle from commit 458d7fb with SHA256
  `4277932ce73c3e596b2978a09768257802550fc8295c5759e047a34a98b3a599`.
- Both use the reference XYZ 1500-step YAML, MATD3 BASE, PQL false,
  critic_chunk_size 0, 6 UAV, 20 targets/15 obstacles, 4096 env, batch8192,
  replay1M 75% CPU, 20k episodes, seed44 deterministic, shared actor/critic
  LayerNorm, T4x2, FP32/AMP off, training-only, W&B and best-only HF.
  The paired `SPEC` values were independently checked: **only** init flag
  and informational tracking tags differ. Both create new per-run W&B IDs.
- Kaggle preflight checks GPU T4x2 and private `WANDB_API_KEY`/`HF_TOKEN`
  Secrets, and requires a user-supplied real `NOTEBOOK_REF`. No tokens are
  in the payload. Both include checksum verification before installing source.

**Verification:** 50 passed in focused MATD3/XYZ/reference/embedded-source
tests. Both notebooks pass nbformat validation, Python code compile, source
SHA verification, exact file-content checks for learner/YAML, and matching
Downloads versus outputs/kaggle copies. A real Kaggle T4x2 run, VRAM peak,
convergence and orthogonal effect on mission performance have NOT been tested.
**No Kaggle job was launched or submitted, no existing run changed.**

**Next:** User manually imports each notebook in a private Kaggle UI session,
enables Internet/T4 x2, authorizes the two secrets, edits NOTEBOOK_REF,
runs preflight and Save & Run All. Compare at matched environment steps
and matched evaluation layouts; monitor mission score, raw return, search,
delivery, Q, critic gradient and action saturation.

---

# 2026-11-06 — Handoff snapshot of current W&B runs

Read-only W&B check at 22:26:48 ICT. MASAC BASE MASKAWARE `f6a196a7` was
running at 225,280 environment steps. Its logged config confirmed seed 44,
4096 envs, 20k episode budget, 1500-step horizon, `warp_packet` and CBF,
shared actor and critic with actor LayerNorm, `pql.enabled=false`,
`runtime.critic_chunk_size=0` and `masac.mask_aware_discrete_entropy_target=true`.

At 22:25:12 ICT the fixed-target MASAC BASE `2734dbf3` and replica `62cfc303`
were running at 30,253,056 and 18,194,432 steps respectively. MATD3 BASE
`b358fc3f` and PQL `30bdc182` were both finished at 20,000 episodes. This
handoff describes the training code and reports verified through `2e0162c`.
Recheck live state before taking action. The next MASKAWARE comparison
uses common environment-step budgets and matching evaluation scenarios.

---

# 2026-11-06 — MASAC entropy source verification and MATD3 reuse

Checked the learner at `715992bd66fb422a68767ea86fb432aaea5a26a4` and both
MASKAWARE notebook payloads. Both continuous and discrete temperatures
are learned with Adam on log-alpha. The notebooks override the discrete
target using valid destination counts. Kela, Liu, and Valcarce use the
same target form in arXiv:2411.08529v3, Appendix D.2, Eq. (17). The
provenance now identifies the paper, the BenchMARL coefficient and the
TorchRL loss separately.

MATD3 BASE `b358fc3f` and PQL `30bdc182` used source `48a5ea2`. The source
diff confirms identical learner, environment, training loop and reference
YAML through `715992b`. Reuse both runs as controls for the fresh MASAC
comparison. The source verification changes documentation only.

---

# 2026-11-06 — MASAC NOCHUNK mask-aware entropy fix

Enabled `masac.mask_aware_discrete_entropy_target=true` in the MASAC BASE and
PQL UI notebooks in `~/Downloads` and `outputs/kaggle`. Both generate fresh
W&B run names tagged `MASKAWARE` and use the same source bundles from
`4c9f67ca`. The reference profile with fixed entropy remains the control.
The only learner setting changed is the discrete entropy target. BASE and
PQL keep their respective modes, seed 44, no critic chunks, 4096 envs,
batch 8192, 1500 steps, 20k episodes, shared networks, reward and CBF.

- The two original notebooks resolved the flag to false. Updated notebooks
  pass nbformat, Python compilation, source checksum, and resolved config
  checks. Their original source archives are byte-identical.
- On 32 recorded observations from `2734dbf3` with six active UAVs and only
  silence legal, 256 actor and temperature updates drove alpha_d from
  1 to 1.079826 with the fixed target. The mask-aware target held it at 1.
  Both actors had identical weights after this controlled test.
- Short CPU training from each notebook's resolved config ran four episodes
  and eight learner updates, then reloaded a finite NOCHUNK checkpoint with
  the expected BASE/PQL mode and mask-aware setting.
- Focused checks passed 105 tests and 3 subtests. The variant documentation
  and regression coverage are in commit `12208368e48db30fec506ce8ee259def6cfd830f`.

Next comparison is a fresh MASKAWARE run versus the fixed-target run under
the same seed, layouts and environment-step budgets. At 3.6864M and 7.3728M,
compare alpha_d, alpha_c, Q scale, search, delivery and raw return. The CPU
checks verify temperature behavior and launch settings. Full Kaggle GPU
learning and recovery of mission performance remain to be measured.

---

# 2026-11-06 — MASAC BASE NOCHUNK stability diagnosis

Audited W&B `2734dbf3` against MATD3 BASE `b358fc3f` and the MASAC BASE replica `62cfc303`. The active training source commits are `1a55c7d`, `48a5ea2` and `4c9f67c`. The learner, environment and training paths match across these revisions. The runs use shared actor/critic, actor LayerNorm, no chunks, seed44, 4096 envs, batch8192 and 1500 steps. BASE/PQL is not a factor for these BASE comparisons. Matched evaluator seed and full layout hashes verified for all five common checkpoints. Both MASAC runs matched every return, outcome and length at 3.6864M and 7.3728M.

- At 18.432M transitions MASAC/MATD3 returned -3823.97/+181.34, found 2%/65% and delivered 0%/63%. MASAC/MATD3 commanded stall rates were 90.67%/1.05% and CBF rates 98.19%/51.00%. MASAC's first and third groups of 4096 training episodes delivered 42.26% and 4.94%. The uploaded screenshot's sampled raw return -3295.16691 and 15%/10% outcomes match episode 9543.
- At 20:27 ICT the 2734dbf3 sample at 23,326,720 transitions showed alpha_d 789900, alpha_c 476.66, H_d 0.01018 and max |Q| 9.08M. At 20:42:46 ICT the run was running at 24,199,168 transitions and 12,685 completed episodes, alpha_d 1320046, alpha_c 755.23, H_d 0.00732 and max |Q| 11.50M. The latest original evaluator still showed zero deliveries at 22.1184M.
- Source `uav_network/algorithms/masac.py` sets fixed discrete entropy target 0.2 log(7) = 0.38918 and updates alpha_d even on singleton action masks. The shared critic bootstrap includes the entropy terms for all six UAVs. Logged alpha_d rose 8.12 → 69.41 → 587.95 → 45191.56 near 3.69M, 7.37M, 11.06M and 18.43M. This mechanism explains discrete-temperature drift and target amplification. It does not isolate the initial actor failure.
- Replayed exact HF actor snapshots on the original five layouts, with separate greedy and sampled MASAC diagnostics. The MASAC 3.6864M snapshot is HF revision `1f36e6e3a331ca0a712662bdaad6eee6de6619af`, SHA-256 `9f883fc9f8807d70a99a39bf75e0f9c202077d8d41e58e4ce446f6df86d6694e`. Delivery was 1% greedy and 22% sampled. At 11.0592M, HF revision `4fda53a9ce5ea0a999116a4ce8586181a5497a53`, both modes delivered 1%. Among active steps, singleton masks covered 99.987% in early greedy rollout and 98.78% in later greedy rollout. Maximum attainable mean discrete entropy over those occupancies was 0.00019 and 0.00934 nats. Reruns used exact actor bytes and layout hashes but some trajectories differed from hosted CPU eval. The matched performance comparison uses original W&B eval, and local rollouts diagnose behavior.
- Ran 256 actual MASAC actor and temperature updates on the same batch of 32 real observations with all six active and only silence valid. The fixed target raised alpha_d 1 → 1.079826, while `masac.mask_aware_discrete_entropy_target=true` kept it at 1.0. Both actor weights were identical afterward. Fresh seed44 CPU MASAC and MATD3 deterministic actions on five resets both had max |XYZ command| 0.5645 and zero saturated components. Both architectures use standard Linear initialization.
- Focused tests for MASAC, MATD3, reference baseline and critic gradients passed 35 tests plus 3 subtests. Audit plot `MASAC_MATD3_BASE_diagnosis.png` uses the 20:27 capture. Per-seed diagnostics and scripts are under `/tmp/uav-masac-audit-20261106`.

Next scientific comparison is a fresh MASAC A/B with only `masac.mask_aware_discrete_entropy_target` toggled, same seed and matched transition checkpoints. Track feasible entropy, temperatures, entropy contribution to critic target, and deterministic and sampled mission outcomes. Keep MATD3 BASE as the independent control and use the same world, reward, safety, seed and evaluation budgets. The full hybrid action benchmark is a declared adapter to the BenchMARL/SAC and Hybrid SAC references and the Ackermann/Fujimoto MATD3/TD3 references. For an upstream continuous-only control with both learners, XuanCe provides MASAC and MATD3. A common fixed routing rule would define a separate protocol for that comparison.

---

# 2026-11-06 — MATD3 mission score, paired holdout and safety audit

Source changes are in commit `16a4687561a3e16878071a6880ee50ae4137d0da`. Training now logs `outcome/mission_score` and periodic evaluation reports `evaluation/mission_score`. The diagnostic is 10 points per unique target found and 50 per target delivered to the GCS. The code also declares separate return and final battery panels. Focused verification passed 71 tests and four subtests.

The existing W&B personal view was updated in place. A historical panel computes `reward/discovery + reward/delivery`, exactly the fixed mission score for the audited runs. Separate episode panels show raw unsafe commands, CBF interventions and final battery. The saved view spec was read back and matched the expected additions.

- BASE `b358fc3f` and PQL `30bdc182` were trained from `48a5ea2`, seed 44. In the screenshot, the sampled raw return -202.40102 and 45%/45% outcomes match BASE episode 4876. Episode 3058 is -799.31769. The early plot uses exact episode indices in fixed 50-episode bins. Comparing episode windows 1–500 with 1501–2000, BASE mission score rose 8.78 while return fell 109.18. Extra safety penalties accounted for 92.06 and blocked motion for 20.39. The later window was 58.708 steps longer.
- At 18,432,000 transitions both best snapshots matched the historical W&B SHA-256 byte-for-byte. HF revision `84a9b8c3f528ae4b5bf4afb660ff66c836db0093`. BASE SHA `f015eff99694c3503a1b077511e746643c11c4eab32d5920806ad33a1643a44b`. PQL SHA `a28078a2e1e48a2c78620a4ac6037ad2afdfa2bb66605d274ccce5626bf1b669`.
- Thirty new CPU Warp scenarios used seeds 2100000044–2100000073, with equal layout hashes within every pair. BASE/PQL mean returns -49.0539/-481.9719, mission points 757/577.6667, search 64.3333%/53%, delivery 62.8333%/47.1667%. BASE minus PQL delivery is 15.6667 percentage points with paired scenario bootstrap interval [9.3333, 22]. Post-filter violations and complete missions were 0/30 each. Battery exhaustion 30/30 versus 29/30. One PQL scenario reached step 1500 with an active UAV.
- Exact PQL 3,686,400-transition snapshot recovered from HF revision `f2c8a238c04bc1a571bca8bfb2e740ba61ad7ba7`, SHA `556c37c655a6c8f5aa1ed548fc0d0bc73742c483007bb094da26bc9f820c96e7`. All five original rollout returns reproduced exactly. Actor components saturated near [-1,+1,+1], and 8958/9000 UAV-steps per scenario proposed a command above the 150 m ceiling. Tracing captured actor output, nominal velocity, CBF candidate, executed velocity and fallback flags.
- Removing only the intervention count penalty in fixed-policy PQL reruns on the 30 holdout layouts moved mean return from -481.9719 to -107.4713. Each paired outcome, motion count, geometric safety metric and final battery matched. The separate from-scratch CPU pilot used 16 envs, batch 128, 16384 transitions and 993 critic updates per arm with matched initial actors. Under the common original evaluation reward, the 0.5/0 penalty arms delivered 8.67%/5.83%, scored 127/91.33 mission points and attempted unsafe commands on 76.97%/84.87% of UAV-steps. Both had zero post-filter violation episodes and zero all-battery-exhaustion episodes. The delivery difference zero minus control was -2.83 percentage points with paired scenario bootstrap interval [-6, +0.67]. This pilot is an early diagnostic for a single training seed and reduced compute.
- Direct Warp battery smoke at horizon 1500 verified that one depleted UAV leaves the episode running, and all six depleted triggers `terminated=True`, `truncated=False`, `battery_exhausted` at step 1.

Next controlled test uses the original PQL scale and matched transition budget. The only objective override for the experimental arm is `--set reward.shield_intervention=0.0`. A separate safety-filter comparison can evaluate 3D DWA or a predictive safety filter on the same geometry and report mission, post-filter violations, battery and throughput. Warp XYZ explicitly rejects `dynamic_window` today, so a 3D DWA comparison needs its own implementation on that backend.

Follow-up at `0d9ccc5` re-read the first 2,000 exact episode records in each
run. All recorded `battery_exhausted`. In BASE the largest difference
between the logged return and its summed reward fields was 0.0000116.
A fresh one-versus-six depleted UAV smoke on the Warp CPU training
kernel reproduced the battery stop at step 1 for six. The stopping code
was already present at launch commit `48a5ea2`. The energy and tracking
checks passed 15 tests and four subtests.

The scalar XYZ environment dispatches `dynamic_window`. Warp XYZ dispatches
the CBF projection and has no dynamic window search branch. A safety
comparison on Warp requires a backend implementation of the candidate
filter. Candidate references include [Fox et al. on dynamic
window](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance)
and [Wabersich and Zeilinger on predictive safety
filters](https://doi.org/10.1016/j.automatica.2021.109597).
The current MATD3 runs remain the controls for any separate safety study.

Local diagnostic scripts and per-seed evidence are in `/tmp/uav-matd3-audit-20261106` on the user machine. Figures are `MATD3_early_episode_audit.png` and `MATD3_mission_score_and_holdout.png`.

---

# 2026-11-06 — Prepared MASAC BASE/PQL NOCHUNK Kaggle UI notebooks

User requested two **manual Kaggle UI** training notebooks, one MASAC BASE and one MASAC PQL, both with `runtime.critic_chunk_size=0`; do not launch either via Kaggle API.

- `~/Downloads/MASAC_BASE_NOCHUNK_Kaggle_UI.ipynb` and `~/Downloads/MASAC_PQL_NOCHUNK_Kaggle_UI.ipynb`; copies under `outputs/kaggle/` with the same names.
- Both embed source generated from Git HEAD `4c9f67ca5c20`, with 412 source files and checksum verification; each has three cells: instructions, CUDA/Secrets/NOTEBOOK_REF preflight, and the entire embedded-source training bootstrap.
- BASE/PQL resolve `pql.enabled=false/true` respectively; matched shared actor/critic, seed44, 4096 envs, batch8192, 1500 steps, 20k episodes, 2x T4, FP32, no compile/AMP, training-only, separate W&B run identities, HF checkpoints. Both override chunk to zero without modifying canonical YAML.
- Validation completed locally: nbformat.validate, Python compile for both code cells, full 412-file embedded archive SHA256, config resolution and mode assertions. **No Kaggle training or CUDA memory test performed**, and account-specific Kaggle Secrets plus real NOTEBOOK_REF must be provided by user.
- Explicit user workflow: sign in to two distinct Kaggle UI accounts, create private notebooks, import one per account, enable T4x2 + Internet, authorize `WANDB_API_KEY` and `HF_TOKEN`, replace NOTEBOOK_REF with actual owner/slug, preflight, then Save & Run All.

---

# 2026-11-06 — Kaggle MATD3 BASE/PQL NOCHUNK dual launch

User explicitly requested MATD3 BASE and MATD3 PQL training on two available Kaggle accounts, with all critic chunking disabled. At 2026-11-06T09:42:51+00:00 UTC the cloud launcher accepted both submissions from source commit `48a5ea2` on `experiment/hfbest-embedded-source-steps1500-20261105`:

- BASE: `account_03`, `fjwoigw/notebookb576080e44`, W&B `b358fc3f`, manifest `outputs/kaggle/notebookb576080e44/launch.json`.
- PQL: `account_04`, `hjgjhjhjd/notebookee3ce97b75`, W&B `30bdc182`, manifest `outputs/kaggle/notebookee3ce97b75/launch.json`.

Both accounts authenticated and reported 30.00h remaining when selected. Both submissions used canonical `configs/comparison-reference-masac-matd3-xyz-sensing.yaml` plus `runtime.critic_chunk_size=0`, `pql.enabled=false/true` as appropriate, and independent `runtime.output_dir` paths; actor/critic remain shared. Both request `gpus_per_run=2`, seed 44, 20k episodes, 1500 max steps, batch8192, 4096 envs, FP32, training_only + Hugging Face checkpoint; no CPU visualization or GitHub supervisor.

Post-submission `kaggle kernels status` returned `running` for both kernel refs. Kaggle logs were empty at the check, and W&B had not yet created either run, **so learner updates, peak VRAM, and training completion are not yet verified**. The canonical YAML remains chunk2048; the zero-chunk setting exists in the job manifest only. Do not submit a duplicate to these accounts. Follow-up: check kernel status/logs, W&B learner counters and gradients, and possible no-chunk OOM; confirm that best checkpoints are uploaded.

---

# 2026-11-06 — Kaggle API new-run names switched to UI-style notebook IDs

User explicitly requested that future Kaggle API training launches stop encoding
algorithm/stage/config/timestamp into the Kaggle notebook title/slug, and use
the same anonymous name style as a freshly created Kaggle UI notebook.

**Implemented:** Ordinary direct API launches now generate a fresh
`notebook<10 lowercase hex>` slug per launch, e.g. `notebookb97fd08763`.
The SDK submission `new_title` already mirrors the slug. W&B run names,
experiment settings, scientific seed, source bundle, the GPU T4x2 request,
checkpoint behavior and training objective have not changed. The Kaggle API
payload remains the existing Python `script` kernel/embedded bootstrap; the
change is the visible notebook identity, not conversion to editable `.ipynb`
cells or update-in-place versioning. Prior Kaggle notebooks remain untouched.

**Safety-preserving exception:** `--auto-continue` GitHub-managed lineages
retain the reserved `gh-` prefix. GitHub's ownership filter must not mistake
any user's ordinary `notebookXXXXXXXXXX` for a managed workflow. Derived
continuation/CPU/controller names also remain unchanged; do not rename
existing linked cloud jobs without a dedicated lineage migration.

**Source commit:** `ee314033b145ae56ceeee8936e52e63dc81103ed` on
`experiment/hfbest-embedded-source-steps1500-20261105`.
Tests written RED first: pre-change launch identity failed because it still
included `tra-g3-...`. After the change, 87 tests + 29 subtests passed in
focused API/continuation/supervisor checks; expanded cloud suite: **126 passed,
31 subtests passed**. `git diff --check` clean. Offline tests only; no Kaggle
kernel submitted and no active run modified.

**Next:** Upon explicit authorization for another training launch, the new
name is produced automatically. No need to re-upload/update earlier jobs.

---

# 2026-11-06 — Real matched chunk2048/shared implementation complete

User explicitly requested actual2048 critic chunks for BOTH learners, shared
actor/critic in current config, and every remaining oldNov2/current config
difference excluding chunk. Source base:52e500a65108e04df6f1ed0fd01dc64a51e70e4a.
Actual implementation source commit:c792aa07b07d2860e5556e75eac5b64929999a06.
Published on the existing experiment branch at2026-11-05T19:45:46.907571+00:00.
Full-suite evidence applies to that source; this delivery update changes docs only.

Canonical configs/comparison-reference-masac-matd3-xyz-sensing.yaml now selects
shared actors/shared critics/LNtrue and chunk2048 for both. Other canonical
hyperparameters unchanged, including vectorizeFalse/compileFalse/AMPoff.
Two1024 proposal YAMLs remain historical proposals; no launcher default switch.
Dataclass defaults and legacy configs/checkpoints are unchanged.

Added pure twin-output checkpoint helper with original batch slicing, axis1
for per-critic actions. MATD3 target/regression/all actor-Q routes now consume
the setting; MASAC regression now chunks in addition to its existing expectedQ
paths. No sampling/noise inside recomputation. Losses,clipping,optimizer count,
target schedule and TD3 delay unchanged. Same2048 bounds original rows before
different algorithm expansions; it does not imply equal VRAM.
Matched-study preparation/selection guards changed0->2048 so they accept the
new canonical recipe. Existing10case sharing axes remain explicit, not replaced.

Verification: RED11failed/8passed proved the missing chunk/sharing behavior.
GREEN41focused; new test file23passed incl4realCPUWarp shared6agent
MASAC/MATD3 BASE/PQL smokes, O198/S565, smallwidth16/B4/E4/horizon4/chunk2.
Target/regression/actor call bounds,uneven11/4,RNG,clipping,Adam states,
TD3 delay and reduced saved storage verified. Canonical2048 construction checked.
Consumer guards14passed after fixing their old0pins. First full run loaded the
pre-edit guards and failed12related tests; final fresh full suite is clean:
1310passed,103skipped,8warnings,1632subtests in265.92s,exit0.
Fresh read-only whole-change review found no Critical/Important/Minor findings;
independently checked all4config snapshots/counts and PQL/checkpoint propagation.

Report:docs/methods/MATCHED_SHARED_2048_CONFIG_DIFF_2026-11-06.md.
Evidence:2026-11-06-shared2048-remaining-config-differences.json and
2026-11-06-shared2048-completion.json under docs/superpowers/evidence/.
Remaining serialized differences excluding chunk:MASAC16each,MATD317each;
includes unused other-algorithm fields and3metadata keys. All world/sensor/
network/energy/reward/PQL fields match corresponding old configs.
Other source differences remain declared; no rollback or entropy/clipping change.

No production cloud job launched; no CUDA/full8192peak-memory/convergence proof.
Historical MASAC OOM remains unconfirmed without traceback. Next: report exact
remaining differences; wait for user's choice before changing other hyperparams
or launching a new controlled experiment.


---

# 2026-11-06 — Matched configuration proposals prepared; no launch/default switch

User asked for all old/new config differences and their training effects,
chunk/vectorize fairness, and common MASAC/MATD3 configurations, including a
MASAC02Nov-like option. Explained shared study budgets vs algorithm-specific
alpha/noise/delay and source-specific vs matched optimizer recipes.
Source base:3d09bc65d26bc7921d5232c934474f16bfe226a7; learner code unchanged.

Created two explicit PROPOSAL profiles, not switched into any launcher:
configs/comparison-proposed-matched-shared-ln-memory1024.yaml:
sharedactors/critics,LNtrue for both,B8192,E4096,replay1M/CPU75%,warmup131072,
8updates,LR3e-4,gamma.99,tau.005,seed44,vectorizeFalse,chunk1024,overlapFalse,
AMPoff/compileFalse/fusedFalse. Keeps source-specific SACnone/TD3per-tensor.5.
configs/comparison-proposed-matched-shared-ln-memory1024-masac02novlike.yaml:
same common setup; MASACalpha.2,targetratio.9/maskawareTrue and BOTH algorithms
globalclip10. Declared matched-hparams variant, not old-code reproduction:
current losses/logstd/no-alpha-cap/estimators remain; outputnormFalse.
Chunk1024 currently consumed by MASAC only. Same scalar setting does not make
MATD3 actually chunk; no code added merely to make flags appear symmetric.
Shared batching exists independently of vectorize flag; MATD3 critic/target
helper can still wrap1module in vmap when enabled. No speed guarantee.

Validated8profile/algorithm/PQLflag combinations and11common fields; CPU models
construct, finite bounded actions,1actor/1TwinCritic,2hiddenLayerNorms.
Source/structure guards:13passed/557subtests in2.15s; diff check clean.
This is config/construction validation, not new PQLtraining integration or
GPUfull8192memory/convergence proof. No production jobs or canonical changes.
Guide:docs/methods/FAIR_CONFIG_MASAC_MATD3_2026-11-06.md.
Evidence:docs/superpowers/evidence/2026-11-06-proposed-matched-configs.json.
Next: provide both proposals/explanations to user; activate/train only under
new task authorization. Compare matched transitions and held-out seeds;
same episode or wallclock caps alone do not equal sample budget.

---

# 2026-11-06 — Four-run old/new/current source audit completed

User requested detailed comparison of four Nov2 MASAC/MATD3 BASE/PQL runs
against latest failed/completed cloud runs and current source, to preserve a
source-backed baseline without rolling back historical heuristics.
Observed online Nov6 ICT; source audit at db5345a63a5e5fab5843109dc46a952e49f1814b.

Old runs all source05beef6c: MASAC BASEc8d5ce6b/PQL2ce9fb9e;
MATD3 BASEdb61bd0c/PQLd51b0350. New BASE574f36c6/PQLab2e953d
have identical learner/environment/communication trees. Current adds optional
hidden LayerNorm only to learner/model paths; it was absent from failed runs.
Reference YAML enables norm but defaults both sharing flags false; all audited
cloud runs share both. Direct reference overlap defaults true vs new runs false.

MASAC old stopped time_budget at16384/16362episodes, rolling delivery4.5/15%.
New BASE latest progress2506752transitions/4648updates (summary4632);
PQL143360/32; replica126976/0recorded. No OOM traceback in retrieved W&B
files; Kaggle CLI console reads failed authentication. GPU1 sampled maxima
grew~4.43GiB to~8.00GiB, below14.56GiBcapacity; sparse samples do not establish
absolute peak or exclude transient OOM. Last progress of all3 within4.2sec
needs external termination investigation; no confirmed alternative cause.
MASAC chunk1024->0 disables checkpointing and bounded344064-row shared Q
enumeration; MATD3 does not use this chunk flag. MATD3 dead-tanh mechanism
from prior checkpoint evidence remains supported by fresh actor-grad0 metrics.
No single-factor training attribution established.

Verified64focused CPU tests passed in4.77s; supplemental6agent shared
chunk0/4 oracle with real O198/S565/C4, B11/width16/3updates, both norm flags:
same RNG and max parameter difference0 in those bounded cases.
No localCUDA/full8192 GPU peak/OOM/convergence proof or production launch.
No learner/config edits. Provenance clarified author regularizer targets
distribution parameters/logits, not neural weights. Updated report/evidence only.
Source-size/SHA guard:2 passed/554 subtests in0.16s; git diff --check passed.

Report: docs/methods/MASAC_MATD3_CODE_COMPARISON_2026-11-06.md.
Evidence: docs/superpowers/evidence/2026-11-06-four-run-code-audit.json;
config diffs: same folder/2026-11-06-four-run-config-diffs.json.
