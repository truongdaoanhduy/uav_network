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
