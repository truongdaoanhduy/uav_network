# XYZ CUDA PREFLIGHT — REFERENCE FIX / BASE RETRY RUNNING

Updated 2026-10-31T01:39:48.135518+07:00. User authorized Kaggle GPU execution; no new approval needed.
Runtime fix 5fee6c753b4e7cd86801a68ab5a0fb72521bda86; parent42c1dac.
Branch feat/cpu-replay-20261022; retry source9e7b88c52637fc5ae3571e3e567271a93cd0c965.

- First BASE benchmark:kesfljhk/ben-g3-both-base-h75-10301741-894057,
  observed cancelled01:11:53+07 near1800s ceiling; cancellation reason not exposed.
  Actual hardware2TeslaT4;10CUDA/native-world tests passed, last Warp check unfinished.
  No XYZ E4096 benchmark JSON,20k production training or convergence proof.
- Root diagnosis:by-value WorldState descriptor copies produce38.69MB PTX,
  801058local stores; default CUBIN compiler SIGKILL after473.5s (OOM unconfirmed).
  Prototype uses wp.ref for140arguments/85helpers/17files; kernel signatures,
  math,RNG,struct layout and checkpoint schemas unchanged. CUBIN AOT passes
  20.77s,peak646.32MiB without GPU. O1/O2/noinline/PTX knobs not adopted.
- Final bare pytest1146passed,102skipped,1557subtests,264.89s,exit0.
  Prototype suite also green; own diagnostic clones moved outside collection.
  Four new real XYZ CUDA checks will run in existing test_cuda.py cloud gate.
  No GPU parity claim yet; independent review0Critical/0Important;1Minor fixed.
- Keep6UAV,1000m,20T/15O,1200steps,E4096,B8192,replay1M,Hybrid75,FP32.
  Learned RF/action4,movement,routing,prior0.5,FOV/Bayes and visualize2D retained.
  Benchmark-only compileoff/W&B-checkpoint-store/log_checkpointfalse; YAMLs unchanged.
- Pin account_01 only; quota27.11h observed01:16:54+07. No account rotation or
  cancellations. User CURRENT_TASK historical suffix remains unchanged/unstaged.

[Usage](../xyz-sensing.md) | [CPU evidence](evidence/2026-10-30-xyz-sensing-verification.md) |
[CUDA/compiler evidence](evidence/2026-10-31-xyz-cuda-preflight.md).
Raw:outputs/xyz-gpu-preflight-42c1dac-20261031/.
Own temporary checkouts:/tmp/uav-xyz-preflight-checkouts-20261031/.

Live BASE retry:kesfljhk/ben-g3-both-base-h75-10301844-63825e (running).
https://www.kaggle.com/code/kesfljhk/ben-g3-both-base-h75-10301844-63825e
Next:poll this job and inspect outputs/xyz-gpu-refs-20261031/base-retry-watch.json.
Require all15CUDA tests,
scenario/mission gates and actual profile updates/VRAM. If BASE passes,run PQL
sequentially on same account; only then decide20k training from measured budget.

---

# Learned RF action and validation-best checkpoint — COMPLETE / CPU VERIFIED

Updated 2026-10-30T09:26:32.602395+07:00. User requested rechecking27–29Oct MASAC/MATD3 at matched
budgets, restoring the original learned-power action and selecting best
rather than blindly using last policy. Verified runtime `951a396a41bc451d2716b7dedcb46b74b9e5677a`
on `feat/cpu-replay-20261022`; profile restoration commit `3ca8f51`.

- Three current20T/15O/1200 profiles now use `power_control=learned`;
  movement/routing/sharing/replay/FP32 remain as before. Guided+3dB stays
  separate `link_budget` ablation. Historical configs/manifests preserved.
- Periodic validation alone ranks success→delivery→search→return; ties keep
  earlier candidate. Exact evaluated actor bytes become best.pt/best.json.
  Model artifact/final CPU eval use selected best, with best alias. No valid
  validation produces explicitly labeled unvalidated_final_fallback.
- Full last learner/replay/optimizer/RNG/environment stays resume_latest.pt,
  with optional best bytes/metadata carried to BASE/PQL continuation. Bank,
  hash, score/metrics and source-step guards remain strict. Manual BASE
  actor-only warm-start is labeled a NEW lineage with fresh critics,
  optimizer/replay/counters; it is not exact resume or35/25→20/15 migration.
- W&B index132 runs;37 unsampled training histories,32 stitched lineages.
  Exact8193–12288 outcomes:shared-both MASAC Hybrid delivery78.87%,
  FullGPU73.01%; shared-critic66.30%/64.48%. Some training episodes succeed;
  success0 does not mean no discovery/relay or no completed train episodes.
  Sources/precision/E/B/rewinds and actual steps are disclosed in report.
- Seven historical checkpoints were downloaded and re-evaluated on the
  same5 validation seeds. Shared-both Hybrid:8192 delivery81.14%,12288
  80.57%,final12353 72.00%; FullGPU:12288 60.57%,final16384 56.57%.
  Shared-actor Hybrid8192 validates32.00% despite strong training; MATD3
  PQL Hybrid12421 validates49.71%. All7 success0/5; small posthoc evidence,
  not an exhaustive or universal architecture winner. Best-of-tested
  copies and scoped selection metadata saved locally per historical run.
- Final pytest1061 passed,97 skipped,1483 subtests,8 warnings,563.24s;
  unittest628 run,71 skipped,OK442.258s. compileall/diff checks clean.
  Independent review3 Important plus one regraded sidecar issue fixed
  with7 RED→GREEN cases. Two old test contracts updated for deferred final
  resume timing/new optional selector state; both pass. Full suite green.
- No production launch/cancel, new sensing/XYZ, or CUDA hardware check.
  Historical model downloads/evaluations were real; new publication tested
  at tracker/SDK boundary only. No remote historical best aliases mutated.

[Matched-run report](evidence/2026-10-30-matched-runs-and-best-checkpoint.md) |
[Structured data](evidence/2026-10-30-matched-run-data.json) |
[Verification](evidence/2026-10-30-learned-power-best.json) |
[Current usage](../controlled-20t15o-study.md).
Plan:plans/2026-10-30-learned-power-best.md.
Local raw audit:outputs/learned-power-best-20261030/.
Historical selected copies:outputs/learned-power-best-20261030/historical-validation/selected/.
Fresh84-case learned-power manifest:outputs/controlled-20t15o-learnedpower-20261030/manifest.json;
prepared only, not submitted. Older guided-power matrices remain historical.

Next action follows the user's new request. Before any requested cloud
launch recheck live jobs/quota/CUDA, pin this source and the intended current
YAML, and preserve old lineages. Do not call these CPU/small-bank results
production convergence or copy35/25 checkpoints into20/15. Broader historical
best search needs more candidate validation; reserve final test bank for
assessment. No main merge/PR requested. Preserve host worktree and initial
uncommitted continuation-record suffix. Read current live state before
operational decisions; the audit is a dated snapshot.

---

# Shared Actor + Critic FULLGPU profile — VERIFIED / NOT LAUNCHED

Updated 2026-10-29. User requested FULLGPU for the new shared-actor/shared-critic MASAC/MATD3 training profile while retaining HYBRID75 for comparison. Runtime commit `23aa1eafe2d273560bd054c7e491ee05410ef963`.

- Added `configs/comparison-20k-fullgpu-1kb-shared-actor-critic.yaml`. Both MASAC and MATD3 keep `share_actor_parameters=true` and `share_critic_parameters=true`.
- FULLGPU replay is explicit: `runtime.replay_device=cuda`, `runtime.replay_cpu_fraction=0`. Run identity resolves to `SHARED-ACTOR-CRITIC-FULLGPU`; tag `fullgpu`; no `cpu75-gpu25`.
- HYBRID shared-both profile remains unchanged for A/B comparison. The FULLGPU profile differs from it only in replay placement and output/group/tag identity; seed44, 20T/15O, 1200 steps, 20k episodes, E4096, B8192, replay1M, updates8, FP32, compile on, BASE, link-budget+3dB remain matched.
- TDD RED occurred because the FULLGPU profile did not exist; GREEN after creation. Focused verification: 44 passed + 34 subtests. Full suite: 1030 passed, 97 skipped, 8 warnings, 1478 subtests in 228.15s. `compileall` and `git diff --check` pass.
- No Kaggle/W&B production training was launched by this task. CUDA production feasibility for this exact shared-both FULLGPU profile is not freshly measured in this task; historical FULLGPU E4096/B8192/R1M runs existed, but do not relabel them as this new 20T/15O profile.

Design: `docs/superpowers/specs/2026-10-29-shared-both-fullgpu-design.md`. Plan: `docs/superpowers/plans/2026-10-29-shared-both-fullgpu.md`.
Next action if launching: fresh-check duplicate jobs/quota/VRAM and start MASAC BASE + MATD3 BASE from the FULLGPU profile; retain the HYBRID shared-both profile as the controlled replay-layout comparator.

---

# Shared actor + shared critic MASAC/MATD3 — CPU VERIFIED / NOT LAUNCHED

Updated 2026-10-29. User selected the full-sharing topology after comparing matched W&B MASAC runs. Added a dedicated controlled profile rather than overwriting the independent baseline:
`configs/comparison-20k-hybrid75-1kb-shared-actor-critic.yaml`.

- MASAC: `share_actor_parameters=true`, `share_critic_parameters=true`.
- MATD3: `share_actor_parameters=true`, `share_critic_parameters=true`.
- All controlled settings remain matched to the canonical 20T/15O baseline: 6 UAV, 1000 m, 20 targets, 15 obstacles, 1200 steps, seed44, 20k episodes, E4096, B8192, replay1M, HYBRID75, FP32, BASE, link-budget+3dB. Only sharing topology plus output/group/tag identity differ.
- Existing shared topology code was already implemented for both learners; no learning-equation change was needed. Shared actor remains UAV-ID conditioned; shared critic is one centralized team twin critic. Run naming resolves to `TRAIN-MASAC-BASE-SHARED-ACTOR-CRITIC-...` and `TRAIN-MATD3-BASE-SHARED-ACTOR-CRITIC-...` with actor-shared/critic-shared tags.
- TDD: new profile contract failed first because the profile did not exist, then passed after creation. Focused controlled-study + parameter-sharing + cloud-identity verification: 43 passed, 34 subtests. Tiny CPU smoke for both algorithms completed 1/1 episode at 2 steps, preserved both sharing flags, and reported no non-finite summary values.
- Fresh full suite after the change: 1029 passed, 97 skipped, 8 warnings, 1475 subtests in 213.47s. `compileall` and `git diff --check` pass.
- No W&B/Kaggle production training was launched by this configuration task. Do not claim new shared-both 20T/15O training results until a fresh production run is explicitly started and evaluated.

Design: `docs/superpowers/specs/2026-10-29-shared-actor-critic-masac-matd3-design.md`.
Plan: `docs/superpowers/plans/2026-10-29-shared-actor-critic-masac-matd3.md`.
Next action for production: fresh-check duplicate jobs/quota/account ownership, then launch one MASAC BASE and one MATD3 BASE from the new profile on seed44; preserve independent baseline results for the controlled comparison.

---

# Controlled 20-target / 15-obstacle 2D study — HORIZON 1200 CONFIGURED / CPU VERIFIED

2026-10-29 follow-up: canonical 20/15 horizon requested at 1200 steps. Historical 35/25 stays at 1800.
The original 1800-step feasibility/test evidence below remains historical evidence; do not relabel it as a fresh 1200 verification.
Periodic evaluation remains every 3,686,400 joint transitions so this change does not alter the experience-based validation cadence.
Fresh verification after the horizon change: focused study/scenario 36 passed, 5 skipped; full pytest 1028 passed, 97 skipped, 8 warnings, 1472 subtests; compileall/diff check pass. A new local 84-case horizon-1200 manifest was generated with zero cloud submissions.

Updated 2026-10-29T10:52:51.081529+00:00. User explicitly requested the five audit recommendations,
20/15 first, and later reference sensing/XYZ. Verified runtime `2dbfdc860d57cf23b391c117d70576f95c323ff8`
on `feat/cpu-replay-20261022`. Existing host worktree retained. Incorporated
published production fixes through `80b690a`; no production job submitted or
old lineage changed by this task.

- Canonical `configs/comparison-20k-hybrid75-1kb.yaml`: 6 UAV, 1000 m,
  20 target/15 obstacle, 1200 steps, link-budget power +3 dB. Keep movement
  and routing learned; keep finite radio/energy, queue, TTL and packet MAC.
  Previous 35/25 profile is `configs/comparison-20k-hybrid75-1kb-35t25o.yaml`.
- Native/Warp guided power and learners ignore the controller-owned power
  slot, including SAC log-probability/entropy, target/replay and PQL copies.
  Optional per-UAV local8x8 coverage has exact resume and selective reset;
  inactive legacy config/snapshots retain their previous meaning.
- Seven separate cases: baseline, learned_power, buffer16, buffer32,
  ttl600, coverage8, battery2x. Three training seeds44/100044/200044;
  MASAC/MATD3 BASE/PQL ->84 prepared configs, zero cloud submissions.
  Validation5 every3686400 transitions at1000000044; final30 at2000000044.
  Final summary verifies checkpoint/config hashes, mode, precision and
  seed bank; reports seed-level sample std and actual budget metadata.
- Final full pytest:1028 passed,97 skipped,1470 subtests,8 warnings,
  301.89s; JUnit0 failures/errors. Project unittest:628 run,71 skipped,
  OK162.819s. Compileall and diff checks pass. Independent review found
  checkpoint/mode binding and missing protocol metadata; both fixed with
  RED->GREEN regressions in one review-fix pass.
- Full-horizon Warp CPU privileged feasibility:5/5 layouts deliver20/20,
  mean138.4 steps, mean final battery88.07%. Final smoke after provenance
  fix:18 train/evaluate cases, all18 verified final checkpoints and positive
  updates;2episodes/E2/horizon8/hidden8x8/step16/5updates/eval2. These are
  pipeline/feasibility checks, not learned convergence or paper results.
- No CUDA hardware on this host; no production20k study training or new
  Bayesian sensing/XYZ implementation. Read-only reference contract is
  documented; do not copy its heavier3km/50target/30obstacle profile.

[Usage and limits](../controlled-20t15o-study.md) |
[Structured evidence](evidence/2026-10-29-20t15o-study.json) |
[Plan](plans/2026-10-29-20t15o-controlled-study.md).
Local prepared manifest: `outputs/controlled-20t15o-study-20261029/manifest.json`.
Local final smoke: `outputs/controlled-20t15o-verification/smoke-final/smoke_summary.json`.

Next action: verify this new path on CUDA, then run a budget-complete paired
baseline study with periodic validation and three training seeds before
selecting resource ablations or claiming convergence. Keep test seeds
held out; do not resume35/25 checkpoints into20/15. Only after the easy2D
stage is measured, add altitude-dependent Bayesian sensing and then XYZ
with motion/LOS/energy/action/observations/checkpoints updated together.
No main merge or PR was requested; preserve this host-managed worktree.
Recheck live service state before any operational action on old lineages.

---

# FP32 default and API-only production cohort — RUNNING / VERIFIED ONLINE

Updated 2026-10-28. User requested FP32 training with AMP omitted by default, then clarified nonprimary web logins are forgotten and only API credentials are available. Resolved usernames through API. Actual production source `ca53c74bcdea557cca681f77082eb6cf323f64cd`, runtime identical to verified d290b715. Canonical YAML no AMP key, cloud no automatic AMP override, effective off/amp-off. Explicit opt-in remains for historical probes. No algorithm/scientific config changes except requested numerical mode.

- New fresh seed44 FP32 roots, same20k/E4096/B8192/R1M/HYBRID75/2T4: MASAC PQL account05/fwelfijwelfj W&Bdff78acf root`fwelfijwelfj/gh-tra-g3-masac-pql-h75-102809-92a283`; MATD3 BASE account08/kljgewkfjl W&Ba85d067a root`kljgewkfjl/gh-tra-g3-matd3-base-h75-1028093-890afc`; MATD3 PQL account09/ewlgkjewkf W&Bb8d0b40a root`ewlgkjewkf/gh-tra-g3-matd3-pql-h75-10280932-1f7593`.
- All3 running on W&B+Kaggle with fresh heartbeat and actual learner progress at09:41UTC: MASAC PQL208896step/136updates; MATD3 BASE487424/696; MATD3 PQL548864/800. Actual full runtime differs only amp_mode fp16->off against their old retry counterpart; all8science sections exact. Latest checked losses/gradient norms finite. Not20kcomplete.
- Old FP16 retries0650d5b1/44558e6c/87a9500e on02/03/04 were NOT cancelled; current segments may run to their existing limits and consume their existing quota. Their accounts are retired from automatic recovery/continuation/final CPU scheduling. Old data/checkpoints preserved. New FP32 roots use different idle accounts because no exact cancellable session IDs or forgotten-account web auth were available. Primary Google login succeeded asduytrngoanh only; no further logins needed/requested after API-only clarification.
- KAGGLE_ENABLED_ACCOUNTS verified account_01,account_05,account_06,account_07,account_08,account_09. Healthy4f98f2c9/a04dbdef/8db188ea still running and supervised. Service probe40665341474/job122738305307 SUCCESS verifies exact API owners,W&B,HF; non-dry40665691175/job122739467717 SUCCESS recognizes all3new exactrefs and healthy01/06/07 active/running; no02/03/04 entries. Workflow active on main. New manifests preserve external GitHub orchestration, auto_continue, finalCPU30 eval/visualization.
- Runtime verification: focused78passed/+460subtests; full960passed/95skipped/8warnings/+1417subtests in501.47s, JUnit0errors/0failures; dualT4diagnostic8df42093 all4canonical cases passed withFP32gradients/parameters/noscalers; review0Critical/0Important. Deferred diagnostic-wrapper JSON minor documented in report. No runtime changes after this verification. Final documentation2tests/+453subtests, diff/evidence assertions pass;12credential values scanned,0hits.

Detailed [API transition report](evidence/2026-10-28-fp32-api-transition.md), [structured evidence including all10 API usernames](evidence/2026-10-28-fp32-api-transition.json), [FP32 verification](evidence/2026-10-28-fp32-default-verification.md). Plan docs/superpowers/plans/2026-10-28-fp32-default.md; ledger outputs/fp32-default-progress.md. Final evidence checks pending before publication.

Next action: preserve these3FP32 roots and healthy01/06/07. Fresh-query services for follow-ups, do not duplicate roots or re-enable retired02/03/04 automatically. Do not claim that FP16 segments were cancelled, that mode change was exact full-state resume, or that20kepisodes have completed. No further login, main merge or runtime work is required for this task.

---

# FP32 default — CPU / DUAL-T4 VERIFIED; production transition BLOCKED

Updated 2026-10-28. Runtime source `d290b7157a85428eacdfa3343c731a343af9fa37` pushed on feat/cpu-replay-20261022. User requested FP32/no AMP needed by default. Removed canonical YAML FP16 key and cloud policy AMP injection; Config already defaults off; tag amp-off. Explicit opt-in retained for historical diagnostics. Algorithm/scientific source unchanged.

- Baseline39passed; RED7expected failures; focused GREEN78passed/+460subtests. Full CPU suite960passed,95skipped,8known warnings,+1417subtests in501.47s; JUnit0errors/0failures.
- Actual dual-T4 diagnostic8df42093, Kaggle fwelfijwelfj/diag-fp32-canonical-1028-c36d75, source d290b715: all4canonical MASAC/MATD3 BASE/PQL cases passed E4096/B8192/R1M/HYBRID75. Real gradients/parameters FP32 and finite; no scalers; no FP16 autocast; compile/PQL worker execution verified. Raw result docs/superpowers/evidence/2026-10-28-fp32-cuda-result.json.
- Fresh reviewer0Critical/0Important. Deferred minor: diagnostic wrapper can overwrite detailed failure JSON if inner preflight fails; all4actual cases passed. Full scope/rulings in report.
- Operational gate remains BLOCKED: no current session IDs exposed by public ref-based Kaggle APIs; cancellation requires exact kernel_session_id. Base kernel ID is not a verified cancellable session ID and was never sent to cancellation. Extension browser timed out; recovered cloud browser has no signed-in Kaggle owner session (private page404, homepage Login). No sign-in credentials entered.
- NO production root cancelled/restarted/overwritten in this FP32 task; no production FP32 replacement launched. All6existing roots still running FP16 at2026-10-28T09:22Z readback, all now have full recovery pointers. Recent retries0650d5b1/44558e6c/87a9500e keep source0042166; healthy4f98f2c9/a04dbdef/8db188ea preserved. Do not mistake FP32 canary for production migration.
- Prepared3relaunch argument sets in ignored outputs/fp32-relaunch-prepared.json; science config equal, only AMP mode off changes, no AMP CLI argument. Plan docs/superpowers/plans/2026-10-28-fp32-default.md; ledger outputs/fp32-default-progress.md.

Detailed [report](evidence/2026-10-28-fp32-default-verification.md), [structured evidence](evidence/2026-10-28-fp32-default-verification.json), [CUDA result](evidence/2026-10-28-fp32-cuda-result.json).

Next action: obtain authenticated owner UI access or verified cancellable session IDs for the exact three retries, cancel only those and verify terminal state, then submit fresh clearly labeled FP32 lineages on the same accounts with external continuation/final CPU eval. Verify W&B effective off, source, positive learner progress and supervisor recognition. Do not weaken strict full-state precision validation or duplicate running roots. No main merge requested.

---

# MATD3 actor reproduction / three authorized restarts — RUNNING / STARTUP VERIFIED

Updated 2026-10-28T08:16:13Z. User explicitly requested a further actor-overflow reproduction and restarts of the three failed experiments. Diagnostic and new production source `0042166fb0408d9efa487ad2ccd31c59b709cd69` is the documentation descendant of verified runtime98724b8; no runtime/config edits in this follow-up.

- Actor probe `aa6066de`, Kaggle `egkljwigojwof/diag-matd3-actor-late-1028-fc3fc4`, completed. Actual finite historical MATD3 BASE model + NEW finite rollout batch reproduced actor overflow at scale32768 in eager and compile-enabled probes. Eager backward trace first nonfinite operation is `aten.mm.default`, [3,8192] x [8192,256] -> [3,256],16nonfinite outputs, zero uninspected operations. FP32 reference weight gradient7.690993309*32768=252018.46875>65504. FP32 loss/norm finite. Historical batch/optimizer were never recovered; do not claim exact old-event replay.
- Fresh seed44 production roots (same scientific/runtime config, 20000episodes,E4096,B8192,R1M,CPU75/GPU25,2T4): MASAC PQL account02 old03df03cc ->new0650d5b1/root`vuliu123456/gh-tra-g3-masac-pql-h75-1028080-7b56ff`; MATD3 BASE account03 old3befcfb8 ->new44558e6c/root`egkljwigojwof/gh-tra-g3-matd3-base-h75-1028-fa6b65`; MATD3 PQL account04 old4f8f7d7d ->new87a9500e/root`gkgijewjf/gh-tra-g3-matd3-pql-h75-10280805-9535a7`. Old full-state checkpoints are absent, so these are fresh lineages, not exact resumes/warm-starts. All three effective W&B scientific sections and full runtime compare equal to each old counterpart.
- 08:15UTC Kaggle+W&B running with positive learner progress: MASAC PQL global_step278528/updates272, MATD3 BASE688128/1088, MATD3 PQL655360/1008. All three have already captured real production nonfinite-gradient events and continued applying updates after AMP skip/backoff; MATD3 BASE/PQL actor and critic snapshots and MASAC PQL critic snapshot are available on their W&B runs. No 20000episode completion claimed.
- External supervisor reconcile40654732673/job122702913311 explicitly recognizes all new exact roots active/running at08:09:44UTC. Overall invocation nonzero concerns only OLD roots with checkpoint_missing; do not confuse this with new-root failure. Manifests retain auto_continue, GitHub orchestration, CPU30 evaluation and visualization.
- Preserve healthy old roots: 4f98f2c9,a04dbdef,8db188ea all running on W&B+Kaggle with fresh heartbeat at08:16UTC. Their config/state was not mutated. Do not launch duplicate roots or cancel them.
- Fresh verification: recipe py_compile passed; project structure/scientific source checks 2passed +449subtests in0.22s; diff check and runtime/config equality against98724b8 passed; evidence consistency assertions passed; 12 credential values scanned across6changed files,0hits. Previously verified runtime98724b8 full CPU suite959passed/95skipped/1408subtests; no runtime modifications here.

Detailed [report](evidence/2026-10-28-matd3-actor-reproduction-and-restarts.md), [structured evidence](evidence/2026-10-28-matd3-actor-reproduction-and-restarts.json), [full diagnostic result](evidence/2026-10-28-matd3-actor-probe-result.json), [executed recipe](evidence/2026-10-28-matd3-actor-probe.py).

Next action: preserve these six active production roots. Recheck live service state for follow-ups; use the now-saved production batch/snapshot for any further numerical failure. Do not infer that an observed skip means the run failed, or claim a lost historical batch was reconstructed. No merge/main migration was requested.

---

# AMP overflow telemetry/capture fix — COMPLETE / VERIFIED ON CPU AND DUAL T4

Completed 2026-10-28. Verified runtime `98724b827ce0dd017bb319dea09aa3c31b90282a` on `feat/cpu-replay-20261022`. User authorized execution of the preceding proposal. Existing host worktree retained; no production restart/cancel or scientific-config change.

## Changes and verification

- Idempotent learner telemetry prevents handled Inf/NaN from crashing strict episode/history/summary/W&B JSON. Original finite metrics retained; missing numeric values carry nonfinite labels/flags.
- AMP pre-clipping norm boundary preserves original gradient evidence and prior optimizer skip/backoff behavior. First critic/actor skip attempts bounded batch/model/gradient/optimizer/scaler/RNG snapshots; storage/upload failure isolated. Replay CLI compares eager FP32/FP16 and records actual parameter gradient statistics. FP32 applied counter is null because AMP counters are bypassed.
- Full recovery first due is capped at1800s; subsequent spacing and attempt cap unchanged. Existing source-pinned jobs keep their old source/timing.
- Fresh whole-change review:2 Important findings (repeated labels and missing gradient tensors), both observed RED->GREEN and fixed. No deferred reviewer minors.
- Final full CPU suite on98724b8:959 passed,95 skipped,1408 subtests,8 warnings,525.47s; JUnit0 failures/0 errors. Compile/diff/scientific source hashes pass;12 credential values scanned,0 hits. Local CUDA unavailable; actual GPU evidence comes from the completed Kaggle probes below.

## Direct numerical evidence

All three diagnostic kernels COMPLETE on Kaggle, all W&B diagnostic runs finished:

- `128d227d`: fresh canonical-sized four-case probe, kernel `egkljwigojwof/overflow-fp16-diagnostics-cbac3c`, source4e3a942. BASE cases did not overflow within bound; MASAC/MATD3 PQL critics naturally overflowed on finite input, and FP32 replay was finite.
- `c031f202`: sourcefd5d207, kernel `gkgijewjf/diag-trained-overflow-1028-33be7a`; actual finite model from historical MATD3 BASE failure, fresh canonical batch. Critic overflow at8192, FP32 norm8542;8 actor batches at32768 did not reproduce actor overflow.
- `9299f2f9`: source98724b8, kernel `egkljwigojwof/diag-overflow-reference-1028-e761e3`; CUDA reference replay of3 captured critic snapshots. Fresh MATD3 PQL weight gradient1.05083346*65536=68867.42; MASAC PQL1.06436467*65536=69754.20, both exceed FP16 max65504. For those two cases, all6/all2 above-range parameter tensors match the nonfinite FP16 gradient tensors. Model/input remain finite. This confirms scaled-backward overflow in observed canaries, not just norm-reduction overflow.

Exact historical failed batches are unavailable. Original MATD3 BASE actor event is still not exactly reproduced; no claim to have identified its first arithmetic operation. Eager replay cannot reconstruct compiled kernels or asynchronous RNG interleaving. Do not reinterpret diagnostic snapshots as resumable training state or invent missing data.

## Live operational snapshot and next action

At2026-10-28T06:52Z, original healthy `4f98f2c9`, `a04dbdef`, `8db188ea` were running on W&B+Kaggle with fresh heartbeat. E4096 BASE now has a verified full backup atstep10256384/episode4096; E2048 roots had no confirmed full checkpoint yet. Preserve them and fresh-query before action. Old failed03df03cc/3befcfb8/4f8f7d7d still lack verified full recovery state; warm-start or restart must be labeled as a new lineage if later requested.

Next launches should pin verified fixed source98724b8 or its documentation descendants. This completed task did not authorize a silent20k production relaunch, main merge, or migration of existing lineages.

Detailed measured evidence and code map: [overflow verification](evidence/2026-10-28-overflow-fix-verification.md), [structured result](evidence/2026-10-28-overflow-fix-verification.json), [diagnostic usage](../training-diagnostics.md).

---

# Additional MASAC E2048/B4096 cohort — RUNNING / CLOUD SUPERVISOR ACTIVE

Updated 2026-10-28T05:29:09Z. User explicitly requested two additional MASAC production roots, BASE and PQL, keeping the preceding canonical HYBRID75/1KiB 20k setup unchanged except `runtime.num_envs=2048` and `masac.batch_size=4096`. The PQL root additionally uses the same mode selector `pql.enabled=true` as the existing PQL cohort.

## Accounts / roots

- `account_06` / `fwklgeewjlfew`: MASAC BASE, root `fwklgeewjlfew/gh-tra-g3-masac-base-h75-1028-a76638`, W&B `a04dbdef`.
- `account_07` / `lkejflwejf`: MASAC PQL, root `lkejflwejf/gh-tra-g3-masac-pql-h75-10280520-a7f572`, W&B `8db188ea`.
- Both roots pin source `55a264cf27505d97f66aae4eadf73f61422169f7`, which contains the previously verified credential-refresh fix; scientific runtime/algorithm code is unchanged from the preceding cohort source.
- Both manifests have `orchestration=github_actions`, `external_supervisor_required=true`, `auto_continue=true`, `cpu_visualize=true`, 2 visible T4 GPUs, final CPU evaluation 30 episodes with seed `2000000044`.

## Verified effective configuration

W&B config readback confirms both runs use `runtime.num_envs=2048`, `masac.batch_size=4096`, episodes 20,000, replay 1,000,000, learning_starts 131,072, updates_per_step 8, scenario-v3, 1KiB report/payload and the same HYBRID CPU75/GPU25 layout. BASE has `pql.enabled=false`; PQL has `pql.enabled=true`. Because the canonical YAML carried stale human tags `env4096`/`batch8192`, both W&B run tags were corrected post-start to `env2048`/`batch4096`; run config and training values were not mutated.

Fresh launch snapshot at 05:29:05–05:29:09Z:

- BASE `a04dbdef`: Kaggle `running`, W&B `running`, global_step 354,304, train/updates 872.
- PQL `8db188ea`: Kaggle `running`, W&B `running`, global_step 202,752, train/updates 520.

GitHub `KAGGLE_ENABLED_ACCOUNTS` now keeps the original four accounts and adds 06/07: `account_01,account_02,account_03,account_04,account_06,account_07`. Hosted service probe confirms Kaggle identities for 06/07 plus W&B/HF are OK. Non-dry supervisor run `40638183428` reports both new roots exactly as `action=active`, `status=running`, so they are independent of the local sandbox and are eligible for the same train -> verified continuation -> final CPU evaluation/visualization flow.

Operational caveat: the same supervisor invocation exits nonzero because older roots on accounts02-04 currently report `recovery_blocked/checkpoint_missing`; account isolation still lets reconciliation continue through accounts06/07, as proven by the active/running entries above. Do not relaunch or cancel the new roots because of the unrelated old-lineage errors.

---

# Four-account accelerated 20k cohort — RUNNING / CLOUD SUPERVISOR ACTIVE

Updated 2026-10-28T03:36:06Z. User explicitly authorized four fresh canonical HYBRID75/1KiB production roots, one Kaggle account per MASAC/MATD3 BASE/PQL, with sandbox-independent continuation and final CPU evaluation/visualization.

## Source / launch correction

- Production roots are pinned to `1aa095f29b8505bbf43ca42033c51ea96d144dd5` on `feat/cpu-replay-20261022`.
- Launch preflight exposed a stale canonical private W&B credential dataset. No kernel had been submitted at that failure point. Root cause: `_ensure_single_credential` verified an existing stale value then failed without refreshing it.
- TDD fix `1aa095f`: existing owned private credential datasets now publish one refreshed version and reread until the selected local credential matches; newly created datasets still fail hard on mismatch. Focused verification: 26 passed + 9 subtests. Full suite: 931 passed, 95 skipped, 8 warnings, 1400 subtests, 0 failures in 513.19s.
- GitHub supervisor secrets were refreshed from the local ignored key file without printing values. `KAGGLE_ENABLED_ACCOUNTS=account_01,account_02,account_03,account_04`.
- Hosted GitHub read-only dry-run `40625191550` succeeded: Kaggle identities for all four accounts, W&B and HF all `status=ok`; no pre-existing supervised lineage on those accounts.

## Running roots

| Account | Case | Kaggle root | W&B learner |
| --- | --- | --- | --- |
| `account_01` / `duytrngoanh` | MASAC BASE | `duytrngoanh/gh-tra-g3-masac-base-h75-102803-a0b159` | `4f98f2c9` |
| `account_02` / `vuliu123456` | MASAC PQL | `vuliu123456/gh-tra-g3-masac-pql-h75-1028032-944054` | `03df03cc` |
| `account_03` / `egkljwigojwof` | MATD3 BASE | `egkljwigojwof/gh-tra-g3-matd3-base-h75-1028-8d7c87` | `3befcfb8` |
| `account_04` / `gkgijewjf` | MATD3 PQL | `gkgijewjf/gh-tra-g3-matd3-pql-h75-10280327-2ad6d8` | `4f8f7d7d` |

All four manifests have `orchestration=github_actions`, `external_supervisor_required=true`, `auto_continue=true`, `cpu_visualize=true`, final CPU evaluation `episodes=30`, `seed=2000000044`, frame stride 5 and 10 fps. They use canonical `configs/comparison-20k-hybrid75-1kb.yaml`, 2 visible T4 GPUs, 20,000-episode budget; PQL roots only add `pql.enabled=true`. Measured acceleration policy remains algorithm-specific and unchanged.

## Fresh live evidence before sandbox handoff

At 03:35:57–03:36:06Z all four Kaggle roots were `running`, all four W&B learner rows were `running` with fresh heartbeat and positive learner progress:

- MASAC BASE `4f98f2c9`: global_step 573,440; train/updates 864.
- MASAC PQL `03df03cc`: global_step 495,616; train/updates 688.
- MATD3 BASE `3befcfb8`: global_step 684,032; train/updates 1,080.
- MATD3 PQL `4f8f7d7d`: global_step 331,776; train/updates 376.

Hosted non-dry supervisor reconcile `40627527455` succeeded and reported all four exact refs with `action=active`, `status=running`. The scheduled `UAV External Supervisor` remains on `main` and runs every 15 minutes; it is independent of the local sandbox. Therefore the local sandbox/laptop may be stopped without stopping these four roots. When a GPU segment ends before the 20,000-episode budget, the supervisor should continue from a verified resumable full-state checkpoint on the same account. When the episode budget completes, it should hand off to the configured CPU 30-episode evaluation/visualization and W&B/HF publication. Cloud quota/service failures can delay a transition; do not create duplicate roots or move a lineage to another account.

## Next action

Do not relaunch/cancel these roots. For status questions, fresh-query Kaggle + W&B and the latest GitHub supervisor report; the progress numbers above are only the launch-time snapshot. If a lineage stops, follow exact checkpoint provenance/recovery gates rather than warm-starting or silently changing scientific config.

---

# Diagnostic coverage audit — COMPLETE / VERIFIED ONLINE

Updated 2026-10-28T01:10:01.014110+00:00. Runtime/source frozen and verified at `56604cc71f294cd94e5251491ab7d1397a95200b`.
User requested audit+supplement logs for crashed/failed/poor training and other meaningful cases; Personal Runs and Workspace refined without clutter. No production training launched/restarted/cancelled, scientific config/eval schedule/checkpoint formats unchanged. Runtime and completion records published on main+feature; remote hash readback and GitHub connector both verified completion commit `95c4fe5b972e3ecb943168790557337749119fad`. This follow-up changes records only.

## New-chat handoff and lineage correction — 2026-10-28

User requested a detailed pasteable handoff, saved in docs/superpowers/HANDOFF_NEW_CHAT_2026-10-28.md. Read it for code map, config, orchestration, constraints and evidence routing. This follow-up changes documentation only; runtime remains56604cc and its924-pass verification.
Fresh CPU lineage/provenance check at08:20:29Vietnam: BASE HYBRID ce8f3be6 ->15db2a32 (45.8095% delivery,0/30success); PQL HYBRID3fcf25c7 ->239f3eb9 (48.9524%,0/30). Prior conversational answer mistakenly called48.95% BASE HYBRID. Updated quality-example JSON/report now map all4MATD3 CPU children correctly. Use artifact provenance rather than adjacency. No new training/cloud/UI mutation.

## Delivered and verified

- Safe evidence serialization/redaction, exception chain/category/resources/signal/run identity. Optional console/disk/journal/async upload failures isolated; real W&B mandatory-online failure contract preserved. NaN/Inf/unknown objects cannot destroy failure evidence; unknown root cause stays false/unknown.
- Selected warning/replay/PQL/reward/action/safety/report/eval evidence with source-step markers. Resource snapshots at most60s at event cadence and fresh on failure; sample age recorded. No background resource sampler during total process freeze.
- BASE/PQL/offline and CPU owned non-terminating native traceback watchdog; default300s with bounded rearm throttle gives270–300s effective inactivity interval. Native stack is evidence, not automatic deadlock/OOM verdict. Learner subprocess PYTHONFAULTHANDLER=1; negative exits named signal; SIGTERM only when actually observed.
- CPU inner progress/protocol/source/count; optional inner-progress W&B failure does not kill actual evaluation. PQL last metrics/seed context without new locks in exception path.
- Personal view `nw-nwusertruongdaoanhduy-w` / `VmlldzoxODA5MjI1Nw==`:17columns,4pins(Name/State/Created/Runtime),27plots+1guide,12individual runs/plot/noaggregation. Algorithm+Mode hidden; eval count replaces redundant Mode. Added cohort n+disk-free plots; actor loss remains in advanced diagnostic profile.
- Original384run IDs/states/historyLineCounts unchanged during publication and after canary. Unrelated views+all IDs preserved. Project final observation385runs(273finished82failed30crashed,0running), only addition logging-onlycheck `f0vws7re`, finished. Probe verified live status/atomic replacement plus actual resource/source/native stack across4remote files; no quality metrics fabricated.
- Signed-in browser Personal UI:17actual column headers, Created/Runtime/eval count visible, Algorithm/Mode absent; real outcome series render. Tab left at user's Personal Workspace. No personal screenshots saved.
- Fresh independent reviewer found2Important, both observed RED→GREEN.77focused passed. Full offline CPU suite on56604cc:924passed95skipped8warnings1396subtests in554.26s; JUnit0errors0failures. compileall/diff pass;14credential values scanned,0hits. No GPU tests claimed this task.

## Limits and continuation

No universal root-cause guarantee: hard kill/power loss/external platform/account events still need Kaggle/OS/service logs. Old pinned jobs cannot gain new code or missing replay/logs retroactively. Canonical periodic mini-eval remains disabled, so training return alone does not establish generalization. FinalCPU30 describes that seed bank; algorithm ranking needs matched protocol/budget and multiple training seeds/uncertainty. No production20k rerun authorized by this audit.

Minor deferred: watchdog may dump up to30s early due bounded timer reset; diagnostic-only/no kill. All coverage decisions and review scope are recorded in the ledger evidence.

Personal: https://forge.coreweave.com/wandb/truongdaoanhduy-uit/uav_search_target/workspace?nw=nwusertruongdaoanhduy
Runs: https://forge.coreweave.com/wandb/truongdaoanhduy-uit/uav_search_target/table?nw=nwusertruongdaoanhduy
Guide: docs/training-diagnostics.md
Coverage: docs/diagnostic-coverage.md
Evidence: docs/superpowers/evidence/2026-10-28-diagnostic-coverage.md (+tests/personal/browser/probe/service/review/ledger records).
No requested work remains blocked; both remote refs verified at95c4fe5, then this documentation-only integration record is pushed using the same atomic method. Retain existing host worktree, credentials and unrelated views.

## Previous completed task record

# W&B Personal Workspace — COMPLETE / VERIFIED ONLINE

Updated 2026-10-28T00:07:15.128134+07:00. Source `9e37b4a9e7ca7a4b45be55f9a37d3501163b2d10`
pushed and verified on main + feat/cpu-replay-20261022; follow-up docs b7cb7ac.
User authorized W&B sign-in and requested Personal Workspace with Algorithm hidden,
then supplied a new W&B key in key.txt. Latest key freshly authenticated as the
correct user/project; no credential values printed, copied into code or committed.

## Final follow-up result

- Personal view nw-nwusertruongdaoanhduy-w, ID VmlldzoxODA5MjI1Nw==:
 17 visible columns; Algorithm hidden, unpinned and removed from order. Name,
 State, Created and Runtime pinned. Existing26 line plots+1guide unchanged.
- Source allowlist excludes config:runtime.algorithm so normal startup/dashboard
 regeneration preserves the preference. Run names/config retain algorithm data.
 Saved primary view was not modified in this follow-up; user-facing link now
 explicitly opens Personal and avoids switching saved views:
 https://forge.coreweave.com/wandb/truongdaoanhduy-uit/uav_search_target/workspace?nw=nwusertruongdaoanhduy
- Browser sign-in completed through secure browserAuth and device confirmation.
 Fresh W&B UI showed signed-in account and Personal Workspace badge. Actual full
 Runs table was inspected: exactly17 column-action headers, no Algorithm. Delivery,
 success and other mission plots render actual training series. Browser left at
 Personal Workspace with charts and marked as the deliverable tab. No auth secrets,
 cookies or identity-flow URLs were persisted.
- Existing tests updated for preference:3 observed REDs, then10 passed in6.94s;
 7 existing SDK deprecation warnings. compileall/diff check pass;12 credential
 values scanned with0hits. No training/learner/checkpoint behavior change or new
 production job; prior full891pass runtime evidence remains below.

## Evidence and next action

API: docs/superpowers/evidence/2026-10-27-wandb-personal-no-algorithm.json
Browser: docs/superpowers/evidence/2026-10-28-wandb-personal-browser-verification.json
Guide: docs/training-diagnostics.md
No requested follow-up remains blocked. Do not ask sign-in permission again or
restart the authentication flow. Wait for a new task; do not launch training
implicitly. Main/working branch include the verified source and completion record.

## Earlier observability work (historical18-column baseline)

The earlier report below predates the user's17-column preference. Runtime source
bc1b647/full-suite ecd597f was already verified; the follow-up only changes display.

## Delivered and verified

- Personal and primary saved W&B views published/read back at 23:21:09 Vietnam,
  27 Oct. Both retain their IDs: Personal VmlldzoxODA5MjI1Nw==; saved primary
  VmlldzoxODA5Nzc1Ng==, nw-gmcixx5iveu-v.
- 18 visible columns, 4 pinned: Name/State/Created/Runtime plus stage/algorithm/mode,
  account/seed, progress, cohort/eval delivery/success, budget complete, stop reason,
  failure type.26 line plots + 1 Vietnamese guide,12 individual runs/plot, no
  aggregation, all real states, newest Created; correct checkpoint/CPU/runtime axes.
- All 384 run IDs/states/historyLineCount unchanged:272 finished82 failed30 crashed,
  0 running/pending at publication. Unrelated saved views unchanged; normal startup
  reconciliation preserves plots. Before specs retained in ignored outputs/wandb.
- Journal/status/guide, selected scalar evidence, independent console/disk/optional
  upload guards, accurate CPU resume/pause/failure counts and primary traceback.
  Explicit status-only now requests at transitions and at most 60s progress,
  including failed attempts. Growing files end + caught-failure/close flush.
- Real probe ybnn37l0 exposed missed atomic live update for 110s; genuine failed
  state retained. Fixed logging-only probe aqp31obe confirmed two revisions while
  running and 3 files / 3 events after finish. No rollout/quality metrics/GPU allocation.
- **Final offline CPU suite:891 passed,95 skipped,8 warnings,1389 subtests in 529.84s.**
  Job 50d61b83acc4, log /tmp/shell_mcp/jobs/50d61b83acc4.stdout.log. JUnit independently
  read 0 failures/0 errors;2375 total cases including subtests/skips. Exact source ecd597f.
- Independent source review: no remaining Critical/Important/Minor. compileall,
  diff check, no-torch supervisor import pass; secret scan 12 real values, 0 hits.
  Scientific config/algorithm/environment diff empty.
- Earlier broad889 passed2 failed exposed synthetic one-character key state leaking
  across tests, causing conservative traceback redaction to replace normal letters.
  Fixture isolation fixed this; production redaction was not weakened. Interrupted
  pre-final runs are not completion evidence.

## Interpretation / limits / next action

Canonical 20k periodic mini-eval remains disabled; missing data stays empty. Judge
mission delivery/success on matched protocol, not loss or GPU utilization alone.
Created/Runtime describe current W&B run, not submit time or total lineage. Full
upload clock is not captured-state age. Hard kill/network loss may prevent final
traceback/upload; no universal root-cause guarantee. New logging applies to future
source-pinned jobs; historic data remains intact and cannot gain retrospective logs.

User has now explicitly authorized the sign-in required for browser visual QA. Do not start a new scientific/retraining task implicitly. Before any
later launch, recheck legitimate account/quota/idle/datasets/T4/no duplicates and
GitHub-synced owner; launch only on user instruction.

Guide: docs/training-diagnostics.md
Evidence: docs/superpowers/evidence/2026-10-27-wandb-observability.md, with live,
logging-probe and test-results JSON beside it.
Design/plan: docs/superpowers/specs/2026-10-27-wandb-observability-design.md and
docs/superpowers/plans/2026-10-27-wandb-observability.md.

## Previous completed reliability work

General MASAC hardening 820b50e, final evidence/help a046a3b. External GitHub
supervisor active cron 15 min; its orchestration is independent of sandbox.
10 local accounts authenticated at 18:17 on 27 Oct; only account_01/account_02 synced
and allowlisted in GitHub at that observation. No production roots launched.
Old MASAC source 77a1d0f heartbeat loss within 11s remains a shared external-incident
hypothesis, not proven Kaggle/OOM/ban. Retained policy-only artifacts cannot restore
lost replay/optimizer/RNG. Do not silently rotate lineage owners or call a seed
restart exact recovery.

See docs/github-actions-full-flow.md and
 docs/superpowers/evidence/2026-10-27-general-failure-hardening.md for prior evidence
and launch restrictions.
