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
