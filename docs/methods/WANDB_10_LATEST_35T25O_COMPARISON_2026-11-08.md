# Live W&B comparison — ten 35-target/25-obstacle BASE runs (captured 2026-11-08T08:50:03.011142+00:00)

## Provenance and zero-retraining rule

This is a **read-only snapshot** from the project
[truongdaoanhduy-uit/uav_search_target](https://wandb.ai/truongdaoanhduy-uit/uav_search_target),
not a fresh training run. Ten most recently created runs match the original
five architecture/update cases for each algorithm: 6 UAVs, 1km, 35 targets,
25 obstacles, 1,500 maximum steps/episode, seed 44, BASE/no PQL,
4,096 GPU vector environments, batch 8,192, replay 1M,
T4×2 notebook recipe, no Critic chunking/AMP/compile/fused optimizers,
actor-only optional LayerNorm.

**Source commit logged by all ten W&B runs:** `a6e07a8f4b199028298b6ffd9a47f2cecd654fc9`.
**Common scenario fingerprint SHA-256:** `5dc95846027125c7e097e2f455c6195844f86bef34cae04445f1acf605d8cedb`.
**Common evaluation protocol SHA-256:** `193106e536d00b260b8e4eaa54b9df77ee2c0d5a6207d4cfeb96e4f957b3ed91`.
**Observation version:** 5.
All ten share the same scientific environment fingerprint and the same
tracked runtime protocol. Every individual case has its own full
`case_fingerprint_sha256` in the [registry](evidence/wandb_10_case_reuse_registry_2026-11-08.json).
The source was the embedded notebook snapshot at the reported commit,
not the later comment/test-only audit commit.

**Never duplicate these experiments merely because a new chat/session starts.**
Reuse the recorded W&B results and checkpoint references as far as they go.
A source comment/documentation change does not invalidate the data.
If any scientifically relevant configuration changes, start a separately named
cohort; if a partial case is explicitly needed at greater horizon,
verify resume/checkpoint availability before extending instead of
restarting by default.

## Exact completed progress

- 10 W&B runs report state `finished` but only **5/10 actually reached 20,000 completed episodes**.
- **5/10** are partial (time-limited and/or early-stopped); their
  W&B `finished` flag is *not* a guarantee of full target completion.
- **All-10 matched training episode budget:** 4,096.
  Training statistic = mean over completed episodes 3,073–4,096.
- **All-10 common fixed-checkpoint validation budget:** 7,372,800 global transitions.
  `eval/mean_*` results use 5 fixed-seed validation episodes.
- `episode/return` and `outcome/*` are **in-sample completed training episodes**,
  not deterministic policy validation. Equal episodes does not imply equal
  transitions because episode length varies. We therefore report both
  matched-episode windows **and** matched-checkpoint validation.
- `best/mean_*` is selected over different available time horizons. It is
  useful for recovering the strongest saved checkpoint but is **not**
  the fair, equal-budget cross-run ranking.

| Run (W&B link) | Architecture | Completed ep | W&B runtime (h) | Training delivery at ep 4,096, last 1024 | Validation delivery at 7.37M transitions | Best validation delivery (unmatched horizon) |
|---|---|---:|---:|---:|---:|---:|
| [MATD3 01](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ccaadcd6) | Shared · H512 · U8 · LN | 20,000 / 20,000 | 2.63 | 40.11% | 25.14% | 58.86% |
| [MATD3 02](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/f55774e2) | Shared · H512 · U16 · LN | 20,000 / 20,000 | 4.87 | 33.09% | 32.00% | 51.43% |
| [MATD3 03](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/55158605) | Independent · H512 · U16 · LN | 16,384 / 20,000 | 10.78 | 23.29% | 1.14% | 22.86% |
| [MATD3 04](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/baf27732) | Independent · H512 · U16 · NoLN | 20,000 / 20,000 | 10.42 | 21.60% | 0.00% | 25.14% |
| [MATD3 05](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/635d7134) | Shared · H256 · U8 · LN | 20,000 / 20,000 | 1.74 | 45.71% | 40.57% | 51.43% |
| [MASAC 01](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/d9c1c5e9) | Shared · H512 · U8 · LN | 11,675 / 20,000 | 10.77 | 44.19% | 3.43% | 46.29% |
| [MASAC 02](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/c70aa6f4) | Shared · H512 · U16 · LN | 4,096 / 20,000 | 10.78 | 37.26% | 5.71% | 19.43% |
| [MASAC 03](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/70eaad3a) | Independent · H512 · U16 · LN | 4,096 / 20,000 | 10.78 | 42.16% | 2.29% | 21.14% |
| [MASAC 04](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/c6cf7916) | Independent · H512 · U16 · NoLN | 4,096 / 20,000 | 10.77 | 41.81% | 1.14% | 12.57% |
| [MASAC 05](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/7f5fd73f) | Shared · H256 · U8 · LN | 20,000 / 20,000 | 8.33 | 44.70% | 4.57% | 52.00% |

Each table value can be traced to the [immutable JSON snapshot](evidence/wandb_latest_ten_2026-11-08.json).
The **W&B runtime** is the recorded run lifetime, not independently
measured GPU-kernel-only compute time.

## Fair matched comparisons within each algorithm

Each difference below means **left case minus right case**.
Positive delivery pp or return indicates that the left case scored
higher under the stated measurement. Training values use the final 1,024
completed episodes up to the pair's minimum observed episode. Validation
values use the latest **identical checkpoint global-step** available to both.
Do not mix these two estimators.

### MATD3

| Cases | Matched ep | Training Δdelivery (pp) | Training Δreturn | Matched eval checkpoint | Validation Δdelivery (pp) | Validation Δreturn |
|---|---:|---:|---:|---:|---:|---:|
| 01 vs 02 | 20,000 | 3.28 | 497.7 | 23.962M | 13.71 | 801.9 |
| 01 vs 05 | 20,000 | 7.24 | 165.1 | 22.118M | -3.43 | -244.9 |
| 02 vs 03 | 16,384 | 42.54 | 2456.2 | 23.962M | 42.86 | 2605.4 |
| 03 vs 04 | 16,384 | -22.19 | -800.9 | 27.648M | -22.29 | -606.9 |

At equal 20,000 completed episodes, **MATD3 case 01 (Shared H512 U8 LN)**
beats **case 02 (Shared H512 U16 LN)** in the trailing training window,
and at shared checkpoint 23,961,600 transitions:
delivery advantage **+13.71 pp**
and return advantage **+801.9**.
Both achieved the 20k target; W&B lifetime was
2.63 h versus
4.87 h. Thus U16 offers no demonstrated
benefit here and costs more wall-clock time.

The strongest MATD3 independent/shared controlled pair is
**case 02 (Shared H512 U16 LN) vs case 03 (Independent H512 U16 LN)**.
At shared checkpoint 23,961,600,
Shared is **+42.86 pp** higher in
delivery rate. Independent case 04 without LayerNorm sometimes beats
independent case 03 but **both remain far below the Shared variants**.
Do not infer from this single seed that independent architectures
are intrinsically incapable of MARL.

**MATD3 case 05 (Shared H256 U8 LN)** achieves the 20k target in
1.74 h, versus case 01 in
2.63 h. It was better in early
same-step validation (40.57%
vs 25.14% at
7,372,800), but its best recorded checkpoint reaches
51.43% compared with
58.86% for H512.
H256 U8 is a useful throughput/early-learning option, not the
highest observed best-checkpoint option.

### MASAC

| Cases | Matched ep | Training Δdelivery (pp) | Training Δreturn | Matched eval checkpoint | Validation Δdelivery (pp) | Validation Δreturn |
|---|---:|---:|---:|---:|---:|---:|
| 01 vs 02 | 4,096 | 6.93 | 578.5 | 7.373M | -2.29 | -120.2 |
| 01 vs 05 | 11,675 | 9.47 | 740.5 | 14.746M | 6.86 | 899.4 |
| 02 vs 03 | 4,096 | -4.90 | -199.4 | 7.373M | 3.43 | -416.7 |
| 03 vs 04 | 4,096 | 0.35 | 75.0 | 7.373M | 1.14 | 461.3 |

The **only MASAC run to finish 20k episodes** was case 05:
Shared H256 U8 LN (20,000 episodes,
8.33 h, best delivery
52.00%).
Shared H512 U8 LN (case 01) reached
11,675 episodes; on an episode-matched
comparison it has +9.47 pp
higher training delivery and at matched checkpoint
14,745,600 has
+6.86 pp higher validation
delivery. It is more expensive and incomplete, so it should
**not** be dismissed as inherently worse, nor declared better at 20k.

MASAC U16 cases 02–04 each stopped after 4,096 completed episodes.
The evidence suffices to document early dynamics and poor throughput
but **does not establish final convergence superiority/inferiority**.
Do not prescribe another identical 20k fresh run from these data;
only extend a stored checkpoint when a new, explicit question warrants it.

## Cross-algorithm interpretation

The most defensible completed-case comparison is **case 05**, because
both algorithms completed 20k with matching Shared H256 U8 LN architecture
and scientific scenario. At the pair's latest common checkpoint
22,118,400 transitions, MATD3 delivery exceeds MASAC by
**+6.86 pp**.
MASAC reaches its strongest validation only later at
25,804,800 transitions; comparing that later
52.00% against MATD3 at an earlier checkpoint is not fair.
The source algorithm remains the causal difference only to the extent
the distinct algorithms also have intrinsically different update cadence.

Do not use MASAC case 01's 11,675 episodes versus MATD3 case 01's
20,000 episodes to claim a final-outcome advantage. At shared episode and
checkpoint budgets, their relative performance varies across learning phases.

## Recommendations based only on these recorded runs

1. **Primary MATD3 baseline:** Case 01, Shared Actor/Critic, H512,
   U8, actor LayerNorm, default initialization. Strongest observed
   validated delivery: 58.86%.
   Fixed-policy validation at its last logged checkpoint:
   58.29%.
2. **Compute-efficient MATD3 baseline:** Case 05, Shared H256 U8 LN;
   substantially faster 20k completion with solid early validation
   and best recorded delivery 51.43%.
3. **Completed MASAC baseline:** Case 05, Shared H256 U8 LN; preserve
   the 20k run and its best checkpoint. Case 01 H512 U8 shows
   promise but is partial, so keep it as a separate research candidate.
4. **Do not use U16 as an automatic improvement:** The matched MATD3
   U8/U16 comparison favors U8 in this specific 35T25O/seed44 setup.
   MASAC U16 evidence is too incomplete to claim long-horizon quality.
5. **Do not re-run these exact cases.** If algorithm code, reward/CBF,
   observation/action, scenario generation, training budget, update rules
   or evaluation protocol changes materially, create a new versioned
   cohort with a new fingerprint. Reusing these already-observed runs
   does **not** require launching new Kaggle training.

The recommendation is conditional on a **single training seed** and
5 periodic validation episodes, without statistical significance
across training seeds. Any future paper-level robustness claim
would require separately justified evaluation; this registry is
not that claim.

## Figures and reproduction artifacts

- [MATD3 delivery validation versus checkpoint transitions](evidence/wandb_ten_matd3_validation_2026-11-08.png)
- [MASAC delivery validation versus checkpoint transitions](evidence/wandb_ten_masac_validation_2026-11-08.png)
- [All 10 at identical 7.3728M checkpoint](evidence/wandb_ten_all_common_validation_2026-11-08.png)
- [Immutable complete metric snapshot](evidence/wandb_latest_ten_2026-11-08.json)
- [Case reuse registry](evidence/wandb_10_case_reuse_registry_2026-11-08.json),
  which records the W&B-reported best checkpoint repository, immutable
  Hugging Face revision and file SHA-256 for all ten cases. This is
  provenance from W&B metadata, not proof of current artifact availability.
- [Pinned HF checkpoint verification](evidence/wandb_ten_hf_checkpoint_verification_2026-11-08.json):
  all 10 Hugging Face commit revisions and the corresponding
  `uav-network/<run_id>/best/checkpoint.pt` tree entries were accessible
  on 2026-11-08. This checks repository metadata only; large checkpoint
  tensors were not downloaded or independently hashed.
- Offline compressed per-episode histories stay at
  `outputs/wandb-ten-20261108/*_episode_history.npz` on the user's
  sandbox laptop. These are local archival data, **not** tracked Git files.
- Script for an explicit refresh: `PYTHONPATH=. python scripts/audit_latest_ten_wandb_baselines.py --ids <ten pinned IDs>`.
  Regenerate report (without any W&B access):
  `python scripts/render_wandb_ten_case_registry.py`.

These reports contain no API keys. **Do not delete W&B source runs or
saved best checkpoints** before a verified backup/reproducibility check.
