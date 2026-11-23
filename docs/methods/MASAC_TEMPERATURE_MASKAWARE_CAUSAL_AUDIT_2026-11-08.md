# MASAC temperature-loss root-cause investigation — 2026-11-08

## Executive result

**No full-budget same-scenario alpha-vs-log_alpha trial exists** in the W&B 35-target/25-obstacle MASAC cohort. Thus **the observed 61% (old) vs 52% (new) best validation is NOT causally attributable to changing temperature loss**.

The strongest isolated evidence in W&B is instead the **masked discrete entropy support**:
two completed 20T/15O MASAC runs on **exact source commit**
`4c9f67ca5c20289a07ebe80fc658fda5738bc3d7`, identical in
relevant science settings except `masac.mask_aware_discrete_entropy_target`,
have best delivery **1%** without mask-awareness
(`62cfc303`) versus **61%** with mask-awareness
(`f6a196a7`). The unmasked run reports
`learner/alpha_discrete=60,630,352` and
`learner/alpha_continuous=27,961.365` at the end,
versus `alpha_d=0.9766` and `alpha_c=0.01026`
for the mask-aware run. This is strong single-seed controlled evidence
that impossible categorical entropy targets destabilized that older
baseline; not proof across independent training seeds.

**Crucial distinction:** both `f6a196a7` (20T/15O)
and `7f5fd73f` (35T/25O) use
`mask_aware_discrete_entropy_target=true`. The 61%-vs-52%
difference cannot be ascribed to enabling mask-awareness.

## Old vs new MASAC — verified W&B source/config

| Parameter | f6a196a7 | 7f5fd73f |
| --- | --- | --- |
| Git source | `4c9f67ca5c20` | `a6e07a8f4b19` |
| World targets/obstacles | **20/15** | **35/25** |
| Training seed | 44 | 44 |
| 6 UAV, map, horizon | 6 / 1000m / 1500 steps | Same |
| BASE Shared Actor/Critic | true/true | true/true |
| Hidden, LayerNorm, update, batch | 256x256 / true / U8 / 8192 | Same |
| Replay, env | 1M / 4096 | Same |
| `mask_aware_discrete_entropy_target` | true | true |
| `temperature_loss` | implicit **log_alpha** in source | explicit **alpha** |
| max training & checkpoint | 36000sec; 128 ep checkpoint | 38767.6sec; 256 ep checkpoint |
| validation interval | 3.6864M transitions | 1.8432M transitions |
| completed episodes | 20k | 20k |
| best validation | **61%** (20 T) | **52%** (35 T) |
| best validation checkpoint global transitions | 22.1184M | 25.8048M |
| best validation search | **67.00%** | **54.2857%** |
| best CBF interventions per UAV step | **0.57694** | **0.67430** |
| absolute mean searched per eval episode | **13.4** targets | **19.0** targets |
| absolute mean delivered per eval episode | **12.2** targets | **18.2** targets |
| delivery / found among avg targets | **91.0%** | **95.8%** |

The newer run is **not a worse communicator** in its best
fixed-validation snapshot: it delivers a larger fraction of the
targets it found. The lower normalized rate originates mostly
in **lower target discovery fraction** on the changed scene,
with more frequent CBF intervention. It actually delivers
**more targets per episode in absolute count**.
The causal source (more obstacles/scene difficulty,
policy/CBF interaction, changed observation sizes,
temperature loss, or combination) is not isolated by W&B.

### Observation input size changes with target count

From SHAs of verified HF best saved Actor snapshots:
`f6a196a7` has actor local observation **198-D**,
shared actor input **204-D** (6 one-hot agent-ID appended),
central critic state **565-D**.
`7f5fd73f` has actor local observation **228-D**,
shared actor input **234-D**, centralized state **830-D**.
In `environment/observations.py`, part of the actor local
input is *per-target known/age memory*, not globally known
true target positions; CTDE remains intact.
Actors cannot be directly transferred across scenarios
without an observation adapter, which would itself introduce
a new confound. **Do not** treat different observation shapes
as the same architecture solely because both hidden layers
have width 256.

## Controlled paired offline intervention: one variable changed

`scripts/diagnose_masac_temperature_35t25o.py` uses:
actual Warp 35T/25O CPU rollout/replay (260 rows),
same mask-aware Shared-MASAC LayerNorm network,
identical initial weights, replay rows, minibatch index
sequences, Torch random sample seeds for both conditions.
Only `masac.temperature_loss` switches between
`log_alpha` and `alpha`.
Hardware bounded for CPU audit: H64x64, batch64,
1 env, 1,200 real `MASAC.update` optimizer steps
rather than 20,000 training episodes.
Unchanged: soft Bellman, Actor/Critic modules, target
entropies, Adam LR, action masking, gradient conventions.
**This is replay-only model stress testing, not a
same-budget end-to-end learning performance comparison.**

| Metric after iterations | log_alpha @160 | alpha @160 | log_alpha @1200 | alpha @1200 |
| --- | ---: | ---: | ---: | ---: |
| alpha continuous | 0.952776 | 0.953198 | 0.697281 | 0.715122 |
| alpha discrete | 0.984045 | 0.984120 | 0.887303 | 0.891063 |
| actor loss | -15.2189 | -15.2244 | -52.9554 | -53.6205 |
| critic loss | 0.9801 | 0.9795 | 0.4791 | 0.4959 |

At 1,200 steps, mean deterministic action RMS difference
is **0.00429** on the fixed held-out observation batch.
The exact same minibatches contain only **1.28%**
agent-states with more than 1 valid destination, so
mask-aware discrete entropy is particularly relevant here.

**Mechanism:** with `alpha=exp(log_alpha)`,
the gradient of `alpha * (H-H_target)` w.r.t.
`log_alpha` is `alpha*(H-H_target)`;
for `log_alpha*(H-H_target)` the gradient
is `(H-H_target)`. They match at `alpha=1`,
but diverge as `alpha` changes.
With Adam, multiplying a gradient by a slowly
varying scalar can be partly canceled by adaptive
gradient normalization: multiplying raw gradient
by `alpha` does **not** mean the Adam parameter update
is necessarily precisely `alpha` times as large.
The short controlled test found no catastrophic
divergence solely from the loss choice under its
restricted conditions. It cannot rule out differences
after 50,000 full-batch updates with H256 and
on-policy replay.

JSON evidence:
- `docs/methods/evidence/masac_temp_35t25o_paired_cpu_2026-11-08.json` (160 updates)
- `docs/methods/evidence/masac_temp_35t25o_paired_cpu_1200upd_2026-11-08.json` (1,200 updates)

## Scientific verdict

1. **Verified strong causal candidate for failed 20T baseline**:
   fixed positive discrete entropy target with almost-always-forced
   silent destination, which drives `alpha_d` up. Enabling
   mask-aware targets changes that source-driven pathological
   objective; `f6a196a7` is much better.
2. **Not the difference between f6a196a7 and 7f5fd73f**:
   both have mask-aware enabled.
3. **Weak evidence for temperature_loss alone causing 9pp normalized drop**:
   no matched full-train W&B pair; in paired real Warp
   replay-only short 35T pilot, `alpha` and `log_alpha` stay close.
4. **Observed bottleneck when scenario changed**:
   discovery fraction falls; delivery efficiency conditional
   on discovery improves; CBF correction rises. Different
   actor/critic observation sizes and layouts are material
   confounders. Increased target density can help some
   discovery events, while more obstacles/harder total
   coverage can hurt: do not assume monotonic difficulty
   from a raw target count alone.
5. **Unresolved evaluation reproducibility**: previous report
   `MASAC_REVIEWER_BASELINE_CORRECTNESS_AUDIT_2026-11-08.md`
   found the same HF actor/seed/layout replays differently
   from archived Kaggle/W&B; do not silently mix local
   replay and original W&B best metrics.

### Next legitimate one-variable full-budget test

For *true* causal attribution of temperature loss:
run paired MASAC BASE **35T/25O** with **same git commit**,
seed44, H256x256 Shared/ActorLN, maskaware=True,
num_envs=4096, U8, batch8192, replay1M,
critic_chunk_size=0, same reward/CBF/network, same
fixed validation seed bank and checkpoint budget;
change **only `masac.temperature_loss`**.
Compare validation at equal **global transitions**
(not best snapshot on mismatched checkpoints);
also report multiple seed limitations, actor action saturation,
CBF intervention, target discoveries, and alpha trajectories.
No GPU/Kaggle training was started for this audit, and
the CPU pilot is not a substitute for this experiment.

## Primary source references

- `uav_network/algorithms/masac.py` (temperature dual),
  `environment/observations.py` (local actor inputs).
- The actual W&B runs:
  https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/f6a196a7
  https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/7f5fd73f
  https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/62cfc303
- Original paper baseline comparison/evaluation warning:
  `docs/methods/MASAC_REVIEWER_BASELINE_CORRECTNESS_AUDIT_2026-11-08.md`.
