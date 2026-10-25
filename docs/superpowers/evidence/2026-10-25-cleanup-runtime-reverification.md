# W&B cleanup and current-source runtime re-verification

Checked 2026-10-25 23:22 +0700. Tested source: `c4d5225eb34104cfa8e13444ae4f5fac0963f75d`, branch `feat/cpu-replay-20261022`.

## Scope and continuity

This continues the user's authorized W&B cleanup and code/runtime verification. Local Git and a fresh remote ref agreed on c4d5225. No owned source, scenario, algorithm, reward or hyperparameter changed after the prior full-code audit. This pass reconciled the remote cleanup, re-read the critical native/Warp observation/report and actor/critic paths, checked the owned-code/config inventory, re-ran the full suite and independently ran four end-to-end micro-smokes. It does not claim a second line-by-line review of unchanged external UavNetSim internals.

The old handoff's 5/35 was an incomplete snapshot. Remote maintenance run `zlh6haz3` had already finished all 34 bulk artifacts, 0 failed; canary `vxc2hc2e` finished its one artifact. No archive or deletion was repeated in this verification session.

## W&B cleanup proof

- 35/35 replacement artifacts contain only `checkpoint.json`.
- All 35 original versions are absent from the complete current model inventory; all 35 replacements are present.
- All 35 finished producer summaries retain the original-version -> replacement-version mapping.
- All 35 manifest revisions pass fresh HF remote file-size and SHA256 metadata checks.
- A fixed random sample of three old references resolved through the production resolver; all three checkpoints were force-downloaded into new directories and matched their immutable manifest size/SHA256.
- Current full inventory: **132 collections, 1,149 model versions**. The public API inspection covers every version, not merely latest aliases.
- `wandb artifact ls --type model truongdaoanhduy-uit/uav_search_target`: exit 0. Empty historical collections are harmless and retained.
- Current run inventory: **315 total = 228 finished + 65 failed + 22 crashed; 0 running/pending**. This is the handoff's 313 runs plus the two maintenance runs. No run/history/config/metric was deleted.
- All artifact types were inspected by name: model, wandb-history, wandb-events, benchmark, run_table, evaluation-audit. Cleanup scope remains the verified legacy model binaries; history/events are retained.

| Measurement | Bytes | Decimal size |
| --- | ---: | ---: |
| Legacy binaries archived/deleted in the completed batch | 313,266,608 | 313.27 MB |
| Replacement W&B manifests | 15,521 | 0.016 MB |
| Net decrease in model logical bytes | 313,251,087 | 313.25 MB |
| Current model logical bytes, all versions | 54,219,565 | 54.22 MB |
| Reconstructed pre-batch model logical bytes | 367,470,652 | 367.47 MB |
| Fresh W&B project `storageBytes` | 3,482,532,950 | 3.483 GB |
| Fresh W&B entity `storageBytes(cached:false)` | 3,485,826,151 | 3.486 GB |

Storage snapshot: 2026-10-25T23:20:42.051997+07:00. Model logical bytes are not total project storage and do not prove an identical billed/deduplicated byte reduction. The pre-batch model size is reconstructed from verified removed/replacement sizes and current inventory; a fresh pre-deletion billing measurement was not available. The remaining project storage includes retained non-model data; no attempt was made to delete run history to reach a smaller number.

### Five heavy artifacts deliberately retained

| Artifact | Bytes | Reason |
| --- | ---: | --- |
| `uav-network-masac-da90a417:v28` | 16,424,427 | source failed |
| `uav-network-matd3-971920c8-recovered:v0` | 19,251,329 | source crashed |
| `uav-network-masac-f8524504-recovered:v0` | 16,362,975 | source crashed |
| `uav-network-matd3-315da96d:v0` | 861,781 | contains final.pt, outside migration schema |
| `uav-network-matd3-480bb79c:v0` | 861,781 | contains final.pt, outside migration schema |

No additional heavy version satisfies even the finished-source/file-schema scope gate. These retained items must not be removed by silently weakening the archive policy.

## Current-source verification

- AST inventory: **100 source modules, 97 test modules, 8 Python scripts**; no parse failures.
- `python3 -m compileall -q uav_network scripts tests`: exit 0.
- **21/21 YAML profiles** loaded and validated.
- Primary config SHA256 remains `3aa6754dedd5150cea8c2a86ce09f3712d1e365672f5af292279f2ab8b806bc6`.
- Fresh full command: `PYTHONPATH=.:tests python3 -m pytest -q tests`.
- Result: **740 passed, 77 skipped, 8 warnings, 1274 subtests passed in 532.46s (0:08:52)**, exit 0. Eight warnings are the existing third-party Pydantic deprecations and intentional failed-worker cleanup warning.
- Fresh smoke: tracking disabled, real native packet network enabled, one environment, two episodes x 40 steps; each then loads its saved policy and evaluates one episode using seed 2000000044.

| Case | Episodes | Global step | Critic updates | Checkpoint load / finite parameters / evaluation |
| --- | ---: | ---: | ---: | --- |
| MASAC BASE | 2 | 80 | 73 | PASS |
| MASAC PQL | 2 | 80 | 392 | PASS |
| MATD3 BASE | 2 | 80 | 73 | PASS |
| MATD3 PQL | 2 | 80 | 392 | PASS |

All four explicitly report episode_budget_complete=true and stop_reason=episode_budget. MATD3 PQL performs 196 actor updates for 392 critic updates, consistent with delay 2. PQL's tiny-smoke ratio is a scheduling/config result; it is not evidence of a new defect or of production-scale fairness/convergence.

No new confirmed source bug, crash, nonfinite saved model, failed checkpoint load/evaluation or PQL worker failure was found. The previously fixed tracking fail-open completion regression is included in the full suite. No new production code change is justified by these results.

## Dec-POMDP / CTDE conclusion and limits

The native and Warp actor observation paths still range-gate link metadata, and local fresh/recovered report sensing does not consult the remote GCS delivery flags. MASAC's joint actor sampler and MATD3's action path use per-agent observation slices; centralized state is passed to critics in learning. The direct counterfactual locality and version-guard regressions are included in the full suite. **No new confirmed actor-information leakage was found in the inspected paths.** This is bounded evidence, not a proof that no possible defect exists.

Centralized critics with decentralized actors are consistent with CTDE; this does not by itself violate Dec-POMDP execution. The reactive MLP restricts the policy class to memoryless behavior; it does not make an otherwise local actor centralized. Ideal local beacon/admission metadata, shared coordinates/GCS location and coupled safety remain explicit simulator assumptions.

Material scientific/modeling limits remain unchanged:

1. Primary simulator is **XY motion on a 1000 m map at fixed 100 m altitude**. It must not be described as a learned 3D flight policy.
2. Sensor is exact deterministic camera footprint + occlusion, without stochastic Pd/Pf/false alarms. An external manuscript claiming those probabilities would need reconciliation.
3. Propulsion remains the documented surrogate `140 + 0.5*v² + 2*a² W`. With 277200 J, approximate endurance before radio/acceleration is 1980 s stationary, 1459 s at 10 m/s, 1098 s at 15 m/s and 815 s at 20 m/s. Sustained movement is battery-tight against the 1800 s horizon; no automatic reward/battery retune was made.
4. PQL means **PQL-inspired asynchronous architecture** around MASAC/MATD3 losses, not canonical PQL. Async scheduling is not bitwise deterministic.
5. One training seed and 30 evaluation scenarios do not establish algorithm-level statistical superiority. Range-pinned dependencies still allow dependency drift.
6. This host reports CUDA unavailable and zero devices. The **77 skipped tests were not executed** here. The earlier real-T4 78-pass diagnostic remains historical evidence at its pinned source, not a new full-CUDA suite result.
7. The existing published 20k scientific policies predate the local-report correction and remain historical. A corrected scientific comparison must start the four matched cases from episode 0 only when requested.

Primary 1 KiB profile and acceleration flags remain unchanged: report/payload1024 B, buffer3072 B, six UAVs,35 targets,25 obstacles,1800 steps,env4096,batch8192,replay1M,CPU75%,seed44,eval30/2000000044; AMP/compile/fused off. No cloud training/evaluation/controller was started, restarted, cancelled or deleted in this verification session.

## Evidence and next action

- [Complete cleanup inventory and immutable archive proofs](2026-10-25-wandb-cleanup-reverification.json)
- [Current-source structural/runtime smoke results](2026-10-25-runtime-reverification.json)
- [Fresh full pytest log](2026-10-25-reverification-full-suite.log)
- [Fresh W&B CLI artifact listing](2026-10-25-reverification-artifact-ls.log)
- [Prior full owned-code audit](2026-10-25-full-code-correctness-audit.md)

Primary references checked for interpretation: [Lowe et al., centralized training/decentralized execution](https://arxiv.org/html/1706.02275v4), [Oliehoek, Dec-POMDP local execution](https://fransoliehoek.net/research/research-highlights/), [Lyu et al., state-critic bias/variance under partial observability](https://arxiv.org/abs/2408.14597), [W&B artifact API](https://github.com/wandb/wandb/blob/main/wandb/sdk/artifacts/artifact.py).

Cleanup and the requested CPU/runtime re-verification are complete. No repeated deletion or further source fix is pending. Next operational work, if requested, is a bounded real-T4 smoke or a fresh matched episode-0 corrected cohort after fresh cloud/account/controller checks. Do not resume old historical policies as corrected results.
