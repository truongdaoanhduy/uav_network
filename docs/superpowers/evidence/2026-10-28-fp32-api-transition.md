# FP32 production transition with API-only account access

Completed startup verification on 2026-10-28. Production source `ca53c74bcdea557cca681f77082eb6cf323f64cd`, runtime/config identical to verified `d290b7157a85428eacdfa3343c731a343af9fa37`. Branch `feat/cpu-replay-20261022`. The user clarified that only API credentials are available for the nonprimary Kaggle accounts and their web sign-in details are forgotten. API introspection resolves their usernames without needing passwords.

## Applied FP32 default

Canonical YAML omits AMP and inherits `RuntimeConfig.amp_mode=off`; the cloud acceleration policy no longer injects FP16. No AMP CLI parameter is present in these three launches. Actual W&B configs read back `amp_mode=off` and tag `amp-off`. Existing optional explicit FP16 remains for controlled probes/historical replay. No model normalization or algorithm equation was added.

For general PyTorch knowledge: initial default floating dtype is float32. Ordinary model/input float32 operations without autocast use FP32. This is AMP off, not an AMP-FP32 mode. Dtype and outer autocast context still matter; disabling AMP does not convert a preexisting half/double tensor. Real nn.Linear/randn/forward example produced float32 throughout. References and earlier full evidence: [FP32 default verification](2026-10-28-fp32-default-verification.md).

## New FP32 production roots

Fresh seed44 starts, initial episode/global_step0, no warm-start/replay/optimizer restore. Actual comparisons against their respective old FP16 runs show equal world/sensor/network/energy/reward/MASAC/MATD3/PQL sections and full runtime differs ONLY by AMP mode. Episodes20000,E4096,B8192,replay1M,HYBRID CPU75/GPU25,two Tesla T4s, evaluation30 seed2000000044 and per-algorithm compile policy are retained. Account choice is operational; these are new FP32 lineages, not exact precision-changing resumes of old checkpoints.

At09:41UTC all three were running on Kaggle and W&B with fresh heartbeat and positive learner progress. Latest checked actor/critic losses and gradient norms are finite; this is a timestamped observation, not a guarantee for the eventual full run.

| Case | Account / API-resolved username | W&B run | global_step | train/updates |
| --- | --- | --- | ---: | ---: |
| MASAC PQL | account_05 / `fwelfijwelfj` | [dff78acf](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/dff78acf) | 208896 | 136 |
| MATD3 BASE | account_08 / `kljgewkfjl` | [a85d067a](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/a85d067a) | 487424 | 696 |
| MATD3 PQL | account_09 / `ewlgkjewkf` | [b8d0b40a](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/b8d0b40a) | 548864 | 800 |

Kaggle refs:

- MASAC PQL: `fwelfijwelfj/gh-tra-g3-masac-pql-h75-102809-92a283`, replaces the requested FP32 counterpart of old0650d5b1/account02.
- MATD3 BASE: `kljgewkfjl/gh-tra-g3-matd3-base-h75-1028093-890afc`, counterpart of old44558e6c/account03.
- MATD3 PQL: `ewlgkjewkf/gh-tra-g3-matd3-pql-h75-10280932-1f7593`, counterpart of old87a9500e/account04.

All3 manifests retain auto_continue, cpu_visualize, external_supervisor_required and orchestration=github_actions. The configured final CPU evaluation is30episodes, seed2000000044, stride5,fps10. Benchmark companion runs60c95c37/47e5d453 finished before their respective PQL learners; benchmark transitions are separate from the fresh training lineage.

## Why idle API accounts were used

Cancelling a current Kaggle session requires the exact kernel_session_id. Public ref-based APIs returned status and notebook metadata but no cancellable current session identity. A base notebook ID was never sent to cancellation. Browser sign-in through the user's chosen Google method established only the remembered primary account duytrngoanh; it could not open the private account03 root. After the user clarified API-only access, no further forgotten-account login was requested.

Before submission, account05's diagnostic was COMPLETE and accounts08/09 had no owned active kernels. Quotas were about29.92h/30h/30h, above the11h launch requirement. Identities were rechecked sequentially; an initial concurrent identity query was discarded before any launch because the helper temporarily changes process environment. Each production launch then ran in its own process. Hosted GitHub service checks independently confirm the exact05/08/09 identities.

This uses new idle accounts to deliver FP32 training immediately. The three old FP16 retry sessions were NOT cancelled. They may finish their currently submitted segments and continue consuming their existing account quota until their existing limits. Their data/history/checkpoints remain intact. Their accounts02/03/04 have been removed from automatic recovery, continuation and final CPU scheduling by the external supervisor. No claim that their GPU processes stopped immediately.

Healthy accounts01/06/07 remain in automatic supervision and their jobs/configs were preserved. Fresh W&B+Kaggle readback shows all six old roots still running with their old FP16 source; all had existing full recovery pointers. Their exact observed progress and artifacts are retained in the structured evidence.

## Independent supervisor verification

`KAGGLE_ENABLED_ACCOUNTS` readback is now `account_01,account_05,account_06,account_07,account_08,account_09`. Existing secrets already covered these accounts; no new credential values were printed or embedded. Old retry accounts02/03/04 are absent.

- [Hosted service probe40665341474](https://github.com/truongdaoanhduy/uav_network/actions/runs/40665341474), job122738305307, SUCCESS: all six enabled Kaggle identities plus W&B/HF statusok; dry reconciliation reports every new exact root active/running at09:35:16UTC.
- [Non-dry reconciliation40665691175](https://github.com/truongdaoanhduy/uav_network/actions/runs/40665691175), job122739467717, SUCCESS: all three new roots and healthy01/06/07 active/running at09:37:53UTC. Its report has no account02/03/04 entry.
- Workflow remains active on main. Existing FP16 source0042166 hands continuation to github_actions rather than submitting a local child, so disabled accounts do not regain an automatic child from the old kernel code.

This verifies configured independent supervision and actual hosted reconciliation. Future continuation still requires confirmed full-state checkpoints and available quota/services. No20000-episode completion or uninterrupted-service guarantee is claimed.

## Verification and evidence

The runtime change was reviewed with0Critical/0Important findings. Focused78passed/+460subtests; full CPU960passed,95skipped,8known warnings,+1417subtests in501.47s, JUnit0errors/0failures. Actual dual-T4 diagnostic8df42093 passed all4canonical BASE/PQL MASAC/MATD3 cases, with real FP32 parameters/gradients, empty scalers, no FP16 autocast, real PQL worker and compiled graph execution. [Raw CUDA result](2026-10-28-fp32-cuda-result.json).

[Structured transition evidence](2026-10-28-fp32-api-transition.json) contains original launch manifests, account/quota preflight, exact live config comparisons, heartbeats, latest metrics, old checkpoint pointers, API-to-username map and both supervisor reports. The earlier blocked snapshot remains as a historical record; this API-only alternative supersedes its operational blocker for launching new FP32 lineages.

Rulings: keep explicit AMP opt-in, preserve healthy01/06/07, and use fresh05/08/09 roots after API-only clarification while retiring old02/03/04 from automatic follow-up. Cost: the old FP16 segments continue to use their existing quota until they end; the new roots start again from seed44. Deferred Minor from the one code review: the diagnostic wrapper can overwrite detailed failure JSON if its inner preflight raises; the gate still fails safely and console traceback remains. All four actual probe cases passed, so it did not affect this evidence. No extra product code was changed after the full runtime verification.

Final documentation/evidence checks:2structure tests/+453subtests passed in0.19s; diff check and runtime/source equality passed; all deployment/config/supervisor consistency assertions passed;12credential values scanned across5changed files,0hits.

## API account identities

No API key or password is included. These names were authenticated from the existing local API credentials:

| Alias | Kaggle username |
| --- | --- |
| account_01 | `duytrngoanh` |
| account_02 | `vuliu123456` |
| account_03 | `egkljwigojwof` |
| account_04 | `gkgijewjf` |
| account_05 | `fwelfijwelfj` |
| account_06 | `fwklgeewjlfew` |
| account_07 | `lkejflwejf` |
| account_08 | `kljgewkfjl` |
| account_09 | `ewlgkjewkf` |
| account_10 | `ewkljfwef` |

Next action: preserve the three FP32 roots and healthy01/06/07. For future status requests, fresh-query Kaggle/W&B plus supervisor reports. Do not relaunch duplicates or claim retired FP16 segments were cancelled. The first new full backup is scheduled by the existing cap; startup proof alone does not mean it has already uploaded.
