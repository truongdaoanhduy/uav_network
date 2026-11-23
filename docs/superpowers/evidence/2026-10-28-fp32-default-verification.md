# FP32 default: verified; live retry transition blocked

Follow-up: the user clarified API-only access. New FP32 roots have now been submitted on idle owned accounts05/08/09, with old02/03/04 removed from automatic continuation; see [completed API-transition evidence](2026-10-28-fp32-api-transition.md). The blocked state below is the preceding timestamped snapshot.

Runtime source: `d290b7157a85428eacdfa3343c731a343af9fa37` on `feat/cpu-replay-20261022`. Base before this task: `e7f9d979d3c49abca4c1a0d04ea7514ece24dffe`. The user requested ordinary FP32 training and explained they also wanted general knowledge about omitted AMP settings.

## Implemented and verified

`RuntimeConfig` already defaulted `amp_mode` to `off`. Canonical `configs/comparison-20k-hybrid75-1kb.yaml` and `apply_measured_acceleration_policy` previously selected FP16. Removed the YAML key and cloud AMP injection, so no AMP command argument is necessary. Canonical tag is `amp-off`; authoritative cloud tags still follow the resolved runtime. Explicit AMP opt-in remains supported for controlled tests and historical replay. Diagnostic scripts that explicitly request FP16 and historical evidence were retained.

No change to model architecture, normalization, scientific equations, rewards, optimizer learning rates, environment, replay allocation, batch size, seed, total episodes or algorithm-specific compile/fused policy. Exact source contracts for MASAC/MATD3 remain verified.

The existing FP32 learner path performs ordinary forward/backward/optimizer updates without an AMP scaler. Tests without setting AMP verify real FP32 activations, parameters and gradients, finite outputs and no scalers for both algorithms.

## Test evidence

- Baseline focused39passed/+8subtests in18.18s.
- RED after changing regression expectations:7failed,33passed,+10subtests in21.50s, all expected failures from forced-FP16 profile/policy/launcher behavior.
- Focused GREEN78passed/+460subtests in20.44s.
- Full CPU suite:960passed,95skipped,8known warnings,+1417subtests in501.47s. Fresh JUnit readback:0errors/0failures. No CPU-only run is presented as GPU proof.

## Actual dual-T4 FP32 execution

W&B diagnostic [8df42093](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/8df42093), Kaggle `fwelfijwelfj/diag-fp32-canonical-1028-c36d75`, source d290b715. All four cases passed at canonical E4096/B8192/replay1M and CPU75/GPU25. Each case runs in a fresh child process. The existing real CUDA preflight executes the world, replay, learner, PQL workers and requested compiled graphs; its diagnostic six-collection workload never enters production checkpoints. PQL diagnostic warmup is3, while production warmup remains32.

All owned learner parameters and remaining gradients are FP32 and finite; scaler dictionaries are empty; `amp_autocast_fp16=0`. The encoder dtype spot check is a separate forward after the preflight, not instrumentation of every compiled arithmetic instruction. Both real worker schedules are exercised by the preflight. Actual real-learner CPU tests separately check forward dtype during updates.

| Case | Critic updates | Actor updates | Compiled graphs | Largest device peak reserved |
| --- | ---: | ---: | ---: | ---: |
| MASAC BASE | 32 | 32 | 3 | 3.77 GiB |
| MASAC PQL | 32 | 32 | 3 | 4.43 GiB |
| MATD3 BASE | 32 | 16 | 3 | 3.85 GiB |
| MATD3 PQL | 32 | 16 | 0 | 4.66 GiB |

Each device has about14.56GiB total memory. The bounded preflight is evidence of successful startup and allocation, not proof of the eventual production memory peak or 20000-episode completion. Raw counters, memory, scaler state and dtype checks: [CUDA result](2026-10-28-fp32-cuda-result.json). Executed [driver](2026-10-28-fp32-cuda-check.py).

## Production state: no FP32 replacement submitted

All six existing production roots were read back on W&B+Kaggle at the timestamps in the [structured evidence](2026-10-28-fp32-default-verification.json). They remain running with their original pinned FP16 configuration, including recent retries0650d5b1/44558e6c/87a9500e. All six now expose full recovery checkpoint pointers. No root was cancelled, overwritten, deleted or relaunched in this task. Source-pinned live jobs do not adopt edited repository defaults.

Three FP32 relaunch argument sets are prepared locally in ignored `outputs/fp32-relaunch-prepared.json`, preserving accounts02/03/04, algorithms/modes, E4096/B8192/R1M, seed44, target20000 and external continuation/CPU evaluation. Effective config comparison shows only runtime AMP mode changes (output directory is a launch location); all scientific sections compare equal. The arguments contain no AMP flag. They have not been submitted, to avoid concurrent duplicate roots.

Cancellation is currently blocked by missing exact session identity. Kaggle's public cancellation API requires `kernel_session_id`; public ref-based status/info expose status, base kernel ID and version ordinal, not the current session ID. The base kernel ID was not used for cancellation. CLI has no cancellation command. The connected browser extension timed out; the cloud browser later recovered but the private notebook returned404 and Kaggle homepage showed Login. No sign-in credentials were entered and no browser cancellation was attempted.

Continue with an authenticated owner UI session for each exact retry root, or a verified cancellable session ID. Verify the root is terminal/cancelled before submitting its prepared FP32 replacement. A cancelled root is already excluded from supervisor recovery by the existing `reconcile_ref` branch. Preserve all old histories/checkpoint artifacts and healthy accounts01/06/07. Changing precision cannot be called an exact FP16 continuation; use a clearly labeled new FP32 lineage rather than silently weakening full-checkpoint mode validation.

## General PyTorch knowledge

PyTorch initially uses `torch.float32` as its default floating dtype. With ordinary FP32 inputs/model and no autocast, typical model operations use FP32; there is no need to write a per-function AMP argument. This is FP32 with AMP off, not an AMP-FP32 mode. Actual dtype still matters: existing half tensors, explicit dtype, NumPy float64 tensors or a changed default can differ. Disabling AMP does not itself cast those tensors. Integer/index/boolean tensors retain their appropriate types.

A real unmodified-default CPU example (`nn.Linear`, `randn`, forward) produced float32 for default dtype, parameter, input and output. Official references: [default dtype](https://docs.pytorch.org/docs/2.14/generated/torch.get_default_dtype.html), [AMP](https://docs.pytorch.org/docs/2.14/amp.html); version-specific Torch2.11 docs were also queried through Context7. FP32 removes the observed FP16 scaled-backward overflow mechanism, but finite-range arithmetic/invalid operations can still produce Inf/NaN.

## Review, decisions and limits

Fresh-context review of e7f9d979..d290b715:0Critical,0Important. Deferred Minor: the diagnostic wrapper may overwrite the detailed preflight failure JSON when the inner check raises; its gate still fails safely and process console traceback remains. All four cases in this task passed, so this limitation did not affect the observed result.

Scope decisions: transition only the three recent retries and retain explicit AMP opt-in. If broader removal was intended, older healthy pinned jobs and the optional internal parameter still exist; the new canonical defaults and omitted-flag launch path are FP32. Actual CUDA/full-suite questions declined by the initial reviewer are now answered by fresh results. Live migration remains explicitly unperformed.

Final evidence checks:2structure tests/+452subtests passed in0.19s; scientific source contracts passed; diff check passed. No production cancellation call or FP32 production submission occurred. Next action is the exact-root cancellation/authentication gate above, followed by three FP32 submissions and fresh W&B/supervisor verification; do not report those operational steps as done.
