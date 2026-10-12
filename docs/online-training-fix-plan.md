# Online training diagnosis and fix

User request: check the Kaggle/W&B runs after laptop shutdown, make them visible
online and verify/restart training. Existing GitHub/local synchronization and
credential-use authorization remain in effect.

Observed at 2026-10-12 15:30 UTC: Kaggle error; both W&B runs failed. MASAC stopped
after 16 episodes, 13,202 logged transitions and 4,601 logged updates. MATD3 stopped
after 25 episodes, 20,402 transitions and 8,201 updates because the remote runner
sent SIGTERM after MASAC failed. Both run IDs are present in the project's API list.

Root numerical reproduction: invalid categorical destinations use float32.min
as log-probability and have probability zero. Multiplication by alpha > 1 produces
infinity before multiplying by zero, so Bellman targets/actor losses become NaN.
Periodic finite checks let an unchecked Adam update poison the model; categorical
sampling then triggers the CUDA probability assertion. Alpha=1.01,2,exp(5) all
reproduce non-finite parameters in a CPU learner regression.

Implementation:

1. Zero log terms at zero-probability destinations before entropy multiplication,
   retaining exact masked probabilities and the adaptive alpha range.
2. Reject non-finite gradient norms before every optimizer step, including updates
   that do not collect scalar metrics. Regression proves Adam state is untouched.
3. Let independent Kaggle learners finish; record successful results/evaluations
   and per-algorithm exit failures. Cover a real failed/successful process pair.
4. Add a real CUDA regression for alpha > 1. Run local suite, independent review,
   publish the fix, then run CUDA checks and start the full profile again.
5. Create a named W&B project workspace with current runs pinned and explicit
   charts for progress, outcomes, safety, energy, network and learner metrics.
   Verify the saved view and actual online history through W&B's API.

Kaggle executes remotely after submission. The laptop is only used to submit,
inspect and download output; training and W&B uploads run inside the kernel.
Do not change old run states to running or manufacture history. A restart must
use fresh run IDs and distinguish old failed runs from live replacements.

Limits: previous failed runs did not reach the 100-episode checkpoint interval.
Existing checkpoints are policy/evaluation checkpoints; optimizer/replay resume
is not implemented. Keep original logs/history, and restart transparently.

Local verification after the fix: 74 tests, 70 passed and four explicit CUDA
skips. Regression checks were observed failing before the fix and passing after.
Independent review found an evaluation failure could prevent preserving the
training results; manifests are now written first and remaining evaluations run.

Saved W&B view: https://wandb.ai/truongdaoanhduy-uit/uav_search_target?nw=gmcixx5iveu
API read-back verified six sections, 26 panels, the `uav-network-2d` group filter,
both old runs pinned/enabled and both IDs returned by the actual project query.
New run IDs will replace the pinned pair after CUDA validation and restart.
