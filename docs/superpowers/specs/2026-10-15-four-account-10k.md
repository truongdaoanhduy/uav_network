# Four-account matched 10k training design

User outcome: four independent Kaggle series, one account per BASE MASAC,
BASE MATD3, PQL MASAC and PQL MATD3. All use the same scenario and hyperparameters,
seed44, target10000 completed episodes, W&B online. Closing laptop/ChatGPT does
not control the jobs. After the target completes, a separate CPU-only kernel
creates videos/plots/metrics in Kaggle Output.

Common profile: configs/comparison-10k.yaml, env4096, batch8192, replay500000,
hidden256x256, learning_starts131072, fixed updates8, PQL collection ratio8,
warmup32, 25targets/15obstacles/H1000. Only pql.enabled and algorithm selection
differ in algorithm config. Compute layout differs by architecture: base uses
GPU0; PQL collectorGPU0 and critic/policy/replayGPU1. Both request the same T4
session. Preserve simulator, reward, movement and model implementation.

Reuse existing private credential datasets, immutable W&B checkpoint artifacts
and Kaggle script bootstrap. Add --auto-continue for single-algorithm train or
validate; it implies CPU handoff. GPU runner checks uploaded summary: complete
budget -> CPU child, clean time-budget stop with positive progress -> GPU child
using full v3 resume, other failures -> explicit failure output. Each GPU child
has fresh W&B run ID linked through root_kernel_ref and continuation_index.
Final CPU URL stays tied to root ref. Pinned source/config/evaluation bank and
selected account persist across hops.

Both trainers save light v2 checkpoint.pt for CPU and full v3 resume.pt with
weights/target networks/optimizers/replay/counters/RNG for continuation. Resume
resets active simulator episodes and says so; no bitwise trajectory claim.
Reported recorded episodes stop exactly at10000 even if final vector step
finishes additional workers. A final vector collection may contain additional
transitions; this is disclosed, not counted as additional completed episodes.

Quota is checked before each GPU submission; insufficient/unavailable quota,
failed artifact upload, missing/stale state and zero-progress stops fail visibly
with retained checkpoint/summary. No promise to repair arbitrary platform failures
or wait through weekly quota reset. CPU metadata has enable_gpu=false,
enable_tpu=false and no accelerator flag; only W&B credential mounted.

Validation: CPU full suite, real four-case short GPU/CPU smoke, and forced
GPU continuation smoke. Launch long series only after relevant smoke verifies
source and artifacts. Compare throughput/time/updates/replay reuse and held-out
search/delivery/completion/energy/safety metrics. Single-seed comparisons describe
these runs; they do not establish algorithm-wide superiority.
