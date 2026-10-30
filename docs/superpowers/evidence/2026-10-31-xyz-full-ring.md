# XYZ full replay ring on two T4 GPUs

All six cases passed at runtime source `481f81f288d8e84a010dba52e4c8e30b5eebd391`.
Each case runs MASAC and MATD3 separately: E4096, B8192, replay1M, FP32/eager,
256 collector rounds = 1,048,576 inserted joint transitions per algorithm.
Every job passed 15 CUDA tests (including 4 XYZ), 4 scenario CUDA and 8 mission tests.
PQL also passed 11 smoke tests. PQL replay counters show 1M resident rows and
48,576 overwritten rows. All emitted learning metrics are finite; explicit
nonfinite counters were not emitted. No profile was shrunk.

| Profile | Mode | Algorithm | Updates/s | Peak PyTorch GB | Result |
|---|---|---|---:|---:|---|
| independent-h75 | BASE | MASAC | 1.557 | 3.98 | passed |
| independent-h75 | BASE | MATD3 | 7.056 | 4.00 | passed |
| independent-h75 | PQL | MASAC | 0.480 | 4.72 | passed |
| independent-h75 | PQL | MATD3 | 0.624 | 4.21 | passed |
| shared-fullgpu | BASE | MASAC | 1.901 | 13.77 | passed |
| shared-fullgpu | BASE | MATD3 | 25.593 | 13.81 | passed |
| shared-fullgpu | PQL | MASAC | 0.460 | 14.49 | passed |
| shared-fullgpu | PQL | MATD3 | 0.682 | 14.06 | passed |
| shared-h75 | BASE | MASAC | 1.851 | 4.03 | passed |
| shared-h75 | BASE | MATD3 | 11.633 | 4.08 | passed |
| shared-h75 | PQL | MASAC | 0.486 | 4.74 | passed |
| shared-h75 | PQL | MATD3 | 0.632 | 4.19 | passed |

BASE rates measure the isolated learner: 2 warmup and 20 measured updates.
PQL rates include collection and replay filling; readiness-only warmup 248 yields
72 critic updates (MASAC 72 actor, MATD3 36 actor). Production PQL warmup remains 32.
These rates are not a matched BASE-vs-PQL throughput comparison or convergence proof.

Shared FullGPU fits this measured workload. PyTorch peaks exclude Warp and driver
allocations; the Warp environment occupies about 2.85GB on the collector GPU.
FullGPU remains a valid comparator; these short runs do not establish learning quality.

Hybrid BASE's existing `cuda_pipeline_verified=false` reflects its configured CPU
replay label; collector/learner still use CUDA0/CUDA1 and hot 25% replay uses GPU.
PQL's historical `target_t4_scenario_verified=false` describes an older fixed
25-target/1000-step/500k-replay stamp; it is preserved rather than reinterpreted.

No 1200-step training episode or 20k training result is established here.
[Machine index](2026-10-31-xyz-full-ring-index.json) and
[full measured results](2026-10-31-xyz-full-ring-results.json) preserve exact URLs,
source, device flags, timings and memory. Raw artifacts remain on the authorized
host in `outputs/xyz-full-ring-20261031/`.
