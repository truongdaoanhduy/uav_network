# Framefree T4 Benchmark Decision — 2026-10-23

## Decision

**KEEP GEN1.** Framefree preserves scientific behavior exactly but fails the throughput promotion gate on the representative active-streaming workload.

Promotion rule from the approved plan:

- reject on parity failure, simulator error, or OOM;
- keep Gen1 if active-streaming median gain is below +2.0%;
- keep Gen1 if low-contention median regression exceeds 2.0%;
- otherwise advance Framefree only to a separate production-shaped gate.

Framefree active-streaming median change is **-25.585%**, so it does not advance.

## Provenance

- Branch: `perf/framefree-bitmap-20261023`
- Exact commit: `d0b8fbbd97df7aa452ccfec12bac758173e15f6d`
- Commit title: `fix(net): make framefree cuda gate public`
- CUDA exactness kernel: `duytrngoanh/uav-framefree-cuda-exact-final`
- Benchmark kernel: `duytrngoanh/uav-framefree-t4-benchmark-final`, version 2
- Accelerator: 2 × Tesla T4
- Scenario: V3, 2,048 environments, 1,000,000-byte reports
- Measured steps per repetition: 32
- Repetitions: 3 per scheduler per corpus, alternating scheduler order
- Corpora: `active_streaming`, `low_contention`

Benchmark version 1 is excluded: its launcher contained a mistyped full commit SHA and failed before benchmark execution. Version 2 used the exact local/remote branch HEAD shown above.

## CUDA exactness prerequisite

Public exactness verification on the same commit succeeded before benchmarking:

- CUDA available: true
- GPU count: 2
- GPUs: Tesla T4, Tesla T4
- `tests/test_framefree_cuda.py`: **27 passed**
- CUDA pytest time: 1040.27 s

The public kernel removes the previous private-kernel `kernels.get` credential blocker.

## Scientific parity and simulator health

All six Gen1/Framefree pairs have identical `scientific_metrics` dictionaries.

| Corpus | Repeat | Scientific parity | Gen1 errors | Framefree errors |
|---|---:|---|---:|---:|
| active_streaming | 1 | exact | 0 | 0 |
| active_streaming | 2 | exact | 0 | 0 |
| active_streaming | 3 | exact | 0 | 0 |
| low_contention | 1 | exact | 0 | 0 |
| low_contention | 2 | exact | 0 | 0 |
| low_contention | 3 | exact | 0 | 0 |

No benchmark repetition reported a simulator error or OOM.

## Throughput

Joint transitions per second:

| Corpus | Repeat | Gen1 | Framefree |
|---|---:|---:|---:|
| active_streaming | 1 | 997.386 | 741.120 |
| active_streaming | 2 | 995.871 | 741.078 |
| active_streaming | 3 | 995.826 | 741.042 |
| low_contention | 1 | 6912.128 | 6913.224 |
| low_contention | 2 | 6912.397 | 6913.038 |
| low_contention | 3 | 6913.501 | 6912.830 |

Medians:

| Corpus | Gen1 median | Framefree median | Framefree vs Gen1 |
|---|---:|---:|---:|
| active_streaming | 995.871 | 741.078 | **-25.585%** |
| low_contention | 6912.397 | 6913.038 | **+0.009%** |

The active-streaming regression is stable across all three repetitions and is far outside the +2.0% promotion threshold.

## Memory

Framefree derived bitmap allocation is **851,968 bytes (0.8125 MiB)** for 2,048 environments, matching the design estimate. Gen1 has no derived bitmap allocation.

## Evidence checksums

Downloaded Kaggle evidence:

- `framefree_t4_benchmark.json` SHA-256: `7a59a27b86c87fdc913af10bddb22a02567cadc8810c1363da07199515261dd5`
- `uav-framefree-t4-benchmark-final.log` SHA-256: `7cf3bff31338e61775b4287674ea0b66dc8a061e7b7f09d50dc0acb2ec28ea38`

## Result

Framefree is scientifically exact but materially slower under active streaming. **Gen1 remains the production scheduler.** Stop this Framefree optimization branch at this gate; do not launch a production-shaped Framefree learner probe and do not alter the four running BASE/PQL production jobs.
