# Receiver Pending Index T4 Decision

## Scope

Task 5 evaluates the experimental `rxindex` scheduler against unchanged Gen1 on the UavNetSim-v1-based Warp/CUDA packet simulator. `rxindex` changes only receiver-pending selection; Gen1 remains the production default and rollback oracle. No production run is restarted or migrated by this task.

## Frozen candidate

- Branch: `perf/receiver-pending-index-20261022`
- Frozen benchmark SHA: `38c8b17d2a7eb11975e17ed7036558c37e9537a4`
- Scenario profile: `configs/comparison-20k-steps1800-scenario-v3.yaml`
- GPU: Tesla T4
- Benchmark shape: 2,048 environments, four measured post-JIT steps, three repeats per scheduler/corpus
- Report workload: 1 MB active streaming
- Schedulers: Gen1 and `rxindex`

GitHub remote head was verified to equal the frozen SHA before launching the T4 jobs.

## CUDA exactness gate

Kaggle kernel: `mflkelwfw/uav-task5-rxindex-exact-20261022-034555`

The fixed active 1 MB CUDA corpus compares Gen1 and `rxindex` canonical snapshot arrays with exact tolerances (`rtol=0`, `atol=0`, `equal_nan=True`) and requires a live 1 MB session so an idle corpus cannot pass accidentally.

Result:

- `1 passed in 267.43s`
- pytest exit code: `0`
- candidate SHA in result: `38c8b17d2a7eb11975e17ed7036558c37e9537a4`
- exactness JSON SHA-256: `51c454977a3e6cfed74d1dab48693e07098c6a3391f821d053cddef17f99a6b2`

CUDA canonical-state exactness therefore passed before throughput benchmarking was allowed to proceed.

## Repeated T4 benchmark

Kaggle kernel: `kjhihgiugy/uav-task5-rxindex-bench-20261022-035238`

Scheduler order alternated by repeat to reduce warmup/order bias.

### Low-contention corpus

| Repeat | Gen1 transitions/s | rxindex transitions/s |
|---:|---:|---:|
| 0 | 6961.5111 | 6971.7487 |
| 1 | 6949.9792 | 6954.3132 |
| 2 | 6952.6552 | 6951.4184 |
| **Median** | **6952.6552** | **6954.3132** |

Median speed ratio: **1.000238x**, or **+0.0238%**.

Median packet-step wall time:

- Gen1: **0.623578 s**
- `rxindex`: **0.623245 s**

Low-contention behavior is effectively neutral.

### Active-streaming 1 MB corpus

| Repeat | Gen1 transitions/s | rxindex transitions/s |
|---:|---:|---:|
| 0 | 430.0921 | 433.8307 |
| 1 | 431.1864 | 433.6093 |
| 2 | 431.3490 | 433.7487 |
| **Median** | **431.1864** | **433.7487** |

Median speed ratio: **1.005942x**, or **+0.5942%**.

Median packet-step wall time:

- Gen1: **18.413471 s**
- `rxindex`: **18.301512 s**

The active-streaming improvement is positive in the repeated benchmark, but small.

## Scientific parity and capacity

For every repeat and both corpora:

- Gen1 scientific metrics == `rxindex` scientific metrics: **PASS**
- simulator error count: **0**
- unexpected terminated worlds: **0**
- OOM/capacity failure: **none**

The benchmark does not alter path loss, LoS/NLoS, SINR/interference, collision handling, CSMA/CA, ACK/retry, propagation delay, TX power, communication energy, report custody, relay hops, RNG, event timestamps, or event priority/insertion ordering.

## Memory cost

Reported device-state allocation:

- Gen1: **1,190,191,104 bytes**
- `rxindex`: **1,283,833,856 bytes**
- Added derived receiver-index state: **93,642,752 bytes = 89.30 MiB**
- Increase relative to Gen1 allocated simulator state: **7.87%**

The extra memory fits the 2,048-environment T4 benchmark, but the cost is material compared with the small throughput gain.

## Decision

**Decision: `promote_to_next_gate`.**

This follows the predeclared Task-6 rule: CUDA/scientific correctness gates passed and repeated active-streaming median throughput is positive versus Gen1.

This is **not a production promotion**. The measured benefit is only **+0.5942%** while adding **89.30 MiB** of derived state. A later production-shaped collector/training gate should keep `rxindex` only if it produces a meaningful end-to-end wall-time improvement under the real MASAC/MATD3 workload. If that gain disappears, Gen1 remains the correct production implementation.

No current Scenario V3 production run is changed by this result.

## Evidence files

- Exactness result: `outputs/task5-rxindex-t4/exactness/downloaded/task5-rxindex-exact.json`
- Benchmark aggregate: `outputs/task5-rxindex-t4/benchmark/downloaded/task5-rxindex-benchmark.json`
- Benchmark JSON SHA-256: `9ffc76c5546df691f88acf6778fddc14e7fd9f24c5f7815ab86c6a0dbae6b268`
- Exactness JSON SHA-256: `51c454977a3e6cfed74d1dab48693e07098c6a3391f821d053cddef17f99a6b2`
