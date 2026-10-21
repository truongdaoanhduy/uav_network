# MAC Scheduler Gen2 — Tesla T4 Gate Result

Candidate: `a0122954b6f9bdbe9a52ede99b2045f1ef561c1f`
Profile: `configs/comparison-20k-steps1800-scenario-v3.yaml`
GPU: Tesla T4
Benchmark: 2,048 environments, 4 measured steps after JIT warmup.

## Correctness gate

- CUDA active 1 MB Gen1↔Gen2 exactness: PASS (`1 passed`).
- Simulator errors: 0 for all A/B cases.
- Gen1↔Gen2 scientific metrics: exact aggregate parity for both corpora.
- No production run was stopped or modified.

## Throughput

| Corpus | Gen1 transitions/s | Gen2 transitions/s | Gen2 / Gen1 | Gen1 packet step | Gen2 packet step |
| --- | ---: | ---: | ---: | ---: | ---: |
| low_contention | 6931.56 | 6922.87 | 0.9987x | 0.6268 s | 0.6280 s |
| active_streaming | 429.43 | 330.53 | 0.7697x | 18.4901 s | 24.1979 s |

Active-streaming Gen2 counters: 10,579,646 attempted chains; 8,802,514 completed chains;
12,328,084 fused events; 1,777,070 competitor fallbacks; max chain length 3.

The active-streaming scientific totals were identical, including 1,441,792 generated packets,
1,409,484 received packets, 2,818,929 PHY successes, 0 PHY failures/collisions, and
8427.138506742876 J communication energy.

## Decision

**FAIL promotion gate.** Gen2 preserves semantics but is materially slower than Gen1 on the
active-streaming T4 corpus (~23% lower throughput). Do not restart or replace current production
runs with Gen2. The next performance work should profile the Gen2-only overhead (especially
millions of macro-dispatch/counter operations) before attempting another scheduler design change.
