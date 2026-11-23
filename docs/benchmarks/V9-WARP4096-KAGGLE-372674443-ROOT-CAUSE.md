# NoOracle V9 — GPU T4x2 Warp training-environment root cause

Evidence: Kaggle therwed/notebookcee1ab3313, completed run scriptVersionId=372674443:
https://www.kaggle.com/code/therwed/notebookcee1ab3313?scriptVersionId=372674443

Raw JSON from Kaggle output is committed at
docs/benchmarks/data/V9-WARP4096-KAGGLE-372674443.json.

Both source variants were profiled serially on the same NVIDIA Tesla T4 GPU0,
with 4096 Warp environments, 8 unmeasured warmup steps, 96 measured vector
steps, full NoOracle PHY/MAC/network enabled, and 3 independent runs.
This is environment-only with random actions, not full MATD3 training.

| Version | Trial 1 transitions/s | Trial 2 | Trial 3 | Median |
|---|---:|---:|---:|---:|
| Original pinned V9 1346ad47 | 4241.95 | 4228.44 | 4236.71 | 4236.71 |
| Previously accelerated V9 7d4faa0 | 7924.34 | 7928.73 | 7895.93 | 7924.34 |

Original -> accelerated: +87.04% simulated environment transitions/second.
The GPU trajectory parity output is MATCHED for the V9 scientific arrays
at the instrumented checkpoints.

Kernel median times across three 4096-world by 96-step trials:

| Kernel | Pinned V9 | Accelerated V1 | Decrease |
|---|---:|---:|---:|
| packet_step / PHY-MAC | 76,774.48 ms | 34,959.54 ms | 54.47% |
| sense_bayesian -> indexed | 12,503.17 ms | 11,070.47 ms | 11.46% |
| observe | 2,394.85 ms | 2,365.38 ms | 1.23% |
| sense_reports | 89.91 ms | 10.78 ms | 88.01% |

Primary bottleneck was the network packet_step Warp kernel, 82.7% of
original measured kernel time and 70.55% after prior optimization. The
4096 environments themselves are not sequential Python processes.
No evidence suggests disabling UavNetSim is a valid solution: network
is an essential scientific feature.

## Why packet_step has residual avoidable work

Inside the PHY/MAC flow, the function streaming.clear_pending is invoked
twice per vector step. Pinned NoOracle V9 has 6 UAV, 140 hypothesis slots,
and 7 possible destinations, yielding 6 * 140 * 7 = 5,880 session records
per independent world. The old implementation reconciles ALL 5,880 on
every invocation even when no cells have been confirmed. The NoOracle
report allocator is append-only and slot-contiguous until world reset.
Therefore a session on an as-yet-unallocated report slot cannot exist
under reachable simulation state.

In new experimental V2 branch, clear_pending scans only allocated report
cell targets (same source/target/destination iteration order), keeps
the original full session scan for historical target-ID scenarios, and
preserves all session state transitions including invalid-session
cleanup and PHY/MAC event ordering.

CPU parity on old accelerated V1 versus new V2:
- 16 environments * 400 steps: full selected scientific checksums identical.
- 4 environments * 34 steps with hand-injected pending reports, valid
  and invalid live MAC sessions: ALL WorldState array hashes identical.
- CPU32 * 220 single trial: V1 2599.74 vs V2 2914.70 transitions/second.
  This is diagnostic, NOT a claimed GPU speedup.

Files prepared:
- MAC session V2 CUDA A/B benchmark notebook (for user run; no training).
- Independent source-pinned V2 full MATD3 training notebook (fresh 20k
  episodes; W&B/HF runs separate from e079eace).

The old CUDA +87.04% is verified; incremental V2 improvement is
NOT yet validated on T4. Do not change seed, Actor/Critic, report
semantics, network, sensor, reward, replay, or evaluation datasets.
Residual learner/replay bottlenecks require a separate full-training
CUDA profile to estimate end-to-end speedup.

Warnings from pip in Kaggle output were installed-package compatibility
warnings, not benchmark failures: both source variants produced complete
profile output and matching CUDA trajectory hashes.
