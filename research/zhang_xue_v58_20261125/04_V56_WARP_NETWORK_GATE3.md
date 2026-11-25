# v5.6 Gate 3 — executable Warp packet network, evidence and blockers

## Evidence obtained on 2026-11-25

- NVIDIA Warp 1.18.0 **CPU backend**, local user's sandbox (CUDA driver unavailable).
- The original monolithic v56 reset/kernel compile exceeded the short compiler timeout;
  the corrected decomposition compiled the *reset* module in 3.09 seconds and
  the *sensor/injection* module in 6.18 seconds.
- Both Warp modules **executed**, reset `2` environments with `ledger_gen=-1`,
  then one source sampled cell 12 and queued a packed 8-byte evidence entry.
- The distinct real **event kernel** compiled in 68.86 seconds on Warp CPU and
  completed a 7-radio 1-hop `UAV0 -> GCS6` scenario: direct route=6,
  DATA delivered at GCS=1, MAC ACK=2, E2E ACK=1, overflow=0.
- All stages are written with Warp `@wp.kernel` / `@wp.func` and device arrays,
  not a Python per-environment MAC simulation disguised as a GPU environment.

## What is and is not complete

Implemented in experimental source: 7 radios, separate GCS, explicit CSMA/CA
states, CCA+DIFS+frozen contention, scheduled DATA/ACK TX, link-gain-dependent
SINR, 64-entry bounded queues, up-to-62 evidence entries, HELLO/DV,
path-vector relay, TTL, dedup, GCS receiver provenance and application-level ACK.

**Validated as a Warp 1.18.0 CPU-backend behavioral smoke (not source-level parity):** two independent worlds, one with `UAV0 -> UAV1 -> GCS6` and one fully disconnected; verified route 0->6 via 1, 1 DATA delivered, 4 MAC ACK, 1 E2E ACK, no forwarding relay fusion, isolated reset; `exit=0` after fixing staggered HELLO timing and the `reset_pending` `%7` vs `%6` out-of-bounds error.

**Scale regression blocker:** E=32 Warp CPU smoke exceeded a 75-second execution limit without completed results. Determine which segment (allocation/reset vs event kernel) dominates via event/allocator profiling; do not extrapolate E=2 speed to E=4096.

**NOT validated for production:** event-level 2-hop parity against the CPU reference under
identical PHY/timing/RNG; collision/hidden terminal stress; MAC ACK loss,
queue overflow and store-carry-forward; 1500-step clock/epoch wrap; full GPU
state reset and memory use; actual CUDA compilation/performance on Tesla T4;
CPU–GPU numerical parity across all cases; terminal/truncation replay integration.

There is **no** claim that the current 7-radio Warp implementation is a
UavNetSim byte-identical clone or an IEEE 802.11 certification. The Warp HELLO
implementation reads a neighbor's current advertised DV state at delivery;
the CPU reference carries a snapshot in its packet. This source-level difference
is among the first parity blockers to resolve.

The `WarpSevenRadioNetwork.run(...)` method refuses an unverified event kernel
unless explicitly called with `experimental=True`. This guard MUST NOT be
removed until the parity and NVIDIA T4 tests have passed with evidence.

## Validation

```
python -m pytest -q tests/test_v54*.py tests/test_v55*.py tests/test_v56_warp_gate.py
# On an environment with warp-lang installed:
python tests/v56_network_topology_cases.py
```

The quick pytest gate avoids automatically compiling a large event kernel.
`tests/v56_network_topology_cases.py` is a manually run two-world Warp CPU/CUDA
behavioral smoke (not a CPU-vs-Warp parity test) and will fail loudly on evidence mismatches.

## CUDA performance plan

Stage 1: E=1 and E=2 two-hop on CPU + CUDA (same ledger/route counters).
Stage 2: collision/TTL/ACK loss/reset and deterministic trace first divergence.
Stage 3: E=32, 256, 1024, 2048 allocations, actual peak VRAM + kernel
execution time + overflow rate on *one* Tesla T4, then 2xT4 separate rollout
lanes. Only after that integrate MATD3/MASAC and long training.

Do not sum the 2xT4 memories. The experimental SoA contains
~563 KB/environment, **not** a measured full allocation. 2048 envs are
approximately 1.07 GiB of these arrays alone; replay, PyTorch weights,
activations, optimizer, context, CUDA graph pools and evaluation are extra.
The v5.5 actor observation is 514 dims and state is 3084 dims, making
1M-transition duplicate-state replay too large for one 16GiB T4. Use the
included `StateDerivedReplayBuffer` with a measured capacity limit.

## References

- Zhang et al., IEEE Transactions on Intelligent Vehicles (2024), DOI 10.1109/TIV.2024.3352581
- Xue et al., Electronics (2026), DOI 10.3390/electronics15091875
- Zeng et al., IEEE Transactions on Wireless Communications (2019), DOI 10.1109/TWC.2019.2902559
- Yates et al., IEEE JSAC (2021), DOI 10.1109/JSAC.2021.3065072
- UavNetSim original: https://github.com/Zihao-Felix-Zhou/UavNetSim
- Warp docs: https://nvidia.github.io/warp/stable/user_guide/interoperability/pytorch.html
