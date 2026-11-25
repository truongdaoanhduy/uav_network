# Zhang-Xue v4.9 — Warp CSMA backoff freeze vs actual pinned SimPy MAC

Date 2026-11-25. Branch experiment/zhang-xue-v49-native-csma-freeze-20261125, based on v4.8 commit d12895b540c936031def0ff0c2b6b6253dc817a5. No training, no GPU job, no changes to main or pinned upstream UavNetSim-v1 source SHA 64bc3a2424f2976418731ee94f9a8fd3e8810b22.

## Why this implementation matters

Prior v4.0 real-SimPy matched-RNG hidden-terminal evaluation across 64 seeds achieved 64/64 DATA PDR match but 35 later-retry start-time mismatches up to 550 microseconds. Native classic-v1 source mac/csma_ca.py::CsmaCa.mac_send checks the occupied SimPy channel resource while decrementing a DIFS-plus-random-backoff timer. The separate listen coroutine polls the channel at 1us and interrupts mac_send on occupancy; wait_idle_channel polls every SLOT_DURATION=20us. After interruption, the remaining part of DIFS restarts in full, whereas interruption during backoff freezes and preserves the remaining backoff count and adds DIFS after the channel becomes idle. Attempt k uses CW=32*2**(k-1)-1, with randint(0,CW-1).

The new Warp device kernel in uav_network/communication/gpu/classic_v1_freeze_warp.py processes a single sender's *external* busy-carrier intervals (up to four per simulated packet) and its supplied native-CW-consistent random draw. Output is expected first transmission time and interruption count. It supports attempts1-5 and E4096 independent worlds on Warp CPU, validates malformed timings and busy interval overlaps. It does NOT fabricate RNG, real concurrent channel occupancy, routing decisions, ACK arrivals or carrier busy intervals.

## Tests and measured results

Tests in tests/test_zhang_xue_v49_backoff_freeze.py first FAILED importing nonexistent Warp helper, then passed after implementation. They execute the actual original SimPy CsmaCa.mac_send generator, not an assistant-created imitation, with SimPy Resource occupied by an external carrier UAV 400m away, controlled source random draw, and actual PHY send-start recorder.

Evidence: docs/superpowers/evidence/zhang_xue_v49_native_csma_freeze_cpu_20261125.json.

Nine scenarios including no traffic, interrupted DIFS, interrupted backoff, two busy intervals and busy at t=0 all had source SimPy first-TX time exactly equal Warp kernel clock; maximum absolute error across nine cases: **0 microseconds**. A tenth test checks E4096 independent first attempts, an eleventh checks E4096 mixed attempts1–5 exponential contention windows (11/11 focused tests PASS).

Especially source-derived seed45 fifth retry: original source draws 456 slots and begins backoff at 40036us. A relay application E2E ACK occupies the channel at 48530–49020us, [8494,8984]us relative to source MAC wait start. Real upstream MAC and Warp both transmit 9720us after beginning this backoff, absolute **49756us**, reproducing previously observed 550us delay from standalone no-interference projection 49206us. The discrepancy was caused by omitted real relay traffic, not erroneous randint arithmetic.

## Gate5 NOT complete — no training

This v4.9 source-faithful component is separate; it is not yet integrated into the v3.9 simulate_one_hop_retry_batch event scheduler used by v4.5/4.8. External busy carrier intervals here are provided, but a true UavNetSim-like event scheduler must generate channel occupancy from simultaneous DATA, reverse link MAC ACK, HELLO/DV and application E2E ACK frames and match native interrupt/resume ordering and queues. An isolated source replay does NOT prove multi-UAV source-exact MAC parity, full network Gate5, CUDA performance or learned MARL convergence.

Other blockers: full dynamic arbitrary 6-UAV multi-hop under realistic mobility, fully device-resident MAC/queue scheduler without host synchronization, full production energy/action/reward and CUDA E4096 diagnostics. Only then start MATD3/MASAC training/eval/test.
