# Zhang–Xue v4.2: persistent Warp evidence, local DS and no-oracle DV routing

Date 2026-11-25. Branch `experiment/zhang-xue-v42-warp-evidence-routing-20261125`, based on v4.1 commit `555f2a816c637103c4a81b14b2d6608bbf1fad36`. **No training**. Original pinned UavNetSim-v1 untouched.

## Implemented and tests

- `zhang_xue_origin_evidence_warp.py`: actual persistent Warp `[E,receiver,origin,400]` int32 counters (separate positive/negative). Local sensing copies only the sender's own counts from the Warp camera map. Remote origin entries become visible to one recipient only after a corresponding successful DATA receipt. Per-origin `atomic_max` means retransmissions and duplicate relay copies cannot inflate confidence. Warp computes receiver-local MAX-plus-SUM DS posterior (equal m_f=m_e=0.8) matching existing `ds_belief.py`, with normalization, bounded high counts and no target truth access. These are genuine Warp kernels, on CPU here.
- `zhang_xue_payload_store_warp.py`: immutable, 62-entry source-origin snapshots stored per GPU-resident queue slot, identified by packet ID. A packet survives blackout with original counts; later sensing cannot rewrite it. Recycled slots cannot reuse a prior packet payload. Receiver merges run as Warp kernels directly against device-resident belief without first decoding via NumPy. Host is used for input validation and control signals; not full GPU-only simulator.
- `zhang_xue_distributed_routing_warp.py`: no map coordinates or ground-truth links. Direct links require bidirectional received HELLO frames and expire by TTL. Routes greater than one hop require **an actually delivered** neighbor distance-vector frame; no centralized BFS/oracle. Updates/expiry computed on Warp arrays.
- `verify_zhang_xue_multihop_flow_v42.py`: 3-node line UAV0↔UAV1↔UAV2. One-hop HELLO and DV controls pass the real Warp MAC; before received DV the source cannot reach UAV2. Warp camera creates source counts, queue captures actual sensor evidence, MAC sends DATA 0→1→2, each recipient merges origin evidence on Warp, and a 90B application E2E ACK returns 2→1→0. The relay copies packet payload and original DATA creation time/TTL. Four interleaved blackout schedules across E4096 worlds.

## Verified evidence

- 82/82 focused regression tests across v3.7–v4.2 and source-pinned SimPy passed on local CPU.
- E4096 2-hop deterministic preflight: 16,384 successful isolated HELLO sends and 8,192 successful DV sends; initial source has no 2-hop route; after DV it learns via UAV1. 4096 DATA generated; 3072 reach final UAV2, 3072 E2E ACK return to original UAV0, 1024 DATA expire at TTL, 3072 DATA and ACK relay forwards each. No blackout sends, no cross-world or receiver-local evidence leak, queues empty. Measured ~4.2s on Warp CPU for this tiny controlled workload (NOT training throughput).
- Evidence JSON `docs/superpowers/evidence/zhang_xue_v42_4096_multihop_map_queue_e2e_ack_cpu_20261125.json`.
- Warp v4.0 isolated 64-seed native SimPy collision parity still matches DATA PDR but 35 retransmission start times differ up to 550µs. This is not disproved by the new easy route case.

## Unresolved Gate 5 — MUST NOT START TRAINING

- Multihop remains a **deterministic pilot**: each user world has one DATA; Python host orchestrates queue admission, relay and one-hop Warp MAC calls. Full 6-UAV concurrent traffic, dense collisions, source rate control and on-device networking need work.
- Routing control packets are isolated 90B MAC trials, not continuous HELLO/advert queues with CSMA contention and source-exact packet format. No full source-exact SimPy event/ACK/backoff and channel freeze parity.
- No producer-side packet scheduling for all 400 dirty cells, per-peer ACK watermarks, periodic resending under multiple targets, loss recovery across many episodes, queue pressure under high offered load, or real app-level reward accounting.
- Scenario pilot uses a fixed camera truth for test determinism and artificial timed blackout (source truth is NOT in the payload or routing). It is **not** sampled realistic 200-step Zhang–Xue scenario with learned actions, DWA/CBF replacement, energy, battery, reward or training.
- Warp CUDA not yet tested on a real NVIDIA GPU. Memory use of full `[4096,6,6,400]` evidence doubles is substantial; GPU stream and host transfers need profiling.
- Kaggle account2 to be used for GPU validation where feasible; never reveal credential content.

## Next steps

1. Compare full source SimPy MAC event trace on competing multi-hop traffic including E2E ACK and HELLO; implement exact CCA freeze and ACK channel occupancy.
2. Integrate persistent payload+DV with all 6 UAVs, concurrent sends, multi-step episode maps and source ACK watermarks; keep source-owned evidence and receiver privacy.
3. Real CUDA E64/E1024/E4096 checks (device memory, identical results to CPU and no host synchronization bottleneck), use Kaggle account2 only if available; then full 200-step no-training environment smoke.
4. Gate full E2E tests and reproducibility, only then train MATD3/MASAC (evaluation and test after training), preserving Git checkpoints.
