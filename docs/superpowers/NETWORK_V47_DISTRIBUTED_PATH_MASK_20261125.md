# Zhang-Xue v4.7 — Warp distributed path-vector loop prevention

Date: 2026-11-25. Branch experiment/zhang-xue-v47-pathvector-loop-guard-20261125 from v4.6 commit 307e785645cf0c491aaea2400d2461ada11153be. No training, no Kaggle/credentials, no main edits.

## Proven original issue (TDD RED then GREEN)

Poisoned reverse introduced in v4.6 correctly prevents two-node count-to-infinity after link failures but did NOT prevent three-node route loops in a triangle. Controlled timeline with HELLO TTL=100µs:
1. UAV0↔UAV1, UAV1↔UAV2, UAV2↔UAV0 and UAV0↔UAV3 present.
2. UAV1 learns destination3 via UAV0, then UAV2 learns destination3 via UAV1.
3. Triangle HELLO links refresh but UAV0↔UAV3 expires at t108.
4. A stale, still-valid UAV2 advertisement for destination3 is delivered to UAV0 at t109. Previous code created UAV0→UAV2→UAV1→UAV0, even with poisoned reverse. Same issue reproducibly occurred across E4096 independent Warp CPU worlds.
5. Three new RED tests reproduced actual loop. This is a concrete behavioral bug, not a speculative paper recommendation.

## Device-side solution

The existing distributed Warp routing module now tracks a six-bit path mask for each [env, UAV, destination]. Direct link paths contain source and destination; a 2-hop path contains all three nodes. Each *physically received* path-vector advertisement stores its sender's current path metadata in the receiver's neighbor-advertisement table (new device array). A sender poisons an advertisement if the route path already includes the receiving neighbor, rather than merely checking the immediate next hop. When selecting next hop, a UAV rejects any advertisement whose path already contains itself, lacks the peer or lacks destination. The receiver constructs its own immutable result path mask for that destination.

The v46 poisoned reverse rule remains active and original bidirectional HELLO/DV reception/TTL checks stay intact. No global map, target truth or physical channel access is read by this routing kernel to construct a route.

This is a model-level *path-vector* defense and not a claim about upstream classic UavNetSim's wire-format compatibility. Current HELLO/DV proof uses 90-byte controlled MAC trials and Python metadata handoff; a real packet must include bounded path-mask bytes before research Gate5 is complete. Stale asynchronous path-vector observations and arbitrary forwarding still require per-packet TTL/hop budget protections; loop freedom for all possible transitions has not been formally proven.

## Verified

- v47 TDD tests/test_zhang_xue_v47_pathvector_loop_guard.py passes direct 3-node cycle regression, valid two-hop path source/receiver mask, E4096 isolation.
- v46 two-node poisoned-reverse tests and v42 2-hop DATA/E2E ACK / original distributed routing tests remain green.
- CUDA not executed locally; full original SimPy MAC parity, 6-agent dynamic multihop forwarding under real no-fly/blackout, production reward/energy, actor/critic and training remain blockers. Do not start training.
