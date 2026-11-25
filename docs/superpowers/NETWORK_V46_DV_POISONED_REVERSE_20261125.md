# Zhang-Xue v4.6: distributed Warp DV routing loop prevention

Date: 2026-11-25. Branch experiment/zhang-xue-v46-dv-poisoned-reverse-20261125, parent v4.5 commit 1020f2d34179ffeebb86ef08e45543270cfaf42d. Main and all prior experiments remain untouched.

## Proven regression (test first)

With HELLO TTL=100 microseconds, topology first has 0↔1 and 1↔2; 1 advertises a route to 2 to 0 at t=90. At t=105 the 1↔2 direct link has expired but 0↔1 and the stale 1→0 DV announcement are still within TTL. Without split horizon, 0 believes destination2 is reachable via1. At t=106, 0 advertises that route BACK to 1 and 1 installs route2 via0, creating a two-node forwarding loop 1→0→1.

RED reproduced the error in actual Warp CPU kernel: next_hop[1,2]=0 rather than -1, including 4096 independent parallel worlds. This is not an invented paper concern; the previous implementation demonstrably failed those tests.

## Change

The existing Warp DV receive kernel _receive_distance_vector now receives the sender's next-hop array. For each physically received DV packet destined to neighbor RX, if sender's next-hop for advertised destination is that same RX, it writes an INF=99 unreachable distance in the neighbor's receive-side advertised table. Otherwise it writes the true sender distance. The original physical received HELLO/DV barrier, expiry, and split source-local view remain unchanged. The source never reads global positions or target truth when constructing DV.

Only the received advertisement is poisoned; no global route forcibly overwritten or fabricated.

## Passing focused regressions

- tests/test_zhang_xue_v46_poisoned_reverse.py: replay timed link expiry, confirm no 1→0→1 loop, verify valid direct and third-party routes remain usable, and E4096 independent-world loop prevention.
- Existing tests/test_zhang_xue_v42_distributed_route.py and tests/test_zhang_xue_v42_multihop_flow.py still pass including 2-hop DATA/E2E ACK from actual HELLO and advertised DV.

This fixes only the two-node count-to-infinity situation, NOT a formal proof of arbitrary multi-node loop freedom. No actual GPU kernel run on CUDA; all tests use Warp CPU. The full source-equivalent UavNetSim MAC and full Gate5 are still blocked. Do not start training.
