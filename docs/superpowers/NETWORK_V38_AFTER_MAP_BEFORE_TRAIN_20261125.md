# Zhang–Xue v3.8 — network engineering after map, BEFORE any training

## Correct user priority, enforced by task scope

**Map first → complete realistic UavNetSim/Warp network and Gate 5 → then
(and only then) MATD3/MASAC training → evaluation/checkpoint/W&B/HF.**

The v3.7 map milestone is separately committed/pushed at
`82edcde14fb83447ac1de3df66942af669b6136f` on
`experiment/zhang-xue-v37-warp-map-first-20261125`.

The new v3.8 branch extends that frozen map commit. It does not change
map geometry/DS logic or launch training.

## Rechecked classic UavNetSim-v1 baseline

Pinned original Python SimPy CSMA/CA source SHA:
`64bc3a2424f2976418731ee94f9a8fd3e8810b22`.

Hidden terminal test: 2 sources at x=0 and x=800, receiver x=400,
62-entry 566-byte application frames, channels both 1. Six seeds 44–49.
Actual V1 native SimPy PDR = 0.0 each seed; old simplified
MiniDiscreteEventNet PDR=1.0 each seed; **56 collision events**.
Evidence: `docs/superpowers/evidence/zhang_xue_v38_network_hidden_terminal_baseline_20261125.json`.
**Gate 5 still FAILS**, no model is trained on that invalid optimistic network.

## New source-grounded Warp first-attempt component

`uav_network/communication/gpu/zhang_xue_mac_first_attempt_warp.py`
provides `schedule_initial_csma_frame_attempts`, an actual Warp CPU/GPU
batch path for 6 first-attempt unicast frames in each independent world:

- App-selected destination (NO_SEND=-1 in this internal MAC contract);
- DIFS50us, SLOT20us, CW first-attempt slots 0..30;
- source/carrier sensing with exact 750m rule from pinned V1;
- approximate CCA deferral to end of occupied DATA+SIFS+MAC ACK reservation;
- variable DATA frame airtime at 2Mbps;
- co-channel/adjacent interfering receiver SINR from earlier Gate 5D
  numerical kernel (channel index gap less than five), 6dB threshold;
- separate valid-mask and success outputs for every attempted frame.

This is a **limited first-attempt scheduler**. It is not a faithful
reproduction of the complete `CsmaCa.mac_send` state machine. In
particular frozen-backoff calculations are simplified and it does NOT
execute MAC ACK reception/ACK timeouts, repeated retries, packet queues,
TTL, multihop Greedy routing or application E2E ACK/watermarks.
Never use this module in production training or claim full Gate 5.

## Verified

Unit contracts: `tests/test_zhang_xue_v38_mac_first_attempt.py`.
Controlled independent scenarios:
1. Two hidden transmitters 800m apart, simultaneous DATA to relay,
   overlapping channel1: 0/2 success;
2. Same geometry, channels1/6: 2/2 success;
3. Two sources within sensing range with staggered CW backoff:
   later sender defers until first DATA+MAC-ACK reservation, 2/2 success.

MAP remained untouched after v3.7 checkpoint.

4,096 independent Warp **CPU** environments:
- 8,192 initial packet frames evaluated;
- same-channel hidden-terminal PHY PDR0;
- orthogonal-channel PHY PDR1;
- isolated measurement 24ms first call (includes kernel overhead),
  2.4ms warmed cache on local machine. This is NOT true network
  throughput or GPU performance;
- evidence: `docs/superpowers/evidence/zhang_xue_v38_first_attempt_4096_cpu_20261125.json`.

## Blocking next steps BEFORE training

1. Full timed state machine: carrier busy freeze/restart, max5 retries,
   MAC ACK collision/loss/timeout and per-packet TTL expiration.
2. App-level queues (including blackout persistence), own-origin
   packet serialization, per-receiver Max-plus-Sum fusion, routed E2E
   ACK and watermark semantics.
3. Native v1 Greedy routing uses destination coordinates from global
   simulator; eliminate that routing oracle or document legitimate
   application-layer localization before claiming full Dec-POMDP.
4. Gate 5 comparisons: success and hidden-terminal failure cases,
   at least dozens of independent seeds with matched MAC settings and
   PDR, delay, hops, retries, queues, airtime, ACK, TTL and rank
   ordering. Test both 1-hop and 2-hop.
5. End-to-end E4096 map + full Warp network + actor/replay logic,
   then GPU execution (CUDA missing locally). **Only after it passes
   should training start**.

This file deliberately reports incomplete work rather than hiding
the remaining mismatches.
