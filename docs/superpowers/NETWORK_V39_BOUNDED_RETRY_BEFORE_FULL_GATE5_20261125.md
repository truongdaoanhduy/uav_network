# Zhang–Xue v3.9: bounded Warp retry and receiver-local evidence smoke

Checkpoint date: 2026-11-25.
Branch: experiment/zhang-xue-v39-warp-mac-retry-20261125.
Starting v3.8 clean checkpoint: 4a461f6, parent map v3.7: 82edcde.
**Do not merge main or launch training until complete Gate 5.**

## Implemented and tested
- v3.7 1km/20x20/6-UAV/10-target map stays unchanged:
  fixed altitude 100m, 90-degree nadir camera, 2 no-fly circles,
  2 Manhattan blackout zones, DS sensor counts with mf=me=0.8.
- uav_network/communication/gpu/zhang_xue_mac_retry_warp.py:
  genuine Warp one-hop packet trial, per-world 6 sources, DIFS 50us,
  slot20us, 750m CCA, five attempts, 2Mbps DATA airtime, overlapping
  channel receiver SINR, hidden terminals, partial reverse MAC ACK
  interference check, ACK timeout, static blackout and TTL expiry.
  Independent worlds are batched in Warp; host CUDA driver unavailable.
- scripts/verify_zhang_xue_map_to_network_v39.py: scene reset,
  Warp movement, sensor/DS, source-origin count packet payload,
  bounded Warp MAC and receiver-specific count merge ONLY after
  actual packet reception. Merging a duplicate cumulative snapshot is
  idempotent. Receiver fusion in this probe uses **NumPy**, not Warp.
- scripts/verify_zhang_xue_network_v39.py: repeat actual pinned classic
  UavNetSim-v1 SimPy contention case, compare with Warp and save JSON.
  Upstream pin: 64bc3a2424f2976418731ee94f9a8fd3e8810b22.

## Reproducible evidence
Native six hidden-terminal seeds44-49 PDR: [0,0,0,0,0,0].
Warp isolated one-hop retry PDR: [0,0,1,0,1,0].
**2/6 outcomes mismatch**, native 60 attempts, Warp 59 attempts.
Random-seed formula alone does not reproduce the original SimPy
HELLO/setup RNG consumption or network event sequence.
**Gate 5 FAIL; no valid full network parity claim.**

E4096, 1-step map to one-hop network CPU preflight:
307291 sensed source observations, 24576 packets formed and received
in the EASY central dense topology, source-owned receiver table entries
match, replay max merge idempotent, sampled DS values finite/normalized.
First-step scenario has zero UAVs inside the sampled blackout zones:
this DOES NOT test reconnect, queueing, or hidden-terminal PDR.
The run is no actor, no replay/training, no multi-hop or app E2E ACK.

Evidence JSON in docs/superpowers/evidence/:
- zhang_xue_v39_simpy_vs_warp_retry_cpu_20261125.json
- zhang_xue_v39_4096_map_to_onehop_network_cpu_20261125.json

## Explicit blockers before training
1. Extract native MAC packet-level times and actual random draw streams.
   Test matched traffic/HELLO/seed and compare collision/ACK/retry/PDR/
   delay over many benign and hidden-terminal topologies.
2. Implement source-equivalent carrier backoff freeze/restart, true
   ACK event collisions, receiver channel resources and timeouts.
   Current CCA and ACK are bounded approximations, not SimPy parity.
3. Implement persistent application queues, packet TTL, blackout/
   reconnect, multi-hop routing and genuine destination E2E ACK,
   source-origin watermarks, retransmission and anti-double-reward.
4. Move receiver maps and Max-plus-Sum DS fusion to persistent Warp
   network memory, enforce no leaked evidence in all agents and steps.
5. Full environment: actor outputs, motion, sensing, PHY/MAC/routing,
   app delivery, DS, reward, reset, held-out validation; benchmark
   on small and E4096 across time, including a CUDA GPU run.
6. Only when Gate5 passes, start fresh MATD3/MASAC training then eval,
   test, checkpoint/W&B/HF, keeping prior runs and main intact.

Status: **research prototype partially working; full Gate5 blocked**.
