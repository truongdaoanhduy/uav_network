# Zhang-Xue v4.8: application forwarding envelope and 5-hop DATA/E2E ACK

Date 2026-11-25. Branch experiment/zhang-xue-v48-hop-envelope-six-uav-20261125, based on v4.7 SHA f3026381ec5043c77d381a641c4542caefe1d848. Separate branch, no main modification, no model training, no GPU job.

## New persistent Warp on-device application envelope

File uav_network/communication/gpu/zhang_xue_forward_envelope_warp.py:
- Immutable per-slot fields [E,holder,queue_slot]: original source UAV, ultimate app destination UAV, hop count and six-bit visited-hop mask. Link-layer queue separately carries next-hop MAC receiver, original creation time and TTL in integer microseconds. Immutable source-owned evidence is in the existing per-slot payload store.
- Admission checks receiver/source bounds, no self-application packets, visited includes both holder and source, valid mask size, popcount must equal hop_count+1. Stale queue-slot reuse never supplies a previous packet's envelope.
- Warp classifies actual physically received DATA into terminal delivery, relay candidate, loop rejected and hop-budget rejected. A packet may traverse max 5 distinct physical links in a 6-UAV world. It cannot be forwarded to a previously visited UAV, even when the DV routing table is stale. MAC ACK is still separate from application receipt/ACK.

TDD: five envelope tests including blackout persistence, immutable data, stale slot reuse, forged hop count/visited mask, loop and maximum hop budget, exact TTL and E4096 independent world isolation. These tests were written and observed RED before implementation.

## Six-UAV full-chain controlled integration

File scripts/verify_zhang_xue_v48_six_uav_multihop.py tests a folded, non-oracle six-node chain within a 1km box:
UAV0→UAV1→UAV2→UAV3→UAV4→UAV5 for DATA, with an independent 90B application E2E ACK returning UAV5→UAV4→UAV3→UAV2→UAV1→UAV0.

- Both-direction HELLO frames actually go through Warp MAC before any neighbor is learned (10 HELLO frames per world); eight received path-vector control frames build the five-hop end-to-end routes, which are initially absent.
- Warp camera-derived belief counts, NOT global target truth, are encoded into an immutable origin-labelled cumulative evidence packet (max 62 entries). Every relay preserves original source, ultimate destination, creation time, TTL, current visited mask and original observation payload, and selects the next hop from its own received DV table. Each successful DATA hop updates its receiver-local Warp MAX-plus-SUM DS map only on real physical reception. Destination5 counts one final DATA and only then queues one application E2E ACK that traverses five independent MAC hops back to source0.
- In 1/4 of worlds radio UAV2 becomes unavailable AFTER it received its DATA from UAV1, so the packet remains queued and expires exactly at its original 10s TTL. No data at UAV5, no fabricated E2E ACK and no incorrect watermark for that group.
- No Oracle lookup or global target truth is used for routing; physical MAC uses actual coordinates to calculate signal/CCA, correctly separated from local routing observations.

## Verified CPU evidence

tests/test_zhang_xue_v48_six_uav_multihop_e2e.py passes E16, E1024 and E4096. E4096 seed53:
- 40,960 physically received HELLO frames, 32,768 received DV adverts;
- 4096 original DATA, 3072 at ultimate receiver5, 1024 expired in relay UAV2;
- 17,408 DATA hop receipts, 14,336 DATA relay admissions;
- 15,360 E2E ACK hop receipts, 12,288 ACK relay admissions, 3072 terminal application E2E ACKs received by source0 and 3072 matching source-peer watermark advancements;
- zero transmissions by blacked-out sender, no forwarding loops, no duplicate application reward, no premature recipient DS evidence or source watermark, queue empty at end;
- elapsed ~4.3 s Warp CPU for the tiny controlled network pilot (not training throughput).
- evidence JSON docs/superpowers/evidence/zhang_xue_v48_4096_fivehop_e2e_blackout_cpu_20261125.json.

## Remaining Gate 5 blockers — no training yet

This is one application message/world, controlled static folded topology, isolated HELLO and DV MAC frames, Python dispatcher copying metadata/payload between device-resident Warp kernels. It is not a fully concurrent 6-agent five-hop traffic workload with realistic dynamic no-fly and blackout geometry, nor a source-faithful UavNetSim MAC simulator. The v4.5 random moving 1km six-sender pilot uses dynamic direct-neighbor HELLO routes but not this 5-hop forwarding scheduler.

Still needed: combine arbitrary concurrent multi-hop forwarding and ACK queues under dynamic movement; original SimPy carrier backoff freeze and APP ACK contention parity; device-to-device scheduling removing Python/NumPy host sync; full production energy/action/reward and randomized 200-step episodes; real Warp CUDA check (Kaggle credentialed submission blocked by safety controls); only then MATD3/MASAC train/eval/test. Full Gate5 remains BLOCKED and no training started.
