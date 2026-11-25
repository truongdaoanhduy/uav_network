# Zhang–Xue v4.5: sampled 1km 200-step moving six-UAV map → network → reward probe

Date 2026-11-25. Separate branch experiment/zhang-xue-v45-sampled-moving-network-20261125 from clean v4.4 SHA a1801302ec8359319d2e13f28796f6aa1823fe77. No training jobs, W&B/HF mutations, original source changes or credential commits.

## Controlled pre-gate scenario made more realistic

The pilot script scripts/verify_zhang_xue_v45_sampled_scene_network.py reuses production ZhangXueConfig and sample_scenario_batch from zhang_xue_map_geometry.py, not a new invented map. Each independent reset has 1,000 x 1,000m, six UAVs, ten static targets, two sampled circular no-fly zones and two sampled radio blackout diamonds, 20x20 DS belief, 100m altitude/FOV90/Pd=.85/Pf=.03, 200 steps at 1s. Targets are only used by Warp sensor and a separately called truth-based score. No target truth enters packet content or the local HELLO route selection.

UAV actions in this pretraining pilot are independent random 2D directions, not MATD3/MASAC policy outputs. Each motion step uses the existing Warp motion and segment-safe no-fly/collision/boundary checks, then updates real scenario-dependent blackout radio state. All six exchange source-only evidence under real Warp bounded one-hop MAC SINR/CCA/5 retries and source-peer persisted E2E ACK watermarks. Neighbor candidates are selected only from bidirectional physically received HELLO link state; HELLO refreshes every four steps. Thus messages are not delivered by an oracle lookup of global geometry.

Fixed a real convergence-traffic bookkeeping issue: multiple application DATA from different UAVs may simultaneously arrive at the same receiver. Instead of writing one APP ACK per receiver slot and silently overwriting earlier ACKs, the new probe admits one ACK attempt for each physically received DATA, and explicitly counts queue-full rejections. A source may receive multiple valid E2E ACKs from different peers during one timestep; these are processed independently. A missing or stale ACK cannot advance remote synchronization.

Full reward engine is NOT integrated. Test-only score is 10 for each newly confirmed true target, -5 for each first falsely confirmed empty cell, with confirmation threshold 0.95 from existing config. Only this callback reads target truth. Coverage, TP and FP are diagnostic, not paper evaluation under a learned policy.

## Reproduced evidence — Warp CPU, NOT CUDA

- E128 × 200 steps, seed44: 128 distinct sampled worlds. Mean coverage 79.207%, 964 distinct true-target confirmations out of 1280, 36 false confirmations, mean restricted confirmation score +73.906. Data generated 66,416; physically received 57,287; 57,287 APP ACK attempted; 55,946 enqueued; 1,341 explicitly rejected due queue-full; 51,105 physically received APP ACK, 50,723 matched and advanced source-peer watermarks, 382 stale ACKs ignored. 192,000 targeted one-hop HELLO attempts; 115,628 successful. 16,212 UAV-radio blackout steps; 5,837 prevented unsafe moves; 3,986 prevented out-of-world moves. Zero final no-fly/inter-UAV/separation/boundary violations, zero attempted transmit by blackout UAV and zero belief leak into non-receivers. Simulation counts: 265,489 CCA deferrals, 54,939 collided attempts, 8,652 queue-full packet rejections overall. ~12.06 seconds *CPU* on the user's machine.
- E4096 × 4 steps, seed53: all 4,096 worlds have ten targets. 98,304 DATA generated; 73,682 received; 73,682 APP ACK attempted/enqueued; 24,576 valid E2E ACKs; mean coverage 16.401%, 5,148 true target confirmations and 392 false confirmations, zero final unsafe positions or receiver evidence leak. ~8.61 seconds CPU, not Warp GPU throughput.

Raw JSON:
docs/superpowers/evidence/zhang_xue_v45_sampled_128x200_cpu_20261125.json
docs/superpowers/evidence/zhang_xue_v45_sampled_4096x4_cpu_20261125.json

TDD: tests/test_zhang_xue_v45_sampled_moving_network.py covers E16×200, E128×20, E1024×4 and exact ACK accounting, no-oracle direct neighbors, spatial safety, source evidence isolation and truth-only score.

## Gate 5 still BLOCKED, DO NOT TRAIN

- Network schedule and sender/ACK selection orchestrated in Python on top of Warp CPU kernels, with synchronous device-to-host reads; not fully fused 4096-world CUDA full episode. Local CUDA unavailable; permitted Kaggle account 2 API works via isolated CLI2.2.5 but private upload blocked by tool safety policy. Credential-free GPU preflight builder exists on v4.3.
- Direct one-hop learned neighbors only, not source-equivalent multihop dynamic routing with full HELLO/DV traffic and queue/forwarding among six moving UAVs. Multi-hop was only verified in static 3-node v4.2 pilot.
- MAC is bounded approximate Warp SINR/CCA; source-exact UavNetSim-v1 SimPy has 35 late-retry timing mismatches up to550µs in 64-seed one-hop parity. Need native control packet competition/event freeze and ACK timing under dense load.
- Pilot restricted truth score is NOT full reward function; energy/battery and actor local observation safety/action masks remain unintegrated, actor/critic/replay/training/eval/test not executed. Do not claim paper metrics from random actions.

This checkpoint can be repeated without modifying main and all earlier experiment branches remain intact.
