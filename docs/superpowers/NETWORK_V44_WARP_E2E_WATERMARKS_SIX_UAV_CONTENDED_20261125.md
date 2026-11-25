# Zhang–Xue v4.4 — source-peer Warp ACK watermarks and six-UAV contention

Date 2026-11-25. Branch experiment/zhang-xue-v44-e2e-watermarks-concurrent-20261125, based on v4.3 6c2ada5cec6b42e7ab4e2fbcb9de2ec877dd31cc. No optimization, W&B, HF or model training.

## Implemented

- zhang_xue_e2e_watermarks_warp.py stores peer-specific origin acknowledgement counts on Warp [env, source, peer, 400 cells] arrays. Positive/negative sensor evidence for the chosen peer is selected into bounded chunks of at most 62 cells. Each source-peer has at most one outstanding application DATA sequence with a frozen snapshot.
- Only a physically returned, matching **application E2E ACK** updates positive/negative watermarks. Per-hop MAC ACK, missing/stale/duplicate app ACK and relay forwarding do not mark evidence synchronized. Sensor increments recorded after packet generation remain dirty. Application timeout may release a pending snapshot for resend.
- verify_zhang_xue_v44_six_uav_contention.py uses a controlled 6-node static ring. Twelve isolated MAC HELLO frames discover direct peer links. Six UAVs then concurrently generate source-local MAP evidence DATA, contend on a common Wi-Fi channel via Warp CCA/SINR/retries, merge only at actual receiver and return real 90-byte APP E2E ACKs. Source-peer watermark ACK identities are checked; no target truth is serialized. Artificial partial blackout on some worlds.
- TDD regressions cover stale/missing E2E ACK, peer isolation, new sensor evidence after send, chunks of 62, timeout/retry, reset and E4096 independence. Integrated tests cover E16x200, E256x12 and E4096x4.

## Pilot evidence, Warp CPU

| Pilot | DATA admitted | DATA physically received | E2E ACK physically returned | Matching watermark updates | CCA deferrals | MAC collided attempts |
|---|---:|---:|---:|---:|---:|---:|
| 256 env x 200 steps, seed44 | 126091 | 119974 | 119215 | 119215 | 830738 | 606787 |
| 4096 env x 4 steps, seed47 | 45622 | 42186 | 40922 | 40922 | 334601 | 218818 |

E256x200 released 6385 timed-out application packets for resend, and had zero stale app ACK, zero queue-full, max queue length 2, zero blackout transmissions and no cross-UAV remote evidence leaks. E4096x4 max queue length1; no timeout in short horizon, same safety assertions. These are *simulated Warp MAC counters*, not real physical network measurements and not full source-original UavNetSim parity.

Evidence:
docs/superpowers/evidence/zhang_xue_v44_256x200_six_uav_wm_cpu_20261125.json
docs/superpowers/evidence/zhang_xue_v44_4096x4_six_uav_wm_cpu_20261125.json

## Gate 5 NOT PASSED — training must remain blocked

- Fixed static ring and fixed neighbor-choice actions, not learned actor or dynamic multi-hop movement/network in a randomized 1km search scenario.
- Warp PHY/MAC surrogate not exactly matched to native SimPy source in ACK timing, channel occupancy and backoff freeze; prior v4.0 found 35 late-start mismatches up to550us across64 seeds.
- Packet enqueue, router updates, ACK creation and delivery tracking still use Python glue and synchronous host copies, not an efficient fused GPU scheduler.
- No actual NVIDIA CUDA execution validated. Kaggle account2 token works with isolated CLI2.2.5 and GPU quota ~29.53h, but the tool security layer blocked direct private Kaggle upload; do not try to bypass controls. Credential-free preflight bundle v4.3 remains available.
- No full 200-step sampled moving UAV search task with reward-only target truth, energy constraints, actor safety or end-to-end training/evaluation has been validated.

Next: source-faithful SimPy MAC parity for concurrent app control + relay ACK, fully on-device multi-hop scheduler, sampled scenario motion/energy/reward, real CUDA diagnostics once permitted and full real Gate5 before training.
