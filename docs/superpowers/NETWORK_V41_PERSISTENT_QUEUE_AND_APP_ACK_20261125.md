# Zhang–Xue v4.1 — persistent Warp queue and application E2E ACK smoke

Date: 2026-11-25. Branch: `experiment/zhang-xue-v41-persistent-warp-queue-20261125`. Based on v4.0 native-metric fix commit `169348ffc8fc30cf8c4bffca33f4dc6184f66ca3`.

## Implemented

- `uav_network/communication/gpu/zhang_xue_persistent_queue_warp.py`: persistent bounded packet-metadata arrays on real NVIDIA Warp (CPU backend locally; CUDA not validated). Per independent world and UAV (6 per world), capacity 1–200, default 50. Admission rejects duplicates and queue-full; FIFO by monotonically increasing serial; enforces single outstanding MAC packet per source. Inputs carry packet ID, destination, DATA/E2E_ACK kind, exact wire length, creation time, TTL. DATA wire is 70+8×entries (0..62), E2E_ACK wire length exactly 90B. All times integer microseconds (`int64`).
- The queue survives blackout across calls and keeps packets waiting; resumed UAV can send. TTL removal uses `now >= creation + ttl`, with per-episode `reset`. Completion is checked against in-flight packet ID to prevent stale ACK from consuming reused slots; `resolve` distinguishes MAC ACK, MAC retry exhaustion, and defer/return to wait. Sender-side MAC status is not equivalent to application DATA receipt.
- `scripts/verify_zhang_xue_queue_mac_v41.py`: a **controlled single-hop** Warp queue → bounded MAC → receiver-generated 90B application E2E_ACK → return ACK diagnostic. Four interleaved environment groups (1/4 send now, 1/4 recover at 3s, 1/4 at 9s, 1/4 remain in blackout and expire at 10s) across seven simulated timestamps. ACK identity and duplicate suppression are checked on the Python test harness, not a production Warp watermark kernel.

## Evidence

- 12 focused v4.1 tests pass (blackout, recovery, TTL boundary, duplicate/full/FIFO, stale completion, queue reset, 64-bit time, frame length, 4096-world isolation, end-to-end metadata ACK smoke).
- 4,096 Warp **CPU** world smoke: 4,096 DATA generated, 3,072 DATA received, 3,072 application E2E ACK generated/received, 6,144 link-MAC ACKs, 1,024 TTL expirations, zero blackout transmissions, empty queues at end. Report: `docs/superpowers/evidence/zhang_xue_v41_4096_queue_blackout_ack_cpu_20261125.json`.
- This deterministic near-perfect one-hop case is **not a general PDR result** or a GPU performance result. CPU timing includes warm-cache and host-side orchestration; do not extrapolate to E4096 GPU training.

## Critical limitations (Gate 5 NOT PASSED)

- Queue persists **metadata only**; source-owned evidence payload, idempotent per-receiver belief and E2E watermark are not in a complete Warp pipeline.
- No dynamic mobility, HELLO neighbor discovery, no-oracle multi-hop path selection or relay forwarding. All tested DATA goes directly from UAV0 to UAV1, with APP ACK returning from UAV1 to UAV0.
- No faithful native SimPy MAC freeze/backoff service or ACK-resource parity for general traffic; v4.0 64-seed controlled parity has 35 later-TX time differences up to 550μs. The MAC/queue integration currently uses host tensor transfers per step, not fully fused GPU scheduling.
- Static/controlled blackout group schedule; not the sampled 2D scenario across a full episode. No actor, learning, replay, reward, W&B/HF or Kaggle CUDA executions.

## Next blockers — do not train

1. Connect persistent Warp queue to **actual** per-origin evidence payload storage, per-peer last-ACK sequence, watermark and receiver-only Max-plus-Sum DS update after data receipt. Assert no target truth leaks into actor/critic.
2. Implement neighbor discovery and strictly non-oracle routing, actual relay packet transfers and return E2E ACK over 2+ hops. Preserve packet creation time/TTL on every hop; verify blackout/reconnect and duplicate receipt.
3. Implement real-time carrier freeze/restart and channel occupancy for simultaneous DATA, MAC ACK, and E2E ACK; compare against pinned original UavNetSim-v1 across matched seeds, independent topologies, traffic sizes, PDR/latency/retries/hops/ACK and TTL.
4. Run complete untrained map/motion→network→belief→reward with E1024 and E4096; run genuine CUDA Warp smoke on eligible GPU; benchmark speed and memory. Only then start MATD3/MASAC training, evaluation and final test.

All preceding map checkpoints and `main` remain untouched.
