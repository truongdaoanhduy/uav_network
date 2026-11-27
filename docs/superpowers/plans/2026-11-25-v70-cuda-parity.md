# CUDA packet parity & parallel RL training (future gate plan)

Every gate has a precise artifact and must fail closed on missing evidence.

1. **P0-1 Kernel launch**: run `scripts/gpu/warp_packet_cuda_gate.py --envs 4` on T4; test reset, dtype/shape, HELLO routing, frame reception, no OOB or failed compile. Evidence: CUDA stdout, traceback, snapshot. Previous CPU source tests do not count.
2. **P0-2 Exact radio trace**: implement timestamped one-env trace in Warp plus CPU and upstream UavNetSim under *identical configurable* channel/time/queue, compare RNG stream or use scripted deterministic backoff sequence. Separate source model differences (Q=32/200, tx power scalar/vector).
3. **P0-3 Belief/packet parity**: cell generation, UQ8.8, multi-origin watermark, ACK and duplicated same timestamp; verify both no-DM and DM paths. For identical sampling only, test max absolute/relative error with explicit float16/UQ quantization bounds.
4. **P0-4 GPU world port**: map sensing, 2D safety/mobility, battery, DS belief, reward, observation, done/reset; validate each with one GPU world against CPU under same seed (finite precision tolerance), no oracle actor inputs.
5. **P0-5 Learner path**: tensor adapter into actor, GPU batch env step, transition/replay and optimizer, snapshot save/restore, eval 50 episodes. Test MATD3 and MASAC separately using 32 env and nonzero finite grads.
6. **P0-6 Scale**: 256→1024→2048 env per GPU; free-memory/peak VRAM via NVML, throughput steps/s, interrupt/restart. T4×2 processes must have independently validated memory/seed/no cross-device pointer assumption.
7. **P1-7 Publication**: 3+ train seeds and 30–50 eval episodes per checkpoint, matched v67 ablations vs 3km stress, discovery, GCS delivery, precision/recall, coverage AUC, mean/95% latency, AoI age, energy, PDR, queue, reward and safety violations. Include confidence intervals and logs/manifest.

No stage inherits PASS from a different stage. Missing GPU, unverified parity or altered scenario is `BLOCKED`, not `PASS`.
