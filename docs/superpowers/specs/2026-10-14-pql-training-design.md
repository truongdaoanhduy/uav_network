# PQL-inspired MASAC/MATD3 training
Approved scope: Option A in the user's uploaded 2026-10-14 handoff; this request authorizes implementation.

## Goal and invariants
Keep independent decentralized actors, centralized twin critics, hybrid masks, 1-step targets, reward, environment, network, energy and safety unchanged. Add separate collector, critic and policy components with bounded asynchronous work and a synchronous reproducibility mode. Preserve existing fixed training as the default.

## Rates and devices
A collector unit is one vector step (horizon 1), not one UAV or individual replay row.
Target: 4096 worlds, batch 8192, replay 500000 joint rows, warmup 32 collections, 8 completed critic updates per eligible collection. MASAC policy/alpha update every critic event. MATD3 policy and targets follow policy_delay=2 exactly.
Sample usages per newly inserted row = completed_critic_updates * batch_size / inserted_rows; steady target = 8*8192/4096=16. Warmup gets no retrospective credit.
GPU 0 owns Warp and inference. GPU 1 owns replay and both learner components when two GPUs exist; one CUDA device is a functional fallback. CPU is only the functional-test path. Explicit missing devices fail.

## Ownership and synchronization
Collector has an inference-only actor copy. Critic owns replay, online/target Q networks and critic Adam. Policy owns actor/temperature Adam and read-only Q snapshots. Immutable cloned snapshots carry monotonically increasing versions; CUDA queues are synchronized at publication boundaries. Bounded collection and policy queues apply feedback backpressure when a consumer is slower. No transition drops, shared mutable parameter reads, or concurrent replay mutation/sampling.
Targets keep their original update clocks; asynchronous snapshots introduce bounded lag and are disclosed. Synchronous split updates must match the pre-refactor update exactly.

## Persistence and diagnostics
PQL checkpoints retain model/targets, all Adam states, update counters, controller carry, versions, progress, RNG and optional full replay. Policy-only legacy v2 checkpoints stay supported. Resume resets simulators with next unused seeds and reports this; it is not an exact continuation of mid-episode simulator state.
Log measured collector/critic/policy rates, ratios, sample uses, queue/wait time, lag, replay age and actual unsampled evictions, alongside existing mission and learner metrics. W&B remains online when enabled.
Save at drained collection boundaries; do not create many identical checkpoints when thousands of episodes complete together.
Periodic W&B artifacts contain light checkpoint.pt; the final artifact also includes resume.pt once so cloud warm-start can restore complete training state.

## Verification and deployment
Freeze the old update functions at afeaa3d as test oracles. Compare losses, weights, targets, Adam and random state for both vectorized and plain paths. Exercise worker errors, queue pressure, mask semantics, ring wrap, warmup, delayed targets, checkpoint/resume, CPU Warp training and CUDA paths when available.
8192 is an unverified T4 batch target. Benchmark the exact 25-target/15-obstacle case for both algorithms, one/two-GPU mapping, peak allocated/reserved VRAM, synchronized update times and measured achieved rates before production. Report OOM without silently changing the batch. No long production run before passing the benchmark.

## Source basis and alternatives
Reference: https://github.com/Improbable-AI/pql at 93c2b583b6e9999dd45c77f3f8342ed538d73fb5.
Paper: https://proceedings.mlr.press/v202/li23f.html .
Increasing replay alone extends residence time but does not improve expected sample usage at fixed insertion/update throughput. Keeping the combined learner would be simpler, but the approved three-component design needs split APIs. Stdlib worker threads, bounded queues and device streams avoid Ray and CPU-per-world workers.
Rapid eviction is a measurable hypothesis, not a universal diagnosis. Current baseline 512*16 updates with batch512 already has 16 sample uses per inserted row, equal to the proposed target. Architecture changes alone do not prove better convergence.
