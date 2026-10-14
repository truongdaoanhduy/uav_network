# GPU Packet Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Run batched search/motion/packet environments on Kaggle CUDA and remeasure 128–4096 env capacity.

**Architecture:** NVIDIA Warp device state and deterministic per-world kernels, parallel across worlds. PyTorch MASAC/MATD3 share tensor storage and stream. Native simulator is the oracle; port is opt-in and supports CPU-only replay visualization.

**Tech Stack:** Python, PyTorch, NVIDIA Warp tested/pinned wheel, current pinned UavNetSim oracle.

**Spec:** docs/superpowers/specs/2026-10-14-gpu-packet-env.md

## Global constraints
- Source repo truongdaoanhduy/uav_network; existing worktree branch gpu-native-20261014.
- Six UAVs; current motion-long profile, batch512/replay100000, seed44; native long-run job continues.
- FP64 event/geometry/energy, float32 actor/replay interfaces; no lossy simulator substitution.
- No environment CPU workers in warp_packet backend; no host action/observation copies per step.
- Sequential counts128,256,512,1024,2048,4096; retain results, stop first failure.

## Review focus
- Equal-time DATA/ACK/listener transitions: deterministic timestamp/priority/insertion ordering.
- Unaffordable/expired transmissions: no phantom delivery/energy debt.
- Partial packet delivery and a receiver filling after admission: retain source custody.
- Reset masks and replay aliases: unaffected worlds unchanged and current obs not overwritten.
- Unsupported settings/queue overflow: explicit failure, never silent truncation.

## Task1: Numeric state and exact packet engine
Files: environment/gpu/state.py; communication/gpu/{rng,reports,phy,mac}.py; tests/test_gpu_packets.py.
Produces WorldParams/WorldState, allocation helpers, reset_network/begin_step/sense_reports/packet_step/end_step kernels as defined by spec.
- [ ] Write native-oracle tests: direct+relay, fragmentation, weak links/retries, collision, TTL, admission/queue and depleted energy; confirm missing backend fails.
- [ ] Implement packed report/packet state, exact native backoff RNG and deterministic event scheduler on Warp CPU/CUDA.
- [ ] Run CPU packet comparisons with explicitly supplied layouts/seeds/actions. Reject unimplemented settings.
- [ ] Confirm overflow/exhaustion cases fail or stop coherently with original counters.
- [ ] Review source/data before committing.

## Task2: Geometry, funded safety, sensing, observations
Files: environment/gpu/{geometry,motion,observations,world}.py; tests/test_gpu_physics.py.
Consumes common structs from Task1 (shared schema defined in spec). Produces motion_step/sense/observe/centralized/reset_geometry kernels.
- [ ] Write failing tests comparing scripted native positions, velocities, battery, safety counts, FOV and obs/state.
- [ ] Port scalar geometry and deterministic dynamic-window reservations including braking/reachability.
- [ ] Port GPU reset with NumPy PCG64 state/uniform sequence and original rejection order.
- [ ] Verify blocked, partial-energy, peer-conflict, boundary and occluded-target cases.
- [ ] Run targeted tests; review own diff.

## Task3: Device vector and learner/collector integration
Files: environment/gpu/vector.py; configuration.py; environment/vector.py; algorithms/masac.py,matd3.py; experiments/training.py,env_scale.py; CPU visualization selection; pyproject.toml/setup extras; tests/test_gpu_vector.py,test_cuda.py.
Consumes Tasks1/2 kernels. Produces opt-in runtime.environment_backend=warp_packet and tensor observation/action path.
- [ ] Test zero workers, shared CUDA pointers, current/next double-buffer isolation, masked reset and CPU-only evaluate.
- [ ] Add runtime selector/default validation and optional pinned dependency.
- [ ] Add tensor-return action option without changing native learners' contract.
- [ ] Integrate all-CUDA random actions/replay; avoid per-agent/step host conversions.
- [ ] Verify both real learners update and checkpoints reload/CPU visualize.

## Task4: Kaggle parity, capacity and evidence
Files: benchmark module/CLI; docs/gpu-packet-results-20261014.md and evidence JSON; ignored launch/output scripts.
- [ ] Full CPU suite and source review; publish clean pinned branch.
- [ ] Separate private Kaggle T4 job: install exact wheel, run CPU/CUDA/native parity and CUDA learner checks.
- [ ] Ordered sweep128→4096, unchanged model/batch/replay; stop actual CUDA OOM or explicit resource failure.
- [ ] Independently recompute raw telemetry summaries; preserve case phases and skipped counts.
- [ ] Record actual CPU/GPU/VRAM/RAM/sample rates and limitations, create concrete reviewable PR.
