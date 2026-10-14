# GPU packet environment design

User goal: remove the measured CPU environment bottleneck on Kaggle T4, run motion/sensing/observations AND packet simulation on CUDA, retain MASAC/MATD3, then rerun 128,256,512,1024,2048,4096 in order. Stop on CUDA OOM; persist every preceding result. CPU limits remain explicit failures. Existing long training is separate.

This changes the environment backend architecture. Add an opt-in NVIDIA Warp backend (distinct from archived Salesforce WarpDrive), one batched state allocation and zero CPU simulator worker processes. Keep the native UavNetSim backend as the behavioral oracle. Use FP64 geometry, energy and event timestamps; use float32 actor observations and rewards. Own a serial scheduler per world, parallelize worlds; do not parallelize same-world channel updates with nondeterministic atomics.

## Contracts

Create environment/gpu/state.py with Warp structs WorldParams and WorldState plus allocation helpers. It is owned by the packet implementation worker. Common fields used by physics and wrapper:
- n, targets_count, obstacles_count, map_size, dt, max_steps, max_speed, max_accel, safety_distance, obstacle_clearance, motion_safety (1 dynamic_window,0 reject), altitude, footprint, occlusion, target_slots, obstacle_slots, peer_range, gcs_range, report_bytes, buffer_bytes, report_ttl, admission_mask, recovery_enabled, energy_enabled, capacity, idle_power, hover_power, speed_coefficient, accel_coefficient, launch_radius, launch_spacing, target_exclusion, obstacle_min_radius, obstacle_max_radius, placement_attempts, polygon_sides.
- WorldState.positions, velocities: array2d(vec2d) [E,N].
- targets: array2d(vec2d) [E,T]; obstacles: array2d(vec3d) [E,O]; gcs: array(vec2d) [E].
- battery, flight_energy: array2d(float64) [E,N]; visible,known,recovery,report_present,report_origin,report_hops: array3d(int32) [E,N,T].
- report_created: array3d(float64) [E,N,T]; report_order: array3d(int64) [E,N,T]; delivered: array2d(int32) [E,T].
- steps,terminated,blocked,clipped,errors: array(int32) [E]; motion_counts: array2d(int32) [E,9] (obstacle,peer,stall,shield_all,shield_obstacle,shield_peer,shield_boundary,shield_infeasible,reserved).
- report_counts: array2d(int32) [E,3] expired,dropped,regenerated; inflight: array2d(float64) [E,N]; communication_energy: array2d(float64) [E,N+1].
- observation: array3d(float32) [E,N,D]; state: array2d(float32) [E,S]; reward: array(float32) [E].
- pcg_state: array2d(uint64) [E,4] PCG state low/high and increment low/high, initialized from NumPy PCG64(seed).state in the wrapper; seeds: array(int64) [E].
Packet-specific arrays can be added by the owner without changing the common fields. Parameters use explicit float64 where numeric, except named integer fields. Pass structs to kernels.

Physics worker produces motion_step(s,p,continuous), sense(s,p), observe(s,p), centralized(s,p), and reset_geometry(s,p,mask), all Warp kernels. continuous is array3d(float32) [E,N,3], mask array(int32) [E]. motion_step updates funded safe motion and flight cost. sense writes visibility only; packet functions own report insertion. reset_geometry draws PCG64 uniform values and mirrors native rejection-sampling order. Unsupported dimensions/configurations fail explicitly, never silently switch protocol/safety.

Packet worker produces reset_network(s,p,mask), begin_step(s,p), sense_reports(s,p), packet_step(s,p,continuous,destination), end_step(s,p). destination is array2d(int64) [E,N]. begin_step snapshots reward counters and expires reports at current time; sense_reports follows GPU sense; packet_step injects selected reports and advances to (steps+1)*dt; end_step increments steps, expires at end, zeroes exhausted velocities, computes reward/termination. p may be extended with packet/PHY/reward constants. Common helper allocator exposes all kernels' buffers on CPU or CUDA.

## Packet invariants

Port the pinned mission bridge and actual CSMA/CA semantics: FIFO report custody and source TTL, payload fragmentation/reassembly, per-packet deadlines, queue caps, admission, per-node seeded Python MT19937 backoff draws, DIFS/backoff freeze and 1us carrier listener, slot idle polling, sender resource reservation, SIFS and ACK frame/timeout, retry limits, duplicate reception, interference weighted by airtime overlap, native free-space/urban NLOS path gain, polygon LOS, selected transmit power for DATA and ACK, and battery debit on each affordable transmission attempt.

Keep deterministic event timestamp/priority/insertion ordering inside each world. Warp RNG is not a replacement for native MAC random.Random(seed+node+5). No hidden queue/event overflows: set errors, fail the case and retain evidence.

## Device path and integration

Runtime.environment_backend defaults native; opt-in value warp_packet. make_vector returns a single device-vector object for this backend. No AsyncVectorEnv worker pool. Two observation/state buffers prevent replay from reading overwritten current observations. MASAC/MATD3 can return device tensors; native calls preserve the existing NumPy contract. Replay accepts tensor actions/observations directly. Random rollout actions/masks are generated on-device. Only episode/logging/evaluation boundaries transfer summaries to CPU.

Use the exact current model/batch/replay/world defaults in the capacity sweep. GPU kernel memory and learner/replay allocations are measured together. NVIDIA driver580/T4sm75 are the cloud target; pin a tested Warp wheel. CPU-only visualization uses the same Warp kernels on CPU with a single world, rather than pretending a ported checkpoint used an identical native simulator.

## Required evidence

CPU native versus Warp CPU scripted-layout motion/FOV/observation/energy tests; native packet oracle tests for direct and relay delivery, fragmentation, busy channel/collision, weak-link retries, TTL, queue pressure and exhausted battery; Warp CPU versus CUDA parity/repeatability. Verify learner updates remain real, targets/optimizers unchanged, observations/actions/replay stay on CUDA. Run CUDA native regression tests, full CPU suite and fresh read-only review.

Rerun the ordered count sweep on a separate Kaggle job, with stable-interval VRAM/utilization/CPU/RAM and collection/update rates. Report counts actually run and skipped. If a semantic gate fails, fix it or explicitly report the unsupported/mismatching behavior; do not advertise the backend as scientifically equivalent before evidence.

Research background and local profile remain in docs/gpu-native-env-research-20261014.md and outputs/gpu-native-20261014. Native reference measurements live in docs/environment-scale-20261014.md.
