# Zhang–Xue v3.6 — Gate 5 evidence and blockers (2026-11-25)

## Source
Classic UavNetSim-v1 (Python SimPy 802.11b), pinned commit `64bc3a2424f2976418731ee94f9a8fd3e8810b22`: https://github.com/EdwardS201/UavNetSim-v1. Different from the newer Sionna UavNetSim project.

## Reproducible measured outcomes

| Controlled test | Real classic v1 SimPy | Reference / Warp | Decision |
|---|---|---|---|
| Gate 5A, 0–400–800 m line, one small DATA | DATA1/1, E2E ACK1/1, 2 hops, latency **1.635 ms** | MiniDiscreteEventNet DATA1/1, 2 hops, **1.604 ms** | Partial PASS: difference 31 µs under 0.5 ms threshold |
| Gate 5B, 6 UAV free-space SNR, 0.5W, 2.4GHz, N=4e-11W, 6dB | Original v1 `general_path_loss` double precision | Real Warp FP32 **CPU** kernel; max error **6.8073e-6 dB** and 0/36 link status differences | PHY-only PASS, not MAC or CUDA parity |
| Gate 5C, 2 hidden senders at x0/800 simultaneously send 62-entry packets to relay x400, 6 seeds44–49 | DATA **0/2 per seed**, **56** interference/collision events in total, 10 TX attempts per seed | MiniDiscreteEventNet DATA **2/2 per seed** | **Critical FAIL**: reference ignores collisions and MAC retransmissions |
| Real 6-UAV learner integration | MATD3 and MASAC real SimPy CPU pilot with replay and gradient updates | Three-step training and three-step evaluation per algorithm | Plumbing PASS, not convergence |

Sources for recorded numbers: `docs/superpowers/evidence/zhang_xue_gate5a_real_simpy_vs_reference_20261125.json`, `zhang_xue_gate5b_warp_friis_cpu_20261125.json`, `zhang_xue_gate5c_hidden_terminal_20261125.json` and algorithm-specific CPU smoke JSON.

### Mechanism of failure

Classic source `phy/large_scale_fading.py::sinr_calculator` includes overlapping-channel transmitters as interference. Sources 800m apart cannot hear each other's carrier under v1's 750m sensing range but both reach the intermediate node 400m away; their same-channel long DATA transmissions collide. The original `CsmaCa` at `mac/csma_ca.py` attempts stop-and-wait retransmission up to five times per sender. The simplified MiniDiscreteEventNet schedules packets individually, lacks concurrent interferers/backoff and therefore predicts ideal PDR. The six-seed test is a **counterexample**, not a population estimate of learned policy PDR.

### Explicit deviations and risks

- `MarlSearchDrone` uses dependency injection instead of `Drone.__init__` to prevent unwanted Poisson traffic, GaussMarkov motion and hard-coded DSDV. It retains upstream real `Greedy`, `CsmaCa`, `Phy`, `Channel`, per-hop ACK.
- Adapter deliberately staggers first HELLO by **1.5 ms × UAV ID** to avoid deterministic initial hidden-terminal HELLO losses; original periodic HELLO thereafter unchanged. Cite this modification in any paper.
- The actor receives peer poses only through **received local HELLO neighbor table**, but upstream Greedy routing still reads the true destination coordinates to choose a next hop. This limits strict Dec-POMDP claims.
- UavNetSim-v1 uses process-global `utils.config` and SimPy polling at 5/10µs; cannot be substituted for 4,096 parallel Warp rollouts.
- Warp tested on **CPU** because local CUDA driver unavailable. Pairwise Friis parity does NOT prove PHY SINR under interferers or MAC/queue/retry/TTL parity.

### Gate 5D: scheduled multi-transmitter SINR (new Warp CPU kernel)

A new NVIDIA Warp FP32 CPU kernel at
uav_network/communication/gpu/classic_v1_interference_warp.py
calculates receiver SINR using simultaneous on-air frames on co-channel or
adjacent IEEE 802.11b channels. The source condition is absolute channel-ID
difference less than 5. This is a fixed-schedule PHY evaluator, NOT a MAC.

- Hidden terminals with two 400m links overlapping on channel 1:
  -0.5283353 dB on each frame, 0/2 successful.
- Same channel with no transmission-time overlap: 8.8819027 dB, 2/2 pass.
- Channels 1 and 6 with overlapping frames: 8.8819027 dB, 2/2 pass.
- A separate NumPy FP64 PHY implementation agreed for 128 random worlds
  x 4 scheduled transmissions (512 evaluated frames), absolute tolerance
  .002dB. The action remains valid only for a *given* frame schedule.

Evidence: docs/superpowers/evidence/zhang_xue_gate5d_warp_interference_cpu_20261125.json

Still missing: a vectorized Warp MAC that generates schedules with CCA,
backoff, ACKs, retries, TTL and buffers; integration into 1km/6UAV GPU
training; multi-seed SimPy vs Warp parity across topology and traffic.
The actual Warp kernel ran on CPU due missing local NVIDIA driver.

## Gate 5E: exact pinned-source CCA occupancy parity

Classic UavNetSim-v1 uses check_channel_availability in
utils/util_function.py: a node finds the channel busy iff another node
currently holds any channel_state resource within strictly less than
750m (v1 default SENSING_RANGE). The classic function does not filter
channel number at this CCA step; receiver interference SINR does
filter channel overlaps separately. A new Warp FP32 CPU kernel
uav_network/communication/gpu/classic_v1_cca_warp.py reproduces this
snapshot rule. The source-comparator subprocess executed the actual
pinned upstream function on 128 seeded random worlds with six nodes
per world: 768 comparisons, 707 busy and 61 free, zero mismatches.
Boundary tests confirm exactly 750m is permitted, while 749.9m is
blocked, and two hidden terminals 800m apart cannot sense one another.
Source: docs/superpowers/evidence/zhang_xue_gate5e_warp_cca_cpu_20261125.json.

This does not yet schedule DIFS, contention window, repeated busy
listens, per-hop MAC ACKs, application E2E ACKs or packet TTL. It is
an independently verified ingredient of the remaining vectorized MAC
implementation, NOT a full Gate5 pass.

## Required next action

Implement a collision-aware, vectorized Warp network surrogate with receiver SINR from simultaneous senders, channel overlap, carrier sensing and hidden-terminal behavior, queue/TTL/retries and separate MAC ACK vs application E2E ACK. Compare matched seeds, packet sizes, routes, PDR, latency, hops, airtime and ACK across benign and collision stress tests before E4096 GPU training. Do not silently replace failure with an easier metric. Gate 5 **full SimPy↔Warp parity remains failed/not ready** despite passing subgates 5A and 5B.

No change to `main`, historical W&B/HF runs or old 800m Warp scenario.
