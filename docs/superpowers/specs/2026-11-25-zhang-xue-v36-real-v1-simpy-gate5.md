# Zhang–Xue v3.6 real classic SimPy UavNetSim-v1 adapter and Gate 5

## Provenance and safe rollback

- Source: https://github.com/EdwardS201/UavNetSim-v1
- Pinned SHA: `64bc3a2424f2976418731ee94f9a8fd3e8810b22`
- License: MIT, copyright 2025 Zihao Zhou; keep upstream LICENSE intact.
- **Not** the GitHub project `Zihao-Felix-Zhou/UavNetSim` (newer Sionna RT implementation).
- Bootstrap: `bash scripts/fetch_uavnetsim_v1_classic.sh` clones to ignored `outputs/reference_sources`; use `UAV_CLASSIC_V1_ROOT` for an existing verified checkout.
- Original 2D/MATD3/MASAC pilot v3.5 Git commit: `0803334c1d4e`. Separate v3.4 source branch at `320bc5b`.
- The current v3.6 Git branch must use incremental commits, push with normal non-force `git push`, and never merge into `main` automatically.

## Requirements

Build a real **CPU SimPy v1 network adapter** under a new module using upstream `CsmaCa`, `Phy`, `Greedy`, and `Channel`, with no original `Drone.__init__` process side effects. Dynamically configure pin-checked PHY and non-destructive radio blacklist. Add application-layer 512B payload, separate E2E ACK, own-origin evidence, per-receiver merging and age-aware retransmission. Ensure native Data PDR excludes E2E ACK from denominator and numerator.

Construct reproducible 3-node 0–400–800m test topology to demand a **real 2-hop** path at 0.5W, 2.4GHz and 6dB (ideal dmax ~557m). In real SimPy, include HELLO control airtime, slot/backoff, retransmissions, MAC ACK, packet lifetime, collision and waiting queues as supported by source. Compare to v3.5 reference only at metric/invariant level, **not assert bitwise parity**, because reference lacks CSMA/CA event detail.

## Scientific constraints and explicit stop conditions

- This is a single-episode **CPU true-network test**, not thousands of vectorized Warp instances.
- Use no pretrained UAV MATD3/MASAC checkpoint in a different action/state space.
- Real classic `Drone` has hard-coded `Dsdv`, `GaussMarkov3D`, and Poisson traffic; subclass/DI must **never invoke parent constructor**.
- v1 network modules use global `utils.config`; do not concurrently run different PHY instances in same Python interpreter. Use subprocess isolation for controlled tests/evaluation.
- Greedy next-hop uses neighbor cache; upstream also reads destination coordinates (needs consent/assumption as globally known destination position). Report as a partial Dec-POMDP assumption, don't hide it.
- Radio blackout cannot be emulated with original `sleep=True` (permanently stops feed/receive); add independent radio gate while preserving queues/deadlines.
- Only claim Gate 5 *parity* if measured quantitative tolerances at matched settings and topology; on failed parity, report failed metrics.
- Keep legacy 800m/25T Warp environment untouched. No claim full 1km Warp, 4096-envs throughput or GPU convergence until actually implemented and tested.

## Checkpoints

- [x] v3.4 and v3.5 branches individually on GitHub, SHA verified.
- [x] new v3.6 branch local and classic upstream pinned in separate clean checkout.
- [x] bootstrap script to retrieve unchanged upstream with exact SHA.
- [x] Real SimPy packet tests (actual CSMA/CA + E2E ACK and blackout), including 2-hop U0 to U1 to U2 and E2E ACK reversed.
- [x] Gate 5A: real V1 SimPy vs simplified reference on fixed 2-hop (PDR1 vs1, hops2 vs2, DATA delay1.635ms vs1.604ms, mismatch31us). Not full Warp parity.
- [ ] Gate 5B: Warp PHY matching and multi-topology/multi-seed contention.
- [ ] Gate 5 full end-to-end SimPy-Warp parity on same 1km scenario.
- [x] Pushed three incremental tested v3.6 commits: pin, SimPy networking, six-UAV integration.
- [ ] Publish final Gate5 evidence and limitations.
