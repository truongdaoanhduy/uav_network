# Zhang–Xue v3.5 CPU Integration Plan — 2026-11-25

## Scope
New isolated 1km/10T CPU experiment, preserving old 800m/25T Warp.
Reference mini-PHY is NOT UavNetSim SimPy MAC or GPU Warp.

## Completed (tested)
- [x] Isolated v3.5 branch from preserved v3.4 ref commit.
- [x] RED missing-module test then GREEN 1km 6UAV env implementation.
- [x] Source-own belief and packet delivery, exact ACK sequences.
- [x] RED→GREEN: radio-out-of-range and blackout local observations.
- [x] RED→GREEN: pending packet lifecycle timeout.
- [x] RED→GREEN: MASAC 0.6 ln(K_valid) entropy target.
- [x] Actor→environment→replay→actor/critic update pilot CPU.
- [x] 16 new, 14 reference, 159 scoped tests pass (1 scoped skip).
- [x] 60 seeds × 40 steps random stress.
- [x] Parent-branch forensic reproduction of two old failures.
- [x] Evidence reports, current-task continuity, source documentation.

## Deferred
- [ ] True UavNetSim-v1 SimPy routing/MAC/PHY event hookup.
- [ ] HELLO/neighbor cache in fully decentralized protocol.
- [ ] GPU Warp 1km kernels and CUDA parity Gate5.
- [ ] Full 20k/5 seeds training, Eval100, paper ablations.

Full 1974 test suite is NOT green. Do not claim it is.
