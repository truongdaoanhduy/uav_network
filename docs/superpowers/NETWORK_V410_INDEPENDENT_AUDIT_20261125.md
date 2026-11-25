# Independent audit of Zhang–Xue v4.9 — 25 November 2026

Input: user ZIP uav_v49_full_audit_and_solutions_20261125.zip; read reports 01–05, proposed Python patch and supplied tests. Baseline: v4.9 SHA b2d211bfc13215ad0abac56f9d1b7ef0a3cdf9b0. Actual classic source: pinned UavNetSim-v1 64bc3a24, NOT the vendored v2.0.0. Changes isolated in experiment/zhang-xue-v410-reviewed-audit-fixes-20261125.

## Claim-by-claim disposition

| Issue | Verdict | Code action / reason |
|---|---|---|
| 1.1 target-only discovery/sharing information reward | CONFIRMED BUG. 390/400 empty cells could not produce dense information reward due truth masking. | CPU RewardEngine now sums positive per-cell reductions over the complete 400-cell map. Legitimate absence observations rewarded; existing TP/FP and airtime logic preserved. Additional regression proved a false-confirmed empty cell formerly generated +18.9 reward net; now that specific confirmed false-positive cell cannot earn information reward on the same step and FP penalty remains. Reward weights still require controlled ablation. |
| 1.2 200-step termination must be truncation | REJECTED as blanket fix. A 200-step fixed-horizon TASK correctly terminates; Pardo ICML2018 and Gymnasium both require remaining time observed to preserve Markov property. Current actor already includes remaining time fraction in base[4]. | Do NOT change termination or off-policy bootstrap. If 200-step cutoff later becomes TRAINING-only for an infinite-horizon task, change it to truncation with bootstrap then. |
| 1.3 missing no-fly-zone local actor input | CONFIRMED. UAV moved/blocked by NFZ but actor had no NFZ input. | Idealized local sensor range 300m provides two nearest visible NFZ center offsets and distances, six dimensions; actor obs 113 to 119. Unseen distant NFZ changes do not alter observation. Replace dead constant base[5] with locally observed NFZ fraction. CPU reference only; Warp production actor not yet integrated. |
| 1.3 global blackout Phi and AoI | CONDITIONAL, not proven programming bugs. Injecting exact global blackout map from simulator would be decentralized-oracle leakage. AoI must derive from received packet timestamps. | Do not add 16 global truth Phi or 3 fake AoI scalars; first implement observable local blackout history and genuine time-stamped packet delivery. |
| 1.4 fixed UAV arbitration priority | CONFIRMED. Ascending-ID resolution systematically prioritizes lower IDs in collisions. | Alternate ascending versus descending on even/odd steps for both CPU pilot and Warp motion kernel; parity test added. It remains simplified sequential collision projection, not second-order physics. |
| 1.4 first-order dynamics and constant Pd/Pf | MODEL ASSUMPTION, NOT UNIVERSAL BUG. Zhang and Xue may assume different dynamics/sensors. | Preserve 50 m/s, 200-step, fixed 0.85/0.03 to keep current baseline comparable; model changes require clear experimental protocol and retraining. |
| 2.1 HELLO 4 seconds / TTL 12 seconds at 50 m/s | REAL RISK of stale links: opposite flight at 50 m/s gives 400m relative drift per 4s. Pinned native Greedy source actually uses HELLO 0.5s and neighbor expiry 2s; report attribution to QFRP paper exact 0.5/1.0s remains unverified. | v4.5 moving pilot now takes optional hello_period_steps=1, hello_ttl_us=2000000 to exercise mobility-aware 1s/2s setting while maintaining old 4/12s defaults so old baseline reproducible. Current Warp movement uses true step index. This is NOT source-exact 0.5s HELLO; substeps and real scheduled control frames still pending. |
| 2.2 MAC CCA uses DATA sender rather than physical ACK TX, and half-duplex | PHYSICAL BUG, but native SimPy source uses sender channel resource throughout DATA+SIFS/ACK wait, so universal patch BREAKS source replay. Early patch broke seed54 parity. | Added physical_carrier_sense=True for physical ACK transmitter location, filtering blacked-out on-air TX and checking ACK phase half-duplex. Keep False default source-compatible with pinned classic-v1 SimPy and its seed PDR tests. Neither is a full continuous multiagent scheduler. |
| 2.2 blacked-out sender occupying CCA | DEFENSIVE EDGE, not demonstrated active bug because static blackout nodes cannot start DATA and cannot emit ACK. | Physical option filters blacked-out on-air transmitter; native source mode preserved. Dynamic blackout mid-air remains unverified. |
| 2.3 v2 UavNetSim interchangeability | CONFIRMED DISTINCTION: vendored version is 2.0.0 with Sionna RT dependency; actual classic pinned SimPy is separate v1. | Keep v1 for exact parity; do not embed Python SimPy/Sionna RT within 4096 environment training loop. |
| 3.1 alleged 394 allocations, 67 synchronous D2H, 138 int64 vars | PARTIALLY VALID COUNTS, WRONG INTERPRETATION. Independent static scan reproduced 394 occurrences of wp.array/wp.zeros/wp.empty, 67 occurrences of .numpy(), 138 textual occurrences of int64 across ten files. These are not executed-per-step allocations, synchronizations or unique variables. | No runtime profiling or quantified speedup claim; measure on T4 once driver/approved platform available. |
| 3.2 CUDA graphs, SoA and blanket int32 conversion | RECOMMENDATIONS ONLY. Installed NVIDIA Warp 1.18.0 API exposes ScopedCapture, Torch interoperability, get_cuda_kernel_properties and kernel grid_stride. No local GPU/driver to test CUDA behavior. | Retain 64-bit absolute microsecond timestamps to avoid overflow, unless proves bounded epoch segmentation. Avoid promising 50k–100k envsteps/s, register gains, or copy-free step before profile. |
| 3.4 two-fidelity Warp versus original SimPy | VALID STRATEGY, not a validated result. Isaac-Net 2026 is an arXiv preprint, not proof of Q1 status or native MAC equivalence. | Keep SimPy offline native comparison and Warp GPU training separate; do not claim under-10% P95 or all-step trace fidelity without measurements. |
| 4.1 MATD3 tanh, MASAC alpha, replay UTD/size/normalization | HYPOTHESES requiring learner implementation and training logs; references like PQL, FastTD3, SimBa do not prove the exact values for this project. | No arbitrary hyperparameter/architecture rewrite before full Gate5 and controlled ablation. |
| Paper labels Q1/A* | INCORRECT blanket venue labeling. WCSP 2024 is a conference, not Q1 journal; ICML/ICLR/NeurIPS are conferences; Isaac-Net 2026 arXiv is a preprint. Electronics Q quartile needs independent year-specific source verification. | Cite primary paper venue and date, do not promote all 32 ZIP references as verified Q1. |

## Critical source-model discrepancy

Pinned classic-v1 mac/csma_ca.py CsmaCa.mac_send takes self.channel_states[self.my_drone.identifier].request() and retains that sender resource while yielding DATA frame airtime plus SIFS/ACK wait. Physical carrier sensing of the receiver ACK location is *not equivalent* to this SimPy abstraction. A single global patch broke source-seed54 PDR; a dual-mode flag resolves this conflict. Original classic source is authoritative for code replay; physical signal propagation is a separate model and validation target.

## Sources

1. Zhang et al., IEEE T-IV 2024: https://ieeexplore.ieee.org/document/10388472
2. Xue et al., Electronics 2026, paper published 29 April 2026: https://www.mdpi.com/2079-9292/15/9/1875
3. Pardo et al., ICML 2018 finite vs artificial time limits: https://proceedings.mlr.press/v80/pardo18a.html
4. Official Gymnasium guide: https://gymnasium.farama.org/v1.2.2/tutorials/gymnasium_basics/handling_time_limits/
5. Pinned classic v1 public source: https://github.com/EdwardS201/UavNetSim-v1
6. Current upstream v2 source: https://github.com/Zihao-Felix-Zhou/UavNetSim
7. NVIDIA Warp Torch/CUDA graph documentation: https://nvidia.github.io/warp/stable/user_guide/interoperability/pytorch.html and https://nvidia.github.io/warp/stable/user_guide/runtime.html
8. NVIDIA Turing optimization guide: https://docs.nvidia.com/cuda/turing-tuning-guide/
9. Isaac-Net preprint Oct 2026: https://arxiv.org/abs/2610.02370
10. Original UavNetSim-v1 source tested locally: classic commit 64bc3a24.

## Reproducibility / release gate

These patches affect individual CPU and Warp pilot components. Actor observation changed 113 to 119 dimensions, so old actor checkpoints are not load-compatible without network changes/retraining. Old v4.5 HELLO baseline remains default 4/12s, mobile mode optional. No local NVIDIA driver: CUDA, device memory graph capture and E4096 T4 performance unverified. **Full Gate5 combining continuous physical MAC, five-hop dynamic routing, actor motion/sensor, local DS, shared-channel ACK and persistent GPU queues is NOT passed. Final MATD3/MASAC training/evaluation/test is NOT launched.**
