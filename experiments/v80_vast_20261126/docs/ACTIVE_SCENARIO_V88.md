# Active UAV study (V87/V88) — enforced scope

Only this scenario is active. Do not substitute historical configuration or infer active code from an older commit or W&B run name.

| Parameter | Active value / source |
|---|---|
| Map | 1000 × 1000 m (`map_size_m: 1000`) |
| UAV and GCS | 6 UAV + 1 GCS (`num_uavs: 6`) |
| Static targets | 15 (`num_targets: 15`) |
| Belief grid | 20 × 20, 400 cells (`grid_w: 20`) |
| Flight altitude | fixed 50 m (`uav_alt_m: 50`) |
| Episode horizon | 900 steps (`safety_horizon_s: 900`) |
| Flight speed / acceleration | 15 m/s / 5 m/s² |
| Sensor | Pd=.85, Pf=.03; 50 m footprint |
| DS | mf=me=.80, confirm=.90, eta=.02 (baseline under audit) |
| RF | fixed TX .1W; 12dB illustrative per-building attenuation; CSMA/CA, DV HELLO, ACK via custom Warp implementation |
| Algorithms | Continuous MATD3 and MASAC with decentralized actors, centralized critics |
| GPU | Only Vast.ai for new tests; prior Kaggle archives forensic only |

## Implementation order and scientific guards

0. Verify source SHA, YAML, project policy and exact CLI before changing code.
1. RED test reproduces a specific bug and GREEN test proves the fix; then run full CPU/CUDA regression.
2. Validate episode metrics before `auto_reset_gpu`, test checkpoint and independent held-out seeds.
3. Confirm packet/RF mechanics with controlled event traces; custom Warp communication is **not yet proved equivalent** to upstream UavNetSim.
4. Experimental static belief persistence and HELLO peer kinematics are **opt-in future ablations**, not part of active default; never feed a decentralized actor hidden true-world peer positions.
5. Run end-to-end GPU smoke E4096 and longer multi-seed evaluation before claiming convergence or scalability.

Do not interpret zero boundary/building collisions as all-safe if UAV–UAV pair flags are nonzero. Label the current `mission_snapshot` as a *post-reset live-world snapshot* rather than terminal episode statistics; use `completed_episodes_summary` for terminal metrics.
