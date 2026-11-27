# V93 active candidate contract — 27 November 2026

Status: CANDIDATE_NOT_FROZEN. The V88 active-scope document is historical.

| Field | Current value |
|---|---|
| YAML | research/zhang_xue/configs/scenario_v91_bayes_1km.yaml |
| Map / agents | 1000m square / 6 UAV + GCS |
| Static targets / legal cells | 15 / 392 |
| Altitude / horizon | 50m / 900 one-second steps |
| Motion / limits | XY, 15m/s, 5m/s², 50m separation |
| Sensor | circular 50m footprint, Pd=.85, Pf=.03, 3D cylinder occlusion |
| Belief | origin-indexed persistent raw counts; per-cell Bayes log odds |
| Prior / confirm | 15/392 / .90 |
| RF | fixed .1W; nominal 12dB per building; CSMA/CA research surrogate |
| Actions | continuous (E,6,2); communication (E,6): silent/GCS/PEER |
| Observations | raw local 549, mask 3, wrapper encoded 553, joint state 3318 |

No TX-power action, vertical motion, landing or takeoff is learned.
GCS is a routed application destination. PEER is evidence dissemination, not the switch enabling all multi-hop routing.
The centralized environment safety filter may use world geometry; actors must use only their permitted local observation.

Safety bits: 1 boundary, 2 building, 4 pair, 8 braking warning, 16 invalid action, 32 optimistic next-step pair viability failure.
A rejected transition retains previous positions and velocities and terminates with the existing safety penalty.
The bit32 triangle-inequality check is necessary only; it is not a global recursive-feasibility guarantee.

Use explicit YAML in every audit and record source/config hashes. Do not use historical runner defaults as the active scientific configuration.
