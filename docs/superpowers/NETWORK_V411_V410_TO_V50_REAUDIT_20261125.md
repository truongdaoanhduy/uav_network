# V4.11 v4.10→v5.0 independent source and paper re-audit (25 November 2026)

Parent commit: 7f03e9bfb3b9ded4c9ee5b3d0ccfb0104a8bb5a9. Branch: experiment/zhang-xue-v411-reverified-halfduplex-nfz-20261125.

## Confirmed and fixed using RED→GREEN actual code tests

1. PHYSICAL ACK half-duplex: sender k in phase 4 waits for ACK; actual ACK TX node is dest[k]. If same node receives another DATA while TX ACK it must collide even if channels differ (one radio). Direct Warp test U0→U1 DATA50-130us, ACK U1→U0 140-260us; U2→U1 DATA150-230us on different channel had incorrectly been received in original. New Warp checks overlapping DATA and ACK. Phase[rx] == 4 kept: receiver is receiving ACK, also radio-busy. Native source-compat False preserved.
2. NFZ observation: two slots of four fields (dx,dy,distance,present flag); empty slots (0,0,FOV/L,0), not (0,0,0). CPU actor dimensions 119→121; checkpoint architecture must be updated.
3. CPU reset RNG: reset(None) used to replay exact same map; now continues existing per-env RNG, explicit reset(seed=X) deterministic.
4. MAC FP32 µs precision: float32 exact +1µs limit 2^24µs, ULP16µs at 200s. Present bounded one-hop MAC resets local clock, so don't claim existing 200s damage. Future absolute shared schedule requires int32 episode-relative µs with max1e9 µs bound, and int64 global time if needed. The change isn't a blanket replacement.

## Wider scientific findings

- Active Zhang CPU config = 1000m/6U/10T/200s/50m/s/2 NFZ discs/2 blackout diamonds/no GCS. Existing SCENARIO_AND_ARCHITECTURE_SPEC.md = 800m/6U/25T/550s/20m/s/urban 22 buildings/GCS. Treat as different experiments.
- Zhang 2024 T-IV p3 assumes full communication and p5 Eq16 combines positive/negative observations with MAX; p7 Eq27 signed global uncertainty reduction. Local MAX_PLUS_SUM and per-cell positive clipped reward are variants, not exact replication.
- Xue 2026 Electronics p6 global blackout Φ is explicitly known to all UAVs, p11 Eq14 reward DS entropy global decrease, p12 actor input six 2D maps; current strict local actor and network MAC are research extensions and not original Xue implementation.
- Source v1 SimPy and vendored UavNetSim v2 SionnaRT distinct; default source compatible False preserves native replay.
- Full unified Warp network with simultaneous HELLO/DV/DATA/MAC_ACK/APP_E2E_ACK and actual CUDA graph does not exist yet; Gate5 NOT complete, NVIDIA local driver unavailable, MATD3/MASAC full train/eval/test not launched.

References: https://doi.org/10.1109/TIV.2024.3352581 ; https://doi.org/10.3390/electronics15091875 ; https://github.com/Zihao-Felix-Zhou/UavNetSim ; https://gymnasium.farama.org/main/api/env/ ; https://nvidia.github.io/warp/v1.18/user_guide/runtime.html .

Tests: tests/test_zhang_xue_v411_reaudit_findings.py imports real Warp and CPU environment classes; original user ZIP 4/4 pass only tested standalone reference helpers and did not directly verify actual simulator classes.
