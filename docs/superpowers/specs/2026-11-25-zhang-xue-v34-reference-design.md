# Zhang–Xue v3.4 reference-fix design — 2026-11-25

**Intent:** Preserve all existing UAV research sources and address demonstrated v3.2 protocol/reward defects with reproducible tests before attempting a production environment rewrite.

**Chosen approach:** New isolated Git worktree from clean Streamfix v3 HEAD `d3e08b1`, with the original v3.2 reference ZIP extracted into `research/zhang_xue_v34_reference/`. Rejected approaches: (1) change the dirty main/study branches, risks overwriting unrelated experiments; (2) claim a standalone ZIP already constitutes complete production integration.

**Contracts:** 6 homogeneous UAV scenario remains a higher-level target; this packet mini-harness tests 3 peers. Source packets have immutable binary payloads and finite TTL. Queue capacity covers both transmit and waiting. Rejected packet gets no TX event. Blackout/no-route sends park packets and release when radio becomes usable on subsequent `run_until` calls. E2E ACK is distinct from MAC ACK, and PDR measures data only. The R1 reward accepts per-cell information attribution and keeps simulator truth only at the reward boundary. Safety/geometry costs are nonnegative magnitudes and subtracted.

**Non-goals:** Reproducing SimPy UavNetSim MAC or PHY, Warp/SimPy parity (user deferred Gate 5), changing policy/critic architectures, running GPU training, changing W&B or HF, or modifying the previously dirty copies.

**Acceptance:** Pre-fix red tests for queue admission, retry, capacity, penalty signs, false-positive information credit, and immutable wire fields; green all tests after correction; existing relevant production tests still pass; no source mutation outside isolated worktree. The reference package must label all unimplemented production requirements.
