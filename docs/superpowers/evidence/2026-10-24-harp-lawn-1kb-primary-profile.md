# HARP-LAWN 1 KiB primary training profile decision — 2026-10-24

## Source check

The user-provided HARP-LAWN paper (DOI 10.1109/JSTSP.2026.3713806) states in Table II that its UavNetSim-v1 simulation uses CSMA/CA, 2 Mbps, retransmission limit 7, and **Data packet payload length = 1024 bytes**. The paper does not define a separate 1 MB target-report object. Its Simulation Setup explicitly says the environment is implemented using UavNetSim-v1.

Important scope: matching 1024 bytes here aligns the packet/report scale used by the paper, but does not make this project identical to HARP-LAWN. HARP-LAWN also states that its reported link-quality results use an interference-free assumption, whereas this project retains its existing packet collision/interference logic.

## Existing project evidence

W&B historical 1024-byte production runs already demonstrated that 4096 environments are feasible at the current mission scale (35 targets, 1800 steps, warp_packet): multiple BASE/PQL runs completed 25,000 episodes in roughly 4.3–7.2 hours.

The existing isolated T4 diagnostic at `docs/superpowers/evidence/2026-10-24-t4-component-isolation.json` measured:

- active 1 KiB / dynamic: 9726.6712 joint transitions/s
- active 1 MB / dynamic: 38.74995 joint transitions/s

Thus the forced-active 1 MB packet path was about 251x slower than the matched 1 KiB case in that diagnostic. This is a simulator diagnostic rather than a full-training throughput ratio, but it directly isolates the packet workload.

## Decision

Use `configs/comparison-20k-hybrid75-1kb.yaml` as the primary profile for future 20k BASE/PQL comparison training.

Preserve `configs/comparison-20k-hybrid75-1m.yaml` and all existing 1 MB runs unchanged for reproducibility and future communication stress-test/ablation work.

The new primary profile preserves the training/environment scale and changes only the report workload semantics needed for the 1 KiB decision:

- report_bytes = 1024
- payload_bytes = 1024
- buffer_bytes = 3072 (three reports, preserving the prior three-report capacity intent)
- num_envs = 4096
- episodes = 20000
- max_steps = 1800
- replay capacity = 1,000,000
- batch size = 8192
- replay CPU fraction = 0.75
- environment backend = warp_packet

No active 1 MB production job is mutated or restarted by this config change.
