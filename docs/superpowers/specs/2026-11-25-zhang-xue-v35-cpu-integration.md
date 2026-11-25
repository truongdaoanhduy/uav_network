# Zhang–Xue v3.5: CPU integration pilot — 2026-11-25

## Scope and source lineage

The user's preserved v3.2 reference specification is in
`research/zhang_xue_v34_reference/config_v3.2.yaml`. The previous repaired
v3.4 reference is preserved in that same directory, committed at `320bc5b`.

**This v3.5 task integrates that reference into a NEW production-package
environment and the EXISTING Vanilla MATD3/MASAC learner/replay interfaces.**
The existing 800 m, 25-target, Warp slotted scenario remains unchanged,
since its compiled constants and observation dimensions are incompatible
with v3.2's 1000 m, 10-target scientific pilot.

## Implemented CPU path

- `uav_network/environment/zhang_xue_coop.py` implements a reproducible
  6-UAV, 20×20×50m 2D world with 10 targets, horizon200 and dt1s; 2 no-fly
  disks (r50m) and 2 blackout Manhattan diamonds (radius3 cells).
- Continuous `(v_x,v_y)` action normalized to 50m/s with boundary,
  obstacle-segment, and inter-UAV separation checks (50m).
- Discrete peer selection 0=NO_SEND, 1..5 ordered peers excluding self;
  source-only packet entries, strict 512B MTU and uint16 counters.
- Local independent per-UAV sensing with Pd0.85, Pf0.03, DS masses0.8,
  threshold0.95. Own-source MAX plus SUM across origins by default.
- Packet transmission uses previous-step cumulative counts; receiver
  updates only after a delivered reference packet; E2E ACKs are accounted
  separately and acked sequences are exact (not assumed contiguous).
- Unacknowledged per-peer counters remain dirty and can be resent; pending
  packet state expires after a bounded data+ACK deadline window.
- Local observation includes local DS patch and pooled map, normalized
  motion/time/radio state, per-peer local dirty summaries; exact remote
  coordinates hidden outside ideal single-hop radio sighting and for
  blackout peers. Centralized critic input is concatenation of local
  observations, **not true target positions**.
- Reward R1 uses simulator truth **inside reward only** to prevent
  positive credit for false positives, and subtracts safety/boundary
  penalties; MASAC uses 0.6×ln(K_valid) discrete entropy target.

## Actual learner/replay execution

`uav_network/experiments/train_zhang_xue_pilot.py` runs the new
environment with existing `VanillaMATD3`, `VanillaMASAC`, and
`GPUHybridReplayBuffer` on **CPU**. Actor and centralized critic take
observation/state sizes discovered from the new pilot at runtime; the
legacy 800m Warp checkpoints are NOT compatible with this state shape.

Run from repository root:

```bash
/home/aduy/Documents/NCKH/uav_network/.venv/bin/python \
  -m uav_network.experiments.train_zhang_xue_pilot \
  --algorithm matd3 --episodes 2 --horizon-steps 12 \
  --learning-starts 4 --batch-size 4 --hidden-size 32 \
  --eval-episodes 2 --seed 44
```

Repeat with `--algorithm masac`. No Kaggle/W&B/HF operations are
performed. The output explicitly labels `cpu_reference_not_uavnetsim`.

## Verified 2026-11-25, local Ubuntu / PyTorch CPU

- TDD: production module initially failed to import; tests then became
  GREEN. Distinct RED→GREEN cases for outside-radio remote coordinates,
  expired pending ACK state, and MASAC 0.6 entropy parameter.
- New test files: **16/16 passed**.
- Preserved v3.4 reference tests: **14/14 passed**.
- Scoped 2D/M2/slotted/vanilla regression: **159 passed, 1 skipped**,
  excluding **one known pre-existing SHA fingerprint assertion**.
- Randomized invariant stress: **60 seeds × 40 steps = 2400 steps**,
  11,139 data send attempts, 10,929 E2E ACKs, maximum pending 27,
  no queue overflow, non-finite belief, map exit or broken min-separation
  assertion. This is reference networking, not UavNetSim parity.
- Real CPU MATD3 smoke: 21 learner updates, 24 environment steps,
  actor parameter delta norm 0.088381, deterministic eval return
  18.09704 and data PDR 1.0 **in simplified reference PHY**.
- Real CPU MASAC smoke: 21 learner updates, 24 environment steps,
  actor parameter delta norm 0.156392, eval return 17.56904 and data
  PDR 1.0 **in simplified reference PHY**.
- `compileall` for all new Python files passed.
- Full project test collection: 1,975 tests collected. Entire test
  suite **NOT GREEN**: stopped at an old 3D marginal camera-FoV
  probability test that *also fails with identical values on the
  unchanged Streamfix v3 parent*. The old SHA fingerprint test likewise
  fails on that original parent.
- CUDA is unavailable locally, so NO Warp GPU training claim can be made.

## Important gaps / blockers

**No production SimPy / UavNetSim CSMA/CA**, collisions, retransmission
timers or precise mobility-channel event timing. The ported
`MiniDiscreteEventNet` is deterministic and approximates MAC overhead:
its PDR should not be compared with realistic network papers.

**No Warp GPU implementation** of the new 1km Zhang–Xue scientific
scenario, no SimPy↔Warp parity (Gate 5 explicitly deferred), no scalable
4096-env kernel. No full RL convergence or 5-seed benchmark.

Radio peer localization assumes instant one-hop ideal ranging/beacons
when both radios are on; no actual HELLO/neighbor-cache protocol. Routing
inside the mini-harness uses simulator topology for greedy next-hop;
this is a test harness simplification, NOT a validated Dec-POMDP routing
protocol. World motion/network topology is frozen inside each dt step,
not event-time 802.11 operation. No active AoI-DS decay (eta0, gamma1),
network energy coupling, or GCS task-specific application delivery.

**Experiment interpretation:** Existing 800m Warp scenario and this
new 1000m research pilot are two different MDPs and cannot be merged
into the same statistical run/metrics table or checkpoint directory.

## Next work

1. Pin actual UavNetSim-v1 vendored SHA and implement an application
   packet/ACK callback compatible with SimPy v1, disable default
   traffic/Gauss-Markov mobility, and test timeout/queue contention
   with dynamic coordinates and real MAC timing.
2. Specify protocol-time observations/neighbor cache and app-level
   retransmission/anti-entropy, preserving the own-source-only rule.
3. Only after a real CPU ground-truth simulator is stable, implement
   matching 1km kernel state in Warp and compare PHY/MAC/report timings
   in Gate 5 if user re-enables it.
4. Run restricted single-GPU smoke, then evaluation seeds independent
   of training; only later full multi-GPU training.

Upstream:
https://github.com/EdwardS201/UavNetSim-v1
https://arxiv.org/abs/2507.09852
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html
