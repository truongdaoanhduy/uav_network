# Scenario V2 validation — 2026-10-17

## Scope

Scenario V2 changes the shared UAV task formulation only. MASAC and MATD3 remain the standard baseline implementations. The V1 diagnostics profile and the four diagnostics-v1 lineages are not modified. This validation is a formulation/parity gate, **not** a claim that either algorithm is better.

Validated code commit before evidence-only documentation: `24e455833d621f451a16f83484d12b309633d9ea` on branch `scenario-v2-common-20261017`, derived from production baseline `e7b3d89b3ef0cffcece049ced84ea06659704284`.

## Invariants

- `configs/comparison-25k-diagnostics.yaml` SHA-256 remains `6253e377e72d2caf709c017dfdcd016b45b4b895605a4c257a704c364dd3d48b`.
- `uav_network/algorithms/masac.py` remains `e6f38a562c7ae4a1bccaa5324689876450f4a75b4f4eb5f63e36c1b869660419`.
- `uav_network/algorithms/matd3.py` remains `fee6b3692d0ac6115c340ab397bb8098c811a97331e0bca2cb0f8aab729ecf4b`.
- Scenario V2 has a distinct evaluation scenario hash from V1.
- V1 defaults retain linear 0.1–0.4 W power mapping, old observation size and no new reward terms.

## Implemented common Scenario V2 changes

1. TX action remains one normalized continuous scalar for both algorithms, but V2 maps it logarithmically over **0.001–0.4 W**. Action `0` maps to **0.02 W**.
2. The protocol peer/GCS radius remains **500 m**. Power controls nominal feasibility/reliability inside that radius; it does not remove the protocol range.
3. Nominal max-power admission excludes links whose no-interference configured link budget cannot meet the threshold even at `Pmax`.
4. Local actor observations conditionally add, for each peer then GCS: nominal required-power fraction, progress toward GCS, and free receiver-buffer fraction. The destination mask remains the final `num_uavs+1` entries.
5. Full 35-target/25-obstacle Scenario V2 actor observation dimension is **163**. A reduced 2-target/1-obstacle learning smoke used dimension **97**.
6. Physical battery accounting remains propulsion + actual UAV RF transmissions. A separate normalized communication-efficiency reward uses actual RF energy divided by one Pmax full-report reference emission; it does not inflate physical joules.
7. The reference energy for one current 1024 B DATA report at `Pmax=0.4 W` is **0.0017312 J** under current header/bit-rate settings.
8. Dynamic-window hard safety remains. A shared shield-correction reward penalizes normalized commanded-to-safe velocity correction while replay continues storing the commanded policy action.
9. No direct reward is added for relay hops. Relay remains emergent from final delivery, time, movement cost, link-cost information, communication efficiency, TTL/buffer/retry effects.
10. Telemetry now records required power, actual-vs-required dB margin, below-required attempts, peer relay progress toward/away from GCS, accumulated shield correction, and normalized reward components. Pooled fractions use raw counts.

## Physics checks

Using the configured 2.4 GHz / 22 MHz / 7 dB NF / 6 dB threshold model, a nominal LOS 500 m link requires approximately **0.00442144 W (4.42 mW)** without interference. This is why the previous 0.1 W minimum made LOS power selection largely trivial inside 500 m, while the V2 1 mW minimum creates meaningful low-vs-higher power decisions.

Explicit deterministic microtests verified:

- near LOS can succeed at low power;
- a farther in-range LOS link can fail at the same low power and recover at higher power;
- selected power below nominal requirement causes real PHY failures/retries while custody stays at the source;
- interference can still break a nominally feasible high-power link;
- Native and Warp match on V2 link-state/admission features;
- Native and Warp match V2 reward components and physical RF/battery accounting;
- peer custody telemetry distinguishes toward-GCS and away-from-GCS transfer;
- shield correction is zero for pass-through/no-active cases and positive when safety changes the command.

The explicit Task 7 validation set finished **11 passed, 2 subtests passed**.

## End-to-end learner smoke

A reduced CPU/Warp smoke used the same V2 semantics for both standard learners, with 1 episode and 3 environment steps only to validate pipeline compatibility:

- MASAC: observation dimension 97, 1 episode completed, global step 3, 2 updates, checkpoint written.
- MATD3: observation dimension 97, 1 episode completed, global step 3, 2 updates, checkpoint written.

The tiny returns are deliberately **not compared** because this smoke is not a learning-quality experiment.

## Test evidence

Before freezing the V2 profile, the authoritative active suite completed:

`444 passed, 47 skipped, 385 subtests passed in 231.76s`.

Task 6 focused profile/comparison checks completed `32 passed, 239 subtests passed`. Final review also made mean TX-power margin a **descriptive** metric: negative margin indicates under-power while excessive positive margin indicates waste, so paired comparison never uses it to auto-favor an algorithm.

## Runtime limitation

This local runtime reports no NVIDIA CUDA driver, so physical CUDA execution was not available. Warp CPU execution and Native/Warp semantic parity are validated here. Long Kaggle GPU runs should only be launched after the frozen branch is available to that pipeline.

## Research interpretation

There is no acceptance gate requiring MATD3 to match MASAC. If matched long Scenario V2 experiments later show `MASAC > MATD3`, that is a valid algorithm-comparison result. Scenario V2 is successful if the common task is physically meaningful and both algorithms receive identical validated semantics.
