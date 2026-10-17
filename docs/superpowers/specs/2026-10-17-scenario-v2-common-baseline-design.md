# Scenario V2 common-baseline design

Date: 2026-10-17
Base revision: `e7b3d89b3ef0cffcece049ced84ea06659704284`
Implementation branch: `scenario-v2-common-20261017`

## 1. Research decision

Scenario V2 must improve the **shared task formulation**, not rescue one learner. MASAC and MATD3 remain standard baseline implementations. Every environment, reward, observation, network, safety, seed-bank and evaluation change described here applies identically to both algorithms. A result where MASAC still outperforms MATD3 is valid evidence about the algorithms on this task and is not a defect to tune away.

The running diagnostics-v1 jobs and `configs/comparison-25k-diagnostics.yaml` are immutable baseline evidence. Scenario V2 gets a new config/version and must not silently change V1 defaults or invalidate V1 checkpoints.

Algorithm-source invariants at design time:

- `uav_network/algorithms/masac.py` SHA-256: `e6f38a562c7ae4a1bccaa5324689876450f4a75b4f4eb5f63e36c1b869660419`
- `uav_network/algorithms/matd3.py` SHA-256: `fee6b3692d0ac6115c340ab397bb8098c811a97331e0bca2cb0f8aab729ecf4b`

Scenario V2 implementation must not modify those files.

## 2. Evidence motivating V2

The completed W&B runs and videos show that relay itself is implemented and used. MASAC can produce real multi-hop delivery (including 4-hop reports), while MATD3 often chooses silence or shallow/direct delivery. That difference is allowed to remain an algorithm result.

The shared scenario nevertheless contains three formulation problems that make the intended communication-control task weaker than described:

1. **TX power has little physical decision value in the current LOS action set.** Current settings are 2.4 GHz, 22 MHz bandwidth, 7 dB noise figure, 6 dB SINR threshold, 0.1--0.4 W TX power and a 500 m hard peer/GCS candidate range. With the pinned free-space LOS model and no interference, the approximate threshold ranges are:
   - 0.1 W LOS: 2377.9 m
   - 0.4 W LOS: 4755.7 m
   - required LOS power at 500 m: about 0.00442 W

   Therefore every LOS link admitted by the current 500 m mask is already comfortably feasible at the **minimum 0.1 W**. The actor cannot learn a meaningful near-low/far-higher LOS power law because the configured minimum power is already far above the nominal requirement. Urban NLOS is instead extremely harsh: about 70.9 m at 0.1 W and 112.6 m at 0.4 W. This is a scenario-level physics/action-scaling problem, not a MATD3-specific problem.

2. **The actor lacks an explicit, consistent link-cost signal.** `destination_mask()` advertises range/admission only and explicitly has no SINR. Peer features expose relative position/velocity/battery, so a network can infer some geometry, but it receives no nominal required-power estimate and no direct relay progress signal. The task asks a policy to jointly select receiver and minimum sufficient power while hiding the quantity it is meant to optimize.

3. **The safety shield prevents collisions but does not teach the policy to stop depending on it.** Current completed runs can have zero actual boundary/safety violations while shield-intervention rates remain very high. The reward penalizes blocked motion but not a safe-filter correction that prevents the block. Replay correctly stores the commanded action, so the right fix is a shared learning signal for correction magnitude rather than rewriting replay to executed motion.

A fourth scale issue is also material: physical UAV RF energy is orders of magnitude smaller than propulsion energy. The physical battery model should remain truthful, but a study whose stated objective includes communication-energy-aware routing needs an explicit, interpretable communication-efficiency objective instead of expecting a tiny raw-joule term to dominate learning.

## 3. Goals and non-goals

### Goals

- Keep MASAC and MATD3 unchanged as algorithm baselines.
- Make TX power a genuine continuous control decision inside the existing local communication radius.
- Let an actor identify which eligible receiver is cheaper/closer to the GCS without privileged target information.
- Preserve real LOS/NLOS, interference, CSMA/CA, ACK, retry, TTL and custody behavior.
- Keep hard motion safety while making excessive shield reliance costly to the shared policy objective.
- Preserve V1 behavior and checkpoint compatibility when all V2 feature flags are disabled.
- Maintain NumPy/native and Warp packet-environment parity.
- Evaluate scenario correctness independently from which algorithm wins.

### Non-goals

- No MATD3-specific Gumbel-Softmax, entropy bonus, architecture, LR/noise, epsilon or critic changes.
- No MASAC-specific changes.
- No reward whose purpose is to make MATD3 catch MASAC.
- No unconditional reward per relay hop.
- No change to target count, obstacle count, buffer size or report TTL in the V2 core comparison; those remain separate ablations.
- No replacement of the safety shield with APF or a global planner.
- No claim that a single seed establishes algorithm-wide superiority.

## 4. Considered approaches

### A. Algorithm-specific repair

Examples: alter MATD3 discrete gradient, add Gumbel-Softmax, tune exploration or change its network only.

**Rejected for baseline.** It destroys the clean question “how do standard MASAC and MATD3 compare on the same task?” and risks tuning the loser to the benchmark.

### B. Large scenario redesign

Examples: enlarge the map, change target/obstacle counts, change buffer/TTL, change NLOS model and change routing reward simultaneously.

**Rejected for V2 core.** Too many confounders make it impossible to attribute improvements. Buffer/TTL and alternative channel models can be separate ablations after the core formulation is validated.

### C. Physics-first shared Scenario V2

Keep the mission geometry, routing mechanism and standard learners. Correct the power-action scale, add shared link-state features, add an interpretable normalized communication-efficiency cost, and penalize safety-filter correction magnitude. Do **not** directly reward relay hops; relay should emerge when it improves delivery/energy/time.

**Recommended.** It fixes the task the algorithms see without changing either algorithm or forcing a specific routing topology.

## 5. Scenario V2 design

### 5.1 Versioning and backward compatibility

Add new configuration fields with V1-compatible defaults. Create a new config, tentatively `configs/comparison-25k-scenario-v2.yaml`, derived from diagnostics-v1 but with V2 fields explicitly enabled. Do not edit `configs/comparison-25k-diagnostics.yaml`.

Proposed new fields:

```yaml
network:
  tx_power_mapping: linear          # V1 default; V2 uses log
  power_limited_admission: false    # V1 default; V2 true
  link_state_observation: false     # V1 default; V2 true
reward:
  communication_efficiency: 0.0     # V1 default; V2 > 0
  shield_correction: 0.0            # V1 default; V2 > 0
```

V1 observation shape, reward and action semantics must remain byte-for-byte compatible when these flags/weights remain at defaults.

### 5.2 Power action semantics

Keep the existing action branch `continuous[..., 2] in [-1, 1]` for both algorithms. Only the environment maps that normalized action to watts.

For V2 use a **logarithmic power mapping** rather than linear watts:

```text
u = (a_power + 1) / 2
P = P_min * (P_max / P_min) ** u
```

Recommended V2 range:

- `P_min = 0.001 W`
- `P_max = 0.4 W`
- peer/GCS protocol candidate radius remains 500 m

Why this range:

- 1 mW LOS threshold range is about 237.8 m.
- 10 mW LOS threshold range is about 751.9 m.
- a 500 m LOS link nominally needs about 4.42 mW.
- 0.4 W remains available for difficult/interference/NLOS attempts and preserves the old maximum.
- log mapping gives useful action resolution near the milliwatt values actually relevant to ordinary LOS links; a linear 1--400 mW mapping would compress the important 1--10 mW region into a tiny fraction of the action interval.

The 500 m protocol radius remains a separate networking constraint. V2 therefore means “power controls reliability/feasibility **within** the neighbor radius,” not “power removes all protocol range limits.” This wording must be used in the paper/report.

### 5.3 Nominal max-power admission

With `power_limited_admission=true`, a destination is eligible only when all existing conditions hold **and** the nominal no-interference power required by the configured LOS/NLOS path-gain model is no greater than `P_max`.

This does not guarantee delivery: selected power may still be too low, interference can reduce SINR, CSMA/CA can collide, ACK can fail, retries can exhaust, TTL can expire, and buffers can reject custody. It only prevents asking the actor to select a link that cannot meet the nominal SNR threshold even at maximum power.

The mask remains at the final `num_uavs + 1` entries of the observation so existing MASAC/MATD3 masking code remains unchanged.

### 5.4 Shared link-state observation

When `link_state_observation=true`, append fixed-order features for every non-silent destination (ordered peers, then GCS) **before** the destination mask:

1. `required_power_fraction`: nominal no-interference required power represented in the same log-power coordinate as the action and clipped to `[0,1]`.
2. `gcs_progress`: `(distance(source,GCS) - distance(destination,GCS)) / map_diagonal`, clipped to `[-1,1]`. For the GCS destination, destination distance is zero.
3. `receiver_free_buffer_fraction`: peer free report-buffer capacity in `[0,1]`; GCS is `1`.

These are neighbor/beacon/channel-estimate features, not privileged target coordinates. Existing target information remains local FOV knowledge. The ideal receiver-admission handshake already assumes receiver metadata is advertised reliably; V2 makes the relevant cost information explicit rather than forcing the policy to reconstruct it implicitly.

Do not add a recent-ACK/SINR recurrent feature in V2 core. The real PHY remains stochastic/interference-dependent; the nominal power feature is enough to make the immediate decision well posed without adding stateful link memory. Recent-link feedback can be a later ablation.

### 5.5 Communication-efficiency objective

Physical RF battery drain remains unchanged and continues to use actual transmitted bits, selected power, retries and ACKs. V2 adds a **separate dimensionless reward term** so communication efficiency is scientifically visible to learning without pretending RF propulsion-scale joules are physically larger than they are.

Define a report-reference TX energy from configured packetization and `P_max`:

```text
E_ref = energy to transmit one complete report's DATA fragments once at P_max
communication_efficiency_cost = delta_UAV_RF_energy / E_ref
```

Then add:

```text
reward -= reward.communication_efficiency * communication_efficiency_cost
```

This automatically charges high power, repeated attempts and UAV-generated ACK energy according to actual network activity while leaving physical battery accounting untouched. `reward.communication_efficiency = 1.0` is the initial V2 calibration target: roughly one full-power report-equivalent costs one reward unit versus `+50` for final GCS delivery. A short scale smoke must verify that this term is non-negligible but does not dominate discovery/delivery before long training.

Keep the legacy raw-joule communication term for backward compatibility. V2 may retain its existing value because it is numerically negligible; the new normalized term is logged separately.

### 5.6 Safety-shield learning signal

Keep the dynamic-window safety filter and the boundary tangent fallback. Replay continues to store the **commanded** policy action; that is the action the learner chose. Do not replace it with executed safe velocity.

For each step compute the mean active-UAV correction:

```text
c_shield = mean_active(||v_safe - v_nominal||_2 / max_speed)
```

and add:

```text
reward -= reward.shield_correction * c_shield
```

Log both per-step/episode accumulated correction and the existing intervention counts. Initial V2 coefficient target: `0.1`, followed by a short scale smoke. This makes repeatedly commanding into boundaries/obstacles costly even when the hard shield successfully prevents an actual blocked move.

### 5.7 Relay remains emergent in the core baseline

Do **not** add `+X` for peer custody transfer in Scenario V2 core. A fixed hop reward can bias topology, reward unnecessary relay and confound the algorithm comparison.

Relay is instead encouraged indirectly and fairly by:

- final GCS delivery reward,
- mission/step cost,
- propulsion cost,
- explicit GCS-progress information,
- explicit required-link-power information,
- normalized communication-efficiency cost,
- real TTL/buffer/retry consequences.

If, after the physics-first V2 is validated, **both** algorithms still fail to learn useful relay despite available opportunities, create a separately named PBRS ablation. The only acceptable dense relay shaping for that ablation is potential-based shaping over target delivery progress; it must not be silently folded into the main V2 baseline.

### 5.8 Keep buffer and TTL unchanged in V2 core

Keep:

- report size 1024 B,
- buffer 3072 B (three reports/UAV),
- report TTL 300 s,
- destructive custody transfer,
- rediscovery recovery,
- max retries 5.

After V2 core results exist, buffer `3 vs 6 reports` and TTL `300 vs 600 s` may be tested as explicit scenario ablations. They must not be changed together with the V2 physics fix because that would confound attribution.

## 6. Native/Warp implementation parity

Production training uses `warp_packet`, so every V2 scientific behavior must exist in both implementations.

Likely implementation points:

- `uav_network/configuration.py`
  - new config fields and validation.
- new pure helper such as `uav_network/communication/link_budget.py`
  - deterministic nominal gain/required-power/log mapping helpers for CPU/native observation and tests; must not query/mutate the live PHY cache.
- `uav_network/environment/observations.py`
  - conditional link features, observation dimension and max-power admission.
- `uav_network/communication/backend.py`
  - shared power mapping for injected packets and eligibility consistency.
- `uav_network/environment/motion.py` / `avoidance.py` / `env.py`
  - nominal-vs-safe correction statistic and reward component.
- `uav_network/experiments/tracking.py`, `telemetry.py`, `comparison_metrics.py`
  - bounded W&B/evaluation metrics for the new terms.
- `uav_network/environment/gpu/state.py`
  - mirrored params/state/reward statistics.
- `uav_network/environment/gpu/observations.py`
  - identical conditional features and max-power admission.
- `uav_network/communication/gpu/phy.py` and packet action path
  - same log power mapping / nominal required-power semantics.
- `uav_network/communication/gpu/metrics.py`
  - mirrored reward/diagnostic extraction.

`uav_network/algorithms/masac.py` and `uav_network/algorithms/matd3.py` must remain untouched.

## 7. Telemetry additions

Add metrics that answer the scientific questions directly rather than inferring them from videos:

- successful peer custody transfers that reduce distance-to-GCS,
- successful peer custody transfers that increase distance-to-GCS,
- mean/sum custody GCS-distance improvement,
- actual DATA TX power histogram/mean (already present),
- nominal required-power mean on attempted links,
- actual/required power ratio or dB power margin on attempts,
- attempts below nominal required power,
- weak-signal vs interference failures and retry limit (already present),
- direct vs relayed GCS delivery and hop histogram (already present),
- physical UAV RF energy,
- normalized communication-efficiency reward cost,
- shield correction magnitude in addition to intervention count,
- report expired/dropped/buffer peaks and delivery delay (already present).

All pooled ratios must use raw numerator/denominator aggregation, following the existing `OutcomePool` design.

## 8. Evaluation contract

Scenario correctness and algorithm ranking are separate questions.

### Scenario-correctness gates

Before long training, deterministic microtests/smokes must prove:

1. V1 flags off reproduce the old observation shape, power mapping, reward and destination behavior.
2. Under V2, a near LOS link succeeds at low power while a farther LOS link can fail at the same low power and succeed at a higher allowed power.
3. A nominal link requiring more than `P_max` is masked when power-limited admission is enabled.
4. Selecting power below the advertised nominal requirement can still produce a real PHY failure/retry; the mask does not hide suboptimal chosen power.
5. Interference can still cause failure even when nominal required power is satisfied.
6. Native and Warp produce matching destination masks/link features for fixed geometry within numeric tolerance.
7. Native and Warp produce matching reward components for fixed deterministic transitions.
8. Shield correction is zero when the policy command is passed through and positive when the shield changes motion.
9. V1 boundary tangent fallback remains safe.
10. MASAC and MATD3 source files remain unchanged.

### Long-run comparison contract

For MASAC and MATD3 use the same:

- Scenario V2 hash/config,
- train seed/budget,
- final deterministic 30-seed evaluation bank,
- target/obstacle geometry distribution,
- max steps,
- observation/action semantics,
- reward,
- network physics.

BASE and PQL are execution/training-flow conditions and must be compared with matched budgets as already done.

Report at minimum:

- search, GCS delivery and full mission success,
- delivery-given-discovery and delivery time,
- direct/relayed delivery, hop histogram and relay progress,
- PDR/PHY success, retries, collision/interference/weak-signal failures,
- actual and required TX power distributions and power margin,
- physical RF and propulsion energy and J/GCS delivery,
- battery remaining/exhaustion,
- report expired/dropped/buffer pressure,
- shield intervention and correction magnitude,
- wall-clock/system throughput separately from policy quality.

There is **no acceptance requirement that MATD3 match MASAC**. A clean result `MASAC > MATD3` is a successful comparison if both experienced the same validated Scenario V2.

## 9. Test plan

Add/extend focused tests before implementation code (TDD):

- configuration validation for new flags/mapping/weights,
- exact linear V1 and log V2 power mappings,
- analytic LOS link-budget thresholds,
- max-power admission on/off,
- link-feature ordering and destination-mask tail invariant,
- native/Warp observation parity,
- low-power fail / higher-power success micro-link,
- interference failure despite nominal feasibility,
- normalized RF reward reference,
- shield pass-through/correction reward,
- reward-component accounting and pooled telemetry,
- V1 checkpoint observation-dimension compatibility when V2 disabled,
- scenario hash differs V1 vs V2,
- algorithm source hashes unchanged.

Then run the full authoritative suite as `python3 -m pytest -q tests`. Do not use bare `pytest` from the repository root because archived source trees under `outputs/` contain duplicate test basenames and are not part of the active suite.

## 10. Rollout sequence

1. Preserve current diagnostics-v1 jobs/results unchanged.
2. Implement V2 behind backward-compatible config fields using TDD.
3. Run CPU/native and Warp parity/unit suite.
4. Run short deterministic physics/safety smokes, not learning-quality tuning.
5. Run a small matched MASAC/MATD3 V2 smoke only to verify end-to-end learning pipeline and W&B telemetry; do not tune one learner to outperform the other.
6. Freeze the V2 commit/config.
7. Launch matched long MASAC/MATD3 BASE/PQL runs under the same budget and seed policy.
8. Evaluate final checkpoints on the identical held-out 30-seed bank.
9. Compare V1 vs V2 as **scenario ablation**, then compare MASAC vs MATD3 inside each fixed scenario.
10. Only after that consider separately named PBRS, buffer/TTL or channel-model ablations.

## 11. Design decision summary

Scenario V2 fixes the shared problem, not the losing algorithm. The core changes are:

- log-scaled 1 mW--0.4 W power action with the existing 500 m protocol radius,
- nominal max-power link admission,
- explicit nominal required power + GCS progress + receiver buffer features,
- normalized communication-efficiency objective while preserving physical battery energy,
- shared safety-shield correction penalty,
- expanded diagnostics/evaluation,
- unchanged MASAC/MATD3 implementations,
- unchanged V1 defaults and immutable diagnostics-v1 evidence.

Direct relay reward and MATD3-specific variants are deliberately excluded from the core baseline.
