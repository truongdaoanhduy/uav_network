# v5.5 scientific contract: actor, DS, AoI, reward

This project deliberately distinguishes a reproducible paper baseline from extensions.

## Actor information budget

`ZhangXue7NodePacketEnv` returns six decentralized observations, shape `[6,514]`:
14 local kinematic/network features and 5 x 10 x 10 pooled receiver-private maps
(target-mass, empty-mass, unknown-mass, receiver AoI, own location). The centralized
state is a concatenation of these six observations and thus carries no separate
oracle target indicator. `true_targets` is used only by the simulator for sensing,
terminal flags and evaluation labels. Do not feed GCS ledger/target truth directly
to the policy. Route length/queue metrics are from the individual radio.

Pooling four source cells into one output cell loses high-frequency boundaries and
small targets; this tradeoff requires 10x10 vs 20x20 ablation. A fully global map
is not the same as a receiver-private, communication-delivered map. Private map
formation requires actual packet provenance and E2E ACK watermarks.

## DS and AoI contract

Paper Zhang 2024 Eq. 16 fuses positive/negative counts with `max` over participating
UAVs; our `zhang_max` configuration is an explicit baseline. The network extension
`sum` preserves source provenance and adds independently observed origin-tagged
counts, avoiding double counting when the same packet traverses multiple relays.

Xue 2026 Eq. 9 discounts each buffered observation by `exp(-eta*delay)`. Our
approximation discounts *a cumulative per-origin snapshot* once at receiver using
its frozen delivery age; it is neither mathematically identical to per-measurement
buffering nor a derivation from a Bayesian prior. `paper_fixed` masses `0.8/0.8`
are the controlled source baseline. `sensor_likelihood_ratio` is experimental,
with support `1-Pf/Pd` (positive) and `1-(1-Pd)/(1-Pf)` (negative); calibrate it
with Brier/reliability curves before claiming calibrated probability. In DS,
`(0,0,1)` represents vacuous uncertainty, **not** a Bayesian `P(target)=0` prior.

AoI maps use elapsed time from the generation time of the latest *received* origin
observation, not a relay timestamp. For never-delivered cells, elapsed mission
time is a **censoring convention** and not unbiased statistical AoI. Avoid calling
that metric unbiased unless the initialization convention is explicitly defined.

## Reward

`r_peer_information` is awarded for new destination-integrated evidence, not
submitted SEND actions, bounded to 12 new cells/step. Confirmed true and false
GCS targets are counted separately, as are sensed/discovered/local and delivered.
Reward is not yet tuned: compare GCS delivery %, discovery %, coverage %, AUC,
MAC delay/AoI, safety veto fraction and episode return on fixed seed sets.

## Motion/battery caveat

Rejected motion currently sets UAV velocity to 0 instantly, which can violate the
5m/s² acceleration limit. Zeng 2019 steady-level propulsion formula omits wind,
turning/climb and acceleration power; 180kJ is a scenario parameter, not a UAV
hardware validation. Do not train a physical-safety claim until this is fixed.
