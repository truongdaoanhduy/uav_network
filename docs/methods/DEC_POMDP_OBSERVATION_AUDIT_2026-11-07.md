# Actor observation audit: 20 to 35 targets

Observed source: f7c0133 (runtime source exported in89f3301).
The user's question concerns whether larger actor inputs expose hidden state.

## Why actor dimension increases

The current independent actor receives observation_i only.
For this configured scenario, observation_dim = 158 + 2*num_targets.

| Feature group | 20 targets | 35 targets |
|---|---:|---:|
| UAV i's locally acquired target/report knowledge |20|35|
| UAV i's custody/recovery/remaining-report-TTL status |20|35|
| All other feature groups |158|158|
| Total |198|228|

Increasing15 to25 obstacles alone leaves actor dimension unchanged.
Obstacle features stay3 slots x5 values=15, including visible heights.
They are the nearest3 obstacles satisfying the current sensor-footprint
visibility rule, not all25 global obstacle coordinates.
The per-target vectors contain local knowledge/report status, not target
coordinates or a team-wide discovery vector. Unseen target IDs have no
acquired knowledge; the scenario target count is public mission configuration.

Actor inputs also include own kinodynamics/battery, local noisy belief
candidates, local belief memory/coarse summaries, in-range neighbor/radio
metadata, and valid-destination masks. The coarse map summarizes UAV i's
own belief grid; it is not the ground-truth map or all agents' beliefs.
The radio/admission channels use the already declared ideal local handshake
assumption, with no separately charged beacon airtime. These assumptions
need to be stated in the observation model.

Relevant source:
- environment/observations.py: observation_dim/report_status/destination_mask.
- environment/xyz_observations.py: xyz_features.
- environment/gpu/observations.py: observe and centralized are separate kernels.
- algorithms/masac.py and matd3.py: actors consume per-agent observation rows.
- algorithms/models.py: critics concatenate centralized state and joint action.

## CTDE / Dec-POMDP interpretation

Dimensionality itself does not decide partial observability. The question is
whether observation_i can be obtained from agent i's declared sensors,
local history and permitted communications at execution time.
A fixed target count and a local per-target memory vector do not by themselves
expose hidden locations. Widening hidden layers also does not change the
MASAC or MATD3 update equations. These remain the declared hybrid-action UAV
adaptations, not exact reproductions of every detail of an original paper.

The actor/critic information split is consistent with CTDE for the audited
inputs. This is a source/dataflow assessment plus finite probes, not a formal
proof for every possible rollout or an audit of all simulator assumptions.
Primary CTDE source: Lowe et al., NeurIPS2017,
https://proceedings.neurips.cc/paper_files/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
(the actor uses local information; extra training information enters critic).

## Separate safety-layer assumption

The current CBF consumes global geometry and all-agent positions/velocities
outside the actor. Native xyz_motion.py passes full geometry into
filter_xyz_acceleration; Warp gpu/xyz_motion.py loops over all obstacles
and peers. The local actor's228 inputs are not this CBF input.

Therefore, do not describe the ENTIRE deployed safety pipeline as using
only decentralized local sensing. A truthful current description is CTDE
policies with a safety filter using privileged global geometry.
As a simulated transition mechanism, such a filter does not automatically
make the formal environment cease to be a Dec-POMDP. It is nevertheless an
additional deployment assumption: fully onboard/local safety would require
a separately scoped local-information filter and suitable guarantees.
No CBF or learning behavior was changed in this audit.

## Verification

12 counterfactual/control cases passed over Native and Warp CPU:
- Changing hidden target coordinates: actor inputs/actions unchanged.
- Changing an out-of-footprint obstacle, away from local radio paths:
  actor inputs/actions unchanged.
- Changing another UAV's private known vector: observer actor unchanged.
- Changing a GCS receipt without informing the observer: actor unchanged.
- Positive controls: changing own acquired knowledge or a visible obstacle
  DOES change the actor observation.
The centralized critic state changed in all cases.
Deterministic raw MASAC/MATD3 observer actions matched for hidden changes.

Dimension checks:
20t15o -> actor198,state565,critic631.
20t25o -> actor198,state605,critic671.
35t15o -> actor228,state790,critic856.
35t25o -> actor228,state830,critic896.

Related regression tests:59 passed,3 CUDA-hardware cases skipped.
No GPU training or Kaggle job was needed/submitted.
Evidence: docs/superpowers/evidence/2026-11-07-decpomdp-observation-audit.json.
Reproduce the diagnostic from the repo root:
```bash
PYTHONPATH=. python docs/superpowers/evidence/2026-11-07-decpomdp-observation-probe.py
```
