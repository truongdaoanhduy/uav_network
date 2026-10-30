# XYZ motion and Bayesian camera sensing with 2D visualization

Status: IMPLEMENTED / CPU VERIFIED on 2026-10-30; approved by user before implementation.
Date: 2026-10-30 (Asia/Ho_Chi_Minh).
Base runtime: `3382feee81047c82d000e5c598ef685aec435297`.
Working branch: `feat/cpu-replay-20261022`.

## Intent and requested behavior

Duy requested the sensing/FOV and XYZ target-search behavior from
`/home/aduy/Documents/NCKH/uav_search_target`, with visualization remaining 2D.
A higher UAV covers a larger area but receives noisier observations. It cannot
confirm a target from high altitude. A lower UAV sees less area and can confirm
when its local posterior and a new positive observation meet the thresholds.
Each UAV has its own grid initialized to probability 0.5.

This is implementation scope, not an instruction to launch cloud training.
The design below records engineering choices approved by the user's “duyệt”
after review. Baseline descriptions below refer to the pre-implementation runtime.

## Current implementation and inspected sources

The baseline runtime inspected before implementation has XY positions/velocities and a fixed sensor height of
100 m. Native `environment/sensing.py` and Warp `environment/gpu/geometry.py`
perform exact geometric target detection. `env._sense()`/Warp report sensing
can immediately create a report. Native actor observations also call the
geometric visibility helper; changing report creation alone would still expose
unconfirmed target coordinates to the policy.

Actions, actor outputs, critics, random exploration and replay currently assume
three continuous slots `[ax, ay, power]`. `action_utils.py` also assumes the
active flag is observation index 9. The Warp state uses vec2d positions and a
single global height. Native radio synchronization similarly supplies a fixed
height. The current renderer draws a fixed-radius FOV and XY coordinates.

Reference files read directly:

- `uav_marl/world/sensing.py`, SHA256
  `1b3075477600c32277369373fc09b1364f659aa055c16a5ef20ed50f27b50d3c`.
- `configs/task/uav_search.yaml`, SHA256
  `546bdf9b6253b421fcaedeb9773ab94af60159a70712f2e21bd04cf64920f33a`.
- Confirmation handling in `uav_marl/world/reports.py` and
  `uav_marl/envs/uav_search.py`.

Reuse the mathematical model, not the reference's wildcard import chain,
notebook migration scaffolding, 3-km/50-target/30-obstacle scenario, 1-MB reports,
APF implementation or training hyperparameters.

## Approaches

1. Add a height-dependent camera to the fixed XY environment. Small change,
   but the policy cannot choose to descend and cannot satisfy the XYZ request.
2. Add a distinct XYZ/Bayesian configuration with matching Native/Warp
   implementations and keep the existing XY profiles loadable. This satisfies
   controllable altitude and allows a controlled comparison with the XY baseline.
3. Replace every legacy environment/checkpoint with XYZ. This removes a branch
   but invalidates historical checkpoints and comparisons.

Approach 2 was approved and implemented. The requested XYZ behavior rules out approach 1;
retaining existing research evidence rules out silently replacing approach 3.
The approval was received before product code was written.

## Scenario and configuration

Create explicit XYZ-sensing counterparts of the three current profiles:
independent Hybrid75, shared actor/critic Hybrid75, shared actor/critic FullGPU.
Retain their original XY YAMLs as controls. Each XYZ profile is a new lineage.

Keep 6 UAV, map 1000 m, 20 targets, 15 obstacles, horizon 1200 steps, report and
payload 1024 B, buffer 3072 B, TTL 300 s, battery 277200 J, learned RF power,
FP32, seed/bank protocol, sharing flags and replay placement from its parent.
Keep 20k episodes, E4096, B8192 and replay1M as requested configuration values;
these are not an assertion that the larger XYZ profile fits a T4.

Proposed reference-based defaults:

| Setting | Value |
| --- | --- |
| Initial altitude | 100 m |
| Allowed altitude | 0..150 m |
| Ground target/GCS altitude | 0 m |
| Obstacle height range | 30..120 m |
| Grid cell size | 25 m: 40 x 40 cells on this map |
| Belief prior | 0.5 independently for every UAV/cell |
| Full camera FOV | 90 degrees |
| Altitude anchors | 0, 50, 100, 150 m |
| Detection probability Pd | 0.99, 0.90, 0.80, 0.70 |
| False-positive probability Pf | 0.01, 0.10, 0.20, 0.30 |
| Fine confirmation height | <=50 m |
| Confirmation threshold | posterior >=0.99 plus a fresh positive observation |

New settings must be validated for shape, ordering, finite values and valid
probabilities. Above-ground motion retains the existing max-speed 20 m/s and
per-axis acceleration limit 2 m/s^2, extended to all three axes. No new landing,
charging or return-to-home task is introduced. At z=0 the camera has zero area
and does not sense; the altitude controller can command ascent again.

## FOV, measurements and Bayesian updates

The ground footprint is a nadir circle with radius
`z * tan(full_fov / 2)`. Interpolate Pd/Pf at the actual current altitude.
Use the reference's circle/rectangle intersection area for partial-cell
coverage, including map-edge cells. A center-only cell approximation is not
silently substituted for the inspected reference model.

Only active UAVs at z>0 take measurements. Measurement support depends on
FOV/grid geometry and camera LOS, not the hidden target location. Cell-center
camera rays and target rays use finite-height cylinders as in the reference.
The simulator may use ground truth to sample the observation likelihood;
actor features must never receive the hidden `has_target` flag or candidate IDs.

For fraction q of a cell covered, use the reference likelihoods:
`pf_eff = 1 - (1 - base_pf)^q` and
`pd_eff = q * base_pd + (1-q) * pf_eff`.
For an actual target inside the footprint with LOS, sample a positive with
base_pd; otherwise sample with pf_eff. Update the cell using pd_eff/pf_eff:

- Positive: `p_new = pd_eff*p / (pd_eff*p + pf_eff*(1-p))`.
- Negative: `p_new = (1-pd_eff)*p / ((1-pd_eff)*p + (1-pf_eff)*(1-p))`.

Use the reference epsilon bound, 1e-8, and a numerically stable implementation.
Unmeasured/occluded cells keep their prior. Coverage is a separate visited
indicator; seeing a cell does not set its target probability to 0.5 or 1.
Repeated high-altitude observations can change the posterior substantially,
but cannot bypass the <=50-m confirmation gate.

Sample exactly once on reset and once after each completed motion step.
Reading observations, rendering and info must not take extra measurements.
Use a dedicated reproducible sensor RNG keyed by episode/world, UAV, tick and
cell; avoid races from Warp threads updating one shared RNG. Masked resets must
reset/sense only selected worlds and leave every other world's beliefs and RNG
state unchanged. Native and Warp must use the same sampling contract.

## Confirmation and reports

Only active UAVs satisfying the altitude, posterior and fresh-positive gates
can confirm. True confirmations additionally require the actual target inside
the FOV with LOS. Multiple targets in one cell follow the reference: confirmation
can identify each visible target, while each target's discovery reward remains
one-time and report custody/TTL follows the existing ReportStore rules.

High-altitude positive observations produce candidate cells, not discovered
target IDs, exact target coordinates or reports. Regenerating an expired/lost
report also requires a fresh valid low-altitude confirmation.

False positives and false confirmations remain possible and are recorded as
simulator diagnostics. Like the reference, false confirmations create no real
target report. Do not reset the actor's posterior to 0.01 using hidden truth:
that reference post-processing would reveal a truth-based verification result
outside this measurement model. Keep updating that local cell from subsequent
sensor measurements instead. Do not add a new reward term in this change.

## XYZ motion, geometry and communication

For XYZ mode, positions/velocities are actual 3D state; targets/GCS are ground
points and obstacles have radius and height. Continuous actions become
`[ax, ay, az, power]`; discrete receiver/silence semantics remain unchanged.
The old XY action remains `[ax, ay, power]`. There is no reinterpretation of an
old RF slot as az. Generalize shared action utilities, actor heads, log-probability
and entropy accounting, target actions/noise, joint-action critics and replay
shapes from the configured dimension. Preserve the active-flag contract or
replace literal indices consistently across every caller.

Extend dynamic-window safety and exact swept collision checks to the XYZ flight
volume, finite-height cylinders, altitude limits and 3D peer separation. Keep
bounded acceleration/speed and funded-motion battery accounting. A safe climb
over an obstacle can be accepted; descending through its top cannot. The
propulsion formula already sums squared velocity/acceleration: include z rather
than charging only XY movement. No unrequested climb-specific energy model.

Use actual endpoint XYZ for radio distance, propagation delay, required power,
SINR/pathloss, beacons/range admission and packet positions. Apply the existing
regular-polygon RF footprint as a finite-height prism; retain the circular camera
occlusion convention. Native UavNetSim and Warp must agree on each convention.
Keep RF limits, retransmission, ACK, contention, custody and power feasibility.
Far/unreachable GCS links remain masked; relay is still learned, never forced.

## Local observations, critic input and memory budget

Keep CTDE/locality. XYZ actors receive self altitude/vertical velocity, local
peer altitude/vertical velocity, locally visible obstacle heights, only observed
candidate cells and confirmed/report-held target information. The old direct
geometric lookup for unconfirmed target slots must not run in Bayesian mode.
Use a local 5x5 posterior patch centered on the UAV ground cell as a proposed
compact input; fill out-of-map entries with 0.5. Candidate target slots come from
local positive measurements with cell-center coordinates, not exact target truth.
No global fused belief or GCS delivered vector is provided to actors.

Retain full 40x40 per-UAV belief and visited maps in simulator memory and exact
resume state, not in each replay observation. Append the compact local patches
to the centralized critic feature vector so Q sees local confirmation confidence.
This is an explicitly compact critic representation, not a claim that its input
contains the complete Bayesian simulator state. Critic-only global geometry and
delivery state stay privileged as before.

At E4096/N6, float32 full belief storage alone would be157,286,400 bytes (~150 MiB).
The implementation uses float64 beliefs for reference/parity,314,572,800 bytes
(~300MiB); learner precision remainsFP32. Actual allocations are recorded in
[resource evidence](../evidence/2026-10-30-xyz-resources.json).
Putting full maps in both replay observations would add 76.8 GB at replay1M;
therefore do not copy the reference's full maps into replay. Calculate all actual
replay/state/model/working allocations after final feature shapes. Keep existing
resource checks and report precise estimated bytes; never silently shrink replay,
E/B or enable AMP to make an XYZ FullGPU job start. CUDA feasibility requires a
separate measured run; CPU smoke does not establish it.

## Checkpoint, evaluation and compatibility

Assign a distinct XYZ/Bayesian observation/action schema and science fingerprint.
Save complete altitude/velocity/obstacle-height/belief/visited/sensor-RNG state.
Restore must validate all shapes and finite values before mutating live worlds.
Resume of an XYZ world must reproduce its next measurement and trajectory.
Old XY checkpoints remain evaluated with the XY model; missing new settings use
legacy defaults only. Reject cross-XY/XYZ exact resume and incompatible actor
warm-starts with a clear message rather than padding/reinterpreting tensors.

Keep validation-best selection, exact evaluated actor bytes, score order,
separate validation/final seed banks and final full resume behavior. New seed
matrices/manifests must say XYZ/Bayesian and cannot overwrite resolved XY cases.
Do not relabel historical 35/25 or 20/15 XY evaluations as XYZ results.

## 2D visualization

Keep top-down XY plots/GIF/MP4 and report-hop arrows. Project 3D paths onto XY;
label each UAV with current z and battery. Draw each UAV's current ground FOV
using that frame's altitude, including no camera disk at z=0 or when inactive.
Record altitude and sensing state in animation frames; do not draw every frame
using final-episode values.

Add a clearly labeled local belief heatmap for a selected UAV (default UAV0),
with probability scale 0..1 and 0.5 labeled unknown. Do not display a fused map
as if UAVs exchanged it. Show coverage, candidate evidence, confirmed target and
GCS-delivered report as distinct statuses. Ground-truth targets, if shown for
research diagnostics, are labeled ground truth and never counted as detected.
No 3D plot or interactive dashboard is required.

## Validation and implementation seams

Changes touch configuration, Native/Warp world/state/generation/motion/sensing/
observations, communication geometry/admission, actor/critic/replay dimensions,
checkpoint/continuation, profiles, rendering, metrics and docs. Reuse current
modules and algorithms; no new dependencies or alternative training stack.
Keep each owned Python/YAML/Markdown source <=500 lines. The repository also
pins SHA256 of MASAC/MATD3 scientific sources in `test_project_structure.py`;
shape plumbing changes require an explicit arithmetic-equivalence review before
updating those recorded hashes. Do not bypass that guard to make tests pass.

Required observable checks, using real paths rather than only mocked helpers:

1. Same XY at z50/z150 gives different footprint and Pd/Pf; larger FOV is not
   confirmation. High posterior at z>50 never discovers or queues a report.
2. Initial/unvisited/occluded belief stays 0.5; positives/negatives move it in the
   expected direction. False positive has a cell coordinate, no hidden target ID.
3. Fresh low-altitude positive plus sufficient posterior creates exactly the
   intended report; wrong altitude, negative observation or insufficient posterior
   cannot create/recover one. Known/delivered/reward counts are not duplicated.
4. XYZ actions ascend/descend, obey altitude/acceleration/speed limits, account
   for vertical energy and reject swept obstacle/peer collisions correctly.
5. Actual height changes 3D radio distance and finite-height LOS consistently in
   Native and Warp. Relay/GCS custody, TTL and learned power remain operational.
6. Both learners, BASE/PQL, independent/shared actors/critics and replay layouts
   accept the correct XYZ shapes and perform real finite CPU learner updates.
7. Sensor sampling has one-tick semantics, masked reset isolation, seed repeatability,
   Native/Warp CPU agreement and exact continuation including the next measurement.
8. XY legacy policies and snapshots remain usable; XY/XYZ mismatch fails before
   model/environment mutation. Best-checkpoint and validation-bank guards remain.
9. A real 2D render/short animation shows frame-correct z/FOV/belief and keeps report
   arrows/energy annotations; inspect the generated image, not just file existence.
10. Run relevant regression tests and then the full existing CPU suite. Record
    CUDA skips honestly; do not claim convergence or production performance.

## Design review

A fresh independent UAV/MARL builder reviewed this document read-only and
reported "nothing important left". This is a design-completeness check, not
runtime verification or user approval. Structure checks on the proposed doc:
2 passed and 479 subtests passed; product source remains unchanged.

## Approval and next action

No product code, dependencies or training jobs have been changed for this design.
After approval, use `superpowers:writing-plans`, implement in the existing linked
worktree with targeted tests, review once, fix findings, verify, and commit/push on
the working branch. The first delivered XYZ baseline stays at 20/15 and 1200.
