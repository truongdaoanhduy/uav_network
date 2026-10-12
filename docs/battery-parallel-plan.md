# Battery, MATD3 and parallel collection

User extension to the working v0.1 baseline (30 passing tests): retain six
homogeneous UAVs, XY movement, local FOV, real UavNetSim and selected next hops.
Do not import the old project, add a belief map, or put generated runs in Git.
All owned files must remain at most 500 lines.

## Stage 1: energy and eligible actions

Add a configurable battery in joules, idle/hover/speed/acceleration propulsion
costs, remaining charge in observations/state and inactive UAV handling.
Charge propulsion for realized safe motion. A UAV unable to fund a complete
motion step stops, consumes its remaining charge and becomes inactive; repeat
pair safety checks against those stopped UAVs. This is a discrete-step energy
model, not an aerodynamics/landing simulation. No recharge or return-to-home.

Enforce the remaining battery at the native PHY entry point. A complete frame
must be affordable before transmission; retries and ACKs spend energy too.
GCS is mains powered. Empty batteries cannot move, sense, transmit or receive.
All-empty episodes terminate as failures; time limits continue to truncate.

Keep upstream LOS/free-space and obstacle NLOS/urban propagation, optionally
add a configured extra NLOS loss in dB. Test UAV-to-UAV attenuation as well as
UAV-to-GCS. Do not reject every obstructed link: power and SINR determine success.
A destination mask exposes silent, reachable living peers and reachable GCS
when a report is buffered. Policy and random warmup respect the mask.

## Stage 2: learners

Keep independent local actors and per-agent centralized twin critics for both
MASAC and MATD3. Use batched actor inference across environments. MASAC retains
Gaussian movement/power and categorical destinations, with conditional power
entropy when transmitting. MATD3 uses clipped double Q, continuous target noise,
delayed policy/target updates and a straight-through categorical branch for
selected next hops. Exploration uses noise and valid-destination epsilon greedy.
Checkpoints identify their algorithm and observation version; old policies
require retraining after battery/mask observation changes.

## Stage 3: processes and CUDA

Use Gymnasium AsyncVectorEnv with spawn, shared-memory observation/state and
disabled autoreset for multiple environments. Each process owns exactly one
UavNetSim; one parent learner owns CUDA. Store final transitions before masked
reset. Each worker has a distinct deterministic episode seed. One environment
uses SyncVectorEnv. Optional step_async overlap runs learner updates while
workers simulate packets. Joint transition count and replay sample/update ratio
are explicit, rather than silently changing the update ratio with env count.

Add tensor replay on CPU or CUDA, hardware inspection, throughput benchmarks,
CPU smoke profiles and adjustable workstation profiles. SimPy remains on CPU;
do not claim that selecting CUDA makes packet simulation run on GPU. Compilation
is optional and must be measured after warmup. No unmeasured GPU speed claims.

## Verification and delivery

1. Battery depletion, safe stopping, dead sensing/receive/TX and exact RF debit.
2. Direct delivery, selected relay, report custody, obstacle channel attenuation.
3. Both real learner updates, masking, delayed MATD3 actor updates, save/load.
4. Multiple real network workers, terminal transition integrity and partial reset.
5. End-to-end train/evaluate for both algorithms; CPU process throughput report.
6. Independent review, clean tests, line limits, docs, commits and local/GitHub sync.

GPU library research lives in `gpu-acceleration-research.md`. The supplied paper
uses a heterogeneous GATAC scenario and only reports workstation hardware; that
does not establish its number of parallel environments or GPU utilization.
