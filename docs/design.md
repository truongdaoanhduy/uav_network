# Battery-aware 2D search and selected-hop communication

## Scope and configuration

Six homogeneous UAVs search stationary targets through camera FOV, then select
a peer or GCS as the report next hop. Use real pinned UavNetSim, finite battery,
MASAC/MATD3, independent CPU scenarios and one optionally CUDA learner.
Keep XY motion, no beliefs, config/args, grouped folders and <=500 lines/file.
Fetch upstream during setup; dependencies, generated data and runs stay out of Git.
Nothing imports the old project. See `battery-parallel-plan.md` for the extension.

Default world: 1000 m square, N=6, T=10, K=8 circular obstacles, dt=1 s,
800-step time limit. Per-axis acceleration bound 2 m/s², speed norm <=20 m/s.
Segment/circle and continuous pair-relative-path checks reject unsafe motion.
Camera altitude100 m/full FOV90° gives a 100 m ground-footprint radius;
exact detection with conservative circular occlusion. GCS is bottom midpoint.

## Battery and motion lifecycle

`environment/energy.py` computes
`E_flight = dt * (P_idle + P_hover + c_v ||v_real||² + c_a ||a_real||²)`.
Here `a_real=(v_real-v_previous)/dt`, including rejected motion and boundary clips.
Coefficients have units W, W/(m/s)² and W/(m/s²)² respectively.
Defaults: capacity277200 J (77 Wh), idle20 W, hover120 W, c_v=.5, c_a=2.
These are explicit baseline simulation parameters, not a paper's rotor model.

A UAV unable to fund a full step stops and consumes its remaining charge.
Re-run collision resolution with the stopped UAV before accepting peer motion.
This conservative time discretization does not simulate partial-step landing.
No recharge/return-to-home. Disabled energy accounting keeps battery fraction1
while still recording modeled consumption. Dynamic receive/amplifier losses
are not modeled separately; idle power represents baseline electronics.

Reset initializes charge and independent energy meters. A step validates actions,
resolves funded motion, debits propulsion, senses with living UAVs, advances
packet time dt, expires reports, updates reward and returns observation/state.
All UAVs empty: true failure termination. All reports at GCS: success termination.
Time limit: truncation; replay retains bootstrap eligibility.

## Action, observations and state

Actions contain continuous `[N,3]` normalized XY acceleration/power and integer
destination `[N]`. Destination0=silent, 1..N−1=ascending peers excluding self,
N=GCS. Power maps to .1–.4 W. Silent power and inactive continuous controls are
canonical zero in policy/replay. Earlier MAC queues retain their selected power.

Own observation features: pos2, velocity2, relative GCS2, time, buffer fraction,
battery fraction, active. Then known flags T, buffered ages T, FOV target slots
(relpos2/known/valid), obstacle slots (relpos2/radius/valid), peer slots
(relpos2/velocity2/battery/active), destination mask N+1.
Hidden target coordinates never enter actor observations. Inactive sensory slots
are zero. Peer position/charge beacons are an ideal control-plane assumption.
The mask uses own buffer, range and liveness, not hidden target geometry;
obstructed links remain candidates. Custody/buffer admission is enforced by network.

For target slots Vt and obstacle slots Vo:
`O=10+2T+4(Vt+Vo)+6(N−1)+(N+1)`; default95.
Privileged critic state includes geometry, knowledge, ages, delivery, inflight,
charge and active summaries: `S=7N+3T+3K+2NT+1`; default217.
State abstracts the internal SimPy event queue rather than serializing it.

## Packet and propagation boundary

UavNetSim revision `04daafb815eb377409b40b285574eeb62b9a8d58` supplies analytical
A2A channel, CSMA/CA, queues, backoff, ACK, interference, SINR and retry statistics.
Random native motion/traffic are disabled. Lift XY to camera altitude, GCS to z0;
obstacles become polygonal buildings taller than flight altitude. LOS uses
free_space, NLOS uses urban; `obstacle_extra_loss_db` scales NLOS gain only.
Default extra loss0 avoids double-counting the native urban attenuation.

Reports preserve target/source/original timestamp/hops. Defaults report1024 B,
payload1024 B, buffer3072 B and TTL300 s. Larger reports fragment; receiver only
acquires custody after every fragment. Source retains custody on failure.
Admission is rechecked at arrival. Duplicate knowledge suppresses relay loops;
rejecting full reassembly discards its session so retry can make progress.
Policy selects the next hop, never an automatic greedy network path.

`communication/energy.py` guards native PHY before channel transmission.
Each affordable attempt debits UAV charge once using all frame bits; retries,
failed attempts and ACKs count. GCS is unbudgeted and its RF energy is separate.
Unaffordable frames do not enter the channel. Sleeping receivers cannot acquire
custody or send ACK. A last affordable emitted frame may still arrive after
its transmitter exhausts the battery; that radiation was already funded.

## Hybrid learners

Independent local actors and per-agent centralized twin critics (not parameter
sharing). ReLU MLP [256,256], Adam, FP32, gamma.99/tau.005, clipping norm10.
Team reward: discovery10, delivery50, blocked move−.2, expired/dropped−10,
step−.01, UAV communication−.01 per J, propulsion−.01 per kJ.

MASAC samples tanh Gaussian continuous controls and masked categorical destinations.
Own discrete Q expectation enumerates all choices; other agent actions are
sampled/detached. Silent choices use power0. Conditional continuous entropy is
movement entropy + transmission probability times power entropy. Inactive
motion contributes zero. Temperature targets scale by active dimensions and
log valid destination count; freeze unavailable controls using masks, not zero
configured entropy targets.

MATD3 uses min twin target Q, clipped Gaussian target noise on continuous only,
actor/target updates every2 critic updates, and a hard categorical forward with
straight-through softmax gradient. Explore with Gaussian continuous noise and
valid-destination epsilon greedy (1→.05 over100000 joint transitions). This is
a hybrid adaptation of MATD3, not an exact Gumbel-Softmax reproduction.

Version2 checkpoints tag algorithm/config/dimensions and save model parameters
atomically after finite checks. Evaluation runs CPU. Version1 policies require
retraining because the observation schema changed. Replay/RNG/optimizers are
not saved, so these policy checkpoints do not promise exact training resume.

## Parallelism and device ownership

One live UavNetSim per process because upstream uses module globals. E=1 uses
Gymnasium SyncVectorEnv; E>1 uses AsyncVectorEnv with spawn and shared-memory
Dict/Box local observation and critic state. Workers own only CPU simulators.
Six actor forwards each process E observations in the parent; one batched
CPU readback assembles all actions. Parent owns CUDA model/optimizer/replay.

Autoreset is DISABLED: store terminal local observation and next state, then
reset only done slots. (The actual API spelling is `AutoresetMode.DISABLED`.)
Assign unique, advancing episode seeds in worker-index order. Fixed worker
count/schedule is required when comparing deterministic runs.
Overlap `step_async` with updates from previous replay, then `step_wait` and add
the collection batch. Drain the final ready update group before logging/saving.
E joint transitions per vector step; updates_per_step is per vector collection.
Report actual sample/update ratio and actual episode count (possible E−1 overshoot).

Tensor ring replay supports CPU/CUDA. Uniform sampling is with replacement;
CPU→CUDA sampled batches may use pinned memory. CPU-bound copies are blocking.
Periodic loss/gradient/scalar checks reduce synchronization; final save checks
all floating parameters. Optional compile covers actor encoders/critics only.
No automatic GPU conversion of SimPy. Benchmark startup, collection and warmed
learner separately with CUDA synchronization. See `gpu-acceleration-research.md`.

## Evidence

Historical v0.1 plan/evidence: `implementation-plan.md`, `verification.md`.
Current verification and hardware limits: `verification-v02.md`.
CUDA tests run only on CUDA hosts and explicitly skip elsewhere.
