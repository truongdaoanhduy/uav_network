# UAV swarm search and communication

## Requested scope

New Python project at `/home/aduy/Documents/NCKH/uav_network` for six
homogeneous UAVs, cooperative target search, circular obstacles and real
UavNetSim communication. Movement is strictly 2D. Keep the old destination
and transmit-power action semantics; remove Bayesian beliefs and belief maps.
Configuration or CLI overrides control experiment parameters. Every tracked
file stays below 500 lines. Group related files by folder. Fetch external
code during setup and exclude external code and run artifacts from Git.

## Decisions and configurable defaults

- World: 1000 m square, 6 UAVs, 10 stationary targets, 8 circular obstacles,
  1 s steps, 800 steps per episode. Obstacle radii: 30–70 m.
- Motion: acceleration actions `(ax, ay)` in `[-1,1]`, scaled by 2 m/s²;
  velocity norm at most 20 m/s. Segment tests prevent obstacle tunneling and
  continuous pairwise collisions. Rejected moves stop that UAV for the step.
- Camera: nadir camera at fixed altitude 100 m, full FOV 90 degrees;
  ground footprint radius `altitude * tan(FOV/2)`. No heading/altitude action.
  Exact detection inside this disk, with conservative 2D obstacle occlusion.
- Network: peer/GCS eligibility ranges 500 m; GCS at bottom-edge midpoint.
  Each UAV chooses index 0=silent, indices 1..5=other UAVs in ascending ID,
  index 6=GCS. The power action maps `[-1,1]` to 0.1–0.4 W.
- Reports: target ID, original source, original timestamp and hop count.
  Default size 1024 B, packet payload 1024 B, buffer 3072 B/UAV, TTL 300 s.
  Larger reports are fragmented. A receiver acquires report custody only
  after every fragment arrives. Sender retains its report on failed transfer.
  Already known target IDs suppress relay loops. Report discovery is exact,
  not a repeated probabilistic confirmation process.
- Reward: team discovery +10, GCS delivery +50, rejected motion −0.2,
  expired/dropped reports −10, per-step −0.01, communication energy −0.01 J.
  Success means every target report reached GCS. Discovery alone is separate.
  Coverage maps, battery exhaustion and APF are outside this first version.

## Modules and interfaces

`configuration.py` owns validated dataclasses and YAML/CLI loading.
`environment/` owns scenario generation, geometry, motion, FOV, observation,
and the `SwarmSearchEnv.reset/step` lifecycle. `communication/` owns report
custody, upstream setup, forced next-hop routing and fragment bookkeeping.
`algorithms/` owns independent hybrid actors, centralized twin critics,
replay and MASAC updates. `experiments/` owns CLI, training, evaluation,
metrics and PNG rendering. Nothing imports the old project at runtime.

`reset(seed, options)` returns local observations `[N, O]` and mission info.
`step(Action(continuous[N,3], destination[N]))` returns observations, scalar
team reward, `terminated`, `truncated`, info. Time limits only truncate:
the replay stores true termination separately and bootstraps at time limits.

Local actors see own position/velocity, GCS relative position, time, their
own known-target flags and report ages, visible target slots, visible obstacle
slots and nearby peer beacons. Hidden target coordinates never appear in
actor observations. Peer beacons are an explicit idealized control-plane
assumption; mission reports use the packet simulator. Training critics see
the true mission geometry, all UAV knowledge/report summaries and delivery
flags. These summaries abstract the internal SimPy event queue.

## Communication boundary

Pin UavNetSim commit `04daafb815eb377409b40b285574eeb62b9a8d58` (the stable
revision already used by the old project). Run its analytical A2A channel,
CSMA/CA, packet queues, ACKs, retry limits, interference and PHY statistics;
no Sionna GPU/ray-tracing dependency is needed for analytical A2A.
Disable upstream random traffic and mobility. Mission XY coordinates drive
network positions. Represent GCS as node N at ground height. Circular mission
obstacles become polygonal buildings in upstream airspace.
Routing delivers to the selected next hop only; no automatic greedy route
overrides the policy. Power is stored on each enqueued fragment and used by
the real PHY, including energy for unsuccessful attempts and ACKs.
Packets enqueued on previous steps continue asynchronously even after silent
is selected. New fragments only enqueue for the destination selected now.
Report age includes time spent at relays; stale packets cannot resurrect a
report after TTL. Upstream uses module-level configuration: use one network
instance per process, and reject simultaneous instances rather than silently
sharing globals. Network-disabled mode is explicit for isolated sensing tests.

## Hybrid MASAC

Six separate actors use local observations. Each has a tanh Gaussian branch
for three continuous actions and a categorical branch for destination.
Each UAV has its own pair of centralized Q networks and target Q networks
using true mission state and joint hybrid actions. Actors have separate Adam
optimizers; critics and learned entropy temperatures use Adam as well.
Reparameterize continuous actions. Enumerate the current UAV's seven discrete
choices exactly; sample other UAVs' actions. Actor i differentiates only
through its own actions using Q_i. Target Q_i includes that actor's two
entropy terms. This is a documented hybrid-action adaptation of SAC/CTDE,
not a claim that upstream SAC itself defines the network destination action.
Use ReLU MLPs (256,256), gamma .99, tau .005, lr .0003, batch 256, replay
50,000, warmup 2,000 and one update per environment step by default.

## Incremental acceptance

1. Configuration, seeded world, continuous collision tests and FOV hiding.
2. Real direct/relay delivery, fragmentation, TTL, failed-attempt energy and
   reset determinism. No mocked channel substitutes for network tests.
3. MASAC finite gradients/updates, independent actors, categorical learning,
   timeout bootstrapping, checkpoint evaluation and a short real-network run.
4. CLI config overrides, setup repeatability, PNG preview, clean Git scope
   and a tracked-file line-limit check.

## References

- https://github.com/Zihao-Felix-Zhou/UavNetSim — packet-level simulator.
- https://github.com/Bussler/DroneWatch — cooperative 2D search, local sensing,
  circular obstacles and task metrics; no Rust/RLlib code copied.
- https://github.com/yangmin32/GPMF — partial observability motivation;
  its battle/gather/pursuit environments and GAT algorithm are not MASAC.
- https://arxiv.org/abs/1812.05905 — SAC objectives and entropy temperature.
- https://arxiv.org/abs/1810.02912 — multi-agent SAC with centralized critics;
  this baseline uses ordinary MLP critics, without its attention mechanism.
