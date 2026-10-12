# UAV swarm implementation plan

> For agentic workers: use superpowers:executing-plans to implement this plan
> sequentially. Follow a failing-check → implementation → passing-check loop.

**Goal:** Six homogeneous UAVs search through FOV and deliver target reports
through real UavNetSim, controlled by Hybrid MASAC.

**Architecture:** One small Python package, grouped by environment,
communication, algorithms and experiments. Explicit config owns adjustable
parameters. Artifacts and downloaded dependencies stay outside tracked code.

**Tech stack:** Python >=3.11, NumPy, PyTorch, SimPy, PyYAML, matplotlib,
UavNetSim's analytical A2A dependencies. Stdlib unittest for checks.

**Spec:** `docs/design.md`.

## Global constraints

- Default six UAVs; movement XY only, fixed camera altitude.
- Destination indices exclude self and end with GCS; power 0.1–0.4 W.
- No belief grid or hidden target positions in local observations.
- Every tracked text/code file <=500 lines.
- Source grouped in folders; downloaded dependency and outputs ignored.
- Reproduce seed 44 within the same Python/software/device configuration.

## Review focus

1. Dense/impossible world configurations must fail with a clear error.
2. High velocity must not tunnel through obstacles or crossing UAV paths.
3. Unsuccessful packets must retain report custody and consume TX energy.
4. TTL and reset must not resurrect stale packets or share simulator globals.
5. Time limits must bootstrap and checkpoint evaluation must use saved config.

## Task 1: World, sensing and configuration

Files: `uav_network/configuration.py`, `environment/{world,geometry,
motion,sensing,observations,env}.py`, `communication/reports.py`,
`tests/test_environment.py`, `configs/default.yaml`.

Produces `load_config(path=None, overrides=()) -> Config`,
`Action(continuous, destination)`, `SwarmSearchEnv.reset(seed, options)`,
`step(action)`, `state()`, `ReportStore.add/expire/transfer`.

- [x] Write checks for dimensions, seed replay, FOV hiding, acceleration and
  continuous obstacle/peer collisions; invalid CLI fields and impossible worlds.
- [x] Run `python -m unittest discover -s tests -p test_environment.py -v`;
  observe missing-feature failures.
- [x] Implement validated config, world, sensing and network-disabled steps.
- [x] Rerun the same command; expect all checks to pass.
- [x] Commit this independently usable search-only stage.

## Task 2: Real packet communication

Files: `communication/{upstream,routing,backend}.py`,
`scripts/{requirements-network.txt,upstream-revision.txt}`, `setup.sh`,
`tests/test_communication.py`.

Consumes `ReportStore` and mission positions/obstacles. Produces
`UavNetSimNetwork.step(action, positions, time_s)`, `metrics()`, `close()`.

- [x] Write real direct/relay/fragment tests, failed-TX energy, TTL, duplicate
  suppression, single-live-instance and same-seed reset tests.
- [x] Run the communication checks and observe the missing backend failures.
- [x] Implement pinned sparse setup, upstream lifetime, selected routing,
  per-fragment power, reassembly and custody transfer.
- [x] Run communication and environment checks; expect no skipped tests.
- [x] Commit the real packet-network stage.

## Task 3: Hybrid MASAC and experiments

Files: `algorithms/{models,replay,masac}.py`,
`experiments/{cli,training,evaluation,rendering}.py`,
`tests/test_masac.py`, `configs/smoke.yaml`, `README.md`.

Consumes `env.state()`, local observations and hybrid actions. Produces
`MASAC.act/update/save/load`, bounded replay, CLI train/evaluate commands.

- [x] Write finite-update, actor-independence, categorical-gradient,
  timeout-bootstrap and checkpoint-roundtrip checks.
- [x] Run these checks; observe missing learner failures.
- [x] Implement independent actors/twin centralized critics and exact own
  categorical expectations; add streaming metrics, resolved config and PNG.
- [x] Run the entire unittest suite and a 2-episode real-network training run.
- [x] Evaluate its saved checkpoint; expect metrics and final PNG to exist.
- [x] Check shell syntax, repeat setup, all file lengths and Git exclusions.
- [x] Commit and perform an independent whole-project code review.
