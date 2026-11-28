# v6.0 independent seven-radio repository cleanup

Parent: 82512e827b5a4c9f5c6fda1283a628967d0bbf3c (v5.9).
Branch: experiment/zhang-xue-v60-newscenario-only-matd3-masac-20261125.

## Retired from this branch

- 50 top-level scenario configuration files and their old run-specific settings.
- Historical environments, communication backends, cloud scripts, training runners,
  asynchronous learner modules, evaluation dashboards tied to prior scenarios,
  historical papers/evidence data, archived notebooks and vendor source snapshots.
- The unrelated old experiment package and configuration objects are not imported
  by the active actor or scenario code.
- No historical Git history was rewritten. Prior experiments remain in parent commits.

## Retained

- One current scenario, research/zhang_xue, with 6 UAV + GCS radio simulation,
  current target map, occlusion, belief, reward, mobility and optional Warp kernels.
- Canonical YAML: research/zhang_xue/configs/scenario_7radio.yaml.
- Smoke YAML: research/zhang_xue/configs/cpu_smoke.yaml.
- MATD3/MASAC and Vanilla algorithm source mathematics and supporting learner code.
- Generic offline evaluation statistics, convergence, telemetry and plotting helpers.
- New learner-only config/checkpoint v4; it is not an old-world checkpoint loader.
- Local setup.sh installs the two editable Python packages. Deterministic scenario
  generation requires no dataset download.

## Executed gates

- 85 pytest tests PASS, 48 additional unittest subtests PASS.
- Full tracked-source governance PASS, changed-file governance PASS.
- All owned Python source <= 500 lines; old root scenario configs are absent.
- Editable packaging of learner and research scenario installed and imported.
- Warp CPU two-world 2-hop proof: UAV0 next-hop UAV1, GCS data [1,0],
  E2E ACK [1,0], MAC ACK [4,0], overflow [0,0], reset isolation PASS.
- No T4/CUDA driver available in the local sandbox: CUDA performance, 4096
  parallel worlds and coupled training were not verified.

## Important unresolved integration boundary

The original learner expects a 3-channel continuous action (XY + RF power)
and a destination mask in the observation tail. The active environment exposes
a 2D velocity command with fixed TX power and a separate destination mask.
The new learner adapter, full training, evaluation/test and live online tracking
are not yet integrated. The retained generic metric helpers do not imply those
pipelines already exist.
