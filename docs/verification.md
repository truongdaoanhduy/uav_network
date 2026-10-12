# Verification — 2026-10-12

Repository: `/home/aduy/Documents/NCKH/uav_network`.
Execution: user's Python 3.13 environment, PyTorch 2.7.1, CPU.
Pinned UavNetSim HEAD: `04daafb815eb377409b40b285574eeb62b9a8d58`.

## Completed checks

- `bash setup.sh --system-site-packages` completed, then completed again.
  Sparse runtime download is approximately 1.3 MB including Git metadata.
- `.venv/bin/python -m unittest discover -s tests -v`: **30 passed**, no skips.
  World/FOV/continuous collisions: 10 checks. Real packet networking: 12.
  Hybrid MASAC/replay/checkpoint/train-evaluate: 7. File size contract: 1.
- Real-network train with `configs/smoke.yaml`: 6 UAVs, 3 targets, 2 obstacles,
  2 episodes × 40 steps = 80 environment steps, **73 learner updates**.
- CPU evaluation from `outputs/smoke-verified/final.pt` completed;
  `outputs/smoke-verified/evaluation/metrics.json` and
  `final_episode.png` (1200×1200) exist.
- Same seed 44: initial and repeated smoke episode logs match exactly;
  all 206 checkpoint tensors match exactly. This is within this software/device
  environment; it does not assert identical floating-point results across devices.
- Default scenario (6 UAVs, 10 targets, 8 obstacles): a 120-step random-policy
  rollout found and delivered 3 targets. This validates live search/report/relay
  wiring; it is not the learned policy's performance.
- A real channel with an intervening obstacle changed LOS to NLOS and failed
  PHY delivery while retaining sender custody and consuming TX energy.
- CLI config overrides and help worked; `bash -n setup.sh` passed.
- All owned files stay <=500 lines; largest production file is 220 lines.
  `third_party/`, `.venv/` and `outputs/` are ignored by Git.

## Independent review and repairs

An independent reviewer inspected all production modules, design and plan.
No Critical issue was found. Two Important network issues were reproduced
against real UavNetSim and covered by failing tests before the fixes:

1. A fully reassembled report could stall after the receiver buffer filled.
   Rejected reassembly is now discarded; sender retains custody and can retry.
2. An oldest duplicate report could block a later report that the selected
   peer needed. Scheduling now chooses the first eligible report for that
   policy-selected destination.

Both regression checks and the complete suite pass after the repairs.

One Minor review item remains: a manually supplied source-directory
`network.upstream_path` is trusted, rather than independently checking its Git
HEAD. The default setup checkout was verified at the pinned SHA; installed
distribution fallback also validates that SHA. This limitation is in README.

## Interpretation

The short trained policy had 0% search/delivery on its short evaluation.
The smoke demonstrates finite learning updates and a working saved-policy
pipeline; it does not establish convergence. Long training can be configured
independently after choosing the desired scenario/reward parameters.
