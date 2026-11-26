# UAV V79 — Vast.ai CUDA integration and safety audit — 2026-11-26

**Scope:** independent experimental V79 CUDA prototype. Evidence confirms functioning smoke/stress tests, NOT scientific validity, safe-policy convergence, exact UavNetSim equivalence, or complete absence of defects. The main branch remains untouched; source lives at experiments/v79_vast_20261126.

## Scenario actually tested
Source YAML: experiments/v79_vast_20261126/research/zhang_xue/configs/scenario_v77_nfz_multi_1km.yaml.
Six UAV, 1km square map, 50m fixed altitude, 15 static targets, 20x20 target belief grid, 900-step default horizon, vmax 15m/s, amax 5m/s2, safe separation 50m. Sensor Pd=.85 Pf=.03, Dempster-Shafer fixed m=.8, confirm .9, DS AoI eta=.02. RF is deterministic multi_obstacle excess attenuation, nominal 12dB/building, TX 0.1W. Decentralized actor/central critic MATD3 and MASAC with tensor replay and team reward.

Environment: Vast 2x RTX 3060 Laptop 12GiB each, driver 580.159.03, PyTorch 2.14.1+cu130, Warp 1.18.0. Actual GPU runs use cuda:1 for both Warp and learner. GPU0 was occupied (~11.7GiB used) by processes invisible inside the container. Warp reports CUDA peer access not supported. No distributed 2-GPU throughput claim.

## Code fixes and direct evidence
1. CUDA initialization in scripts/gpu/run_v79_real_rollout.py: set_device before reset_peak_memory_stats. Before: both learners failed with Invalid device argument. A minimal PyTorch repro failed before set_device and succeeded after. After: E4096 smoke PASS.
2. Actor own-position map channel in research/zhang_xue/src/zhang_xue/gpu/kernels/actor_maps.py: tile-presence indicator must equal 1, not .25. Pre-fix parity test saw [410] actual .25 expected 1.0; post-fix CPU Warp parity test PASS.
3. New terminal reason bitmask in mission_v79_kernels.py: bit1 safety, bit2 all batteries depleted, bit4 all targets confirmed at GCS, bit8 time-limit truncation. Read before selective auto-reset. This is diagnostic only, not a modification to terminal rules.
4. New safety subtype bitmask: bit1 unsafe boundary/building and bit2 UAV-to-UAV pair risk. The pair and geometry flags can co-occur; numbers below count each flag, not necessarily disjoint unless measured.
5. Masked reset clears diagnostic flags only for reset worlds, preserving unaffected worlds. TDD tests were observed failing before implementation and passing afterward.
6. Rebuilt SHA256 manifests (277 source +119 research checksums), and retained original FAIL logs and source backups as evidence.

## Verified tests on exact staged source
- Full CPU-only pytest suite: 394 passed, 3 skipped, 98 subtests passed, with CUDA_VISIBLE_DEVICES empty to avoid unrelated occupied GPU0 when running CPU optimizer tests.
- CUDA and CPU terminal-reason and masked-reset behavioral tests: 6 passed with RUN_V79_CUDA_GATE=1.
- Project policy: 279 scanned files, 0 violations. Compileall PASS. SHA256 manifests PASS.
- MATD3 and MASAC E4096/B8192: 3 vector steps, 12,288 real transitions, 2 optimizer updates each, 4096 distinct target layouts; horizon=2 purposely stresses reset, NOT 900-step training.
- MATD3 E64/horizon900/300 steps: 299 optimizer updates; 63 safety terminations, geometry 8, UAV pair 55. All were safety, none all-targets-delivered, all-batteries-depleted or horizon truncation.
- MASAC E64/horizon900/300 steps: 299 optimizer updates; 117 safety terminations, geometry 17, UAV pair 100. Again no actual 900-step successful episode proven.
- Zero action static baseline Warp cuda:1 E64 x300: zero safety terminations, zero event overflows. Thus failures correlate with selected motion/policy/shield interaction rather than occurring spontaneously in the stationary-control baseline.
- Controlled NLOS PHY sweep through 75m tall building: 0/6/12/20 dB configuration yielded relative channel gains 0/-6/-12/-20 dB. Demonstrates deterministic attenuation wiring, NOT physical calibration.

## Remaining OPEN concerns, not PASS
A. Safety shield is fail-closed but may produce infeasible instantaneous zero velocity when postcheck fails; 50m safe distance and 5m/s2 constraints require proper viability/feasible-trajectory controller and adversarial multi-UAV tests. Do NOT hide failures by disabling safety termination.
B. Exact CPU–Warp mission parity for motion/energy/sensing/DS/map/PHY/MAC/routing/queue/AoI/reward remains unproven. Test seeds, same initial worlds, same actions, controlled RNG and packet-event traces before paper claims.
C. Full 900-step episode behavior, real training convergence, multi-seed evaluation and checkpoint/resume are not established.
D. The reward uses local UAV ledgers for information gain, not solely GCS-delivered evidence. This can be a valid CTDE team-reward choice but requires explicit statement and ablation, especially for a network-focused paper.
E. Static-target DS evidence decay eta=.02 requires eta=0 ablation and justification; fixed masses m=.8 are not altitude/Pd/Pf-calibrated evidence weights.
F. RF per-building 12dB is deterministic excess attenuation, not ITU material propagation validation. Need SINR/PDR/ACK/queue/routing under varied obstacles and shadowing.
G. Warp UavNetSim-inspired network has not demonstrated event-by-event exact parity with classic SimPy UavNetSim.
H. Only GPU1 was tested; GPU0 memory was occupied, CUDA peer access unavailable, so no 2-GPU speedup or correctness evidence.

## Evidence traceability
All raw stdout logs, per-step JSONL, failure repro, before-after patches, both terminal diagnostics and RF sweep are in the ZIP output on Vast and the corresponding package mirrored to the user's Ubuntu laptop. RUNS.csv in this directory lists verified run results, including early failed experiments.

Repeat from directory experiments/v79_vast_20261126:
    export PYTHONPATH="$PWD:$PWD/research/zhang_xue/src"
    CUDA_VISIBLE_DEVICES='' python -m pytest -q
    RUN_V79_CUDA_GATE=1 python -m pytest -q tests/test_v79_terminal_reason_runtime.py
    python scripts/gpu/run_v79_real_rollout.py --algorithm matd3 --envs 4096 --steps 3 --horizon 2 --batch-size 8192 --hidden 64 --world-device cuda:1 --learner-device cuda:1 --allow-experimental-physics --output outputs/e4096_smoke.json

Reference context (NOT certification of implementation): https://github.com/Zihao-Felix-Zhou/UavNetSim ; https://nvidia.github.io/warp/ ; https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2109-2-202308-I!!PDF-E.pdf
