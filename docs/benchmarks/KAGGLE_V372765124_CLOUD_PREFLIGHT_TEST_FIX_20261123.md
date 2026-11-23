# Kaggle 372765124: runtime source stripped mandatory cloud unit preflight tests

Date: 2026-11-23. Exact user-supplied failing URL:
`https://www.kaggle.com/code/longlonthiu/notebookbbd0aec150?scriptVersionId=372765124`

## Evidence — inspected original reference AND user's actual failure

1. Downloaded the requested original with exactly `kaggle kernels pull therwed/notebook5051433b29 -p <local_dir> -m` using the project's connected local Kaggle account. Original reference has 3 notebook cells, embedded complete source including `tests/`, notebook file 778,324 bytes.
2. Downloaded the latest log and output of `longlonthiu/notebookbbd0aec150` using `kaggle kernels output longlonthiu/notebookbbd0aec150`; source notebook pulled separately. The failing log corresponds to the already-fixed 372762819 `CONFIG` NameError successor: source contains `SPEC["config"]`, and now exits in `uav_network.cloud.remote.run_job` at approx 46.37 sec with `subprocess.CalledProcessError` from `python -m unittest discover -s tests -p test_cuda.py -v`, **exit code 5**. Attached Kaggle `cuda-tests.log` says **Ran 0 tests / NO TESTS RAN**. Pip dependency-resolver conflicts are warnings, NOT the fatal issue in this run.
3. Root cause: the previously introduced `minimal_runtime_source_archive()` excluded **all** `tests/` to fit Kaggle's 1MB source cap. However `uav_network.cloud.remote.run_job` unconditionally runs three unittest preflight commands (`test_cuda.py`, `test_cuda_scenario_v2.py`, `test_training_audit_regressions.py`) before CUDA execution check and training. The absent tests were not discoverable, generating exit code5. This is a packaging regression from the previous assistant build, NOT an algorithm, Bayesian, GPU or Kaggle error.

## Repair — source-bundle contracts aligned with original reference

`minimal_runtime_source_archive()` now includes each mandatory cloud preflight file **plus all its transitively imported test modules**. Exact reviewed closure is 10 source test files:
`test_cuda.py`, `test_cuda_scenario_v2.py`, `test_training_audit_regressions.py`,
`test_benchmark_devices.py`, `test_performance.py`, `test_vector.py`,
`test_vectorized_critics.py`, `test_xyz_radio.py`, `test_xyz_sensing.py`,
`test_gpu_packets.py`.

Static AST imports determine this dependency closure automatically during build, so local preflight imports cannot silently disappear. All runtime UAV, UavNetSim, cloud/eval/forensics and single selected MAP800 posterior-only config remain unchanged and embedded/sha256-verified. Other unrelated 222 test modules and old experiment configs are excluded to meet the strict 1MB Kaggle source limit. This follows the reference notebook's actual preflight workflow while minimizing unneeded archive content.

## Tests — RED before fix, GREEN after

- Reproduced exact failure from a notebook-extracted source archive: all three `unittest discover -s tests -p <name>` returned **exit status 5** and reported `Ran 0 tests`.
- Changed source archive packaging (no changes to MDP or MATD3), verified all ten transitive test files are present in embedded archive.
- Executed **all three real unittest preflight commands** against the exact notebook-extracted runtime source: all returned exit0 on local CPU; GPU-specific tests skipped because this Ubuntu machine lacks usable CUDA.
- Executed actual `uav_network.cloud.remote.run_job` on the extracted notebook source with real subprocess unittest execution; verified it ran three preflight suites successfully and reached the next CUDA execution-check boundary (CUDA and cloud downstream mocked). This closes the prior test gap where only bootstrap was mocked.
- Ran **70 selected Pytest tests passed**, including full 3-cell bootstrap emulation, source hash/under1MB validation, real compiled CPU MATD3 Warp training → Actor checkpoint → held-out CPU Eval, and deep W&B/HF fake-transport integration.
- Output notebook byte size **763,233 bytes**, well below one decimal megabyte. Source is self-contained; no runtime GitHub source download. The SHA256 check remains mandatory and passwords are not embedded.

## Unchanged science

MATD3 ActorBN PlainCritic, hidden256x256 orthogonal, compile Inductor, E4096/U4/B8192, 20000 training episodes target, 800m map, 40x40 belief, altitude0-150m, camera YES + Bayesian posterior>=0.99 with **NO LR>=4** gate, Actor-learned TX0.001-0.8W, NoOracle cell-ID reports, same shared battery model, snapshots every100 vector steps, validation100 per policy across2 T4s and final Test1000, W&B/HF forensic archives.

## Scope/limits

This **fixes the observed `Ran 0 tests` / exit-code5 failure** in Kaggle version372765124. It is not evidence that these CUDA test suites will pass on T4 or that downstream `execution_check`/full E4096 train & E8192 Eval fit GPU memory. Remote T4 execution of the fixed file was **not** performed in this turn. Future CUDA errors require the next exact Kaggle log to diagnose; do not assert full hosted success now.

## Output

New uniquely named notebook to avoid previous uncorrected imports:
`notebooks/MATD3-POSTERIORONLY-MAP800-TX08-T4X2-REF5051433B29-FIX372765124-20261123.ipynb`.

On user's Ubuntu Downloads:
`/home/aduy/Downloads/MATD3-POSTERIORONLY-MAP800-TX08-T4X2-REF5051433B29-FIX372765124-20261123.ipynb`.
