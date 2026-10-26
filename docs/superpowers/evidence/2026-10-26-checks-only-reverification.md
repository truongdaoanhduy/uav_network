# Checks-only scenario/algorithm verification — complete

Updated 2026-10-26T07:34:16.289962+00:00. The user's latest instruction prohibited deletion on W&B/Hugging Face and requested checks only. No source/test/scientific config was changed. Remote operations in this phase were reads and diagnostic-output downloads; the only new cloud execution was a bounded private Kaggle diagnostic with tracking disabled, HFoffline and no credential datasets. Earlier cleanup next actions are cancelled.

## Source and verified runs

Current tested source is `b77a96a0b7bc7f5c8063098234f3dd8f33da9ea7`. It differs from the earlier CUDA-v3 source `4c5b8862b2c71e6dc7ead6b9b99e1402ca2fd137` only in docs/evidence. Fresh AST:100source/99test/8script modules;21YAMLprofiles validate. Primary config SHA256503cc6fb34133b7ffb0a43e3c9238b07a5e587958a49d68c45d3f491f711e952.

Fresh service/output reconciliation reconfirmed v3complete/pass on2TeslaT4, Torch2.11.0+cu128, CUDA12.8, Warp1.18:280scenario tests passed;8worlds x1800vectorsteps and4096worlds x16steps passed;four learner execution/train/save/load/evaluate probes passed; full CUDA suite834passed/21skipped/37warnings/2432subtests,0failures. This was a fresh retrieval of an earlier run, not a new full-suite execution.

New diagnostic `duytrngoanh/uav-full4096-checks-only-20261026`, version2, completed07:26:27UTC on2TeslaT4. Receipt/script hashes match the submitted version:

- Primary1KiB: **6/6 checks pass**,299.79s.4096-world serialization/restore/next-action and next-state checks run separately oncuda:0/cuda:1. All four MASAC/MATD3 BASE/PQL cases retain env4096,batch8192,replay1M,CPU75%,AMP/compile/fused off. Two fresh bounded warmup segments advance4096->8192globalsteps; all4096existing replay rows/RNG/seeds/length/progress are preserved; exact-environment/optimizer-replay restoration flags hold. Primary warmup checkpoints2.497–2.499GB.
- Existing legacy1MB workload: opt-in `CudaFull4096ResumeTests` **5passed,0skipped,280subtests**,681.58s. This is a separate500k/GPU-only stress profile under the corrected source, not the primary1KiB cohort.
- The original21CUDA skips are accounted for:5opt-in cases nowexecuted;15CPU-only cases were covered on the CPU host;1inverse-GPU-availability test intentionally does not apply on2GPU hardware. Last same-source CPU suite774passed/81skipped/8warnings/1285subtests is retained as prior evidence.

Driver version1 failed before world allocation because a reused historical helper forced replay_device=cuda while primary hybrid replay requires replay_device=cpu. Only the standalone driver was corrected; production validation/algorithms/configs remain unchanged. Both failure receipt and successful driver are retained.

## Finding: PQL GPU-only sparse checkpoint storage

Resume correctness passes, but PQL GPU-only serializes unused ring storage. In the legacy4096warmup cases only8192/500000rows are populated:

| Algorithm | BASE bytes | PQL bytes | Excess bytes |
|---|---:|---:|---:|
| MASAC |2,487,767,592|8,932,913,600|6,445,146,008|
| MATD3 |2,490,410,198|8,935,556,142|6,445,145,944|

Root cause: `uav_network/experiments/pql_replay.py:89` returns CUDA slices backed by full data/measurement storages. `torch.save` preserves the backing storage. BASE `fixed_checkpoints.py:13` compacts partial views. Unused rows predict6,445,143,840B, matching both excesses within2.2KB of other metadata. PyTorch2.11 official [serialization notes](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/serialization.rst) confirm view/storage semantics; consulted through Context7.

This is an I/O/storage inefficiency and possible checkpoint disk/time-budget pressure, not evidence of wrong restored actions or Bellman updates. The large excess was not observed in the four primary hybrid cases. Fully populated production checkpoint sizes are not estimated by these warmup measurements. **No fix applied under the checks-only instruction.** A later authorized fix needs a real-CUDA sparse-storage regression first.

## Scenario / Dec-POMDP assessment

Fresh actor-path spot checks retain per-agent observations (`observations[:,i]`) for MASAC/MATD3; global state stays with centralized critics. Existing local-link/report leakage regressions passed in the CUDA scenario stage. No new confirmed actor leakage was found. This is bounded evidence, not a proof that every possible implementation defect is absent.

The privileged feasibility oracle delivered35/35targets on three seeds44,2000000044,2000000045 in186,193,168steps. It is an engineering positive control, not a learned/decentralized baseline. Sensor detection remains exact deterministic footprint+occlusion. Propulsion remains an uncalibrated surrogate;277200J/140W gives1980s stationary endurance, and surviving1800s limits RMSspeed to5.29m/s before acceleration/radio. This is a design/calibration constraint, not an arithmetic defect. Battery/reward were not tuned. Reactive actors, ideal local beacons, one train seed and dependency ranges remain scientific/reproducibility limitations.

The37full-suite warnings break down into14Matplotlib/Pyparsing deprecations,7W&B-workspaces/Pydantic deprecations,14TorchJIT deprecations,1intentional crash-diagnostic async-close warning,1test-only tensor-to-scalar warning. No new unhandled crash/nonfinite/thread failure was observed in the passing runs.

## Remote preservation and limits

Fresh before/after reads match:W&B321runs (234finished,65failed,22crashed),noactive; both HF heads/LFSobject counts/bytes unchanged. Mainhead5fc83edb4ea962d640ae77647468871e82b6bec5;publichead2c68392e5ccc500a2fc13d4d8a2f4b444175bcb8. No further cleanup or repair was performed. Evidence remote-before/after JSON is adjacent.

No20ktraining cohort was launched; no historical policy was warm-started. Production-sized resume tests were warmup snapshots; active-learning behavior was covered by the earlier smaller CUDA tests/probes. PQL asynchrony does not promise bitwise-identical wall-clock schedules. External UavNetSim internals were not newly line-audited; owned wrapper/integration coverage remains the applicable scope.

Evidence prefix:`2026-10-26-checks-only-*` in this directory, including actual receipt,primary result,full opt-in log/XML,submitted driver/kernel,pipfreeze,source/local checks,coverage,energy and the storage finding. Current verification is complete; storage deletion remains prohibited. No further action is queued.
