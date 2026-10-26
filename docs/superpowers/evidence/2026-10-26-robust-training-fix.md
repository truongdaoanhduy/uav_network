# Robust generation and bounded-memory MASAC — verified and eight corrected runs learning

User request: fix random scenario generation across seeds, the diagnosed MASAC memory failures, and recognizable running names. Production code pinned at77a1d0f52ed4e78bc2823c3ada03d661568cd923.

## Implemented behavior

- Automatic native and Warp reset use up to four complete placement attempts.
- A deterministic, immutable bank of up to four complete scenes is certified before automatic collection. If all random attempts fail, the reset selects one certified complete scene using the same PCG draw on native and Warp.
- Failure to certify rejects configuration at startup. Explicit supplied scenes remain strictly validated and are never replaced. No obstacle is dropped; counts/radius bounds/clearance/map/targets are unchanged.
- Certificates reside on the device; reset does not transfer a CPU scene per world. Per-row generation_attempts and generation_fallback diagnostics are saved with the exact world snapshot.
- Raw state allocation remains available for manually supplied physics fixtures; production WarpVectorEnv explicitly requires certification.
- Scene generation_version3 rejects incompatible generation2 training/policy/evaluation checkpoints. Old source remains available for historical lineages.
- MASAC pure destination-enumerating critic calls use primary chunk1024, with non-reentrant activation checkpointing for differentiable chunks. Sampling stays outside, full logical batch8192 and optimizer/clipping boundaries remain unchanged.
- TRAIN/BENCH/CTRL/EVALCPU identities preserve algorithm, BASE/PQL, replay layout, environment/batch/replay sizes, generator version and seed. Mandatory preflight records stagebenchmark and retains a separate identity from reserved production.
- Shared certificate memory is counted once by the state estimator.

## Unchanged scientific settings

Primaryconfigs/comparison-20k-hybrid75-1kb.yaml:
20,000episodes;4096environments;batch8192;replay1M;seed44;35targets;25obstacles;1800steps;1024Breport/payload;3072Bbuffer;warmup131072;8updates.
Map1000m, obstacle radii30..70m and clearance10m; reward/network/energy/actor observations unchanged.
AMP, compile and fused optimizer remain off.
Final CPU evaluation30episodes/seed2000000044.

## CPU evidence

- Scenario RED6failed5passed2skipped; focused GREEN35passed4CUDA-skipped.
- Critic RED3failed1passed; GREEN39passed376subtests checks output/input gradients/full updates/RNG/optimizer counters and retained activation storage.
- Cloud identity RED4failed; focused GREEN90passed31subtests.
- Independent review found actual benchmark identity/stage and low-level allocation seams. Both were confirmed and fixed; shared-memory estimate also corrected.
- Review RED3failed4passed; focused GREEN61passed4skipped46subtests.
- Final full CPU suite: **806passed,95skipped,8warnings,1344subtests;exit0;265.05s**.
- Initial full CPU run5failed/800passed is preserved; it identified the allocation seam. It is superseded by the final passing run.
- Native seed scan **74,126/74,126passed**, no failure, one recovered retry, zero natural certified fallbacks. Covers24,096potential training seeds44..24139,30finaleval seeds2000000044..2000000073, and50,000additional seeds100000..149999.
- Forced fallback, final PCG state parity, masked reset and restored future world/action RNG are separate regressions. Zero fallback in the natural scan does not substitute for those.
- Interrupted first scan is not counted; its workers re-read YAML while a runtime schema was being edited. Fixed scan freezes the configuration and completed independently.

## Hardware gate and operations

Private diagnosticduytrngoanh/uav-g3-robust-memory-check-20261026 submitted once at12:42:25UTC26October, pinned77a1d0f; requested2T4. It mounts no credentials and writes no W&B/HF data.
Stages: focused CPU/CUDA regressions;74126CUDAseed schedule and fullGPU MASACBASE/PQL90second training segments plus exact continuation; fullCUDA repository suite.
Actual hardware and pass/failure are not yet claimed; await verified receipt.

Old running names were clarified withG2and role labels, without changing source/config or cancelling jobs.
Old MASAC PQL60fc6245 has since left the active set; old MASACBASEd17a279d remains observed running at12:44UTC. Fresh-check before reporting current state.

Prepared but NOT SUBMITTED corrected cases use clean detached worktree
/home/aduy/Documents/NCKH/uav_network-robust-g3-20261026 at77a1d0f.
Eight cases are a matched generation3 replacement of the two earlier four-way comparisons; originalG2jobs are preserved. HYBRID accounts16/17/22/23;FULLGPU18/19/20/21. All were idle with>=28GPUhours in preflight; each launch must freshly reconcile again.
The case launcher fails before submission unless both CPUfinalexit0 and matching diagnosticstatuspassed exist; exclusive intents prevent blind duplicate retry.
Auto-continue/finalCPU30eval remain required. No newproductioncase was submitted as of this record.

## Durable operational evidence

outputs/robust-training-20261026/ contains the ledger, native scan, CPUlogs/XML, independent review, source publication, diagnostic launcher/observer/receipt and gated cohort case definitions.
Kaggle failure/verification receipts must be read, not inferred from a disappearing process.
No HF/W&B data deletion or cleanup. No existing production job was cancelled or restarted.

## Final verification and launch

Real2T4 diagnostic passed: focused43pass; production3pass (74126CUDAseeds, BASE104->112updates and PQL112->120updates through exact continuation); fullCUDA877pass24skip37warnings2530subtests. Source77a1d0f. GPU1 PyTorch allocated peaks BASE13858352640B/PQL14657340928B; reserved14193524736B/15273558016B against15636037632Bdevicecapacity. These are bounded execution/resume proofs, not20kcompletion or final learning-quality results.

At 2026-10-26T13:15:10.902765+00:00 all8G3production runs were Kaggle/W&Brunning with updates>0 beyond131072warmup, actual2TeslaT4, matched budgets/protocol/layout, and8healthy cloudcontrollers. Project18Running=9training+9controllers, including one preservedG2BASElineage. Earlier pending/submission paragraphs above are historical preparation records superseded here.

| Case | W&B | Step | Updates |
|---|---|---|---|
| fullgpu-masac-base | [25eecfd1](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/25eecfd1) | 344064 | 416 |
| hybrid-masac-base | [63202ad9](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/63202ad9) | 364544 | 456 |
| fullgpu-masac-pql | [3b39b0c8](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3b39b0c8) | 327680 | 368 |
| hybrid-masac-pql | [6f0bcde3](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/6f0bcde3) | 311296 | 336 |
| fullgpu-matd3-base | [93fdf127](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/93fdf127) | 1032192 | 1760 |
| hybrid-matd3-base | [ce8f3be6](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/ce8f3be6) | 806912 | 1320 |
| fullgpu-matd3-pql | [76b73021](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/76b73021) | 1007616 | 1712 |
| hybrid-matd3-pql | [3fcf25c7](https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/3fcf25c7) | 659456 | 1016 |
