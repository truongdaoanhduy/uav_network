# Source-verified MATD3 baseline for M2 target-belief 2D

Reference: W&B run **18d756ef** `MATD3-EVAL100-ACTORBN-PLAINCRITIC-E4096-U4-B8192-H256x512-S44-18d756ef` (`https://wandb.ai/truongdaoanhduy-uit/uav_search_target/runs/18d756ef`), Kaggle bootstrap `https://www.kaggle.com/code/therwed/notebook5051433b29`.

**Provenance:** Actual W&B `.config` read via authenticated API, not inferred from run name. Extracted the exact 362-file source tarball embedded in Kaggle notebook by parsing (NOT executing) the code and verifying SHA256 `4f66fbb12ac2ff68675f2bb1590fb7066610ced96da6b264272ea5fe6b9e4aa2`. Declared `source_git_commit=0804dd75af39351c5628b4f2abda930dcbfedac7`, `source_patch_sha256=d419d142664f6d2bb274ffc268595042b7f39302eaf8c69e94acbb3a9f2dab6d`. The verified reference files are available at `/home/aduy/Downloads/M2_EXACT_REFERENCE_18d756ef_20261121/verified_source/` and the Kaggle notebook at that folder's `kaggle_notebook/notebook5051433b29.ipynb`.

## What was ported byte-identically

- `uav_network/algorithms/models.py` SHA256 **`c1b65c7d5e64574404f3f0394d8377de92d886e7ab126576153ad82a981b2f8f`**.
- `uav_network/algorithms/matd3.py` SHA256 **`e106277d2695936d983e2b558ad5cefac7a1015b24cca3b8032c727563bfeefa`**.
- `uav_network/algorithms/masac.py` SHA256 **`7e5fd9cb00c664bbea88321dcfdd3d0b75527c992b041a1a42434162a92796ce`** (MASAC source same between pinned bundles).
- `critic_chunks.py` also matches extracted run. The MATD3 and MASAC gradient *logging probes* had their axis names adapted to include the M2 packet-kind fifth continuous action, avoiding the observed MASAC `IndexError`; this does **not** change the optimizer/loss source.
- Extended M2 `Config` with `matd3.actor_batch_norm`, `matd3.critic_batch_norm`, `runtime.collector_overlap`, validation and `tracking.deferred_gpu_eval_keep_all` from the source, preserving existing M2-only features.

## Effective normalization — critical fidelity

For this run `matd3.actor_batch_norm=true`, `matd3.actor_layer_norm=true` and `matd3.critic_batch_norm=false`. In the pinned `DeterministicActor`, `mlp(..., batch_norm=True)` **uses `RolloutBatchNorm1d` instead of `LayerNorm`**; these norms are not applied simultaneously to the Actor. During no-grad rollout/eval, BN uses running statistics without mutating them; Actor training forward updates BN buffers; `MATD3.update_targets` copies running statistics to target Actor. Critic has **no BatchNorm**.

The exact Actor/Critic MLP hidden dims are `[256,512]`, shared Actor/Critic, default PyTorch init, MATD3 target-noise/delayed actor logic unchanged. Model source files are verified byte-for-byte against the actual Kaggle run, but **new M2 255D observation, 1530D observation-only Critic state and XYZ+packet-type+RF 5D continuous action change the input/output matrix shapes. Original 4-channel trained weights cannot be transplanted directly into this new M2 MDP**. The algorithm model and training protocol are the reference, not an equivalent learned checkpoint.

## Exact historical training/eval/test protocol

The reference W&B `.config` snapshot is `configs/reference-18d756ef-wandb-resolved-20261121.yaml`. Derived future-M2 profile is `configs/m2-target2d-exact-18d756ef-20261121.yaml` and passes a whole-config-dictionary equality test after allowing *only* M2 enable, output location, tracking group/tags. Key settings:

- Map **1000m**, 35 targets, 25 physical 3D obstacles, 6 UAVs, max_steps 1500, safety `reject`, `sensor.grid_cell_m=25` (the old Bayesian source already used 25m). The **new** M2 ground-only target DS F/E/U grid is therefore **40×40 cells**, not a new resolution or 3D/SLAM reconstruction.
- MATD3 `E4096`, `U4` (FOUR optimizer updates/vector step, not U8), `B8192`, replay capacity 1,000,000, CPU75/GPU25 hybrid, warmup131072 joint transitions, 20,000 completed episodes, checkpoint256, `max_train_seconds=0` (no software cutoff), seed44, actor BatchNorm effective + plain Critic.
- Collector/learner GPU roles in the actual run: collector `cuda:0`, learner `cuda:1` on T4x2. `runtime.collector_overlap=true`; `vectorize_critics=false`, `critic_chunk_size=0`, `compile_models=false`, AMP off, fused optimizers true.
- Deferred GPU checkpoint snapshot every 100 vector steps; validation of **100 fixed scenarios/checkpoint**, best-checkpoint selection and **4,000 held-out final test scenarios**; exact notebook uses `one_session.py` and `deferred_gpu_evaluation.py`. Evaluation is separate from training. Training periodic eval seed1000000044; final test seed2000000044.
- Original Actor BN rollout/evaluation mode and `actor_output_regularization=0.0` must be preserved. Replay/optimizer/runner source from the pinned tar is the ultimate protocol reference; no one-env B128/U8 diagnostic run may be called equivalent.

## Remaining hard blockers: DO NOT start misleading E4096 M2 training

1. The M2 branch **does not yet have an integrated full GPU Warp M2 environment**. The old `communication/gpu/{mac,phy,events,streaming,reports}.py` provides Warp implementation of UavNetSim-like PHY/MAC/packet processes, **not Python SimPy executing on CUDA**. It uses target IDs, old Bayesian observations and report reward; it cannot produce source-provenanced DS/SPRT verification, packet types, GCS consensus or MAP_DELTA. Stage-1 Warp F/E/U projection and sparse LLR accumulation passed T4x2 numerical tests; those are NOT full GPU rollout.
2. The exact run has an extended runner `experiments/one_session.py` and evaluator `experiments/deferred_gpu_evaluation.py`; the M2 worktree lacks them. These cannot simply be imported because their metrics, target-ID reward and snapshot ABI assume the **old scenario**. They must be ported with new M2 metrics and exact separation of train/val/test. Staging a YAML without this work is **not equivalent**.
3. Exact one-million M2 replay at local 6×255 actor observation, 1530D state and 5D action estimates **24,656 bytes/transition = 17.222GiB CPU + 5.741GiB GPU** for 75/25 split; check actual Kaggle RAM, VRAM and environment packet state before any E4096 load. Do **not** silently reduce buffer/batch/environment counts to work around memory.
4. M2's native reference contains real UavNetSim/SimPy with 1000m target-only belief. Until GPU event-ID/packet-kind/PHY/MAC/relay/GCS state, reward, safety/occlusion, fixed seeded parity and GPU memory tests are complete, keep explicit `NotImplementedError` in `make_vector`, direct `WarpVectorEnv` and the old training entry. The previously submitted Kaggle **Native one-env 120-episode** pilot is not the new matched run and must not be reported as such.
5. Additional 100 fixed-seed validation scenarios and 4000 final test episodes are true planned settings, not proof that run `18d756ef` itself reached its planned 20k training episodes or that the M2 algorithm converged.

## Test status

`pytest -q tests/test_m2_exact_18d756ef_parity_20261121.py tests/test_m2_matd3_training_20261121.py tests/test_matd3.py tests/test_masac.py tests/test_m2_logic_audit_20261121.py tests/test_m2_target_belief2d_20261121.py tests/test_m2_uavnetsim_integration.py tests/test_correlated_m2_protocol.py tests/test_m2_hardening_20261121.py` => **101 passed and 3 subtests passed** (after exact source copy). This verifies model wiring, BN operation, hot/cold replay and Native M2 reference, not full M2 Warp environment or learning convergence.

## Full-suite regression audit after exact source pin (transparent status)

The **first** full-repository run after porting exact `18d756ef` models reported `1550 passed, 21 failed, 105 skipped, 1737 subtests passed`. The earlier M2 branch already had **10 historical failures**, so 21 was **not acceptable** as a claim of bug-free migration. We investigated additional failures rather than re-labeling them as passes.

- A **real runtime bug** was found in `experiments/fixed_checkpoints.restore_fixed_learner`: exact-run gradient/Actor metrics contain both Tensor values and Python scalar hook counters. The legacy loader incorrectly called `.to(device)` on an `int`, causing resume failure. Fixed type-preserving restore with explicit validation and confirmed the previous failing `test_nonfinite_learner_metrics_survive_episode_summary_and_resume[base-matd3]` now passes. No change to the pinned model source.
- Full exact reference algorithm dependencies now include **byte-identical** `optimization.py`, `hybrid_replay.py`, `layerwise_gradient_probe.py`, `factorized_candidate.py` in addition to `models.py`, `matd3.py`, `masac.py`, `critic_chunks.py`. The two five-channel M2 gradient probes are the only adapted source files, to label the packet-kind action; their source changes are limited to the extra axis.
- Original reference runner/evaluator files were preserved in an **immutable checksum-verified ZIP** (`docs/reference_runs/18d756ef/frozen_runner_reference.zip`) rather than as executable Python modules. `tests/test_project_structure.py` pins the exact SHA of upstream `matd3.py` (543 lines), an explicitly documented exception to the repository's 500-line style constraint required for byte-identical model fidelity. The older unmodified 500-line violations remain. The source SHA contract was updated to the user-selected run, not loosened.
- After these fixes, the residual **targeted** compatibility suite (`test_matched_critic_chunks`, `test_masac_valid_routes`, `test_memory_probe`, `test_vectorized_critics`, `test_overflow_telemetry`, `test_project_structure`) recorded **68 passed, 11 failed, 3 skipped, 662 subtests passed**. Of those, 3 are unchanged historical source-size violations; `test_masac_valid_routes`'s older sparse-routing expectation is not the chosen `MASAC.shared_critic_mode=dense`; the vectorized/chunked critic tests compare old forward-hook/call-count and diagnostic-key contracts to the newer exact code's optimized path; `test_memory_probe` compares old vs newer gradient-probe magnitudes. These residual failures require **version-aware review**, not fabricated metric entries or silent changes to the user-selected algorithm.

**Conclusion:** exact source/model, verified replay layout, training hyperparameters and local Native M2 mission tests are grounded. The full M2 Warp E4096 simulator and exact train/validation/final-test flow are **not yet ported**; **no M2 convergence claim**. Do not submit the old one-env Native pilot as matched E4096.
